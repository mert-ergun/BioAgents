"""Supervisor Agent for routing tasks to specialized agents."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from bioagents.agents.supervisor_helpers import (
    check_for_empty_response_loop,
    check_for_missing_tool,
    check_for_repeated_routing,
)
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class RouteResponse(BaseModel):
    """Response from supervisor for routing."""

    next_agent: Literal[
        "research",
        "analysis",
        "coder",
        "ml",
        "dl",
        "report",
        "tool_builder",
        "protein_design",
        "critic",
        "FINISH",
    ]
    reasoning: str
    task_for_agent: str = Field(
        default="",
        description="A clear, specific instruction for the next agent explaining what task they should perform. "
        "This should be actionable and explicit about what data to fetch, analyze, or produce.",
    )


SUPERVISOR_PROMPT = load_prompt("supervisor")


# ── FIX 1: replace the fragile message-string scan with a direct memory check ──
def check_report_complete_in_memory(memory: dict) -> bool:
    """
    Return True when the report agent has already written a successful result
    to shared memory.  This is the authoritative signal that report is done.

    Args:
        memory: The shared memory dict from AgentState

    Returns:
        True if report memory shows success
    """
    report_mem = memory.get("report", {})
    is_complete = report_mem.get("status") == "success" and bool(report_mem.get("data"))
    if is_complete:
        logger.info(
            "Supervisor: report memory shows status=success with data present. "
            "Report is done — routing to FINISH."
        )
    return is_complete


def create_supervisor_agent(members: list[str]):
    """
    Create the Supervisor Agent that routes based on shared memory state.

    Key change: Supervisor now reads from state['memory'] to determine routing,
    not from messages. Messages are used only for orchestration signals.

    Args:
        members: List of available agent names

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm(prompt_name="supervisor")

    options = [*members, "FINISH"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above and the shared memory state below, who should act next? "
                f"Choose from: {', '.join(options)!s}\n\n"
                "SHARED MEMORY STATUS:\n"
                "{memory_status}"
                # ── FIX 2: explicit instruction added to the LLM prompt so the model
                #    itself understands it must not re-route to report once done ──
                "\n\nCRITICAL ROUTING RULE: If the 'report' agent already has "
                "status=success in shared memory, you MUST choose FINISH. "
                "Do NOT route to 'report' again under any circumstances.",
            ),
        ]
    )

    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    def supervisor_node(state):
        """
        The supervisor node function (memory-aware).

        Args:
            state: The current AgentState

        Returns:
            A dict with next_agent, reasoning, and messages
        """
        messages = state["messages"]
        memory = state.get("memory", {})

        # ── FIX 3: check memory FIRST — this is the most reliable signal ──
        if check_report_complete_in_memory(memory):
            return {
                "next": "FINISH",
                "reasoning": "Report agent completed successfully (confirmed via shared memory). Finishing.",
                "messages": [],
                "memory": memory,
            }

        # Check for loops from messages (existing logic)
        is_empty_loop, empty_agent = check_for_empty_response_loop(messages)
        if is_empty_loop:
            logger.error(
                f"Supervisor: Detected empty response loop from agent '{empty_agent}'. "
                "Terminating to prevent infinite loop."
            )
            error_msg = SystemMessage(
                content=f"[SYSTEM] Workflow terminated: Agent '{empty_agent}' failed to produce "
                "a response after multiple attempts."
            )
            return {
                "next": "FINISH",
                "reasoning": f"Loop detected: agent '{empty_agent}' returned empty responses repeatedly.",
                "messages": [error_msg],
                "memory": memory,
            }

        is_routing_loop, looping_agent = check_for_repeated_routing(messages)
        if is_routing_loop:
            logger.warning(
                f"Supervisor: Detected repeated routing to agent '{looping_agent}'. "
                "Attempting to break the loop."
            )
            # For report agent specifically, finish the workflow
            if looping_agent == "report":
                logger.info("Supervisor: Report appears to be looping. Finishing workflow.")
                return {
                    "next": "FINISH",
                    "reasoning": "Report agent completed. Breaking loop by finishing workflow.",
                    "messages": [],
                    "memory": memory,
                }
            # For protein_design agent, finish the workflow since it has gathered data
            elif looping_agent == "protein_design":
                logger.info("Supervisor: Protein design appears to be looping. Finishing workflow.")
                return {
                    "next": "FINISH",
                    "reasoning": "Protein design agent has gathered sufficient data. Breaking loop by finishing workflow.",
                    "messages": [],
                    "memory": memory,
                }
            # For other agents, try to escalate to report if available
            elif looping_agent != "report" and "report" in members:
                return {
                    "next": "report",
                    "reasoning": f"Loop detected: agent '{looping_agent}' was called repeatedly without progress. "
                    "Escalating to report agent to summarize current state.",
                    "messages": [],
                    "memory": memory,
                }
            else:
                error_msg = SystemMessage(
                    content=f"[SYSTEM] Workflow terminated: Detected repeated routing to agent "
                    f"'{looping_agent}' without progress. The task may require manual intervention."
                )
                return {
                    "next": "FINISH",
                    "reasoning": f"Loop detected: agent '{looping_agent}' was called repeatedly. Terminating.",
                    "messages": [error_msg],
                    "memory": memory,
                }

        if "tool_builder" in members:
            should_route_to_builder, reason = check_for_missing_tool(messages)
            if should_route_to_builder:
                logger.info(f"Supervisor: Overriding to route to tool_builder - {reason}")
                return {
                    "next": "tool_builder",
                    "reasoning": f"Programmatic detection: {reason}. Routing to tool_builder to create the missing tool.",
                    "messages": [],
                    "memory": memory,
                }

        # Normal LLM-based routing
        result = supervisor_chain.invoke({
            "messages": messages,
            "memory_status": build_memory_status_summary(memory, members),
        })

        # ── FIX 4: post-LLM safety net — if the LLM somehow still routes to
        #    report despite the prompt instruction, override it to FINISH ──
        if result.next_agent == "report" and check_report_complete_in_memory(memory):
            logger.warning(
                "Supervisor: LLM tried to re-route to report despite it being complete. "
                "Overriding to FINISH."
            )
            return {
                "next": "FINISH",
                "reasoning": "Report already complete. Overriding LLM re-route to FINISH.",
                "messages": [],
                "memory": memory,
            }

        handoff_messages = []
        if result.next_agent != "FINISH" and result.task_for_agent:
            handoff_msg = HumanMessage(
                content=f"[SUPERVISOR TASK] {result.task_for_agent}",
                name="Supervisor"
            )
            handoff_messages.append(handoff_msg)
            logger.info(
                f"Supervisor handoff to {result.next_agent}: {result.task_for_agent}"
            )

        return {
            "next": result.next_agent,
            "reasoning": result.reasoning,
            "messages": handoff_messages,
            "memory": memory,
        }

    return supervisor_node


def build_memory_status_summary(memory: dict, members: list[str]) -> str:
    """
    Build a summary of memory state for supervisor decision-making.

    Args:
        memory: The shared memory dict
        members: List of agent names

    Returns:
        Formatted string of memory status
    """
    summary_lines = []

    for agent_name in members:
        agent_mem = memory.get(agent_name, {})
        status = agent_mem.get("status", "pending")
        timestamp = agent_mem.get("timestamp", "N/A")
        has_data = bool(agent_mem.get("data", {}))
        completeness = agent_mem.get("data", {}).get("completeness", "unknown")
        errors = agent_mem.get("errors", [])

        error_str = f", errors: {errors}" if errors else ""
        summary_lines.append(
            f"  {agent_name}: status={status}, has_data={has_data}, completeness={completeness}{error_str}"
        )

    return "\n".join(summary_lines)