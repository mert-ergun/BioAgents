"""Supervisor Agent for routing tasks to specialized agents."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from bioagents.agents.helpers import extract_best_content, get_message_content
from bioagents.agents.supervisor_helpers import (
    check_code_agent_task_completed,
    check_coder_should_force_finish,
    check_finish_if_code_agent_substantive_repeat,
    check_for_empty_response_loop,
    check_for_missing_tool,
    check_for_repeated_routing,
    check_for_research_failure,
    check_tool_builder_execution_success,
    check_tool_builder_success,
    extract_original_query,
    get_all_created_tools,
)
from bioagents.llms.llm_provider import get_llm, get_structured_output_kwargs_for_routing
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
        "literature",
        "web_browser",
        "paper_replication",
        "data_acquisition",
        "genomics",
        "transcriptomics",
        "structural_biology",
        "phylogenetics",
        "docking",
        "planner",
        "tool_validator",
        "tool_discovery",
        "prompt_optimizer",
        "result_checker",
        "shell",
        "git",
        "environment",
        "visualization",
        "FINISH",
    ]
    reasoning: str
    task_for_agent: str = Field(
        default="",
        description="A clear, specific instruction for the next agent explaining what task they should perform. "
        "This should be actionable and explicit about what data to fetch, analyze, or produce.",
    )


SUPERVISOR_PROMPT = load_prompt("supervisor")


def create_supervisor_agent(members: list[str]):
    """
    Create the Supervisor Agent for routing.

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
                "Given the conversation above, who should act next? "
                f"Choose from: {', '.join(options)!s}",
            ),
        ]
    )

    supervisor_chain = prompt | llm.with_structured_output(
        RouteResponse,
        **get_structured_output_kwargs_for_routing(),
    )

    def supervisor_node(state):
        """
        The supervisor node function.

        Args:
            state: The current AgentState

        Returns:
            A dict with next_agent and reasoning for routing
        """
        messages = state["messages"]

        # FIRST: Check for tool builder execution success (only if not already handled)
        # Skip if SystemMessage already exists (already handled, let normal LLM routing decide next step)
        execution_already_handled = any(
            isinstance(msg, SystemMessage)
            or (hasattr(msg, "__class__") and msg.__class__.__name__ == "SystemMessage")
            for msg in messages[-10:]
            if "[EXECUTION_SUCCESS]" in get_message_content(msg)
        )

        if not execution_already_handled:
            # Only check for execution success if we haven't already handled it
            execution_success, exec_tool_name = check_tool_builder_execution_success(messages)
            if execution_success:
                logger.info(
                    f"ToolBuilder successfully executed tool '{exec_tool_name}' and returned results. "
                    "Routing to report to summarize findings."
                )
                # Mark execution success to prevent re-detection
                success_marker = SystemMessage(
                    content=f"[EXECUTION_SUCCESS] ToolBuilder successfully executed tool '{exec_tool_name}' and task is complete."
                )
                return {
                    "next": "report",
                    "reasoning": f"ToolBuilder successfully executed tool '{exec_tool_name}' and completed the task. "
                    "Routing to report to summarize the results.",
                    "messages": [success_marker],
                }

        failed_agents: set = state.get("failed_agents") or set()

        force_finish, finish_reason = check_coder_should_force_finish(messages)
        if force_finish:
            logger.info(f"Supervisor: Forcing FINISH after code agent exhaustion — {finish_reason}")
            return {
                "next": "FINISH",
                "reasoning": finish_reason,
                "messages": [],
            }

        finish_repeat, repeat_reason = check_finish_if_code_agent_substantive_repeat(messages)
        if finish_repeat:
            logger.info(
                "Supervisor: Forcing FINISH after substantive code-agent repeat — %s", repeat_reason
            )
            return {
                "next": "FINISH",
                "reasoning": repeat_reason,
                "messages": [],
            }

        task_done, done_reason = check_code_agent_task_completed(messages)
        if task_done:
            logger.info("Supervisor: Forcing FINISH — code agent completed task — %s", done_reason)
            return {
                "next": "FINISH",
                "reasoning": done_reason,
                "messages": [],
            }

        is_empty_loop, empty_agent = check_for_empty_response_loop(messages)
        if is_empty_loop:
            logger.error(f"Supervisor: Detected empty response loop from agent '{empty_agent}'.")
            failed_agents = failed_agents | {empty_agent.lower()}

            if empty_agent.lower() == "report" and "report" in failed_agents:
                best = extract_best_content(messages)
                if best:
                    logger.info("Report failed; injecting best available content and finishing.")
                    summary_msg = AIMessage(
                        content=(
                            f"## Results\n\n"
                            f"The report agent was unable to synthesize a final report. "
                            f"Below is the most substantive output from the workflow:\n\n{best}"
                        ),
                        name="Report",
                    )
                    return {
                        "next": "FINISH",
                        "reasoning": f"Agent '{empty_agent}' failed with empty responses. "
                        "Injecting best available content and finishing.",
                        "messages": [summary_msg],
                        "failed_agents": failed_agents,
                    }

            alternative = None
            if (
                empty_agent.lower() != "report"
                and "report" not in failed_agents
                and "report" in members
            ):
                alternative = "report"
            elif (
                empty_agent.lower() != "coder"
                and "coder" not in failed_agents
                and "coder" in members
            ):
                alternative = "coder"

            if alternative:
                logger.info(
                    f"Empty loop from '{empty_agent}', trying alternative agent '{alternative}'."
                )
                return {
                    "next": alternative,
                    "reasoning": f"Agent '{empty_agent}' returned empty responses. "
                    f"Routing to '{alternative}' as fallback.",
                    "messages": [],
                    "failed_agents": failed_agents,
                }

            best = extract_best_content(messages)
            if best:
                summary_msg = AIMessage(
                    content=(
                        f"## Results\n\n"
                        f"Multiple agents failed to produce a response. "
                        f"Below is the most substantive output from the workflow:\n\n{best}"
                    ),
                    name="Summary",
                )
                return {
                    "next": "FINISH",
                    "reasoning": f"Agent '{empty_agent}' and fallbacks failed. "
                    "Injecting best content and finishing.",
                    "messages": [summary_msg],
                    "failed_agents": failed_agents,
                }

            error_msg = SystemMessage(
                content=f"[SYSTEM] Workflow terminated: Agent '{empty_agent}' failed to produce "
                "a response after multiple attempts. This may indicate a configuration issue "
                "or the agent lacks the necessary tools to complete the task."
            )
            return {
                "next": "FINISH",
                "reasoning": f"Loop detected: agent '{empty_agent}' returned empty responses repeatedly. "
                "Terminating to prevent infinite loop.",
                "messages": [error_msg],
                "failed_agents": failed_agents,
            }

        is_routing_loop, looping_agent = check_for_repeated_routing(messages)
        if is_routing_loop:
            logger.warning(
                f"Supervisor: Detected repeated routing to agent '{looping_agent}'. "
                "Attempting to break the loop."
            )

            if looping_agent == "ToolBuilder":
                execution_success, exec_tool_name = check_tool_builder_execution_success(messages)
                if execution_success:
                    logger.info(
                        f"ToolBuilder successfully executed tool '{exec_tool_name}' and returned results. "
                        "Routing to report to summarize findings."
                    )
                    success_marker = SystemMessage(
                        content=f"[EXECUTION_SUCCESS] ToolBuilder successfully executed tool '{exec_tool_name}' and task is complete."
                    )
                    return {
                        "next": "report",
                        "reasoning": f"ToolBuilder successfully executed tool '{exec_tool_name}' and completed the task. "
                        "Routing to report to summarize the results.",
                        "messages": [success_marker],
                    }

                tool_success, tool_name = check_tool_builder_success(messages)
                if tool_success:
                    created_tools = get_all_created_tools(messages)
                    original_query = extract_original_query(messages)

                    if created_tools:
                        tools_list = ", ".join([f"`{t}`" for t in created_tools])
                        if original_query:
                            task_instruction = (
                                f"Use the newly created tools {tools_list} to complete the original task: {original_query}. "
                                f"First, search for these tools using search_custom_tools, then execute them with execute_custom_tool."
                            )
                        else:
                            task_instruction = (
                                f"Use the newly created tools {tools_list} to complete the task. "
                                f"First, search for these tools using search_custom_tools, then execute them with execute_custom_tool."
                            )
                    else:
                        task_instruction = (
                            f"Use the newly created tool `{tool_name}` to complete the original task. "
                            f"Search for it using search_custom_tools, then execute it with execute_custom_tool."
                        )
                        if original_query:
                            task_instruction = f"{task_instruction} Original task: {original_query}"

                    target_agent = "research" if "research" in members else "analysis"
                    handoff_msg = HumanMessage(
                        content=f"[SUPERVISOR TASK] {task_instruction}", name="Supervisor"
                    )
                    logger.info(
                        f"ToolBuilder created tool(s): {created_tools or [tool_name]}. "
                        f"Routing to {target_agent} to use the tool(s)."
                    )
                    return {
                        "next": target_agent,
                        "reasoning": f"ToolBuilder successfully created tool(s): {created_tools or [tool_name]}. "
                        f"Routing to {target_agent} to execute the tool(s) and complete the task.",
                        "messages": [handoff_msg],
                    }

            looping_lower = looping_agent.lower()
            failed_agents = failed_agents | {looping_lower}
            loop_escape_tried: dict[str, list[str]] = dict(state.get("loop_escape_tried") or {})
            tried_for_agent = list(loop_escape_tried.get(looping_lower, []))

            _DATA_AGENTS = {"web_browser", "data_acquisition", "literature"}
            if looping_lower in _DATA_AGENTS:
                escape_order = ["data_acquisition", "coder", "literature", "report", "research"]
            else:
                escape_order = ["coder", "data_acquisition", "research", "report"]

            alternatives = [
                m
                for m in escape_order
                if m != looping_lower
                and m not in failed_agents
                and m in members
                and m not in tried_for_agent
            ]
            if alternatives:
                alt = alternatives[0]
                tried_for_agent.append(alt)
                loop_escape_tried[looping_lower] = tried_for_agent
                logger.info(
                    "Loop break for '%s': routing to '%s' (escape sequence so far: %s)",
                    looping_agent,
                    alt,
                    tried_for_agent,
                )
                return {
                    "next": alt,
                    "reasoning": f"Loop detected: agent '{looping_agent}' was called repeatedly without progress. "
                    f"Routing to '{alt}' as alternative.",
                    "messages": [],
                    "failed_agents": failed_agents,
                    "loop_escape_tried": loop_escape_tried,
                }
            else:
                best = extract_best_content(messages)
                finish_messages = []
                if best:
                    finish_messages.append(
                        AIMessage(
                            content=f"## Results\n\nThe workflow encountered routing issues. Best output:\n\n{best}",
                            name="Summary",
                        )
                    )
                else:
                    finish_messages.append(
                        SystemMessage(
                            content=f"[SYSTEM] Workflow terminated: Detected repeated routing to agent "
                            f"'{looping_agent}' without progress. The task may require manual intervention."
                        )
                    )
                return {
                    "next": "FINISH",
                    "reasoning": f"Loop detected: agent '{looping_agent}' was called repeatedly. All alternatives exhausted. Terminating.",
                    "messages": finish_messages,
                    "failed_agents": failed_agents,
                    "loop_escape_tried": loop_escape_tried,
                }

        if "tool_builder" in members:
            should_route_to_builder, reason = check_for_missing_tool(messages)
            if should_route_to_builder:
                logger.info(f"Supervisor: Overriding to route to tool_builder - {reason}")
                return {
                    "next": "tool_builder",
                    "reasoning": f"Programmatic detection: {reason}. Routing to tool_builder to create the missing tool.",
                    "messages": [],
                }

        # Before normal routing, check if research already failed—avoid re-sending
        research_failed, research_fail_reason = check_for_research_failure(messages)

        # Detect if planner just responded — prevent routing back to planner
        planner_just_responded = False
        if messages:
            for msg in reversed(messages[-3:]):
                if isinstance(msg, AIMessage) and getattr(msg, "name", "") == "Planner":
                    planner_just_responded = True
                    break
                if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" in str(msg.content):
                    break

        # Window messages for the routing LLM to avoid slow calls on large histories.
        # Keep the first message (user query) and the last 12 messages (recent context).
        if len(messages) > 14:
            routing_messages = messages[:1] + messages[-12:]
        else:
            routing_messages = messages

        # Normal LLM-based routing
        result = supervisor_chain.invoke({"messages": routing_messages})
        if result is None:
            logger.warning(
                "Supervisor: structured routing returned None (likely parse failure); retrying once."
            )
            result = supervisor_chain.invoke({"messages": messages})

        if result is None:
            logger.error(
                "Supervisor: structured routing failed twice; finishing with error instead of crashing."
            )
            return {
                "next": "FINISH",
                "reasoning": "Supervisor could not obtain a valid routing decision from the model.",
                "messages": [
                    SystemMessage(
                        content="[SYSTEM] Supervisor routing failed: the model did not return a valid "
                        "structured routing response after two attempts. The task may be too long or the "
                        "model output was unparsable."
                    )
                ],
            }

        if research_failed and result.next_agent == "research":
            alt = "coder" if "coder" in members else "report"
            logger.warning(
                "Supervisor: Research already failed; overriding routing from "
                "'research' to '%s'. Reason: %s",
                alt,
                research_fail_reason,
            )
            result.next_agent = alt

        if planner_just_responded and result.next_agent == "planner":
            for candidate in ("research", "data_acquisition", "coder", "report"):
                if candidate in members:
                    result.next_agent = candidate
                    break
            # Extract original user query so the execution agent knows what to do
            original_query = ""
            for msg in messages:
                if isinstance(msg, HumanMessage) and "[SUPERVISOR TASK]" not in str(
                    msg.content
                ):
                    original_query = str(msg.content)[:800]
                    break
            result.task_for_agent = (
                f"A plan has been created (see conversation history). "
                f"Begin executing the first data-retrieval steps now. "
                f"Original user request: {original_query}"
            )
            logger.warning(
                "Supervisor: Planner already produced a plan; skipping duplicate routing. "
                "Proceeding to %s with execution task.",
                result.next_agent,
            )

        handoff_messages = []
        if result.next_agent != "FINISH" and result.task_for_agent:
            handoff_msg = HumanMessage(
                content=f"[SUPERVISOR TASK] {result.task_for_agent}", name="Supervisor"
            )
            handoff_messages.append(handoff_msg)
            logger.info(f"Supervisor handoff to {result.next_agent}: {result.task_for_agent}")

        return {
            "next": result.next_agent,
            "reasoning": result.reasoning,
            "messages": handoff_messages,
        }

    return supervisor_node
