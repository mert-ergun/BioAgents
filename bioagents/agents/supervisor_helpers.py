"""Helper functions for the Supervisor Agent."""

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bioagents.agents.helpers import get_message_content
from bioagents.sandbox.coder_helpers import COMPLETION_MARKER
from bioagents.tools.result_status import tool_result_status


def _env_int_local(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on any bad value."""
    import os

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


logger = logging.getLogger(__name__)

MAX_EMPTY_RESPONSES = 1

TOOL_MISSING_PATTERNS = [
    r"no suitable tool found",
    r"no tool found",
    r"tool not available",
    r"cannot find a tool",
    r"no tools found",
    r"failed to find.*tool",
    r"could not find a suitable",
    r"no suitable.*found",
    r"cannot.*search.*by gene name",
    r"missing required parameters",
    r"tool.*not.*exist",
    r"no.*capability",
    r"lacks.*capability",
]


_CODE_AGENT_NAMES = frozenset({"Coder", "ML", "DL"})

# An explicit declaration that the work is finished. The ML-training vocabulary below
# is kept, but it alone never matched a bioinformatics run: a coder that ended with
# "TASK COMPLETE" and a list of output files read as unfinished and was re-delegated.
_TASK_COMPLETION_MARKERS = re.compile(
    r"(TASK\s+COMPLETED?\b|ANALYSIS\s+COMPLETE\b|WORKFLOW\s+COMPLETE\b"
    r"|PIPELINE\s+COMPLETE\b|ALL\s+STEPS\s+COMPLETED?\b"
    r"|METRIC:|Accuracy[:\s]+\d|Loss[:\s]+\d|Epoch\s+\d+/\d+"
    r"|training complete|evaluation accuracy|successfully trained"
    r"|model trained|F1[:\s]+\d|AUC[:\s]+\d|precision[:\s]+\d|recall[:\s]+\d)",
    re.IGNORECASE,
)

# Negations that make an apparent completion claim the opposite of one.
_NOT_COMPLETE = re.compile(
    r"\b(not|never|could ?n.t|cannot|unable to|failed to|without)\s+"
    r"(fully\s+|yet\s+)?(complete|finish)",
    re.IGNORECASE,
)


def _is_substantive_code_agent_output(content: str) -> bool:
    """True if a code agent response looks like a real implementation, not a one-line stub."""
    c = content.strip()
    if len(c) < 400:
        return False
    if "```" in c or "nn.Module" in c or "torch.nn" in c or "sklearn" in c:
        return True
    return len(c) >= 1200


def check_code_agent_task_completed(messages) -> tuple[bool, str]:
    """Finish if the last code agent message contains execution metrics indicating the task is done."""
    if not messages:
        return False, ""
    last = messages[-1]
    if not isinstance(last, AIMessage) and not (
        hasattr(last, "__class__") and last.__class__.__name__ == "AIMessage"
    ):
        return False, ""
    name = getattr(last, "name", None)
    if name not in _CODE_AGENT_NAMES:
        return False, ""
    content = get_message_content(last)

    # The coder stamps this when smolagents actually delivered a final answer, which is
    # a fact about the run rather than a guess from the prose.
    if COMPLETION_MARKER in content:
        return True, (f"Code agent '{name}' delivered its final answer; the task is complete.")

    if len(content.strip()) < 200:
        return False, ""
    if _NOT_COMPLETE.search(content):
        return False, ""
    if _TASK_COMPLETION_MARKERS.search(content):
        return True, (
            f"Code agent '{name}' reported the task complete and produced substantive "
            "output, so it must not be delegated again."
        )
    return False, ""


def check_finish_if_code_agent_substantive_repeat(messages) -> tuple[bool, str]:
    """
    If Coder/ML/DL has returned substantive output in 2+ turns, finish the workflow.

    Prevents supervisor↔code-agent loops where the LLM keeps re-delegating for
    spurious "refinement" after the task is already satisfied.
    """
    if len(messages) < 2:
        return False, ""
    last = messages[-1]
    if not isinstance(last, AIMessage) and not (
        hasattr(last, "__class__") and last.__class__.__name__ == "AIMessage"
    ):
        return False, ""
    name = getattr(last, "name", None)
    if name not in _CODE_AGENT_NAMES:
        return False, ""
    content = get_message_content(last)
    if not _is_substantive_code_agent_output(content):
        return False, ""

    count = 0
    for msg in messages[-14:]:
        if (
            isinstance(msg, AIMessage)
            or (hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage")
        ) and getattr(msg, "name", None) == name:
            count += 1
    if count >= 2:
        return True, (
            f"Code agent '{name}' has already produced substantive output in multiple turns; "
            "finishing to avoid redundant delegation."
        )
    return False, ""


def check_coder_should_force_finish(messages) -> tuple[bool, str]:
    """
    If a code agent (Coder/ML/DL) ended with exhaustion or sandbox denial markers,
    the supervisor must not delegate the same execution again—finish the workflow.
    """
    if not messages:
        return False, ""
    last = messages[-1]
    if not isinstance(last, AIMessage) and not (
        hasattr(last, "__class__") and last.__class__.__name__ == "AIMessage"
    ):
        return False, ""
    if getattr(last, "name", None) not in _CODE_AGENT_NAMES:
        return False, ""
    content = get_message_content(last)
    if (
        "[CODER_STATUS: max_steps_reached]" in content
        or "[CODER_STATUS: repeated_parse_errors]" in content
        or "[CODER_STATUS: timeout]" in content
        or "[CODER_STATUS: empty_response]" in content
        or "[CODER_STATUS: repeated_code]" in content
    ):
        return True, (
            "Code agent finished with step-limit, parse-loop exhaustion, or timeout; "
            "terminating workflow to avoid redundant delegation."
        )
    if "[CODER_STATUS: import_denied]" in content:
        return True, (
            "Code agent could not import a required package in the sandbox; "
            "terminating workflow to avoid an execution loop."
        )
    if "[CODER_STATUS: import_failed]" in content:
        return True, (
            "Code agent could not import a required library (missing shared object / module); "
            "terminating workflow to avoid an execution loop."
        )
    if "[CODER_STATUS: sandbox_unrecoverable]" in content:
        # Kernel is gone and cannot be rebuilt; re-delegating only produces more 404s.
        return True, (
            "Code agent lost its execution sandbox and could not recover it; "
            "terminating workflow rather than re-running code that cannot execute."
        )
    return False, ""


def check_for_empty_response_loop(messages) -> tuple[bool, str]:
    """
    Check if the last agent returned an empty response, indicating a potential loop.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (is_loop_detected, agent_name_that_failed)
    """
    if len(messages) < 2:
        return False, ""

    consecutive_empty = 0
    last_agent = ""

    for msg in reversed(messages[-6:]):
        if isinstance(msg, AIMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage"
        ):
            content = get_message_content(msg)
            agent_name = getattr(msg, "name", "unknown")

            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            is_empty = not content.strip() and not has_tool_calls

            if is_empty:
                consecutive_empty += 1
                if not last_agent:
                    last_agent = agent_name
            else:
                break

    if consecutive_empty >= MAX_EMPTY_RESPONSES:
        logger.warning(
            f"Loop detected: {consecutive_empty} consecutive empty responses from agent '{last_agent}'"
        )
        return True, last_agent

    return False, ""


def check_for_repeated_routing(messages) -> tuple[bool, str]:
    """
    Check message history for signs of repeated routing to the same agent.

    Looks for patterns like the same agent being called multiple times without progress.
    Only counts AIMessage instances with agent names, not tool names.

    Also detects tool-call pattern loops where the research agent repeatedly calls
    tool_universe_find_tools without ever calling tool_universe_call_tool.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (is_loop_detected, agent_name)
    """
    if len(messages) < 4:
        return False, ""

    valid_agent_names = {
        "ToolBuilder",
        "Research",
        "Analysis",
        "Coder",
        "ML",
        "DL",
        "Report",
        "ProteinDesign",
        "Critic",
        "Supervisor",
        "Summary",
        "Literature",
        "WebBrowser",
        "PaperReplication",
        "DataAcquisition",
        "Genomics",
        "Transcriptomics",
        "StructuralBiology",
        "Phylogenetics",
        "Docking",
        "Planner",
        "ToolValidator",
        "ToolDiscovery",
        "PromptOptimizer",
        "ResultChecker",
        "Shell",
        "Git",
        "Environment",
        "Visualization",
    }

    agent_counts: dict[str, int] = {}
    agent_content_sigs: dict[str, list[str]] = {}
    for msg in messages[-20:]:
        if isinstance(msg, AIMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage"
        ):
            agent_name = getattr(msg, "name", None)
            if agent_name and agent_name in valid_agent_names:
                agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
                content = getattr(msg, "content", "")
                if isinstance(content, str) and len(content) > 100:
                    sig = content[:200]
                    agent_content_sigs.setdefault(agent_name, []).append(sig)

    # Research legitimately emits 3 named AIMessages per pass, so flag only at 5+.
    AGENT_APPEARANCE_LOOP_THRESHOLD = 5
    candidates = [
        (agent, count)
        for agent, count in agent_counts.items()
        if count >= AGENT_APPEARANCE_LOOP_THRESHOLD
    ]

    for agent_name, sigs in agent_content_sigs.items():
        if len(sigs) >= 3 and agent_name not in {c[0] for c in candidates}:
            unique = set(sigs)
            if len(unique) <= len(sigs) // 2:
                candidates.append((agent_name, len(sigs)))

    # Many find_tools calls with zero call_tool means sub-agents are stuck searching.
    find_tools_calls = 0
    call_tool_calls = 0
    for msg in messages[-30:]:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name == "tool_universe_find_tools":
                find_tools_calls += 1
            elif name == "tool_universe_call_tool":
                call_tool_calls += 1
    # 4 sub-agents x 2 search rounds = 8; beyond that with no execution is a loop.
    if find_tools_calls >= 9 and call_tool_calls == 0:
        logger.warning(
            "Tool-call loop detected: %d tool_universe_find_tools calls with 0 "
            "tool_universe_call_tool calls in last 30 messages.",
            find_tools_calls,
        )
        candidates.append(("Research", find_tools_calls))

    if not candidates:
        return False, ""

    # Highest count wins; on ties deprioritize synthesis agents to break work loops.
    def _loop_sort_key(item: tuple[str, int]) -> tuple:
        name, cnt = item
        deprior = 1 if name in ("Report", "Summary", "Supervisor") else 0
        return (-cnt, deprior, name)

    candidates.sort(key=_loop_sort_key)
    agent, count = candidates[0]
    logger.warning(
        f"Potential loop: agent '{agent}' appeared {count} times in last 20 messages "
        f"(among {len(candidates)} agent(s) over threshold)"
    )
    return True, agent


_RESEARCH_FAILURE_PATTERNS = re.compile(
    r"(no\s+(genetic|transcriptomic|proteomic|omics)\s+data\s+(was|were|could be)\s+(retrieved|collected|obtained)"
    r"|foundational\s+data\s+collection\s+steps?\s+failed"
    r"|no\s+substantive\s+data\s+was\s+collected"
    r"|could\s+not\s+be\s+successfully\s+(retrieved|executed|completed)"
    r"|this\s+foundational\s+step.*was\s+unsuccessful"
    r"|no\s+specific\s+tool\s+calls?\s+were\s+executed"
    r"|tool\s+searches?\s+were\s+unsuccessful)",
    re.IGNORECASE,
)


def check_for_research_failure(messages) -> tuple[bool, str]:
    """
    Check if the last research agent message indicates a data-collection failure.

    Prevents the supervisor from re-sending the same task to research when the
    previous attempt already reported that no data could be retrieved.
    """
    for msg in reversed(messages[-5:]):
        if not (
            isinstance(msg, AIMessage)
            or (hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage")
        ):
            continue
        name = getattr(msg, "name", "")
        if name != "Research":
            continue
        content = get_message_content(msg)
        if _RESEARCH_FAILURE_PATTERNS.search(content):
            return True, (
                "Research agent already reported data-collection failure. "
                "Re-sending the same task will not produce different results."
            )
    return False, ""


def check_for_missing_tool(messages) -> tuple[bool, str]:
    """
    Check recent messages for patterns indicating a tool is missing.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (should_route_to_tool_builder, reason)
    """
    recent_messages = messages[-3:] if len(messages) >= 3 else messages

    for msg in recent_messages:
        content = ""
        if hasattr(msg, "content"):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        content_lower = content.lower()

        for pattern in TOOL_MISSING_PATTERNS:
            if re.search(pattern, content_lower):
                logger.info(f"Detected missing tool pattern: '{pattern}' in message")
                return True, f"Detected missing tool: pattern '{pattern}' matched"

    return False, ""


def check_tool_builder_execution_success(messages) -> tuple[bool, str]:
    """Find a successful custom-tool step; this does not establish task completion."""
    for msg in reversed(messages[-10:]):
        if isinstance(msg, ToolMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "ToolMessage"
        ):
            content = get_message_content(msg)
            try:
                result = json.loads(content)
                if "tool" in result and "status" in result:
                    # The latest execution is authoritative; do not fall back to an
                    # older success after a failed, pending or diagnostic-only run.
                    if tool_result_status(result) != "success" or "result" not in result:
                        return False, ""
                    tool_name = result.get("tool")
                    if not isinstance(tool_name, str) or not tool_name:
                        return False, ""
                    logger.info(f"Detected tool execution success from ToolMessage: {tool_name}")
                    return True, tool_name
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    return False, ""


def check_tool_builder_success(messages) -> tuple[bool, str]:
    """
    Check if tool_builder has successfully created a tool.

    Uses a lightweight LLM call to determine if ToolBuilder successfully created a tool.
    This is more reliable than pattern matching and handles various message formats.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (tool_created, tool_name)
    """
    for msg in reversed(messages[-10:]):
        if isinstance(msg, ToolMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "ToolMessage"
        ):
            content = get_message_content(msg)
            try:
                result = json.loads(content)
                if result.get("status") == "success":
                    if "tool_name" in result:
                        tool_name = result.get("tool_name")
                        logger.info(f"Detected tool success from ToolMessage: {tool_name}")
                        return True, tool_name
                    message = result.get("message", "")
                    if "validated successfully" in message.lower():
                        tool_name_match = re.search(
                            r"Tool '([^']+)' validated successfully", message
                        )
                        if tool_name_match:
                            tool_name = tool_name_match.group(1)
                            logger.info(
                                f"Detected tool validation success from ToolMessage: {tool_name}"
                            )
                            return True, tool_name
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    # Fallback: lightweight LLM call over the last 5 messages to keep tokens low.
    recent_messages = messages[-5:]
    if not recent_messages:
        return False, ""

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field

        from bioagents.llms.llm_provider import get_llm

        llm = get_llm(prompt_name="supervisor")

        class ToolBuilderSuccessResponse(BaseModel):
            """Response indicating if ToolBuilder successfully created a tool."""

            success: bool = Field(description="True if ToolBuilder successfully created a tool")
            tool_name: str = Field(
                default="", description="Name of the tool if created, empty string otherwise"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are analyzing conversation messages to determine if the ToolBuilder agent "
                    "successfully created a tool. Look for indicators like 'tool registered successfully', "
                    "'tool validated successfully', 'created a new tool', or similar success messages. "
                    "Respond with whether a tool was successfully created and its name if available.",
                ),
                (
                    "human",
                    "Did the ToolBuilder agent successfully create a tool in these messages?\n\n{context}",
                ),
            ]
        )

        tool_builder_context = []
        for msg in recent_messages:
            agent_name = getattr(msg, "name", "")
            if agent_name == "ToolBuilder":
                content = get_message_content(msg)
                if len(content) > 500:
                    content = content[:500] + "..."
                tool_builder_context.append(f"[ToolBuilder]: {content}")

        if not tool_builder_context:
            return False, ""

        context_text = "\n".join(tool_builder_context)

        chain = prompt | llm.with_structured_output(ToolBuilderSuccessResponse)
        result = chain.invoke({"context": context_text})

        if result.success:
            logger.info(f"LLM detected tool success: tool_name={result.tool_name or 'unknown'}")
            return True, result.tool_name or "unknown"

    except Exception as e:
        logger.warning(
            f"LLM-based tool success check failed: {e}, falling back to pattern matching"
        )
        return _check_tool_builder_success_patterns(messages)

    return False, ""


def _check_tool_builder_success_patterns(messages) -> tuple[bool, str]:
    """
    Fallback pattern matching for tool builder success detection.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (tool_created, tool_name)
    """
    success_patterns = [
        r"successfully created.*tool",
        r"tool.*registered successfully",
        r"tool.*validated successfully",
        r"created.*registered.*validated",
        r"i have created.*tool",
        r"created a new tool",
        r"tool.*has been.*created",
        r"tool.*created and.*validated",
    ]

    for msg in reversed(messages[-10:]):
        content = get_message_content(msg)
        agent_name = getattr(msg, "name", "")

        if agent_name == "ToolBuilder":
            if '"status": "success"' in content and '"tool_name"' in content:
                try:
                    json_match = re.search(
                        r'\{[^{}]*"status"\s*:\s*"success"[^{}]*"tool_name"\s*:\s*"([^"]+)"[^{}]*\}',
                        content,
                    )
                    if json_match:
                        tool_name = json_match.group(1)
                        logger.info(f"Detected tool success from AIMessage JSON: {tool_name}")
                        return True, tool_name
                except Exception:  # nosec B110
                    pass

            if '"status": "success"' in content and "validated successfully" in content.lower():
                tool_name_match = re.search(r"Tool '([^']+)' validated successfully", content)
                if tool_name_match:
                    tool_name = tool_name_match.group(1)
                    logger.info(f"Detected tool validation success: {tool_name}")
                    return True, tool_name

            for pattern in success_patterns:
                if re.search(pattern, content.lower()):
                    tool_match = re.search(r"tool.*['\"`]([\w_]+)['\"`]", content.lower())
                    if tool_match:
                        tool_name = tool_match.group(1)
                        logger.info(f"Detected tool success from pattern '{pattern}': {tool_name}")
                        return True, tool_name

    return False, ""


def get_all_created_tools(messages) -> list[str]:
    """
    Extract all successfully created tool names from ToolMessage responses.

    Args:
        messages: List of conversation messages

    Returns:
        List of tool names that were successfully created
    """
    created_tools = []

    for msg in reversed(messages[-20:]):
        if isinstance(msg, ToolMessage) or (
            hasattr(msg, "__class__") and msg.__class__.__name__ == "ToolMessage"
        ):
            content = get_message_content(msg)
            try:
                result = json.loads(content)
                if result.get("status") == "success" and "tool_name" in result:
                    tool_name = result.get("tool_name")
                    if tool_name and tool_name not in created_tools:
                        created_tools.append(tool_name)
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    return created_tools


def extract_steering_messages(messages) -> list[str]:
    """Extract [USER STEERING] directives from conversation history.

    Returns the cleaned steering text (prefix removed) for each steering message found.
    """
    steering_texts: list[str] = []
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        content = get_message_content(msg)
        if "[USER STEERING]" in content:
            clean = content.replace("[USER STEERING]", "").strip()
            if clean:
                steering_texts.append(clean)
    return steering_texts


def check_for_engagement_request(messages) -> tuple[bool, dict | None]:
    """Check if the last agent message contains an engagement request.

    Agents signal they need user input by including [ENGAGEMENT_REQUEST] followed
    by a JSON payload in their AIMessage content.

    Returns (has_request, parsed_request_dict_or_None).
    """
    # Don't re-trigger if user already responded to an engagement
    for m in messages:
        content = get_message_content(m)
        if isinstance(m, HumanMessage) and "[USER RESPONSE]" in content:
            return False, None

    for msg in reversed(messages[-5:]):
        if not isinstance(msg, AIMessage):
            continue
        content = get_message_content(msg)
        marker = "[ENGAGEMENT_REQUEST]"
        idx = content.find(marker)
        if idx == -1:
            continue
        json_text = content[idx + len(marker) :].strip()
        try:
            data = json.loads(json_text)
            if isinstance(data, dict) and "type" in data and "question" in data:
                return True, data
        except json.JSONDecodeError:
            json_match = re.search(r"\{[^{}]+\}", json_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if isinstance(data, dict) and "type" in data and "question" in data:
                        return True, data
                except json.JSONDecodeError:
                    pass
        return False, None
    return False, None


def check_query_ambiguity(messages) -> str | None:
    """Check if the initial user query is too ambiguous to route confidently.

    Returns a description of what's missing, or None if the query is specific enough.
    Only checks on the first supervisor call (when there's no agent output yet).
    """
    # Only check on first routing — if there are AIMessages, agents already responded
    has_agent_output = any(isinstance(m, AIMessage) for m in messages)
    if has_agent_output:
        return None

    # Don't re-ask if user already responded to a previous engagement
    for m in messages:
        content = get_message_content(m)
        if isinstance(m, HumanMessage) and "[USER RESPONSE]" in content:
            return None

    original = extract_original_query(messages)
    if not original:
        return "No query provided."

    original_lower = original.lower().strip()

    if len(original_lower) < 10:
        return "Query is too vague to route. Please provide more details about what you need."

    ambiguous_patterns = [
        (
            r"^analyze\s+(?:this|a|the)\s+protein$",
            "Please specify which protein (UniProt ID, PDB ID, or protein name).",
        ),
        (r"^analyze\s+$", "Please specify what to analyze and which protein/molecule."),
        (
            r"^help$",
            "Please describe what you need help with — e.g., sequence analysis, structure prediction, literature search.",
        ),
        (
            r"^run\s+(?:analysis|ml|dl)$",
            "Please specify what data to use and what analysis to perform.",
        ),
    ]
    for pattern, reason in ambiguous_patterns:
        if re.search(pattern, original_lower):
            return reason

    return None


def extract_original_query(messages) -> str | None:
    """
    Extract the original user query from messages.

    Args:
        messages: List of messages in the conversation

    Returns:
        The original user query string, or None if not found
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            return str(content) if content is not None else None
    return None


# Files that say nothing about what work is already done.
_WORKSPACE_IGNORED = {".git", "__pycache__", ".ipynb_checkpoints", ".cache", "node_modules"}

# Small text artifacts worth previewing so the supervisor sees already-computed results.
_PREVIEWABLE_SUFFIXES = {".json", ".csv", ".txt", ".md", ".fasta", ".tsv", ".yaml", ".yml"}


def _preview_file(path, max_chars: int = 180) -> str:
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            text = handle.read(max_chars + 1)
    except OSError:
        return ""
    text = " ".join(text.split())
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def describe_workspace_artifacts(
    workspace_dir=None, max_files: int = 40, preview_max_bytes: int = 4096
) -> str:
    """List what already exists in the shared workspace, for the supervisor's context.

    The supervisor routed from the message history alone, so files that earlier steps
    had already produced were invisible to it: a run that had written 100 molecules to
    disk was told to build the whole pipeline again from scratch, because nothing in the
    conversation said the file was there. Routing needs the state of the workspace, not
    just the transcript.

    Returns an empty string when the workspace cannot be read or holds nothing — the
    supervisor must keep routing either way.
    """
    try:
        if workspace_dir is None:
            from bioagents.sandbox.sandbox_manager import get_sandbox

            workspace_dir = get_sandbox().workdir
        root = Path(workspace_dir)
        if not root.is_dir():
            return ""

        entries = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in _WORKSPACE_IGNORED or part.startswith(".") for part in path.parts):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            entries.append((path, size))

        if not entries:
            return ""

        entries.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
        shown = entries[:max_files]

        lines = [
            "[WORKSPACE ARTIFACTS — files that already exist in the shared workspace]",
            "These were produced by earlier steps of THIS run and are readable by every "
            "agent at the path shown. Before routing work, check whether what you need "
            "is already here: reuse the file and route only the remaining steps. Do not "
            "rebuild a pipeline whose outputs are already on disk, and do not ask an "
            "agent to re-download data that is listed below.",
            "",
        ]
        for path, size in shown:
            rel = path.relative_to(root)
            line = f"- {rel} ({size:,} bytes)"
            if path.suffix.lower() in _PREVIEWABLE_SUFFIXES and size <= preview_max_bytes:
                preview = _preview_file(path)
                if preview:
                    line += f" — {preview}"
            lines.append(line)

        if len(entries) > len(shown):
            lines.append(f"- … and {len(entries) - len(shown)} more files")

        return "\n".join(lines)
    except Exception:  # pragma: no cover - routing must survive a workspace failure
        logger.debug("Could not describe workspace artifacts for the supervisor.")
        return ""


# The coder writes Python from scratch: slowest and least reproducible, so cap its turns.
MAX_CODER_ROUTES = max(1, _env_int_local("BIOAGENTS_MAX_CODER_ROUTES", 2))

CODE_AGENTS = frozenset({"coder", "ml", "dl"})

# Words that genuinely call for writing code rather than calling a tool.
_CODE_TASK_HINTS = (
    "write a script",
    "write python",
    "custom script",
    "implement an algorithm",
    "simulation",
    "train a model",
    "fine-tune",
    "custom analysis",
    # No dedicated tool exists for these, so the coder is the right destination.
    "t-sne",
    "tsne",
    "umap",
    "pca",
    "cluster",
    "dimensionality reduction",
    "rmsd",
)


def count_code_agent_routes(messages) -> int:
    """How many times a code-writing agent has already answered in this run."""
    total = 0
    for msg in messages:
        name = str(getattr(msg, "name", "") or "").lower()
        if isinstance(msg, AIMessage) and name in {"coder", "ml", "dl"}:
            total += 1
    return total


# Capability phrasings that share no word with the tool name (e.g. "ECFP4 fingerprints").
_CAPABILITY_ALIASES: tuple[tuple[str, str], ...] = (
    (r"\becfp\d*\b|\bmorgan\b|\bfingerprint", "compute_molecular_fingerprint"),
    (r"\bqed\b|\bsa[ _-]?score\b|\bdruglike|\blipinski\b", "compute_molecular_descriptors"),
    (r"\bdescriptor|\blogp\b|\btpsa\b|molecular weight", "compute_molecular_descriptors"),
    (r"\btanimoto\b|similar(ity)? (search|molecules)", "search_similar_molecules_tool"),
    (r"\bsubstructure\b|\bsmarts\b", "search_substructure"),
    (r"\bblast\b|homolog|sequence identity", "run_blast_search"),
    (r"co-?fold|predict the complex|complex structure", "run_boltz"),
    (r"\bdock(ing)?\b|binding affinity|virtual screen", "run_docking"),
    (
        r"\besm\b|saturation (mutagenesis|scan)|variant effect|deep mutational",
        "saturation_scan_esm",
    ),
    (r"\bplot\b|\bchart\b|\bheatmap\b|\bfigure\b|visuali[sz]", "create_scatter_plot"),
)


def _agent_holding(tool_name: str, inventory: dict, member_set: set) -> str | None:
    for agent, names in inventory.items():
        if agent in member_set and tool_name in names:
            return str(agent)
    return None


def suggest_specialist_for_task(task_text: str, members) -> str | None:
    """Find a tool-holding specialist whose tools match the task wording.

    Routing to the coder is only justified when nothing else can do the work, so this
    answers the prior question: does an agent already hold a tool for this?
    """
    text = (task_text or "").lower()
    if not text.strip():
        return None

    try:
        from bioagents.tools.agent_manifest import agent_tool_names

        inventory = agent_tool_names()
    except Exception:  # pragma: no cover - routing must survive a manifest failure
        return None

    member_set = {str(m).lower() for m in members}

    for pattern, tool_name in _CAPABILITY_ALIASES:
        if re.search(pattern, text):
            holder = _agent_holding(tool_name, inventory, member_set)
            if holder:
                return holder

    best_agent, best_score = None, 0
    for agent, names in inventory.items():
        if agent not in member_set:
            continue
        score = 0
        for tool_name in names:
            # Require a full match; partial overlap misroutes (parse_fasta_file vs "parse the PDB").
            words = [w for w in tool_name.lower().split("_") if len(w) > 3]
            if words and all(w in text for w in words):
                score += 1
        if score > best_score:
            best_agent, best_score = agent, score

    return best_agent if best_score > 0 else None


def check_coder_routing_guard(messages, members, task_text: str = "") -> tuple[str | None, str]:
    """Redirect a code-agent route when a specialist or existing results can serve.

    Returns ``(replacement_agent, reason)``; ``(None, "")`` leaves the routing alone.
    """
    text = (task_text or "").lower()
    if any(hint in text for hint in _CODE_TASK_HINTS):
        return None, ""

    used = count_code_agent_routes(messages)
    if used >= MAX_CODER_ROUTES:
        target = "report" if "report" in members else None
        if target:
            return target, (
                f"Code agents have already run {used} times in this workflow "
                f"(limit {MAX_CODER_ROUTES}). Reporting what has actually been produced "
                "instead of writing more code."
            )

    specialist = suggest_specialist_for_task(task_text, members)
    if specialist and specialist not in CODE_AGENTS:
        return specialist, (
            f"Agent '{specialist}' already holds tools matching this task, so it runs "
            "directly instead of the coder writing an implementation from scratch."
        )

    return None, ""
