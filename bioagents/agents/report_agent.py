"""Report Agent - Synthesizes findings from shared memory with tool execution loop."""

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bioagents.agents.helpers import invoke_with_retry, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

REPORT_AGENT_PROMPT = load_prompt("report")

REPORT_AGENT_SYSTEM_PROMPT = """
You are the Report Agent. Your task is to synthesize all findings from shared memory
into a comprehensive report.

CRITICAL INSTRUCTIONS:
1. You have access to results from all agents in the shared memory (provided below)
2. Synthesize their findings into a coherent narrative
3. Create a comprehensive report that integrates all findings
4. Once you have created the report, you MUST respond with ONLY valid JSON

IMPORTANT: Your response MUST be valid JSON. Do NOT include any text before or after the JSON.
Do NOT use markdown code blocks. Return ONLY the JSON object.

OUTPUT FORMAT (MANDATORY):
{
    "title": "Comprehensive Report: [Topic]",
    "executive_summary": "Brief summary of all findings",
    "sections": [
        {
            "name": "Sequence Information",
            "content": "Details about fetched sequences"
        },
        {
            "name": "Biochemical Analysis",
            "content": "Results of property analysis"
        },
        {
            "name": "Key Findings",
            "content": "Summary of important discoveries"
        }
    ],
    "key_findings": ["finding1", "finding2", "finding3"],
    "recommendations": ["recommendation1", "recommendation2"],
    "metadata": {
        "agents_involved": ["research", "analysis"],
        "timestamp": "ISO timestamp string",
        "completeness": "full"
    }
}

After you create this report, END with this JSON. Do not add anything else.
"""


def create_report_agent():
    """
    Create the Report Agent.

    Returns:
        Agent node function
    """
    llm = get_llm(prompt_name="report")

    def report_node(state):
        """Report agent - synthesizes memory into structured report."""
        try:
            memory = state.get("memory", {}) or {}
            messages = state.get("messages", []) or []

            # ── FIX: idempotency guard ──────────────────────────────────────────
            # If report memory already holds a successful result, skip re-running
            # the LLM entirely and return the cached data.  This prevents the
            # agent from doing duplicate work when the supervisor mistakenly
            # routes here a second time.
            existing_report = memory.get("report", {})
            if existing_report.get("status") == "success" and existing_report.get("data"):
                logger.info(
                    "Report agent: result already in memory (status=success). "
                    "Returning cached data without re-invoking LLM."
                )
                cached_raw = existing_report.get("raw_output", "")
                return {
                    "data": existing_report["data"],
                    "raw_output": cached_raw,
                    "tool_calls": [],
                    "error": None,
                    "messages": [AIMessage(content=cached_raw or "Report (cached)", name="Report")],
                }
            # ───────────────────────────────────────────────────────────────────

            # Find user message
            for m in messages:
                if isinstance(m, HumanMessage):
                    break

            windowed = prepare_messages_for_agent(messages, "report", summary_mode=True)
            messages_with_system = [SystemMessage(content=REPORT_AGENT_PROMPT), *windowed]

            return invoke_with_retry("Report", llm, messages_with_system)

        except Exception as e:
            logger.error(f"Report agent error: {e}", exc_info=True)
            err_txt = str(e)
            return {
                "data": create_fallback_report("", memory),
                "raw_output": err_txt,
                "tool_calls": [],
                "error": err_txt,
                "messages": [AIMessage(content=err_txt, name="Report")],
            }

    return report_node


def format_memory_for_report(memory: dict) -> str:
    """
    Format shared memory into readable text for report generation.

    Args:
        memory: The shared memory dict

    Returns:
        Formatted string representation
    """
    lines = []

    # List all agents and their status
    lines.append("Available Agent Results:")
    lines.append("-" * 50)

    for agent_name in sorted(memory.keys()):
        agent_data = memory[agent_name]
        status = agent_data.get("status", "unknown")

        lines.append(f"\n{agent_name.upper()}: {status}")

        if agent_data.get("status") != "success":
            if agent_data.get("errors"):
                lines.append(f"  Errors: {agent_data.get('errors')}")
            continue

        # Include structured data if available
        if agent_data.get("data") and isinstance(agent_data["data"], dict):
            lines.append("  Data:")
            try:
                data_str = json.dumps(agent_data["data"], indent=4)
                for line in data_str.split("\n"):
                    lines.append(f"  {line}")
            except (TypeError, ValueError):
                lines.append(f"  {agent_data['data']!s}")

        # Include raw output as fallback
        if agent_data.get("raw_output") and not agent_data.get("data"):
            raw = agent_data["raw_output"]
            if isinstance(raw, str):
                lines.append(f"  Output: {raw[:200]}")

    return "\n".join(lines)


def parse_report_json(text: str, memory: dict) -> dict:
    """
    Parse JSON from report agent output with robust fallbacks.

    Args:
        text: Raw output from LLM
        memory: Shared memory for fallback creation

    Returns:
        Parsed report dict
    """
    if not text or not isinstance(text, str):
        return create_fallback_report("", memory)

    text = text.strip()

    # Strategy 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "title" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown or text
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict) and "title" in parsed:
                logger.info("Successfully extracted JSON from text")
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to fix common JSON issues
    try:
        # Remove markdown code blocks
        fixed_text = re.sub(r"```json\n?", "", text)
        fixed_text = re.sub(r"```\n?", "", fixed_text)
        fixed_text = fixed_text.strip()

        parsed = json.loads(fixed_text)
        if isinstance(parsed, dict) and "title" in parsed:
            logger.info("Successfully parsed JSON after fixing markdown")
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 4: If all parsing fails, create fallback from raw text
    logger.warning(f"Could not parse report JSON. Creating fallback. Text: {text[:200]}")
    return create_fallback_report(text, memory)


def create_fallback_report(raw_text: str, memory: dict) -> dict:
    """
    Create a fallback report structure when JSON parsing fails.

    Args:
        raw_text: Raw text output from agent
        memory: Shared memory

    Returns:
        Valid report dict
    """
    from datetime import datetime

    # Extract which agents completed
    completed_agents = [k for k, v in memory.items() if v.get("status") == "success"]

    # Build sections from memory
    sections = []
    key_findings = []

    for agent_name in ["research", "analysis", "coder", "ml", "dl", "protein_design"]:
        agent_mem = memory.get(agent_name, {})
        if agent_mem.get("status") != "success":
            continue

        # Add section for this agent's findings
        if agent_mem.get("data"):
            sections.append(
                {
                    "name": f"{agent_name.title()} Results",
                    "content": json.dumps(agent_mem["data"], indent=2)[:500],
                }
            )

            # Extract findings from data
            if isinstance(agent_mem["data"], dict):
                if "key_findings" in agent_mem["data"]:
                    key_findings.extend(agent_mem["data"]["key_findings"])
                elif "fetched_sequences" in agent_mem["data"]:
                    key_findings.append(f"Retrieved sequence data for {agent_name}")

    # If no sections from memory, use raw text
    if not sections and raw_text:
        sections.append({"name": "Analysis Results", "content": raw_text[:500]})

    return {
        "title": "Comprehensive Analysis Report",
        "executive_summary": (
            f"This report synthesizes findings from {len(completed_agents)} agents: "
            f"{', '.join(completed_agents)}. Data retrieval and analysis completed successfully."
        ),
        "sections": sections
        or [
            {
                "name": "Summary",
                "content": "Analysis completed. Results available from completed agents.",
            }
        ],
        "key_findings": key_findings or ["Analysis completed successfully"],
        "recommendations": [
            "Review detailed findings from individual agent reports",
            "Consider further analysis based on findings",
        ],
        "metadata": {
            "agents_involved": completed_agents,
            "timestamp": datetime.now().isoformat(),
            "completeness": "full" if completed_agents else "partial",
        },
    }
