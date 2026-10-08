"""Single source of truth for which tools each agent owns.

Before this module the agent→tool mapping existed in two places that had drifted apart:
local variables inside ``create_graph`` (the real wiring) and a hardcoded catalogue in
``frontend/server.py`` that advertised tool names which no longer existed. Nothing could
answer "does this agent actually have the tool this task needs?" — so the supervisor
routed blind, and agents were sent work they had no way to perform.

Everything that needs the mapping now reads it from here: the graph builds agents from
it, the supervisor is told what each specialist can do, the UI lists it, and the tool
audit checks it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Agents that take no LangChain tool list. The code-writing agents (coder/ml/dl) run
# smolagents CodeAgents with their own tool objects; the rest are pure-LLM agents.
NON_TOOL_AGENTS: frozenset[str] = frozenset(
    {
        "coder",
        "ml",
        "dl",
        "critic",
        "report",
        "summary",
        "planner",
        "tool_validator",
        "prompt_optimizer",
        "result_checker",
        "supervisor",
        "user_input",
    }
)


def build_agent_tool_map(session_context_tool: BaseTool | None = None) -> dict[str, list]:
    """Build the agent → tools mapping used to wire the graph.

    Args:
        session_context_tool: Optional per-request tool for retrieving prior session
            context. When given it is appended to every tool-using agent.

    Returns:
        Mapping of graph node name to the list of tools that agent receives.
    """
    from bioagents.tools.analysis_tools import (
        analyze_amino_acid_composition,
        calculate_isoelectric_point,
        calculate_molecular_weight,
        run_aggrescan3d,
    )
    from bioagents.tools.docking_tools import get_docking_tools
    from bioagents.tools.environment_tools import get_environment_tools
    from bioagents.tools.esm_tools import get_esm_tools
    from bioagents.tools.file_tools import get_file_tools
    from bioagents.tools.genomics_tools import get_genomics_tools
    from bioagents.tools.git_tools import get_git_tools
    from bioagents.tools.literature_tools import get_literature_tools
    from bioagents.tools.paperqa_wrapper import search_local_papers_with_paperqa
    from bioagents.tools.pdf_tools import (
        extract_pdf_text_spacy_layout,
        fetch_webpage_as_pdf_text,
    )
    from bioagents.tools.protein_design_tools import get_all_protein_design_tools
    from bioagents.tools.proteomics_tools import download_uniprot_flat_file, fetch_uniprot_fasta
    from bioagents.tools.shell_tools import get_shell_tools
    from bioagents.tools.structural_tools import (
        download_structure_file,
        fetch_alphafold_structure,
        fetch_pdb_structure,
        get_structural_tools,
    )
    from bioagents.tools.tool_builder_tools import get_tool_builder_tools
    from bioagents.tools.tool_universe import tool_universe_call_tool, tool_universe_find_tools
    from bioagents.tools.transcriptomics_tools import get_transcriptomics_tools
    from bioagents.tools.visualization_tools import get_visualization_tools
    from bioagents.tools.web_tools import get_web_tools

    tool_universe = [tool_universe_find_tools, tool_universe_call_tool]

    agent_tools: dict[str, list] = {
        "research": [
            fetch_uniprot_fasta,
            *tool_universe,
            fetch_webpage_as_pdf_text,
            extract_pdf_text_spacy_layout,
            search_local_papers_with_paperqa,
            fetch_alphafold_structure,
            fetch_pdb_structure,
            download_structure_file,
        ],
        "analysis": [
            calculate_molecular_weight,
            analyze_amino_acid_composition,
            calculate_isoelectric_point,
            run_aggrescan3d,
            *get_esm_tools(),
        ],
        "tool_builder": get_tool_builder_tools(),
        "protein_design": get_all_protein_design_tools(),
        "literature": get_literature_tools() + tool_universe,
        "web_browser": get_web_tools(),
        "paper_replication": get_web_tools() + get_git_tools() + [extract_pdf_text_spacy_layout],
        "data_acquisition": get_web_tools() + get_file_tools() + [download_uniprot_flat_file],
        "genomics": get_genomics_tools() + tool_universe,
        "transcriptomics": get_transcriptomics_tools() + tool_universe,
        "structural_biology": get_structural_tools() + get_esm_tools(),
        "phylogenetics": get_genomics_tools(),
        "tool_discovery": get_tool_builder_tools(),
        "shell": get_shell_tools(),
        "git": get_git_tools(),
        "environment": get_environment_tools(),
        "visualization": get_visualization_tools(),
        "docking": get_docking_tools(),
    }

    if session_context_tool is not None:
        for tools in agent_tools.values():
            tools.append(session_context_tool)

    return agent_tools


def agent_tool_names(session_context_tool: BaseTool | None = None) -> dict[str, list[str]]:
    """Return the agent → tool-name mapping, for display, routing checks and audits."""
    return {
        agent: sorted({getattr(tool, "name", str(tool)) for tool in tools})
        for agent, tools in build_agent_tool_map(session_context_tool).items()
    }


def find_agents_with_tool(tool_name: str) -> list[str]:
    """Return the agents that own ``tool_name``.

    Lets the supervisor answer "who can actually do this?" instead of guessing from an
    agent's prose description.
    """
    return sorted(agent for agent, names in agent_tool_names().items() if tool_name in set(names))


def describe_agent_tools(max_tools_per_agent: int = 40) -> str:
    """Render the agent → tool mapping for the supervisor's context.

    The supervisor must know which concrete tools each specialist holds; routing a task
    to an agent that cannot perform it is the failure this is meant to prevent.
    """
    lines = [
        "[AGENT TOOL INVENTORY — which tools each agent actually has]",
        "Route a task only to an agent whose tools can perform it. If no agent has a "
        "suitable tool, say so or route to tool_builder — do not delegate work that "
        "cannot be done and then accept a narrative result.",
        "",
    ]
    for agent, names in sorted(agent_tool_names().items()):
        shown = names[:max_tools_per_agent]
        suffix = f" (+{len(names) - len(shown)} more)" if len(names) > len(shown) else ""
        lines.append(f"- {agent}: {', '.join(shown)}{suffix}")

    lines.append("")
    lines.append(
        "Agents without this list (coder, ml, dl) write and execute Python in a sandbox: "
        "use them for computation, parsing and analysis that no dedicated tool covers."
    )
    return "\n".join(lines)
