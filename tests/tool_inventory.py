"""Inventory of every agent-callable tool, rebuilt exactly as ``bioagents/graph.py`` does.

Nothing here invokes a tool body; that lives in test_tool_callability.py and tool_audit.py.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import BaseTool  # noqa: TC002 - used at runtime

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
from bioagents.tools.proteomics_tools import (
    download_uniprot_flat_file,
    fetch_uniprot_fasta,
    run_esm2,
    run_esm3,
    run_saprot,
)
from bioagents.tools.rdkit_tools import get_all_rdkit_tools
from bioagents.tools.shell_tools import get_shell_tools
from bioagents.tools.structural_tools import get_structural_tools
from bioagents.tools.tool_builder_tools import get_tool_builder_tools
from bioagents.tools.tool_universe import (
    tool_universe_call_tool,
    tool_universe_find_tools,
)
from bioagents.tools.transcriptomics_tools import get_transcriptomics_tools
from bioagents.tools.visualization_tools import get_visualization_tools
from bioagents.tools.web_tools import get_web_tools

# Below this, a description tells an agent nothing about when to pick the tool.
MIN_DESCRIPTION_LENGTH = 40

# A description "documents its return value" if it mentions returning something.
_RETURNS_PATTERN = re.compile(r"\bReturns?\b|\bReturning\b|\bReturned\b")

# Words that make a hardcoded return string look like a fake success report.
_SUCCESS_PATTERN = re.compile(
    r"\b(success|successfully|completed|generated|finished|done)\b", re.IGNORECASE
)


_TU_TOOLS = [tool_universe_find_tools, tool_universe_call_tool]

#: Every ``get_*_tools()`` factory that feeds an agent, keyed by factory name.
FACTORIES = {
    "get_docking_tools": get_docking_tools,
    "get_environment_tools": get_environment_tools,
    "get_esm_tools": get_esm_tools,
    "get_file_tools": get_file_tools,
    "get_genomics_tools": get_genomics_tools,
    "get_git_tools": get_git_tools,
    "get_literature_tools": get_literature_tools,
    "get_all_protein_design_tools": get_all_protein_design_tools,
    "get_shell_tools": get_shell_tools,
    "get_structural_tools": get_structural_tools,
    "get_tool_builder_tools": get_tool_builder_tools,
    "get_transcriptomics_tools": get_transcriptomics_tools,
    "get_visualization_tools": get_visualization_tools,
    "get_web_tools": get_web_tools,
    # Not in graph.py, but a public factory used by the rdkit validator agent.
    "get_all_rdkit_tools": get_all_rdkit_tools,
}

#: Tools that graph.py passes to agents individually rather than via a factory.
LOOSE_TOOLS: list[BaseTool] = [
    tool_universe_find_tools,
    tool_universe_call_tool,
    fetch_uniprot_fasta,
    download_uniprot_flat_file,
    fetch_webpage_as_pdf_text,
    extract_pdf_text_spacy_layout,
    search_local_papers_with_paperqa,
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
    run_aggrescan3d,
]

#: Tools defined with @tool but not reachable from any agent today. They are
#: audited anyway so that dead-but-shipped tools stay visible.
ORPHAN_TOOLS: list[BaseTool] = [run_esm2, run_esm3, run_saprot]


def build_agent_tool_map() -> dict[str, list[BaseTool]]:
    """Rebuild the agent -> bound tool list mapping, mirroring ``create_graph()``."""
    structural = get_structural_tools()
    web = get_web_tools()
    git = get_git_tools()
    genomics = get_genomics_tools()

    return {
        "research": [
            fetch_uniprot_fasta,
            tool_universe_find_tools,
            tool_universe_call_tool,
            fetch_webpage_as_pdf_text,
            extract_pdf_text_spacy_layout,
            search_local_papers_with_paperqa,
            *[
                t
                for t in structural
                if t.name
                in ("fetch_alphafold_structure", "fetch_pdb_structure", "download_structure_file")
            ],
        ],
        "analysis": [
            calculate_molecular_weight,
            analyze_amino_acid_composition,
            calculate_isoelectric_point,
            run_aggrescan3d,
            *get_esm_tools(),
        ],
        "tool_builder": get_tool_builder_tools(),
        "tool_discovery": get_tool_builder_tools(),
        "protein_design": get_all_protein_design_tools(),
        "literature": get_literature_tools() + _TU_TOOLS,
        "web_browser": web,
        "paper_replication": web + git + [extract_pdf_text_spacy_layout],
        "data_acquisition": web + get_file_tools() + [download_uniprot_flat_file],
        "genomics": genomics + _TU_TOOLS,
        "transcriptomics": get_transcriptomics_tools() + _TU_TOOLS,
        "structural_biology": structural + get_esm_tools(),
        "phylogenetics": genomics,
        "shell": get_shell_tools(),
        "git": git,
        "environment": get_environment_tools(),
        "visualization": get_visualization_tools(),
        "docking": get_docking_tools(),
        # Not part of create_graph(), but a real agent in bioagents/agents/.
        "rdkit_validator": get_all_rdkit_tools(),
    }


AGENT_TOOL_MAP: dict[str, list[BaseTool]] = build_agent_tool_map()


def _collect_all_tools() -> dict[str, BaseTool]:
    tools: dict[str, BaseTool] = {}
    for tool_list in AGENT_TOOL_MAP.values():
        for tool in tool_list:
            tools.setdefault(tool.name, tool)
    for factory in FACTORIES.values():
        for tool in factory():
            tools.setdefault(tool.name, tool)
    for tool in [*LOOSE_TOOLS, *ORPHAN_TOOLS]:
        tools.setdefault(tool.name, tool)
    return dict(sorted(tools.items()))


#: Every tool name -> tool object on the whole agent-reachable surface.
ALL_TOOLS: dict[str, BaseTool] = _collect_all_tools()

#: Tool names reachable from at least one agent.
AGENT_REACHABLE: frozenset[str] = frozenset(
    tool.name for tools in AGENT_TOOL_MAP.values() for tool in tools
)


def owners_of(tool_name: str) -> list[str]:
    """Return the agent names that bind ``tool_name``, or ``["(unreachable)"]``."""
    owners = sorted(
        agent for agent, tools in AGENT_TOOL_MAP.items() if any(t.name == tool_name for t in tools)
    )
    return owners or ["(unreachable)"]


@dataclass(frozen=True)
class DuplicateFinding:
    """A tool name that appears more than once where it should appear once."""

    tool_name: str
    location: str
    count: int
    distinct_objects: int

    def __str__(self) -> str:
        kind = "SHADOWED (distinct objects)" if self.distinct_objects > 1 else "duplicated"
        return f"{self.tool_name}: {kind} {self.count}x in {self.location}"


def find_duplicate_names_in_lists() -> list[DuplicateFinding]:
    """Find tool names listed more than once inside a single bound tool list.

    Providers reject or collapse duplicates, so one of the two tools becomes unreachable.
    """
    findings: list[DuplicateFinding] = []
    sources: dict[str, list[BaseTool]] = {
        f"agent:{name}": tools for name, tools in AGENT_TOOL_MAP.items()
    }
    sources.update({f"factory:{name}": fn() for name, fn in FACTORIES.items()})

    for location, tools in sorted(sources.items()):
        grouped: dict[str, list[BaseTool]] = defaultdict(list)
        for tool in tools:
            grouped[tool.name].append(tool)
        for name, group in sorted(grouped.items()):
            if len(group) > 1:
                findings.append(
                    DuplicateFinding(
                        tool_name=name,
                        location=location,
                        count=len(group),
                        distinct_objects=len({id(t) for t in group}),
                    )
                )
    return findings


def find_shadowed_names() -> list[DuplicateFinding]:
    """Find tool names bound to two *different* tool objects anywhere.

    An agent asking for such a name reaches whichever list it came from.
    """
    grouped: dict[str, set[int]] = defaultdict(set)
    locations: dict[str, set[str]] = defaultdict(set)
    for location, tools in [(f"factory:{n}", f()) for n, f in FACTORIES.items()]:
        for tool in tools:
            grouped[tool.name].add(id(tool))
            locations[tool.name].add(location)
    for tool in [*LOOSE_TOOLS, *ORPHAN_TOOLS]:
        grouped[tool.name].add(id(tool))
        locations[tool.name].add("module-level")

    return [
        DuplicateFinding(
            tool_name=name,
            location=", ".join(sorted(locations[name])),
            count=len(ids),
            distinct_objects=len(ids),
        )
        for name, ids in sorted(grouped.items())
        if len(ids) > 1
    ]


@dataclass
class StructuralReport:
    """Result of the static checks for a single tool."""

    name: str
    owners: list[str]
    has_schema: bool
    description_length: int
    documents_return: bool
    undocumented_args: list[str] = field(default_factory=list)
    args_without_schema_description: list[str] = field(default_factory=list)

    @property
    def description_ok(self) -> bool:
        return (
            self.description_length >= MIN_DESCRIPTION_LENGTH
            and self.documents_return
            and not self.undocumented_args
        )

    @property
    def problems(self) -> list[str]:
        issues = []
        if not self.has_schema:
            issues.append("no args_schema")
        if self.description_length < MIN_DESCRIPTION_LENGTH:
            issues.append(f"description too short ({self.description_length} chars)")
        if not self.documents_return:
            issues.append("description does not document the return value")
        if self.undocumented_args:
            issues.append(f"undocumented args: {', '.join(self.undocumented_args)}")
        return issues


def analyse_structure(tool: BaseTool) -> StructuralReport:
    """Run every static quality check on one tool."""
    description = tool.description or ""
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", {}) if schema is not None else {}

    missing_schema_desc = [name for name, f in fields.items() if not f.description]
    # An argument is documented if the schema or the description names it; both reach
    # the model, so only a silent argument is a defect.
    undocumented = [
        name
        for name in missing_schema_desc
        if not re.search(rf"\b{re.escape(name)}\b", description)
    ]

    return StructuralReport(
        name=tool.name,
        owners=owners_of(tool.name),
        has_schema=(schema is not None and bool(fields)) or schema is not None,
        description_length=len(description),
        documents_return=bool(_RETURNS_PATTERN.search(description)),
        undocumented_args=sorted(undocumented),
        args_without_schema_description=sorted(missing_schema_desc),
    )


class ProbeSentinel:
    """A value no scalar pydantic field can accept."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<ProbeSentinel>"


def build_invalid_payload(tool: BaseTool) -> dict | None:
    """Build an argument dict that must fail ``args_schema`` validation.

    ``None`` when the tool takes no arguments; the payload never reaches the tool body.
    """
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", {}) if schema is not None else {}
    if not fields:
        return None
    required = [name for name, f in fields.items() if f.is_required()]
    if required:
        # Omitting required args is enough; the body is never entered.
        return {}
    return {name: ProbeSentinel() for name in fields}


def _tool_function(tool: BaseTool):
    for attr in ("func", "coroutine", "_run"):
        fn = getattr(tool, attr, None)
        if fn is not None:
            return fn
    return None


def _function_ast(tool: BaseTool) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    fn = _tool_function(tool)
    if fn is None:
        return None
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _returns_hardcoded_success(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Return) or sub.value is None:
            continue
        value = sub.value
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and _SUCCESS_PATTERN.search(value.value)
        ):
            return True
        if isinstance(value, ast.JoinedStr):
            literal = "".join(
                part.value
                for part in value.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if _SUCCESS_PATTERN.search(literal):
                return True
    return False


def unused_parameters(tool: BaseTool) -> list[str]:
    """Return declared parameters the tool body never reads."""
    node = _function_ast(tool)
    if node is None:
        return []
    args = node.args
    params = [
        a.arg
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if a.arg not in ("self", "cls")
    ]
    used = {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }
    # f-strings and attribute access are covered by ast.Name above.
    return [p for p in params if p not in used]


def is_phantom_tool(tool: BaseTool) -> bool:
    """Detect a tool that ignores a declared parameter and returns a canned success string."""
    node = _function_ast(tool)
    if node is None:
        return False
    return bool(unused_parameters(tool)) and _returns_hardcoded_success(node)


def find_phantom_tools() -> list[str]:
    """Return the sorted names of every phantom tool on the surface."""
    return sorted(name for name, tool in ALL_TOOLS.items() if is_phantom_tool(tool))


GRAPH_PATH = Path(__file__).resolve().parents[1] / "bioagents" / "graph.py"

#: Factories this harness skips: they produce smolagents Tools, not LangChain
#: BaseTools, so the coder/ml/dl agents call them differently.
NON_LANGCHAIN_FACTORIES = frozenset({"get_esm_smol_tools"})


def factories_used_by_graph() -> set[str]:
    """Return every ``get_*_tools()`` factory graph.py actually calls."""
    source = GRAPH_PATH.read_text()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("get_")
            and node.func.id.endswith(("_tools", "_tools_list"))
        ):
            names.add(node.func.id)
    return names


def unaudited_graph_factories() -> set[str]:
    """Factories graph.py uses that this inventory does not cover."""
    return factories_used_by_graph() - set(FACTORIES) - NON_LANGCHAIN_FACTORIES
