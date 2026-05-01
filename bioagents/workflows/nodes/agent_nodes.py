"""Workflow nodes that wrap existing BioAgents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from bioagents.agents.analysis_agent import create_analysis_agent
from bioagents.agents.coder_agent import create_coder_agent, create_coder_node
from bioagents.agents.critic_agent import create_critic_agent
from bioagents.agents.data_acquisition_agent import create_data_acquisition_agent
from bioagents.agents.dl_agent import create_dl_agent, create_dl_node
from bioagents.agents.docking_agent import create_docking_agent
from bioagents.agents.environment_agent import create_environment_agent
from bioagents.agents.genomics_agent import create_genomics_agent
from bioagents.agents.git_agent import create_git_agent
from bioagents.agents.literature_agent import create_literature_agent
from bioagents.agents.ml_agent import create_ml_agent, create_ml_node
from bioagents.agents.paper_replication_agent import create_paper_replication_agent
from bioagents.agents.phylogenetics_agent import create_phylogenetics_agent
from bioagents.agents.planner_agent import create_planner_agent
from bioagents.agents.prompt_optimizer_agent import create_prompt_optimizer_agent
from bioagents.agents.protein_design_agent import create_protein_design_agent
from bioagents.agents.report_agent import create_report_agent
from bioagents.agents.research_agent import create_research_agent
from bioagents.agents.result_checker_agent import create_result_checker_agent
from bioagents.agents.shell_agent import create_shell_agent
from bioagents.agents.structural_biology_agent import create_structural_biology_agent
from bioagents.agents.summary_agent import create_summary_agent
from bioagents.agents.tool_builder_agent import create_tool_builder_agent
from bioagents.agents.tool_discovery_agent import create_tool_discovery_agent
from bioagents.agents.tool_validator_agent import create_tool_validator_agent
from bioagents.agents.transcriptomics_agent import create_transcriptomics_agent
from bioagents.agents.visualization_agent import create_visualization_agent
from bioagents.agents.web_browser_agent import create_web_browser_agent
from langchain_core.messages import HumanMessage
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


def _coerce_query(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    raise ValueError("query must be a non-empty string")


def _normalize_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_output = payload.get("raw_output")
    data = payload.get("data")
    error = payload.get("error")
    if not isinstance(raw_output, str) or not raw_output.strip():
        raw_output = _text_from_messages(payload.get("messages"))
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)
    if error is not None and not isinstance(error, str):
        error = str(error)
    if not isinstance(data, dict):
        data = {}
    return {
        "text": raw_output,
        "structured": data,
        "status": "error" if error else "success",
        "error": error or "",
    }


def _text_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif content is not None:
            parts.append(str(content))
    return "\n\n".join(parts).strip()


class _BaseAgentWorkflowNode(WorkflowNode):
    """Shared execution/runtime adapter for workflow-facing agent nodes."""

    _agent_name: ClassVar[str]

    def __init__(self, *, max_steps: int = 20) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._max_steps = max_steps
        self._runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {"max_steps": self._max_steps}

    @property
    def input_schema(self) -> dict[str, str]:
        return {"query": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {
            "text": "str",
            "structured": "dict",
            "status": "str",
            "error": "str",
        }

    def _get_runner(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if self._runner is None:
            self._runner = self._build_runner()
        return self._runner

    def _build_runner(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = _coerce_query(inputs.get("query"))
        state = {"messages": [HumanMessage(content=query)]}
        try:
            payload = self._get_runner()(state)
            if not isinstance(payload, dict):
                raise TypeError(f"{self._agent_name} returned non-dict payload")
        except Exception as exc:
            payload = {
                "raw_output": "",
                "data": {},
                "error": f"{self._agent_name} execution failed: {exc}",
            }
        return _normalize_agent_payload(payload)


@dataclass(frozen=True)
class _AgentSpec:
    workflow_type_id: str
    class_name: str
    display_name: str
    description: str
    runner_factory: Callable[[int], Callable[[dict[str, Any]], dict[str, Any]]]


def _make_agent_node_class(spec: _AgentSpec) -> type[_BaseAgentWorkflowNode]:
    class _GeneratedAgentNode(_BaseAgentWorkflowNode):
        workflow_type_id: ClassVar[str] = spec.workflow_type_id
        _agent_name: ClassVar[str] = spec.display_name

        def _build_runner(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
            return spec.runner_factory(self._max_steps)

        @property
        def metadata(self) -> NodeMetadata:
            return NodeMetadata(
                name=spec.display_name,
                description=spec.description,
                version="1.0.0",
                category="agent",
            )

    _GeneratedAgentNode.__name__ = spec.class_name
    _GeneratedAgentNode.__qualname__ = spec.class_name
    return _GeneratedAgentNode


_AGENT_SPECS: list[_AgentSpec] = [
    _AgentSpec(
        workflow_type_id="agent_coder",
        class_name="CoderAgentNode",
        display_name="Coder Agent",
        description="Run the coding agent in workflow mode.",
        runner_factory=lambda max_steps: create_coder_node(create_coder_agent(max_steps=max_steps)),
    ),
    _AgentSpec(
        workflow_type_id="agent_ml",
        class_name="MlAgentNode",
        display_name="ML Agent",
        description="Use the ML code agent to analyze a query and return text plus structured output.",
        runner_factory=lambda max_steps: create_ml_node(create_ml_agent(max_steps=max_steps)),
    ),
    _AgentSpec(
        workflow_type_id="agent_dl",
        class_name="DlAgentNode",
        display_name="DL Agent",
        description="Use the deep-learning code agent for model-oriented tasks in a workflow.",
        runner_factory=lambda max_steps: create_dl_node(create_dl_agent(max_steps=max_steps)),
    ),
    _AgentSpec(
        workflow_type_id="agent_research",
        class_name="ResearchAgentNode",
        display_name="Research Agent",
        description="Research agent for biological data and literature gathering.",
        runner_factory=lambda _max_steps: create_research_agent([]),
    ),
    _AgentSpec(
        workflow_type_id="agent_analysis",
        class_name="AnalysisAgentNode",
        display_name="Analysis Agent",
        description="Analysis agent for biochemical interpretation tasks.",
        runner_factory=lambda _max_steps: create_analysis_agent([]),
    ),
    _AgentSpec(
        workflow_type_id="agent_report",
        class_name="ReportAgentNode",
        display_name="Report Agent",
        description="Generate final scientific reports from intermediate outputs.",
        runner_factory=lambda _max_steps: create_report_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_tool_builder",
        class_name="ToolBuilderAgentNode",
        display_name="Tool Builder Agent",
        description="Designs and drafts tools from a workflow prompt.",
        runner_factory=lambda _max_steps: create_tool_builder_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_protein_design",
        class_name="ProteinDesignAgentNode",
        display_name="Protein Design Agent",
        description="Protein design planning and generation agent.",
        runner_factory=lambda _max_steps: create_protein_design_agent([]),
    ),
    _AgentSpec(
        workflow_type_id="agent_critic",
        class_name="CriticAgentNode",
        display_name="Critic Agent",
        description="Critiques and challenges proposed solutions.",
        runner_factory=lambda _max_steps: create_critic_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_literature",
        class_name="LiteratureAgentNode",
        display_name="Literature Agent",
        description="Literature and paper-focused reasoning agent.",
        runner_factory=lambda _max_steps: create_literature_agent(extra_tools=[]),
    ),
    _AgentSpec(
        workflow_type_id="agent_web_browser",
        class_name="WebBrowserAgentNode",
        display_name="Web Browser Agent",
        description="Web browsing and page inspection agent.",
        runner_factory=lambda _max_steps: create_web_browser_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_paper_replication",
        class_name="PaperReplicationAgentNode",
        display_name="Paper Replication Agent",
        description="Agent specialized in reproducing paper procedures.",
        runner_factory=lambda _max_steps: create_paper_replication_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_data_acquisition",
        class_name="DataAcquisitionAgentNode",
        display_name="Data Acquisition Agent",
        description="Collects and organizes experimental data sources.",
        runner_factory=lambda _max_steps: create_data_acquisition_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_genomics",
        class_name="GenomicsAgentNode",
        display_name="Genomics Agent",
        description="Genomics-oriented analysis and retrieval agent.",
        runner_factory=lambda _max_steps: create_genomics_agent(extra_tools=[]),
    ),
    _AgentSpec(
        workflow_type_id="agent_transcriptomics",
        class_name="TranscriptomicsAgentNode",
        display_name="Transcriptomics Agent",
        description="Transcriptomics workflow and interpretation agent.",
        runner_factory=lambda _max_steps: create_transcriptomics_agent(extra_tools=[]),
    ),
    _AgentSpec(
        workflow_type_id="agent_structural_biology",
        class_name="StructuralBiologyAgentNode",
        display_name="Structural Biology Agent",
        description="Structural biology and protein structure reasoning agent.",
        runner_factory=lambda _max_steps: create_structural_biology_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_phylogenetics",
        class_name="PhylogeneticsAgentNode",
        display_name="Phylogenetics Agent",
        description="Phylogenetics comparison and lineage analysis agent.",
        runner_factory=lambda _max_steps: create_phylogenetics_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_docking",
        class_name="DockingAgentNode",
        display_name="Docking Agent",
        description="Docking-oriented computational agent.",
        runner_factory=lambda _max_steps: create_docking_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_planner",
        class_name="PlannerAgentNode",
        display_name="Planner Agent",
        description="Breaks goals into executable scientific steps.",
        runner_factory=lambda _max_steps: create_planner_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_tool_validator",
        class_name="ToolValidatorAgentNode",
        display_name="Tool Validator Agent",
        description="Evaluates tool calls and validation outcomes.",
        runner_factory=lambda _max_steps: create_tool_validator_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_tool_discovery",
        class_name="ToolDiscoveryAgentNode",
        display_name="Tool Discovery Agent",
        description="Finds candidate tools for a task.",
        runner_factory=lambda _max_steps: create_tool_discovery_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_prompt_optimizer",
        class_name="PromptOptimizerAgentNode",
        display_name="Prompt Optimizer Agent",
        description="Optimizes prompts for downstream agent performance.",
        runner_factory=lambda _max_steps: create_prompt_optimizer_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_result_checker",
        class_name="ResultCheckerAgentNode",
        display_name="Result Checker Agent",
        description="Checks result quality and consistency.",
        runner_factory=lambda _max_steps: create_result_checker_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_shell",
        class_name="ShellAgentNode",
        display_name="Shell Agent",
        description="Shell execution and command-line workflow agent.",
        runner_factory=lambda _max_steps: create_shell_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_git",
        class_name="GitAgentNode",
        display_name="Git Agent",
        description="Git and repository workflow assistant agent.",
        runner_factory=lambda _max_steps: create_git_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_environment",
        class_name="EnvironmentAgentNode",
        display_name="Environment Agent",
        description="Environment setup and dependency troubleshooting agent.",
        runner_factory=lambda _max_steps: create_environment_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_visualization",
        class_name="VisualizationAgentNode",
        display_name="Visualization Agent",
        description="Visualization generation and charting agent.",
        runner_factory=lambda _max_steps: create_visualization_agent(),
    ),
    _AgentSpec(
        workflow_type_id="agent_summary",
        class_name="SummaryAgentNode",
        display_name="Summary Agent",
        description="Summarizes and finalizes multi-step findings.",
        runner_factory=lambda _max_steps: create_summary_agent(),
    ),
]


AGENT_WORKFLOW_NODES: list[type[_BaseAgentWorkflowNode]] = []
for _spec in _AGENT_SPECS:
    _cls = _make_agent_node_class(_spec)
    globals()[_spec.class_name] = _cls
    AGENT_WORKFLOW_NODES.append(_cls)
