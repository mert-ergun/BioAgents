"""Transcriptomics Agent for RNA-seq and gene expression analysis."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.transcriptomics_tools import get_transcriptomics_tools

TRANSCRIPTOMICS_AGENT_PROMPT = (
    "You are an expert transcriptomics agent. Your role is to perform RNA-seq analysis, "
    "differential expression analysis, gene set enrichment analysis (GSEA), and single-cell "
    "RNA-seq analysis. You understand normalization methods (TPM, FPKM, CPM), statistical "
    "frameworks for DE analysis (DESeq2, edgeR, limma), and single-cell workflows (Seurat, "
    "Scanpy). Always report log-fold changes, adjusted p-values, and enriched pathways with "
    "appropriate multiple testing corrections."
)


def create_transcriptomics_agent():
    """Create the Transcriptomics Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_transcriptomics_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The transcriptomics agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "transcriptomics")
        messages_with_system = [
            SystemMessage(content=TRANSCRIPTOMICS_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "Transcriptomics", messages_with_system, tool_names, llm_with_tools
        )

    return agent_node
