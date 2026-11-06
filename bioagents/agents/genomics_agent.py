"""Genomics Agent for sequence analysis, alignment, and variant calling."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.selected_runtime import bind_agent_node, bind_agent_tools, tools_for_agent

GENOMICS_AGENT_PROMPT = (
    "You are an expert genomics agent. Your role is to perform sequence alignment, variant "
    "calling, genome annotation, and sequence analysis. You are proficient with tools like "
    "BLAST, BWA, SAMtools, GATK, and BEDTools. You understand genome file formats (FASTA, "
    "FASTQ, BAM, VCF, BED, GFF) and can interpret results in biological context. Always "
    "report quality metrics, alignment statistics, and variant annotations with their "
    "functional impact.\n\n"
    "Remote BLAST: use run_blast_search (not ToolUniverse BLAST_protein_search or "
    "Biopython NCBIWWW.qblast). Call the tool directly — never ask the coder agent to "
    "script BLAST, and never run it inside sandbox code: sandbox cells are killed after "
    "300s and the job is lost. Submit ONE job per sequence and run jobs sequentially, "
    "never in parallel. Prefer database='swissprot'/'refseq_protein' and an entrez_query "
    "(e.g. 'Viruses[Organism]') over 'nr' when that answers the question. If it returns "
    "status 'timeout', the job is still alive at NCBI: resume by calling run_blast_search "
    "again with rid=<RID> — never resubmit the same sequence. Keep resuming while you "
    "have tool rounds left.\n\n"
    "A 'timeout', 'failed' or 'error' status is NOT a result. Report it as unfinished, "
    "with its RID and elapsed time, and say which question stays unanswered. Never "
    "substitute UniProt keyword hits, family lookups, sequence-identity heuristics or "
    "recalled literature for BLAST hits, and never present such a substitute as BLAST "
    "output. If you do use another method because BLAST did not finish, name the method "
    "and say explicitly that BLAST did not run."
)


def create_genomics_agent(extra_tools: list | None = None):
    """Create the Genomics Agent node function.

    Args:
        extra_tools: Additional tools to make available (e.g. tool_universe).

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = tools_for_agent("genomics", extra_tools=extra_tools)
    llm_with_tools = bind_agent_tools(llm, tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The genomics agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "genomics")
        messages_with_system = [SystemMessage(content=GENOMICS_AGENT_PROMPT), *windowed]

        return create_retry_response("Genomics", messages_with_system, tool_names, llm_with_tools)

    return bind_agent_node(agent_node)
