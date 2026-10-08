"""Data Acquisition Agent for downloading and managing biological datasets."""

from langchain_core.messages import SystemMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.file_tools import get_file_tools
from bioagents.tools.web_tools import get_web_tools

DATA_ACQUISITION_AGENT_PROMPT = (
    "You are an expert biological data acquisition agent. Your role is to download and "
    "manage datasets from biological databases including GEO, SRA, TCGA, UniProt, and NCBI. "
    "You understand the data formats and access patterns for each database, can construct "
    "appropriate queries, download data efficiently, and organize files for downstream "
    "analysis. Always verify data integrity after download and report dataset metadata "
    "such as sample counts, species, and experimental conditions.\n\n"
    "For full UniProt entries (plain-text .txt / flat files), always use "
    "download_uniprot_flat_file — never fetch_url_content for rest.uniprot.org …/uniprotkb/… .txt "
    "because those responses are huge and stall the workflow. After saving, summarize from the "
    "returned preview or confirm the path and file size.\n\n"
    "IMPORTANT: When downloading AlphaFold structures, ALWAYS use model_v6 URLs (NOT v4 or v3). "
    "Example: https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v6.pdb "
    "Older versions (v1-v4) have been removed from the AlphaFold DB server and return 404 errors. "
    "For protein structures, prefer the download_structure_file tool which handles this automatically."
    "\n\n"
    "## Scope — what you must NOT do\n"
    "Your job ends when the data is downloaded, verified and its location reported. You "
    "acquire data; you do not analyse it.\n\n"
    "YOU CANNOT EXECUTE CODE. You have no Python interpreter, no shell and no compute "
    "sandbox — only tools for fetching files and reading/writing them. Writing a Python "
    "script with write_local_file does NOT run it and produces NO results.\n\n"
    "Therefore you must NEVER:\n"
    "- write analysis, parsing, extraction or calculation scripts and treat them as work done;\n"
    "- attempt structure parsing, SMILES conversion, RMSD or any other computation yourself;\n"
    "- keep writing more scripts when one did not produce the answer.\n\n"
    "Instead, when the task needs computation, STOP and report back: state what you "
    "downloaded, give the exact file paths, and say explicitly that the computation step "
    "needs the Coder agent (or another specialist). Handing the task back with the data "
    "in place is a SUCCESSFUL outcome for you — it is not a failure, and it is far more "
    "useful than scripts that never ran.\n\n"
    "If you find yourself calling write_local_file repeatedly, you have left your scope. "
    "Stop and hand back to the supervisor."
)


def create_data_acquisition_agent():
    """Create the Data Acquisition Agent node function.

    Returns:
        A function that can be used as a LangGraph node
    """
    llm = get_llm()
    tools = get_web_tools() + get_file_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_names = [t.name for t in tools]

    def agent_node(state):
        """The data acquisition agent node function."""
        messages = state["messages"]
        windowed = prepare_messages_for_agent(messages, "data_acquisition")
        messages_with_system = [
            SystemMessage(content=DATA_ACQUISITION_AGENT_PROMPT),
            *windowed,
        ]

        return create_retry_response(
            "DataAcquisition", messages_with_system, tool_names, llm_with_tools
        )

    return agent_node
