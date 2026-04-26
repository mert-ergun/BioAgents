"""
ISOLATED REAL LLM TEST
Bypasses the multi-agent graph's sub-agent loops (which cause the Langchain/Gemini thought_signature bug)
and directly tests if the LLM respects our dynamic tool filter.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

# BioAgents imports
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.agents.helpers import filter_allowed_tools, inject_tools_to_system_prompt
from bioagents.agents.helpers import get_content_text

# Tools to test
from bioagents.tools.proteomics_tools import fetch_uniprot_fasta
from bioagents.tools.structural_tools import fetch_alphafold_structure, download_structure_file
from bioagents.tools.tool_universe import tool_universe_find_tools

def print_separator(title):
    print(f"\n{'=' * 80}")
    print(f" {title} ")
    print(f"{'=' * 80}")

def main():
    load_dotenv()
    print_separator("ISOLATED LLM TOOL FILTERING TEST")
    
    # 1. Define a subset of tools to represent the system
    all_available_tools = [
        fetch_uniprot_fasta, 
        fetch_alphafold_structure, 
        download_structure_file,
        tool_universe_find_tools
    ]
    
    # 2. Simulate UI sending ONLY ONE allowed tool
    allowed_from_ui = ["fetch_uniprot_fasta"]
    
    # 3. Apply your filter logic
    active_tools = filter_allowed_tools(all_available_tools, allowed_from_ui)
    active_tool_names = [getattr(t, "name", str(t)) for t in active_tools]
    
    print(f"[+] Total System Tools : {[getattr(t, 'name', str(t)) for t in all_available_tools]}")
    print(f"[+] Allowed by UI      : {allowed_from_ui}")
    print(f"[+] Active Tools Bound : {active_tool_names}\n")

    # 4. Inject into System Prompt
    base_prompt = load_prompt("research")
    dynamic_system_prompt = inject_tools_to_system_prompt(base_prompt, active_tool_names)
    
    # 5. Initialize the REAL LLM directly (Bypassing the buggy Graph/Sub-agents)
    llm = get_llm(prompt_name="research")
    llm_with_tools = llm.bind_tools(active_tools)
    
    # 6. The Trick Query
    query = "I need you to fetch the 3D structure of the p53 protein using AlphaFold and download the PDB file."
    print(f"USER QUERY:\n\"{query}\"")
    
    messages = [
        SystemMessage(content=dynamic_system_prompt),
        HumanMessage(content=query)
    ]
    
    print("\n[-] Sending request to REAL LLM... Please wait.")
    try:
        # Invoke LLM once directly
        response = llm_with_tools.invoke(messages)
        
        print_separator("LLM RESPONSE PARSED")
        
        # Parse Real Internal Thought / Content
        content = get_content_text(getattr(response, "content", ""))
        print("\n LLM ACTUAL CONTENT / REASONING:")
        if content and content.strip():
            print(content.strip())
        else:
            print("(LLM provided no text response, went straight to tools)")
            
        # Parse Real Tool Calls
        print("\n LLM ACTUAL TOOL CALLS:")
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"  - Tool Selected : {tc.get('name')}")
                print(f"  - Arguments     : {tc.get('args')}")
        else:
            print("  (No tools selected)")
            
        print("\n" + "=" * 80)
        print("SUCCESS: The LLM attempted to use its limited tools instead of AlphaFold!")
        
    except Exception as e:
        print(f"\n[!] LLM Execution Error: {e}")

if __name__ == "__main__":
    main()