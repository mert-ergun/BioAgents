"""Coder Agent for generating and executing code via Jupyter notebooks."""

import logging
from typing import List, Optional, Any, Dict, Callable, Literal
from smolagents import CodeAgent, Tool
from langchain_core.messages import AIMessage
from bioagents.llms.llm_provider import get_llm
from bioagents.tools.smol_tool_wrappers import ToolUniverseSearchTool, ToolUniverseExecuteTool
from bioagents.llms.adapters import LangChainModelAdapter
from bioagents.sandbox.coder_executor import create_executor
from bioagents.sandbox.coder_helpers import (
    extract_original_query,
    extract_available_data,
    build_task_with_output_dir
)
from bioagents.prompts.prompt_loader import load_prompt

logger = logging.getLogger("BioAgents")

CODER_AGENT_PROMPT = load_prompt("coder")


def create_coder_agent(
    tools: Optional[List[Tool]] = None,
    additional_imports: Optional[List[str]] = None,
    model_provider: Optional[Literal["openai", "ollama", "gemini"]] = None,
    model_name: Optional[str] = None,
    max_steps: int = 20
) -> CodeAgent:
    """
    Create the Coder Agent node function.

    Args:
        tools: List of tools available to the agent (defaults to ToolUniverse tools)
        additional_imports: Additional Python packages to allow in the sandbox
        model_provider: LLM provider name. If None, reads from LLM_PROVIDER env var (defaults to 'openai')
        model_name: Specific model name to use
        max_steps: Maximum number of execution steps

    Returns:
        A CodeAgent instance that can generate and execute Python code via Jupyter notebooks
    """
    if tools is None:
        tools = [ToolUniverseSearchTool(), ToolUniverseExecuteTool()]
        
    if additional_imports is None:
        additional_imports = [
            "pandas", "numpy.*", "matplotlib.*", "scipy.*", 
            "bioagents.*", "typing", "json", "os", "os.path", "sys", "pathlib"
        ]

    lc_model = get_llm(provider=model_provider, model=model_name)
    model = LangChainModelAdapter(lc_model)
    executor = create_executor(additional_imports)
    
    # Escape Jinja2 template syntax in instructions to avoid conflicts
    escaped_instructions = CODER_AGENT_PROMPT.replace("{", "{{").replace("}", "}}")
    
    agent = CodeAgent(
        tools=tools,
        model=model,
        executor=executor,
        additional_authorized_imports=additional_imports,
        max_steps=max_steps,
        instructions=escaped_instructions,
    )
    
    return agent


def create_coder_node(agent: CodeAgent) -> Callable:
    """
    Create the Coder Agent node function.

    Args:
        agent: The CodeAgent instance to wrap

    Returns:
        A function that can be used as a LangGraph node
    """
    # Capture system prompt for use in task description
    system_prompt = CODER_AGENT_PROMPT
    
    def coder_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        
        original_query = extract_original_query(messages)
        available_data = extract_available_data(messages)
        output_dir = state.get("output_dir")
        
        task = build_task_with_output_dir(
            original_query,
            available_data,
            output_dir,
            system_prompt=system_prompt
        )
        
        try:
            logger.info(f"Starting coder agent execution with task length: {len(task)} chars")
            result = agent.run(task)
            logger.info(f"Coder agent execution completed. Result type: {type(result)}")
            
            # Format result message - prioritize final_answer if available
            if hasattr(result, 'final_answer') and result.final_answer:
                content = f"Task completed successfully.\n\nFinal answer: {result.final_answer}"
                logger.info("Coder agent returned final_answer")
            elif hasattr(result, 'output') and result.output:
                content = str(result.output)
                logger.info(f"Coder agent returned output (length: {len(content)} chars)")
            else:
                content = str(result)
                logger.warning(f"Coder agent returned unexpected result format: {type(result)}")
            
            return {
                "messages": [AIMessage(content=content)],
                "next": "supervisor"
            }
        except Exception as e:
            import traceback
            error_msg = f"Error executing code: {e}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Coder agent error: {error_msg}")
            return {
                "messages": [AIMessage(content=error_msg)],
                "next": "supervisor"
            }
            
    return coder_node
