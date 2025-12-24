"""Tools for the Tool Builder Agent for discovering, creating, and managing custom tools."""

import json
import logging
import re

from langchain_core.tools import tool

from bioagents.llms.llm_provider import get_llm
from bioagents.tools.tool_registry import (
    ToolDefinition,
    ToolParameter,
    get_registry,
)

logger = logging.getLogger(__name__)


@tool
def extract_tools_from_text(text: str, _section_type: str = "methods") -> str:
    """Extract bioinformatics tool mentions from scientific text.

    Analyzes Methods sections of papers to identify:
    - Software tools being used
    - Tasks they perform
    - Associated databases
    - GitHub/documentation URLs

    Args:
        text: The scientific text to analyze (ideally Methods section)
        _section_type: Type of section (methods, results, introduction)

    Returns:
        JSON string with extracted tool information
    """
    llm = get_llm()

    prompt = f"""Analyze this scientific text and extract all bioinformatics tools, software, and databases mentioned.

For each tool, identify:
1. Name: The exact name of the tool/software
2. Task: What the tool is used for in this context
3. URL: Any GitHub, PyPI, or documentation URL mentioned
4. Database: Any associated database the tool accesses

Text to analyze:
{text}

Return a JSON object with a "tools" array. Each tool should have: name, task, url (or null), database (or null).
Only include actual bioinformatics tools, not general Python libraries like pandas or numpy.
Focus on domain-specific tools like: samtools, BLAST, DESeq2, Scanpy, CellTypist, etc.
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json_match.group()

        return json.dumps({"tools": [], "error": "Could not extract structured data"})

    except Exception as e:
        logger.error(f"Tool extraction failed: {e}")
        return json.dumps({"tools": [], "error": str(e)})


@tool
def research_tool_documentation(tool_name: str, tool_url: str | None = None) -> str:
    """Research a tool to gather documentation and usage information.

    Searches for:
    - GitHub README and documentation
    - PyPI package information
    - API documentation
    - Usage examples

    Args:
        tool_name: Name of the tool to research
        tool_url: Optional URL to start research from

    Returns:
        JSON string with gathered documentation
    """
    # This would integrate with web search in production
    # For now, return a structured template for the LLM to work with

    result = {
        "tool_name": tool_name,
        "url": tool_url,
        "research_notes": f"Research needed for {tool_name}",
        "suggested_actions": [
            f"Search GitHub for '{tool_name}'",
            f"Check PyPI for '{tool_name}' package",
            "Look for official documentation",
            "Find usage examples in tutorials",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def generate_tool_wrapper(
    tool_name: str,
    tool_description: str,
    tool_category: str,
    parameters_json: str,
    return_type: str,
    return_description: str,
    implementation_notes: str,
) -> str:
    """Generate a Python wrapper for a bioinformatics tool.

    Creates a complete Python function that wraps the tool's functionality,
    handling input/output, error handling, and integration with BioAgents.

    Args:
        tool_name: Name for the wrapper function
        tool_description: Description of what the tool does
        tool_category: Category (genomics, proteomics, structural, ml, etc.)
        parameters_json: JSON string describing parameters
        return_type: Return type (string, dict, DataFrame)
        return_description: Description of what's returned
        implementation_notes: Notes on how to implement (CLI wrapper, API call, etc.)

    Returns:
        JSON string with generated code and metadata
    """
    llm = get_llm()

    prompt = f"""Generate a Python wrapper function for this bioinformatics tool:

Tool Name: {tool_name}
Description: {tool_description}
Category: {tool_category}
Parameters: {parameters_json}
Return Type: {return_type}
Return Description: {return_description}
Implementation Notes: {implementation_notes}

Requirements:
1. Create a well-documented Python function
2. Include proper type hints
3. Add comprehensive docstring
4. Handle errors gracefully
5. Return data in a consistent format
6. Make it work as a standalone function

Generate the complete Python code for this wrapper. Include:
- All necessary imports at the top
- The main function implementation
- A simple test case at the bottom (in if __name__ == "__main__" block)

Return a JSON object with:
- "code": The complete Python code
- "dependencies": List of pip packages needed
- "test_code": A simple test invocation
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Extract JSON or code from response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json_match.group()

        # If no JSON, wrap the code in a JSON structure
        code_match = re.search(r"```python\n([\s\S]*?)```", content)
        if code_match:
            return json.dumps(
                {
                    "code": code_match.group(1),
                    "dependencies": [],
                    "test_code": None,
                }
            )

        return json.dumps(
            {
                "error": "Could not generate wrapper",
                "raw_response": content[:500],
            }
        )

    except Exception as e:
        logger.error(f"Wrapper generation failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def register_custom_tool(
    name: str,
    description: str,
    category: str,
    code: str,
    parameters_json: str,
    return_type: str,
    return_description: str,
    dependencies_json: str = "[]",
    source_paper: str | None = None,
    source_url: str | None = None,
    source_software: str | None = None,
) -> str:
    """Register a custom tool in the BioAgents tool registry.

    Saves the tool for future use by all agents in the system.

    Args:
        name: Unique name for the tool
        description: What the tool does
        category: Tool category
        code: Python code implementing the tool
        parameters_json: JSON array of parameter definitions
        return_type: What the tool returns
        return_description: Description of return value
        dependencies_json: JSON array of pip dependencies
        source_paper: DOI or bioRxiv ID if from literature
        source_url: GitHub or documentation URL
        source_software: Name of the wrapped software

    Returns:
        Status message
    """
    try:
        # Parse parameters
        params_data = json.loads(parameters_json)
        parameters = [
            ToolParameter(
                name=p.get("name", "arg"),
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
            )
            for p in params_data
        ]

        dependencies = json.loads(dependencies_json)

        # Create tool definition
        definition = ToolDefinition(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            return_type=return_type,
            return_description=return_description,
            code=code,
            dependencies=dependencies,
            source_paper=source_paper,
            source_url=source_url,
            source_software=source_software,
        )

        # Register in the registry
        registry = get_registry()
        success = registry.register_tool(definition, overwrite=True)

        if success:
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Tool '{name}' registered successfully",
                    "tool_name": name,
                    "category": category,
                }
            )
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to register tool '{name}'",
                }
            )

    except Exception as e:
        logger.error(f"Tool registration failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def validate_custom_tool(tool_name: str, test_args_json: str = "{}") -> str:
    """Validate a custom tool by loading and optionally testing it.

    Args:
        tool_name: Name of the tool to validate
        test_args_json: Optional JSON object with test arguments

    Returns:
        Validation result
    """
    try:
        registry = get_registry()
        test_args = json.loads(test_args_json) if test_args_json else None

        success = registry.validate_tool(tool_name, test_args)

        if success:
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Tool '{tool_name}' validated successfully",
                }
            )
        else:
            tool = registry.get_tool(tool_name)
            error = tool.validation_error if tool else "Tool not found"
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Validation failed: {error}",
                }
            )

    except Exception as e:
        logger.error(f"Tool validation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def list_custom_tools(category: str | None = None, validated_only: bool = False) -> str:
    """List all custom tools in the registry.

    Args:
        category: Filter by category (genomics, proteomics, etc.)
        validated_only: Only show validated tools

    Returns:
        JSON array of tool information
    """
    registry = get_registry()
    tools = registry.list_tools(category=category, validated_only=validated_only)

    return json.dumps(
        {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "validated": t.validated,
                    "usage_count": t.usage_count,
                    "source_software": t.source_software,
                }
                for t in tools
            ],
            "total": len(tools),
        },
        indent=2,
    )


@tool
def search_custom_tools(query: str, limit: int = 5) -> str:
    """Search custom tools by description.

    Args:
        query: Search query describing needed functionality
        limit: Maximum results to return

    Returns:
        JSON array of matching tools with scores
    """
    registry = get_registry()
    results = registry.search_tools(query, limit=limit)

    return json.dumps(
        {
            "query": query,
            "results": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "score": score,
                    "validated": tool.validated,
                }
                for tool, score in results
            ],
        },
        indent=2,
    )


@tool
def get_tool_code(tool_name: str) -> str:
    """Get the Python code for a custom tool.

    Args:
        tool_name: Name of the tool

    Returns:
        The tool's Python code or error message
    """
    registry = get_registry()
    tool = registry.get_tool(tool_name)

    if tool is None:
        return json.dumps({"error": f"Tool '{tool_name}' not found"})

    return json.dumps(
        {
            "name": tool.name,
            "code": tool.code,
            "dependencies": tool.dependencies,
            "parameters": [
                {"name": p.name, "type": p.type, "description": p.description}
                for p in tool.parameters
            ],
        },
        indent=2,
    )


@tool
def execute_custom_tool(tool_name: str, arguments_json: str = "{}") -> str:
    """Execute a custom tool from the registry.

    Args:
        tool_name: Name of the tool to execute
        arguments_json: JSON object with tool arguments

    Returns:
        Tool execution result
    """
    try:
        registry = get_registry()
        func = registry.load_tool_function(tool_name)

        if func is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to load tool '{tool_name}'",
                }
            )

        args = json.loads(arguments_json)
        result = func(**args)

        # Record usage
        registry.record_usage(tool_name)

        return json.dumps(
            {
                "status": "success",
                "tool": tool_name,
                "result": result,
            },
            default=str,
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return json.dumps(
            {
                "status": "error",
                "message": str(e),
            }
        )


def get_tool_builder_tools() -> list:
    """Get all tools available to the Tool Builder Agent."""
    return [
        extract_tools_from_text,
        research_tool_documentation,
        generate_tool_wrapper,
        register_custom_tool,
        validate_custom_tool,
        list_custom_tools,
        search_custom_tools,
        get_tool_code,
        execute_custom_tool,
    ]
