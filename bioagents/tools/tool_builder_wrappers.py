"""Smolagents wrappers for Tool Builder functionality.

These wrappers allow the Coder Agent to interact with custom tools
created by the Tool Builder Agent.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from smolagents import Tool

if TYPE_CHECKING:
    from smolagents import Tool as SmolTool

logger = logging.getLogger(__name__)


def create_smolagent_tool_from_registry(tool_name: str) -> SmolTool:
    """Create a smolagents Tool from a registered custom tool.

    Args:
        tool_name: Name of the tool in the registry

    Returns:
        A smolagents Tool for use with CodeAgent
    """
    from bioagents.tools.tool_registry import get_registry

    registry = get_registry()
    tool_def = registry.get_tool(tool_name)

    if tool_def is None:
        raise ValueError(f"Tool '{tool_name}' not found in registry")

    # Build inputs schema
    inputs_schema = {}
    for param in tool_def.parameters:
        inputs_schema[param.name] = {
            "type": param.type,
            "description": param.description,
            "nullable": not param.required,
        }

    # Capture values for class definition
    _tool_name = tool_def.name
    _tool_desc = tool_def.description
    _tool_inputs = inputs_schema
    _tool_output = tool_def.return_type

    class CustomRegistryTool(Tool):
        name = _tool_name
        description = _tool_desc
        inputs = _tool_inputs
        output_type = _tool_output

        def __init__(self, tool_name_arg: str):
            super().__init__()
            self._tool_name = tool_name_arg

        def forward(self, **kwargs) -> Any:
            registry = get_registry()
            func = registry.load_tool_function(self._tool_name)
            if func is None:
                raise RuntimeError(f"Failed to load tool '{self._tool_name}'")
            result = func(**kwargs)
            registry.record_usage(self._tool_name)
            return result

    return CustomRegistryTool(tool_name)


def get_all_custom_tools_for_coder(
    validated_only: bool = True,
    categories: list[str] | None = None,
) -> list[SmolTool]:
    """Get all custom tools as smolagents Tools for the Coder Agent.

    Args:
        validated_only: Only return validated tools
        categories: Filter by categories

    Returns:
        List of smolagents Tools
    """
    from bioagents.tools.tool_registry import get_registry

    registry = get_registry()
    tools = []

    for tool_def in registry.list_tools(validated_only=validated_only):
        if categories and tool_def.category not in categories:
            continue

        try:
            smol_tool = create_smolagent_tool_from_registry(tool_def.name)
            tools.append(smol_tool)
        except Exception as e:
            logger.warning(f"Failed to create tool '{tool_def.name}': {e}")

    return tools


class CustomToolSearchTool(Tool):
    """Tool for searching custom tools in the registry."""

    name = "search_custom_tools"
    description = (
        "Search for custom bioinformatics tools in the BioAgents registry. "
        "Use this to find specialized tools that can help with your task."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "query": {
            "type": "string",
            "description": "Description of the capability you need",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default 5)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, limit: int = 5) -> str:
        from bioagents.tools.tool_registry import get_registry

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
                        "parameters": [p.name for p in tool.parameters],
                    }
                    for tool, score in results
                ],
            },
            indent=2,
        )


class CustomToolListTool(Tool):
    """Tool for listing all custom tools in the registry."""

    name = "list_custom_tools"
    description = (
        "List all custom tools available in the BioAgents registry. Optionally filter by category."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "category": {
            "type": "string",
            "description": "Filter by category (genomics, proteomics, ml, etc.)",
            "nullable": True,
        },
        "validated_only": {
            "type": "boolean",
            "description": "Only show validated tools",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        category: str | None = None,
        validated_only: bool = False,
    ) -> str:
        from bioagents.tools.tool_registry import get_registry

        registry = get_registry()
        tools = registry.list_tools(
            category=category,
            validated_only=validated_only,
        )

        return json.dumps(
            {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "category": t.category,
                        "validated": t.validated,
                        "usage_count": t.usage_count,
                    }
                    for t in tools
                ],
                "total": len(tools),
                "categories": list({t.category for t in tools}),
            },
            indent=2,
        )


class CustomToolExecuteTool(Tool):
    """Tool for executing a custom tool from the registry."""

    name = "execute_custom_tool"
    description = (
        "Execute a custom tool from the BioAgents registry. "
        "First use search_custom_tools to find available tools, "
        "then use this to run them."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "tool_name": {
            "type": "string",
            "description": "Exact name of the custom tool to execute",
        },
        "arguments": {
            "type": "object",
            "description": "Dictionary of arguments to pass to the tool",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        from bioagents.tools.tool_registry import get_registry

        registry = get_registry()
        func = registry.load_tool_function(tool_name)

        if func is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool '{tool_name}' not found or failed to load",
                }
            )

        try:
            args = arguments or {}
            result = func(**args)
            registry.record_usage(tool_name)

            return json.dumps(
                {
                    "status": "success",
                    "tool": tool_name,
                    "result": result,
                },
                default=str,
                indent=2,
            )

        except Exception as e:
            logger.error(f"Custom tool execution failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "tool": tool_name,
                    "message": str(e),
                }
            )


class CustomToolInfoTool(Tool):
    """Tool for getting detailed information about a custom tool."""

    name = "get_custom_tool_info"
    description = (
        "Get detailed information about a custom tool including "
        "its parameters, return type, and usage examples."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "tool_name": {
            "type": "string",
            "description": "Name of the tool to get info about",
        },
    }
    output_type = "string"

    def forward(self, tool_name: str) -> str:
        from bioagents.tools.tool_registry import get_registry

        registry = get_registry()
        tool = registry.get_tool(tool_name)

        if tool is None:
            return json.dumps(
                {
                    "error": f"Tool '{tool_name}' not found",
                }
            )

        return json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "validated": tool.validated,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default,
                    }
                    for p in tool.parameters
                ],
                "return_type": tool.return_type,
                "return_description": tool.return_description,
                "dependencies": tool.dependencies,
                "source_software": tool.source_software,
                "source_url": tool.source_url,
                "usage_count": tool.usage_count,
            },
            indent=2,
        )


class RequestNewToolTool(Tool):
    """Tool for requesting the creation of a new custom tool."""

    name = "request_new_tool"
    description = (
        "Request the Tool Builder Agent to create a new custom tool. "
        "Use this when you encounter a bioinformatics task that "
        "existing tools cannot handle."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "task_description": {
            "type": "string",
            "description": "Detailed description of what the tool should do",
        },
        "input_description": {
            "type": "string",
            "description": "Description of expected inputs",
        },
        "output_description": {
            "type": "string",
            "description": "Description of expected outputs",
        },
        "suggested_software": {
            "type": "string",
            "description": "Name of existing software to wrap (if known)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        task_description: str,
        input_description: str,
        output_description: str,
        suggested_software: str | None = None,
    ) -> str:
        # This creates a request that the Tool Builder Agent can process
        # In a full implementation, this would add to a queue or trigger the agent

        request = {
            "request_type": "new_tool",
            "task_description": task_description,
            "input_description": input_description,
            "output_description": output_description,
            "suggested_software": suggested_software,
            "status": "pending",
            "message": (
                "Tool request created. The Tool Builder Agent will process this "
                "request and create a custom tool. Use list_custom_tools to check "
                "when the tool becomes available."
            ),
        }

        # Log the request for the supervisor to pick up
        logger.info(f"New tool request: {task_description}")

        return json.dumps(request, indent=2)


def get_custom_tool_wrappers() -> list[Tool]:
    """Get all smolagents wrappers for custom tool functionality.

    Returns:
        List of Tool instances for use with CodeAgent
    """
    return [
        CustomToolSearchTool(),
        CustomToolListTool(),
        CustomToolExecuteTool(),
        CustomToolInfoTool(),
        RequestNewToolTool(),
    ]


def get_coder_tools_with_custom_registry(
    include_tool_universe: bool = True,
    include_custom_tools: bool = True,
) -> list[Tool]:
    """Get a complete tool set for the Coder Agent including custom tools.

    Args:
        include_tool_universe: Include ToolUniverse search and execute
        include_custom_tools: Include custom tool registry access

    Returns:
        List of all tools for the Coder Agent
    """
    tools = []

    if include_tool_universe:
        from bioagents.tools.smol_tool_wrappers import (
            ToolUniverseExecuteTool,
            ToolUniverseSearchTool,
        )

        tools.extend(
            [
                ToolUniverseSearchTool(),
                ToolUniverseExecuteTool(),
            ]
        )

    if include_custom_tools:
        tools.extend(get_custom_tool_wrappers())

    return tools
