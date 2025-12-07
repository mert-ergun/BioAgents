"""Wrappers for ToolUniverse tools to be used with smolagents."""

from smolagents import Tool


class ToolUniverseSearchTool(Tool):
    """Tool for searching bioinformatics tools in ToolUniverse."""

    name = "tool_universe_find_tools"
    description = "Search for bioinformatics tools in ToolUniverse."
    inputs = {
        "description": {"type": "string", "description": "Description of the capability needed."},
        "limit": {"type": "integer", "description": "Max number of tools to return (default 5).", "nullable": True}
    }
    output_type = "string"

    def forward(self, description: str, limit: int = 5) -> str:
        from bioagents.tools.tool_universe import DEFAULT_WRAPPER
        return DEFAULT_WRAPPER.find_tools(description, limit=limit)


class ToolUniverseExecuteTool(Tool):
    """Tool for executing specific ToolUniverse tools."""

    name = "tool_universe_call_tool"
    description = "Execute a specific ToolUniverse tool."
    inputs = {
        "tool_name": {"type": "string", "description": "Exact name of the tool."},
        "arguments_json": {"type": "string", "description": "JSON string of arguments.", "nullable": True}
    }
    output_type = "string"

    def forward(self, tool_name: str, arguments_json: str = "{}") -> str:
        from bioagents.tools.tool_universe import DEFAULT_WRAPPER
        return DEFAULT_WRAPPER.execute_tool(tool_name, arguments_json)

