# Tool selection

The checkout uses the main repository's native workflow plus an operator-owned
tool selection. Task definitions and native benchmark behavior are unchanged.

```dotenv
BIOAGENTS_TOOL_MODE=selected
BIOAGENTS_TOOL_SELECTION_FILE=bioagents/tools/selected_tools.json
```

`selected` enables only the names in the JSON file. `all` enables the main tool
catalogue. The current selection contains the requested 11 local public names and
9 ToolUniverse public names. The public `genmol` alias resolves internally to
`NvidiaNIM_genmol`.

Add an existing registered name to `local_tools` or `tool_universe_tools` to enable
it. Original descriptions and argument schemas are resolved automatically. Local
`@tool` definitions and the existing custom registry are supported. Remove a name
to hide its introduction and schema and reject its calls before execution.

Agent factories, Supervisor inventory, system introductions, discovery, custom
registry dispatch, Web UI listings, and Coder/ML/DL adapters use the same selection.
Unavailable calls return `tool_not_available`, `execution_state=not_started`, and
an English instruction to check the current tool introduction list. Invalid
operator configuration never silently enables all mode.

New graphs read the current JSON. Existing graph nodes and lazy executors retain
their frozen selection. Docker receives a name-only snapshot and configured
credentials needed by the selected SDK backends; credentials are not embedded in
JSON or tool definitions. Ordinary Python libraries remain available. Generated
cells cannot directly import the registered SDK/tool implementation packages;
this is an application interface restriction, not a separate OS sandbox.

Run the offline selection checks from the repository root:

```bash
python tests/run_selected_checks.py
```
