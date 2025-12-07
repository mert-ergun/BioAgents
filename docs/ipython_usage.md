# IPython Usage in BioAgents

## Overview

IPython is an enhanced interactive Python shell that provides a powerful environment for iterative development and testing of BioAgents workflows. It allows you to:

- Keep the Python process alive between queries
- Initialize Docker containers and models **once** (not on every query)
- Run multiple queries interactively without restart overhead
- Iterate quickly on complex multi-agent workflows
- Use autoreload to automatically reload code changes

## Why IPython is Useful in BioAgents

### 1. **Import Time Optimization (CRITICAL)**

The most important benefit of IPython is avoiding repeated import and initialization overhead.

- **Without IPython** (`python examples/coder_agent_interactive_demo.py`):
  - Each run requires **5+ minutes** for:
    - Starting Docker containers
    - Loading LLM models (GPT, Gemini, etc.)
    - Initializing the LangGraph workflow
    - Setting up logging and handlers
    - Importing all dependencies
  - **Total overhead**: 5+ minutes **per query**

- **With IPython**:
  - **First run** (`%run examples/coder_agent_interactive_demo.py`): 5+ minutes (one-time cost)
  - **Subsequent queries**: Only code reloading (~1-2 seconds)
  - Docker containers stay alive
  - Models stay in memory
  - Graph remains initialized
  - **Total overhead**: ~1-2 seconds per query after initial setup

**Key Point**: The heavy initialization (Docker, models, graph) happens **once** at the start. All subsequent queries in the same IPython session skip this overhead entirely.

### 2. **Eliminating Repeated Import Overhead**

The critical advantage: **No repeated 5+ minute waits**.

- **Without IPython**: Every query = 5+ minutes of waiting for imports/initialization
- **With IPython**: First query = 5+ minutes, all others = instant (only query execution time)

This makes iterative development and testing **practical** instead of waiting hours for multiple test runs.

### 3. **Iterative Development**
- Test query variations quickly (no 5+ min wait between tests)
- Debug workflow issues interactively
- Experiment with different agent configurations
- View intermediate results without re-running entire workflows

### 4. **Code Reloading**
- Use `autoreload` to automatically reload code changes
- Test code modifications without restarting IPython (saves 5+ minutes per test)
- Perfect for rapid development cycles

## Setup and Usage

### Step 1: Start IPython

```bash
cd 06.12.2025/BioAgents
uv run ipython
```

### Step 2: Load the Interactive Demo (One-Time Setup)

Once IPython starts, run:

```python
%run examples/coder_agent_interactive_demo.py
```

**⚠️ IMPORTANT**: This step takes **5+ minutes** the first time because it:
- Starts Docker containers (if not already running)
- Loads all LLM models into memory (GPT, Gemini, etc.)
- Creates and compiles the LangGraph workflow
- Sets up comprehensive logging
- Imports all BioAgents dependencies

**But this only happens once!** After this initial setup, all subsequent queries in the same IPython session will be fast because:
- Docker containers remain running
- Models stay loaded in memory
- Graph is already compiled
- Only code changes are reloaded (if using autoreload)

This provides you with a `graph` object and `run_query()` function ready to use.

### Step 3: Enable Autoreload (Optional but Recommended)

```python
%load_ext autoreload
%autoreload 2
```

This enables automatic reloading of code changes:
- `%load_ext autoreload`: Loads the autoreload extension
- `%autoreload 2`: Reloads all modules (except those excluded by `%aimport`) before executing code

**Note**: Use `%autoreload 1` for more conservative reloading (only reload modules imported with `%aimport`).

### Step 4: Re-run Demo After Major Code Changes

**Important**: If you make changes to code that affects initialization (e.g., `graph.py`, `coder_agent.py`, Docker setup), you need to re-run the demo:

```python
%run examples/coder_agent_interactive_demo.py
```

**When to re-run**:
- Changes to `bioagents/graph.py` (workflow structure)
- Changes to agent initialization (`create_coder_agent`, `create_research_agent`, etc.)
- Changes to Docker executor setup
- Changes to prompt loading or system prompts
- Any changes that affect the `create_graph()` function

**When autoreload is sufficient**:
- Changes to agent logic (inside agent functions)
- Changes to tool implementations
- Changes to helper functions
- Changes to prompt content (if loaded dynamically)

**Tip**: If you're unsure, re-running `%run` is safe - it takes 5+ minutes but ensures everything is up-to-date.

### Step 5: Run Queries

Now you can run queries interactively:

```python
run_query("""
Comprehensive protein analysis with multiple outputs:
1. Get sequence for TP53 (P04637)
2. Calculate properties
3. Create a bar chart (save as 'tp53_chart.png')
4. Generate CSV report (save as 'tp53_data.csv')
5. Generate JSON summary (save as 'tp53_summary.json')
""")
```

## Complete Workflow Example

Here's the complete sequence from start to finish:

```bash
# 1. Start IPython
uv run ipython

# 2. In IPython, load the demo (takes 5+ minutes first time)
In [1]: %run examples/coder_agent_interactive_demo.py

# 3. Enable autoreload (optional but recommended)
In [2]: %load_ext autoreload
In [3]: %autoreload 2

# 4. Run your first query
In [4]: run_query("Analyze human insulin P01308")

# 5. Run another query (no restart needed - fast!)
In [5]: run_query("Find tools for protein folding")

# 6. Make code changes to agent logic (autoreload handles it)
In [6]: run_query("Get BRCA1 sequence and create visualization")

# 7. Make major changes (e.g., to graph.py or agent initialization)
#    Need to re-run the demo (takes 5+ minutes again)
In [7]: %run examples/coder_agent_interactive_demo.py

# 8. Continue with queries (fast again)
In [8]: run_query("Test the updated workflow")
```

## IPython Magic Commands

### Essential Commands

- `%run <file>`: Execute a Python file in the current namespace
- `%load_ext autoreload`: Load the autoreload extension
- `%autoreload <level>`: Set autoreload level (0=off, 1=conservative, 2=aggressive)
- `%aimport <module>`: Import a module and mark it for autoreload
- `%time <statement>`: Time the execution of a statement
- `%timeit <statement>`: Time the execution multiple times for accuracy

### Useful for Debugging

- `%debug`: Enter the debugger after an exception
- `%pdb`: Toggle automatic debugger on exceptions
- `%who`: List all variables in the current namespace
- `%whos`: List all variables with details
- `%reset`: Clear all variables from the namespace

## Tips and Best Practices

### 1. **Keep IPython Session Alive**
- Don't close IPython between queries
- Use `Ctrl+C` to interrupt a running query if needed
- Use `exit` or `quit` to properly close IPython

### 2. **Use Autoreload Wisely**
- Enable `%autoreload 2` during development
- Disable it (`%autoreload 0`) when running production queries
- Some modules (like `bioagents.graph`) may need manual reloading

### 3. **Monitor Resources**
- Check Docker containers: `docker ps`
- Monitor memory usage in IPython
- Restart IPython if containers become unresponsive

### 4. **Logging**
- All queries are logged to `logs/interactive_session_<timestamp>.txt`
- Logs include full agent messages, tool calls, and execution details
- Check logs for debugging failed queries

### 5. **Output Files**
- Each query creates a timestamped output directory: `outputs/run_<timestamp>/`
- Files are automatically organized by query
- Use the output directory path shown in the query output

## Troubleshooting

### Docker Container Issues
```python
# Check if containers are running
!docker ps

# Restart containers if needed
!docker restart <container_id>
```

### Model Loading Issues
```python
# Recreate the graph
from bioagents.graph import create_graph
graph = create_graph()
```

### Autoreload Not Working
```python
# Manually reload a module
import importlib
import bioagents.agents.coder_agent
importlib.reload(bioagents.agents.coder_agent)
```

## Comparison: IPython vs Direct Execution

| Aspect | Direct Execution | IPython |
|--------|-----------------|---------|
| **Initial Setup** | 5+ minutes per run | 5+ minutes once |
| **Import/Init Time** | 5+ minutes per query | ~1-2 seconds (after first run) |
| **Query Execution** | 5-30 seconds | 5-30 seconds |
| **Total Time (10 queries)** | 50+ minutes (500+ seconds) | ~6-7 minutes (360+ seconds) |
| **Code Reload** | Manual restart (5+ min each) | Automatic (1-2 seconds) |
| **Docker Containers** | Restart each time | Stay alive |
| **Model Loading** | Reload each time | Stay in memory |
| **Debugging** | Limited | Full interactive debugging |
| **Iteration Speed** | Very slow | Fast |

**Key Insight**: The 5+ minute import/initialization overhead is the main bottleneck. IPython eliminates this for all queries after the first one.

## Example Session

```python
# Start IPython
$ uv run ipython

# Load demo
In [1]: %run examples/coder_agent_interactive_demo.py
🚀 BioAgents Interactive Demo
📝 Logging to: logs/interactive_session_20251207_143022.txt
⏳ Importing modules (this happens once)...
✅ Modules imported!
⏳ Creating graph (starting Docker containers & loading models)...
✅ Graph created successfully!
🎉 Interactive session ready!

# Enable autoreload
In [2]: %load_ext autoreload
In [3]: %autoreload 2

# Run queries
In [4]: run_query("Get TP53 sequence and calculate molecular weight")
...
✅ Query completed in 12.34 seconds

In [5]: run_query("Create a bar chart of amino acid composition for TP53")
...
✅ Query completed in 8.76 seconds

# Make code changes to agent logic (autoreload handles reloading)
In [6]: run_query("Analyze BRCA1 with custom visualization")
...
✅ Query completed in 15.23 seconds

# Make major changes (e.g., to graph.py), need to re-run
In [7]: %run examples/coder_agent_interactive_demo.py
⏳ Creating graph (starting Docker containers & loading models)...
✅ Graph created successfully!

# Continue with updated code
In [8]: run_query("Test updated workflow")
...
✅ Query completed in 12.45 seconds
```

## Summary

IPython provides a powerful development environment for BioAgents that **eliminates the 5+ minute import/initialization overhead** for all queries after the first one.

### The Critical Benefit

**Without IPython**: Every query = 5+ minutes of waiting (Docker startup + model loading + graph initialization)

**With IPython**: 
- First query = 5+ minutes (one-time cost)
- All subsequent queries = Only execution time (~5-30 seconds)
- **10 queries**: 50+ minutes → ~6-7 minutes total

### Key Advantages

- **Eliminate repeated imports**: Docker containers, models, and graph stay alive
- **Save hours of waiting**: No 5+ minute wait between test queries
- **Iterate faster**: Test changes without restarting (autoreload handles code updates)
- **Debug easily**: Use interactive debugging tools
- **Develop efficiently**: Focus on queries, not waiting for initialization

### When to Use

- **Development & Testing**: **Always use IPython** - saves hours of waiting time
- **Production/One-off**: Direct execution is fine if you only need one query

**Bottom Line**: If you're running multiple queries or testing, IPython is **essential** - it transforms a 50+ minute testing session into a 6-7 minute one.

