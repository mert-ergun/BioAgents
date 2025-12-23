"""Coder Agent Interactive Demo.

This demo showcases the Coder Agent's capabilities for generating and executing
Python code via Jupyter notebooks, creating visualizations, and using ToolUniverse tools.

Usage:
    # Option 1: Using IPython (Recommended - avoids 5+ min import overhead per query)
    uv run ipython
    In [1]: %run examples/coder_agent_interactive_demo.py
    In [2]: %load_ext autoreload
    In [3]: %autoreload 2
    In [4]: run_query("Your query here")

    # If you make major code changes (e.g., to graph.py), re-run:
    In [5]: %run examples/coder_agent_interactive_demo.py

    # Option 2: Direct execution (5+ min initialization each time)
    python examples/coder_agent_interactive_demo.py

Requirements:
    - Docker must be installed and running
    - Set up API keys in .env file

See docs/ipython_usage.md for detailed usage instructions and benefits.
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import BioAgents components
from bioagents.graph import create_graph

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"interactive_session_{timestamp}.txt"


class DualOutputHandler(logging.Handler):
    """Handler that writes to both console and file."""

    def __init__(self, log_file_path):
        super().__init__()
        self.log_file_path = log_file_path
        # PTH123: Replace with Path.open()
        # SIM115: Context manager not possible here as handle is kept open
        self.file_handle = Path(log_file_path).open("a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def emit(self, record):
        try:
            msg = self.format(record)
            print(msg, flush=True)
            self.file_handle.write(msg + "\n")
            self.file_handle.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self.file_handle:
            self.file_handle.close()
        super().close()


root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

log_handler = DualOutputHandler(log_file)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
root_logger.addHandler(log_handler)

logging.getLogger("BioAgents").setLevel(logging.DEBUG)
logging.getLogger("bioagents").setLevel(logging.DEBUG)
logging.getLogger("smolagents").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langsmith").setLevel(logging.WARNING)
logging.getLogger("langsmith.client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger("BioAgents")


def log_and_print(text: str, level=logging.INFO):
    """Print text to console and append to log file with timestamp."""
    logger.log(level, text)


# Write initial header to log file
with Path(log_file).open("w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("🚀 BioAgents Interactive Demo\n")
    f.write(f"📝 Log File: {log_file}\n")
    f.write(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    f.write("This log captures:\n")
    f.write("  • All node executions and state transitions\n")
    f.write("  • All agent messages (full content)\n")
    f.write("  • Tool calls and results\n")
    f.write("  • Coder agent execution logs\n")
    f.write("  • ToolUniverse search and execution logs\n")
    f.write("  • HTTP requests and responses\n")
    f.write("  • Exceptions and tracebacks\n")
    f.write("  • STDOUT/STDERR captures\n")
    f.write("\n" + "=" * 80 + "\n\n")

log_and_print("=" * 80)
log_and_print("🚀 BioAgents Interactive Demo")
log_and_print(f"📝 Logging to: {log_file}")
log_and_print("=" * 80)
log_and_print("\n⏳ Importing modules (this happens once)...")

# BioAgents components imported at top of file

log_and_print("✅ Modules imported!\n")

# Create graph once (cached)
log_and_print("⏳ Creating graph (starting Docker containers & loading models)...")
try:
    graph = create_graph()
    log_and_print("✅ Graph created successfully!\n")
except Exception as e:
    log_and_print(f"❌ Error creating graph: {type(e).__name__}: {e}", level=logging.ERROR)
    import traceback

    exc_traceback = traceback.format_exc()
    logger.error(f"Graph creation failed with traceback:\n{exc_traceback}")
    log_and_print("Make sure Docker is running!")
    log_and_print(f"Full error details logged to: {log_file}")
    sys.exit(1)


def run_query(query_text: str):
    """Helper function to run a query and print results nicely."""
    log_and_print(f"\n{'=' * 80}")
    log_and_print(f"🚀 Running query: {query_text}")
    log_and_print(f"{'=' * 80}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_start_time = datetime.now().timestamp()
    output_dir = Path("outputs") / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_and_print(f"📁 Output directory: {output_dir}")
    log_and_print(
        f"⏰ Start time: {datetime.fromtimestamp(run_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    initial_state = {
        "messages": [HumanMessage(content=query_text)],
        "output_dir": str(output_dir.absolute()),
        "run_start_time": run_start_time,
    }

    try:
        last_logged_node = None

        for event in graph.stream(initial_state):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue

                node_key = f"{node_name}_{id(node_output)}"
                if node_key == last_logged_node:
                    continue
                last_logged_node = node_key

                if node_name == "coder":
                    log_and_print(
                        f"\n📍 Node: {node_name.upper()} (smolagents logger will show execution details)"
                    )
                    if "next" in node_output:
                        log_and_print(f"   👉 Next Agent: {node_output['next']}")
                    continue
                log_and_print(f"\n{'=' * 80}")
                log_and_print(f"📍 NODE: {node_name.upper()}")
                log_and_print(f"{'=' * 80}")

                # Log node output structure for debugging
                if node_output:
                    log_and_print(f"   📦 Output structure: {list(node_output.keys())}")

                if "messages" in node_output:
                    for i, msg in enumerate(node_output["messages"]):
                        msg_type = type(msg).__name__
                        content = str(msg.content)

                        log_and_print(f"   📝 Message #{i + 1} ({msg_type}):")

                        if msg_type in ["HumanMessage", "AIMessage"]:
                            if len(content) > 2000:
                                display_content = (
                                    content[:1500] + "\n... [truncated] ...\n" + content[-500:]
                                )
                            else:
                                display_content = content
                            log_and_print(f"      {display_content}")
                            logger.debug(f"      Full content: {content}")

                        elif msg_type == "ToolMessage":
                            tool_name = getattr(msg, "name", "unknown_tool")
                            log_and_print(f"      🔧 Tool: {tool_name}")
                            if len(content) > 1500:
                                display_content = (
                                    content[:1200]
                                    + "\n... [truncated - see log file for full result] ...\n"
                                    + content[-300:]
                                )
                            else:
                                display_content = content
                            log_and_print(f"      📊 Result:\n{display_content}")
                            logger.debug(f"      Full tool result: {content}")

                        elif msg_type == "SystemMessage":
                            log_and_print(
                                f"      {content[:1000]}{'...' if len(content) > 1000 else ''}"
                            )
                            logger.debug(f"      Full system message: {content}")

                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            log_and_print(f"      🔧 Tool Calls ({len(msg.tool_calls)}):")
                            for j, tool_call in enumerate(msg.tool_calls):
                                tool_name = tool_call.get("name", "unknown")
                                tool_args = tool_call.get("args", {})
                                tool_id = tool_call.get("id", "no-id")
                                log_and_print(f"         [{j + 1}] {tool_name} (id: {tool_id})")
                                if tool_args:
                                    args_str = str(tool_args)
                                    if len(args_str) > 500:
                                        args_str = args_str[:400] + "... [truncated]"
                                    log_and_print(f"            Args: {args_str}")
                            # Log full tool calls to file
                            logger.debug(f"      Full tool calls: {msg.tool_calls}")

                # Show routing decision
                if "next" in node_output:
                    next_agent = node_output["next"]
                    log_and_print(f"   👉 Routing Decision: {next_agent.upper()}")
                    if "reasoning" in node_output:
                        reasoning = node_output["reasoning"]
                        log_and_print("   🤔 Reasoning:")
                        # Format reasoning with proper indentation
                        for line in reasoning.split("\n"):
                            log_and_print(f"      {line}")

                # Log any other fields (state changes, metadata, etc.)
                other_fields = {
                    k: v
                    for k, v in node_output.items()
                    if k not in ["messages", "next", "reasoning"]
                }
                if other_fields:
                    log_and_print("   📋 Additional State:")
                    for key, value in other_fields.items():
                        value_str = str(value)
                        if len(value_str) > 300:
                            value_str = value_str[:250] + "... [truncated]"
                        log_and_print(f"      • {key}: {value_str}")
                        # Log full value to file
                        logger.debug(f"      Full {key}: {value}")

        # After execution, move any output files to the output directory
        _move_output_files(output_dir, run_start_time)

        end_time = datetime.now()
        duration = end_time.timestamp() - run_start_time
        log_and_print(f"\n✅ Query completed in {duration:.2f} seconds")
        log_and_print(f"{'=' * 80}\n")

    except KeyboardInterrupt:
        log_and_print("\n⚠️ Query interrupted by user (Ctrl+C)", level=logging.WARNING)
        raise
    except Exception as e:
        log_and_print(
            f"\n❌ Error during query execution: {type(e).__name__}: {e}", level=logging.ERROR
        )
        import traceback

        exc_traceback = traceback.format_exc()
        # Log full traceback at ERROR level (will be in file)
        logger.error(f"Full exception traceback:\n{exc_traceback}")
        # Also log to console (truncated)
        log_and_print(f"   Full error details logged to: {log_file}")
        log_and_print(f"   Error type: {type(e).__name__}")
        log_and_print(f"   Error message: {e!s}")


def _move_output_files(output_dir: Path, run_start_time: float):
    """
    Move output files (images, data files) from current directory to output directory.

    IMPORTANT: Only moves files created/modified AFTER the run started.
    This prevents moving project files like requirements.txt, pyproject.toml, etc.

    Args:
        output_dir: Target directory for output files
        run_start_time: Timestamp when the run started (to filter old files)
    """
    # Important project files to NEVER move (regardless of timestamp)
    protected_files = {
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        ".gitignore",
        ".env",
        "setup.py",
        "Makefile",
        "Dockerfile",
        "pytest.ini",
        "uv.lock",
        "poetry.lock",
        ".git",
        ".venv",
    }

    # Only move clearly identifiable output files (not source files)
    output_patterns = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.pdf",
        "*.svg",  # Images
        "*.csv",
        "*.xlsx",
        "*.xls",  # Data files
    ]

    moved_files = []
    current_dir = Path()

    # Move image and data files (only if created/modified during this run)
    for pattern in output_patterns:
        for file_path in current_dir.glob(pattern):
            # Skip files in output directories and logs
            if "outputs" in str(file_path) or "logs" in str(file_path):
                continue

            # Skip protected files
            if file_path.name in protected_files:
                continue

            # Check if file was created/modified during this run
            try:
                file_mtime = file_path.stat().st_mtime
                # Only move if file was modified after run started (with 5 second buffer for safety)
                if file_mtime < (run_start_time - 5):
                    continue  # File is too old, skip it
            except OSError:
                continue  # Can't get mtime, skip to be safe

            try:
                dest_path = output_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                moved_files.append(dest_path)
            except Exception:  # nosec B110
                # File might be in use or already moved
                pass

    # For JSON files, only move if they look like data outputs AND are new
    json_output_patterns = ["*_data.json", "*_result.json", "*_output.json", "data_*.json"]
    for pattern in json_output_patterns:
        for file_path in current_dir.glob(pattern):
            if "outputs" in str(file_path) or "logs" in str(file_path):
                continue
            if file_path.name in protected_files:
                continue

            # Check timestamp
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime < (run_start_time - 5):
                    continue
            except OSError:
                continue

            try:
                dest_path = output_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                moved_files.append(dest_path)
            except Exception:  # nosec B110
                pass

    if moved_files:
        log_and_print(f"\n📦 Moved {len(moved_files)} output file(s) to: {output_dir}")
        for f in moved_files:
            log_and_print(f"   • {f.name}")


# EXPORT VARIABLES TO IPYTHON NAMESPACE
# This ensures graph and run_query are available even if %run is used
try:
    import builtins

    builtins.graph = graph
    builtins.run_query = run_query
except Exception:  # nosec B110
    pass

log_and_print("=" * 80)
log_and_print("🎉 Interactive session ready!")
log_and_print("=" * 80)
log_and_print("\nAvailable objects:")
log_and_print("  • graph: The compiled LangGraph workflow (ready to use)")
log_and_print("  • run_query(text): Helper function to run a query and print output")
log_and_print("\nExample commands:")
log_and_print("  >>> run_query('Find a tool for protein folding using the coder')")
log_and_print("  >>> run_query('Analyze human insulin P01308')")
log_and_print("\n" + "=" * 80)

# If running as a script, drop into IPython or pdb
if __name__ == "__main__":
    try:
        # Try to start IPython if available
        from IPython import start_ipython

        start_ipython(argv=[], user_ns=locals())
    except ImportError:
        # Fallback to code.interact
        import code

        code.interact(local=locals())
