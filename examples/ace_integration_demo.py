"""Test script for ACE integration with BioAgents graph.

This script tests the ACE (Agent-Curator-Environment) integration with BioAgents.
The behavior depends on the ACE_ENABLED setting in your .env file:

Configuration (.env):
    - ACE_ENABLED=true/false: Enable or disable ACE integration
        * true, "1", or "yes" → ACE enabled (self-evolving capabilities active)
        * false (default) → ACE disabled (normal operation, zero overhead)

    - ACE_CURATOR_FREQUENCY=N: How often curator runs (default: 5)
        * Lower values = more frequent updates (e.g., 2 for testing)
        * Higher values = less frequent updates (e.g., 10 for production)

    - ACE_PLAYBOOK_DIR: Directory for playbooks (default: bioagents/playbooks)

    - ACE_PLAYBOOK_TOKEN_BUDGET: Token budget for playbooks (default: 80000)

When ACE is enabled:
    - Agents track their executions
    - Reflector analyzes agent outputs and tags playbook bullets
    - Curator periodically adds new instructions based on learned patterns
    - Playbooks are automatically created/updated in bioagents/playbooks/
    - Learned best practices are merged into agent prompts

When ACE is disabled:
    - System works normally without any ACE overhead
    - No playbooks are created or updated
    - Agent prompts remain static (from XML files)

Note: This script automatically detects ACE_ENABLED from .env and tests accordingly.
"""

import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from bioagents.graph import create_graph
from bioagents.learning.ace_integration import clear_ace_cache


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for beautiful terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Agent colors
    SUPERVISOR = "\033[95m"  # Magenta
    RESEARCH = "\033[94m"  # Blue
    ANALYSIS = "\033[96m"  # Cyan
    REPORT = "\033[93m"  # Yellow
    TOOL_BUILDER = "\033[92m"  # Green
    PROTEIN_DESIGN = "\033[96m"  # Cyan
    CRITIC = "\033[91m"  # Red
    CODER = "\033[93m"  # Yellow

    # Status colors
    SUCCESS = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    INFO = "\033[94m"  # Blue

    # ACE-specific colors
    ACE = "\033[95m"  # Magenta
    PLAYBOOK = "\033[96m"  # Cyan


def print_banner(title: str, char: str = "="):
    """Print a formatted banner."""
    width = 100
    print(f"\n{Colors.BOLD}{char * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{title:^{width}}{Colors.RESET}")
    print(f"{Colors.BOLD}{char * width}{Colors.RESET}\n")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{'-' * 100}{Colors.RESET}")
    print(f"  {Colors.ACE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'-' * 100}{Colors.RESET}")


def get_agent_color(node_name: str) -> str:
    """Get color code for an agent node."""
    node_lower = node_name.lower()
    if "supervisor" in node_lower:
        return Colors.SUPERVISOR
    elif "research" in node_lower:
        return Colors.RESEARCH
    elif "analysis" in node_lower:
        return Colors.ANALYSIS
    elif "report" in node_lower:
        return Colors.REPORT
    elif "tool_builder" in node_lower or "toolbuilder" in node_lower:
        return Colors.TOOL_BUILDER
    elif "protein_design" in node_lower or "proteindesign" in node_lower:
        return Colors.PROTEIN_DESIGN
    elif "critic" in node_lower:
        return Colors.CRITIC
    elif "coder" in node_lower or "ml" in node_lower or "dl" in node_lower:
        return Colors.CODER
    elif "tools" in node_lower:
        return Colors.INFO
    elif "end" in node_lower:
        return Colors.SUCCESS
    else:
        return Colors.INFO


def format_node_name(node_name: str) -> str:
    """Format node name with color."""
    color = get_agent_color(node_name)
    return f"{color}{Colors.BOLD}{node_name.upper()}{Colors.RESET}"


def test_ace_integration():
    """Test ACE integration based on .env configuration."""
    # Check current ACE_ENABLED status
    current_ace_enabled = os.getenv("ACE_ENABLED", "false").lower()
    is_ace_enabled = current_ace_enabled in ("true", "1", "yes")

    if is_ace_enabled:
        print_banner("🚀 Testing ACE Integration (ACE Enabled)")
    else:
        print_banner("🚀 Testing ACE Integration (ACE Disabled)")

    # Set curator frequency for testing (only if not already set and ACE is enabled)
    if is_ace_enabled and os.getenv("ACE_CURATOR_FREQUENCY") is None:
        os.environ["ACE_CURATOR_FREQUENCY"] = "2"  # Run curator every 2 executions for testing

    # Clear ACE cache to ensure fresh initialization
    clear_ace_cache()

    # Create graph (ACE will be initialized based on .env setting)
    print(f"{Colors.INFO}Creating workflow graph...{Colors.RESET}")
    graph = create_graph()

    # Test query
    test_query = "What is the molecular weight of TP53 protein?"

    print_section("Test Query")
    print(f"{Colors.BOLD}Query:{Colors.RESET} {test_query}")

    if is_ace_enabled:
        print(f"\n{Colors.INFO}⏳ Running graph with ACE enabled...{Colors.RESET}")
    else:
        print(f"\n{Colors.INFO}⏳ Running graph with ACE disabled...{Colors.RESET}")

    # Run graph with streaming to track agent execution
    initial_state = {
        "messages": [HumanMessage(content=test_query)],
        "next": "",
        "reasoning": "",
        "output_dir": None,
    }

    try:
        # Show execution trace
        print(f"\n{Colors.BOLD}{'─' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}EXECUTION TRACE{Colors.RESET}")
        print(f"{Colors.BOLD}{'─' * 100}{Colors.RESET}\n")

        total_steps = 0
        agent_flow = []
        final_state = None

        # Stream through workflow to track progress
        for step_output in graph.stream(initial_state, {"recursion_limit": 50}):
            total_steps += 1
            node_name = next(iter(step_output))
            node_state = step_output[node_name]
            agent_flow.append(node_name)

            # Print colored step trace
            color = get_agent_color(node_name)
            print(
                f"  {Colors.BOLD}Step {total_steps}:{Colors.RESET} {color}{Colors.BOLD}{node_name.upper()}{Colors.RESET}"
            )

            # Show supervisor routing decisions
            if node_name == "supervisor" and "next" in node_state:
                next_agent = node_state.get("next", "FINISH")
                reasoning = node_state.get("reasoning", "")
                next_color = get_agent_color(next_agent)
                print(
                    f"    {Colors.SUPERVISOR}→ Routing to: {next_color}{next_agent}{Colors.RESET}"
                )
                if reasoning:
                    reasoning_wrapped = textwrap.fill(
                        reasoning,
                        width=90,
                        initial_indent="    Reasoning: ",
                        subsequent_indent="               ",
                    )
                    print(f"{Colors.DIM}{reasoning_wrapped}{Colors.RESET}")

            # Show agent processing
            elif node_name != "supervisor" and not node_name.endswith("_tools"):
                agent_color = get_agent_color(node_name)
                print(f"    {agent_color}Processing...{Colors.RESET}")
                # Show message preview if available
                if node_state.get("messages"):
                    last_msg = node_state["messages"][-1]
                    if hasattr(last_msg, "content"):
                        content = str(last_msg.content)
                        if len(content) > 0:
                            preview = content[:100] + "..." if len(content) > 100 else content
                            print(f"    {Colors.DIM}{preview}{Colors.RESET}")

            # Save state with messages
            if node_state.get("messages"):
                final_state = {
                    k: v.copy() if isinstance(v, list) else v for k, v in node_state.items()
                }
                if "messages" in final_state:
                    final_state["messages"] = list(node_state["messages"])

        # Use final state or fallback to initial
        if not final_state:
            final_state = initial_state.copy()

        result = final_state

        print_section("Execution Results")
        if is_ace_enabled:
            print(f"{Colors.SUCCESS}✅ Graph execution completed (ACE enabled)!{Colors.RESET}")
        else:
            print(f"{Colors.SUCCESS}✅ Graph execution completed (ACE disabled)!{Colors.RESET}")
        print(f"\n{Colors.INFO}📊 Total steps: {Colors.BOLD}{total_steps}{Colors.RESET}")
        print(
            f"{Colors.INFO}📊 Final messages: {Colors.BOLD}{len(result.get('messages', []))}{Colors.RESET}"
        )

        # Show agent flow
        if agent_flow:
            flow_colored = " → ".join([format_node_name(node) for node in agent_flow[-15:]])
            print(f"\n{Colors.INFO}🔄 Agent flow: {flow_colored}{Colors.RESET}")

        # Check if playbooks were created (only if ACE is enabled)
        if is_ace_enabled:
            playbook_dir = Path("bioagents/playbooks")
            if playbook_dir.exists():
                playbooks = list(playbook_dir.glob("*_playbook.yaml"))
                if playbooks:
                    print(
                        f"\n{Colors.PLAYBOOK}📚 Playbooks created: {Colors.BOLD}{len(playbooks)}{Colors.RESET}"
                    )
                    for pb in playbooks:
                        print(f"   {Colors.PLAYBOOK}• {pb.name}{Colors.RESET}")
                else:
                    print(
                        f"\n{Colors.WARNING}⚠️  No playbooks created yet (curator may not have run){Colors.RESET}"
                    )
            else:
                print(f"\n{Colors.WARNING}⚠️  Playbook directory not found{Colors.RESET}")

        print(f"\n{Colors.SUCCESS}✨ Test completed successfully!{Colors.RESET}")

    except Exception as e:
        print(f"\n{Colors.ERROR}❌ Error during execution: {e}{Colors.RESET}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Load .env file from BioAgents root directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    # Test based on .env configuration
    test_ace_integration()

    print_banner("🎉 Test completed!")
