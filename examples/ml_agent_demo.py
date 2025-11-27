"""
Example: ML Agent for Bioinformatics

This example demonstrates how to use the ML agent to design, execute, and optimize
machine learning pipelines for bioinformatics tasks.

The ML agent can:
1. Design complete ML pipelines with various algorithms
2. Execute code safely in sandboxed Docker containers
3. Analyze results and suggest optimizations
4. Iterate to improve model performance

Prerequisites:
- Docker must be installed and running
- Required Python packages: see requirements.txt
"""

import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from bioagents.agents.ml_agent import create_ml_agent
from bioagents.llms.langsmith_config import (
    get_langsmith_config,
    print_langsmith_status,
    setup_langsmith_environment,
)
from bioagents.tools.ml_tools import (
    design_ml_pipeline,
    execute_ml_code,
    optimize_ml_pipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def create_sample_data():
    """Create sample bioinformatics dataset for demonstration."""
    import numpy as np
    import pandas as pd

    # Simulate protein classification data
    np.random.seed(42)
    n_samples = 100

    # Features: length, GC content, hydrophobicity, charge
    data = {
        "length": np.random.randint(50, 500, n_samples),
        "gc_content": np.random.uniform(0.3, 0.7, n_samples),
        "hydrophobicity": np.random.uniform(-2, 2, n_samples),
        "charge": np.random.uniform(-5, 5, n_samples),
        "alpha_helix_percent": np.random.uniform(0.1, 0.6, n_samples),
        "beta_sheet_percent": np.random.uniform(0.1, 0.5, n_samples),
    }

    # Create target based on features (membrane protein vs soluble protein)
    # Membrane proteins: higher hydrophobicity, more alpha helices
    membrane_score = (
        (data["hydrophobicity"] > 0) * 2
        + (data["alpha_helix_percent"] > 0.4) * 2
        + np.random.randn(n_samples) * 0.5
    )
    data["protein_type"] = (membrane_score > 2).astype(int)

    df = pd.DataFrame(data)

    # Add some noise and class balance
    noise_indices = df.sample(frac=0.1).index
    df.loc[noise_indices, "protein_type"] = 1 - df.loc[noise_indices, "protein_type"]

    logger.info(f"Generated {len(df)} samples")
    logger.info(f"Class distribution: {df['protein_type'].value_counts().to_dict()}")

    return df


def should_continue(state):
    """Determine if we should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"

    return "continue"


def create_ml_workflow():
    """Create the ML workflow graph."""
    tools = [design_ml_pipeline, execute_ml_code, optimize_ml_pipeline]

    ml_agent = create_ml_agent(tools)
    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", ml_agent)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": "__end__",
        },
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


def run_example_1_basic_pipeline():
    """Example 1: Design and execute a basic ML pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 1: Basic ML Pipeline")
    logger.info("=" * 60 + "\n")

    # Get LangSmith config
    langsmith_config = get_langsmith_config()

    # Create sample data
    df = create_sample_data()
    data_csv = df.to_csv(index=False)

    app = create_ml_workflow()

    request = f"""
I have a protein classification dataset with the following features:
- length: protein sequence length
- gc_content: GC content ratio
- hydrophobicity: hydrophobicity score
- charge: net charge
- alpha_helix_percent: percentage of alpha helix structure
- beta_sheet_percent: percentage of beta sheet structure

The target variable is 'protein_type' (0: soluble, 1: membrane).

Please:
1. Design an ML pipeline using Random Forest
2. Execute it with the data provided below
3. Analyze the results

Here's the complete dataset in CSV format:

{data_csv}
"""

    logger.info("Starting ML workflow...")
    logger.info(f"Request size: {len(request)} characters")

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    def run_workflow():
        import sys

        step_count = 0
        logger.info("Starting stream iteration...")
        sys.stdout.flush()

        # LangSmith will automatically trace via environment variables set by setup_langsmith_environment
        for event in app.stream(
            {"messages": [HumanMessage(content=request)]},
            {"recursion_limit": 15},
        ):
            step_count += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"STEP {step_count}")
            logger.info(f"{'=' * 60}")
            sys.stdout.flush()

            for node_name, node_output in event.items():
                logger.info(f"\n--- {node_name.upper()} ---")
                sys.stdout.flush()

                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            if node_name == "tools":
                                logger.info(f"Tool Response:\n{msg.content[:2000]}...")
                            else:
                                logger.info(f"Agent Response:\n{msg.content}")
                            sys.stdout.flush()
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                logger.info(f"Calling Tool: {tool_call['name']}")
                                if "args" in tool_call:
                                    logger.info(
                                        f"  Args preview: {str(tool_call['args'])[:200]}..."
                                    )
                                sys.stdout.flush()

            logger.info(f"Completed step {step_count}, waiting for next event...")
            sys.stdout.flush()

        logger.info("Stream iteration completed")
        sys.stdout.flush()
        return step_count

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_workflow)
            try:
                step_count = future.result(timeout=300)  # 5 minute timeout per step
                logger.info(f"\n✓ Workflow completed in {step_count} steps")
            except FuturesTimeoutError:
                logger.error("\n✗ Workflow timed out after 5 minutes")
                logger.error("This usually means the LLM is not responding")
                raise TimeoutError("Workflow execution timed out") from None
    except KeyboardInterrupt:
        logger.info("\nWorkflow interrupted by user")
        raise
    except Exception as e:
        logger.error(f"\n✗ Error in workflow: {e}")
        import traceback

        traceback.print_exc()
        raise


def run_example_2_optimization():
    """Example 2: Design, execute, and optimize an ML pipeline."""
    import pandas as pd

    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 2: ML Pipeline with Optimization")
    logger.info("=" * 60 + "\n")

    df = create_sample_data()
    df_majority = df[df["protein_type"] == 0]
    df_minority = df[df["protein_type"] == 1].sample(n=30, random_state=42)
    df = pd.concat([df_majority, df_minority], ignore_index=True)

    logger.info(f"Imbalanced class distribution: {df['protein_type'].value_counts().to_dict()}")

    data_csv = df.to_csv(index=False)

    app = create_ml_workflow()

    request = f"""
I have an imbalanced protein classification dataset:
- Features: length, gc_content, hydrophobicity, charge, alpha_helix_percent, beta_sheet_percent
- Target: protein_type (0: soluble, 1: membrane)
- Class distribution is imbalanced (more soluble than membrane proteins)

Please:
1. Design an ML pipeline using XGBoost
2. Execute it and analyze the results
3. If there are issues with performance due to class imbalance, suggest optimizations

Here's the complete dataset in CSV format:

{data_csv}
"""

    logger.info("Starting ML workflow with optimization...")
    logger.info(f"Request size: {len(request)} characters")
    try:
        step_count = 0
        # LangSmith will automatically trace via environment variables set by setup_langsmith_environment
        for event in app.stream(
            {"messages": [HumanMessage(content=request)]},
            {"recursion_limit": 15},
        ):
            step_count += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"STEP {step_count}")
            logger.info(f"{'=' * 60}")

            for node_name, node_output in event.items():
                logger.info(f"\n--- {node_name.upper()} ---")
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            if node_name == "tools":
                                if len(msg.content) > 1000:
                                    logger.info(
                                        f"Tool Response (truncated):\n{msg.content[:500]}...\n...\n{msg.content[-500:]}"
                                    )
                                else:
                                    logger.info(f"Tool Response:\n{msg.content}")
                            else:
                                content_preview = (
                                    msg.content[:300] + "..."
                                    if len(msg.content) > 300
                                    else msg.content
                                )
                                logger.info(f"Agent: {content_preview}")
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                logger.info(f"Calling Tool: {tool_call['name']}")
                                if "args" in tool_call:
                                    args_str = str(tool_call["args"])
                                    logger.info(f"  Args size: {len(args_str)} characters")

        logger.info(f"\nWorkflow completed in {step_count} steps")
    except KeyboardInterrupt:
        logger.info("\nWorkflow interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        import traceback

        traceback.print_exc()
        raise


def run_example_3_algorithm_comparison():
    """Example 3: Compare multiple algorithms."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 3: Algorithm Comparison")
    logger.info("=" * 60 + "\n")

    df = create_sample_data()
    data_csv = df.to_csv(index=False)

    app = create_ml_workflow()

    request = f"""
I want to compare different ML algorithms for protein classification.

Dataset features: length, gc_content, hydrophobicity, charge, alpha_helix_percent, beta_sheet_percent
Target: protein_type

Here's the complete dataset in CSV format:

{data_csv}
"""

    logger.info("Starting algorithm comparison workflow...")
    logger.info(f"Request size: {len(request)} characters")
    try:
        step_count = 0
        # LangSmith will automatically trace via environment variables set by setup_langsmith_environment
        for event in app.stream(
            {"messages": [HumanMessage(content=request)]},
            {"recursion_limit": 15},
        ):
            step_count += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"STEP {step_count}")
            logger.info(f"{'=' * 60}")

            for node_name, node_output in event.items():
                logger.info(f"\n--- {node_name.upper()} ---")
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            if node_name == "tools":
                                if len(msg.content) > 1000:
                                    logger.info(
                                        f"Tool Response (truncated):\n{msg.content[:500]}...\n...\n{msg.content[-500:]}"
                                    )
                                else:
                                    logger.info(f"Tool Response:\n{msg.content}")
                            else:
                                content_preview = (
                                    msg.content[:300] + "..."
                                    if len(msg.content) > 300
                                    else msg.content
                                )
                                logger.info(f"Agent: {content_preview}")
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                logger.info(f"Calling Tool: {tool_call['name']}")
                                if "args" in tool_call:
                                    args_str = str(tool_call["args"])
                                    logger.info(f"  Args size: {len(args_str)} characters")

        logger.info(f"\nWorkflow completed in {step_count} steps")
    except KeyboardInterrupt:
        logger.info("\nWorkflow interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run the basic ML agent example."""
    import sys

    try:
        import docker

        client = docker.from_env()
        client.ping()
        logger.info("✓ Docker is available")
    except Exception as e:
        logger.error(f"✗ Docker is not available: {e}")
        logger.error("Please install and start Docker to run this example")
        logger.error("\nTo install Docker:")
        logger.error("  - Linux: sudo apt install docker.io")
        logger.error("  - Mac/Windows: Install Docker Desktop")
        return
    
    # Set up LangSmith monitoring if enabled
    try:
        setup_langsmith_environment()
        print_langsmith_status()
    except ValueError as e:
        logger.warning(f"LangSmith setup warning: {e}")

    example_num = 1
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
        except ValueError:
            logger.error("Invalid example number. Usage: python ml_agent_demo.py [1|2|3]")
            return

    try:
        if example_num == 1:
            logger.info("Running Example 1: Basic ML Pipeline")
            run_example_1_basic_pipeline()
        elif example_num == 2:
            logger.info("Running Example 2: ML Pipeline with Optimization")
            run_example_2_optimization()
        elif example_num == 3:
            logger.info("Running Example 3: Algorithm Comparison")
            run_example_3_algorithm_comparison()
        else:
            logger.error(f"Invalid example number: {example_num}. Choose 1, 2, or 3.")
            return

        logger.info("\n" + "=" * 60)
        logger.info("Example completed successfully!")
        logger.info("=" * 60)
        logger.info("\nTo run other examples:")
        logger.info("  python examples/ml_agent_demo.py 1  # Basic pipeline")
        logger.info("  python examples/ml_agent_demo.py 2  # With optimization")
        logger.info("  python examples/ml_agent_demo.py 3  # Algorithm comparison")

    except KeyboardInterrupt:
        logger.info("\n\nExample interrupted by user")
    except Exception as e:
        logger.error(f"Error running example: {e}")
        raise


if __name__ == "__main__":
    main()
