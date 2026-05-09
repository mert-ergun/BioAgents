# test_api_interception.py
import json
import os
from langchain_core.messages import HumanMessage
from bioagents.graph import create_graph

def test_api_interception():
    print("\n=== TEST STARTING ===")
    
    # 1. Ensure the API key is NOT in the environment (for test validity)
    if "TAMARIND_API_KEY" in os.environ:
        del os.environ["TAMARIND_API_KEY"]
    
    # 2. Build the graph (loads tools and agents as defined in graph.py)
    print("Building the multi-agent graph...")
    graph = create_graph()

    # 3. Trigger the agent to explicitly run AlphaFold 2
    initial_state = {
        "messages": [
            HumanMessage(content="Please use AlphaFold 2 to predict the structure for the sequence 'ACEGHIK'. Use Tamarind Bio as the provider.")
        ]
    }

    print("Starting workflow. The Supervisor should assign the task to the relevant agent...")
    
    # 4. Start the stream
    # Normally this flows to the end. With our intervention, it should 
    # interrupt() at the 'user_input' node.
    try:
        # stream() returns a generator. We iterate through each step.
        for step in graph.stream(initial_state, {"recursion_limit": 50}):
            # Print the executed node for visual tracking
            for node_name, node_state in step.items():
                print(f"-> Node Executed: {node_name}")
                
                # If the node has messages, check the last message
                if "messages" in node_state and node_state["messages"]:
                    last_msg = node_state["messages"][-1]
                    content = getattr(last_msg, "content", "")
                    
                    if "[ENGAGEMENT_PENDING]" in str(content):
                        print("\n" + "="*60)
                        print("SUCCESS: System successfully paused for API Key!")
                        print(f"LLM Output:\n{content}")
                        print("="*60 + "\n")
                        
                        # To take the test further: When the system pauses,
                        # we could resume it by feeding a fake key.
                        print("Phase 2 of Test: System is ready to accept a fake key and resume.")
                        print("Terminating the test here as the interception is confirmed successful.")
                        
                        return 
                        
    except Exception as e:
        print(f"\nERROR (Unexpected): {e}")

if __name__ == "__main__":
    test_api_interception()