#!/usr/bin/env python3
"""Test the format conversion capability for caffeine."""

from bioagents.agents.rdkit_validator_agent import create_rdkit_validator_agent
from langchain_core.messages import HumanMessage

print("\n" + "="*80)
print("TESTING CAFFEINE FORMAT CONVERSION")
print("="*80)

# Test: Full agent output with caffeine
print("\n[TEST] Full Rdkit_Validator Agent Output with Caffeine")
print("-" * 80)
agent = create_rdkit_validator_agent()
state = {
    'messages': [
        HumanMessage(
            content='Show me multiple representations of caffeine. Convert between SMILES, InChI, and InChIKey formats.'
        )
    ],
    'memory': {}
}

print("Running agent...")
agent_result = agent(state)
output = agent_result['messages'][0].content

print("\nAgent Output:")
print(output)

# Check for format conversions in output
if "InChI" in output and "InChIKey" in output:
    print("\n✅ SUCCESS: Format conversions are included in agent output!")
elif "InChI" in output or "InChIKey" in output:
    print("\n✓ PARTIAL: Some format conversions found in output")
else:
    print("\n❌ Format conversions not found in output")

print("\n" + "="*80 + "\n")
