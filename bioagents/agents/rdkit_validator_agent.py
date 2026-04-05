"""RdkitValidator Agent - Pre-flight chemistry validation in suggestive mode.

Validates chemical inputs (SMILES, SMIRKS) and flags warnings without blocking.
This agent can be inserted into workflows to catch chemistry issues early.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from bioagents.llms.llm_provider import get_llm
from bioagents.tools.rdkit_tools import (
    get_rdkit_validation_tools,
    validate_smiles,
    validate_smirks,
    repair_invalid_smiles,
)

logger = logging.getLogger(__name__)


def create_rdkit_validator_agent(llm: ChatOpenAI | None = None):
    """
    Create a chemistry validation agent using rdkit-agent.
    
    In SUGGESTIVE MODE: Flags issues but allows proceeding.
    Users/agents can choose whether to apply repair suggestions.
    
    Args:
        llm: Optional language model instance. If None, uses default from llm_provider
    
    Returns:
        Agent node function for the workflow
    """
    if llm is None:
        llm = get_llm()
    
    def rdkit_validator_node(state: dict) -> dict:
        """
        RDKit Validator agent node.
        Validates molecules from user messages or state.
        """
        try:
            messages = state.get("messages", [])
            
            # Extract SMILES from the most recent user message
            smiles_list = []
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    content = msg.content
                    # Parse SMILES from message (simple extraction)
                    # Looking for patterns like: CCO, c1ccccc1, etc.
                    words = content.split()
                    for word in words:
                        # Check if word looks like SMILES (contains chemical notation)
                        if any(char in word for char in ['C', 'c', 'N', 'n', 'O', 'o', 'P', 'p', 'S', 's', 'F', 'Cl', 'Br', 'I']):
                            smiles_list.append(word.strip("(),. "))
                    break
            
            validation_results = _validate_chemistry_inputs({
                "smiles": smiles_list,
                "smirks": [],
                "request_type": "validate"
            })
            
            # Format output message
            output_text = f"Validation Report:\n"
            output_text += f"Total inputs: {len(smiles_list)}\n"
            output_text += f"Valid: {len(validation_results['validated_inputs'])}\n"
            output_text += f"Issues: {validation_results['issue_count']}\n"
            output_text += f"Recommendation: {validation_results['recommendation']}\n\n"
            
            if validation_results['issues']:
                output_text += "Issues found:\n"
                for issue in validation_results['issues']:
                    output_text += f"  - {issue['input']}: {issue['severity']} - {issue['reason']}\n"
                    if issue['repair_suggestion']:
                        output_text += f"    Suggestion: {issue['repair_suggestion']}\n"
            
            # Store in memory
            memory = state.get("memory", {})
            memory["rdkit_validator"] = {
                "status": "success",
                "data": validation_results,
                "output": output_text
            }
            
            return {
                "messages": [AIMessage(content=output_text)],
                "memory": memory,
            }
            
        except Exception as e:
            logger.error(f"RdkitValidator error: {e}", exc_info=True)
            memory = state.get("memory", {})
            memory["rdkit_validator"] = {
                "status": "error",
                "error": str(e),
                "output": f"Validation failed: {str(e)}"
            }
            return {
                "messages": [AIMessage(content=f"Validation error: {str(e)}")],
                "memory": memory,
            }
    
    return rdkit_validator_node


def _validate_chemistry_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Validate chemistry inputs.
    
    Args:
        inputs: Dict with keys:
            - smiles: Single SMILES or list of SMILES
            - smirks: Single SMIRKS or list of SMIRKS
            - request_type: "validate" or "validate_and_repair"
    
    Returns:
        Validation report dict
    """
    smiles_list = inputs.get("smiles", [])
    smirks_list = inputs.get("smirks", [])
    request_type = inputs.get("request_type", "validate")
    
    # Normalize to lists
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list] if smiles_list else []
    if isinstance(smirks_list, str):
        smirks_list = [smirks_list] if smirks_list else []
    
    issues = []
    validated_inputs = []
    
    # Validate SMILES
    for smiles in smiles_list:
        try:
            result = validate_smiles.invoke({"smiles": smiles})
            result_dict = json.loads(result)
            
            if result_dict.get("success"):
                data = result_dict.get("data", {})
                is_valid = data.get("overall_pass", False)
                
                if is_valid:
                    validated_inputs.append(smiles)
                    issues.append({
                        "input": smiles,
                        "type": "SMILES",
                        "severity": "info",
                        "reason": "Valid SMILES",
                        "is_valid": True,
                        "repair_suggestion": None,
                    })
                else:
                    # Try to get repair suggestion
                    repair_suggestion = None
                    if request_type == "validate_and_repair":
                        try:
                            repair_result = repair_invalid_smiles.invoke({"input_str": smiles})
                            repair_dict = json.loads(repair_result)
                            if repair_dict.get("success"):
                                repair_suggestion = repair_dict.get("data", {}).get("canonical_smiles")
                        except Exception:
                            pass
                    
                    issues.append({
                        "input": smiles,
                        "type": "SMILES",
                        "severity": "warning",
                        "reason": data.get("summary", "Invalid SMILES"),
                        "is_valid": False,
                        "repair_suggestion": repair_suggestion,
                    })
            else:
                issues.append({
                    "input": smiles,
                    "type": "SMILES",
                    "severity": "error",
                    "reason": result_dict.get("error", {}).get("message", "Validation error"),
                    "is_valid": False,
                    "repair_suggestion": None,
                })
        except Exception as e:
            logger.error(f"Error validating SMILES '{smiles}': {e}")
            issues.append({
                "input": smiles,
                "type": "SMILES",
                "severity": "error",
                "reason": str(e),
                "is_valid": False,
                "repair_suggestion": None,
            })
    
    # Validate SMIRKS
    for smirks in smirks_list:
        try:
            result = validate_smirks.invoke({"smirks": smirks})
            result_dict = json.loads(result)
            
            if result_dict.get("success"):
                data = result_dict.get("data", {})
                is_valid = data.get("overall_pass", False)
                
                if is_valid:
                    validated_inputs.append(smirks)
                    issues.append({
                        "input": smirks,
                        "type": "SMIRKS",
                        "severity": "info",
                        "reason": "Valid SMIRKS",
                        "is_valid": True,
                        "repair_suggestion": None,
                    })
                else:
                    issues.append({
                        "input": smirks,
                        "type": "SMIRKS",
                        "severity": "warning",
                        "reason": data.get("summary", "Invalid SMIRKS"),
                        "is_valid": False,
                        "repair_suggestion": None,
                    })
            else:
                issues.append({
                    "input": smirks,
                    "type": "SMIRKS",
                    "severity": "error",
                    "reason": result_dict.get("error", {}).get("message", "Validation error"),
                    "is_valid": False,
                    "repair_suggestion": None,
                })
        except Exception as e:
            logger.error(f"Error validating SMIRKS '{smirks}': {e}")
            issues.append({
                "input": smirks,
                "type": "SMIRKS",
                "severity": "error",
                "reason": str(e),
                "is_valid": False,
                "repair_suggestion": None,
            })
    
    # Determine recommendation
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    
    if error_count > 0:
        recommendation = "has_errors"
    elif warning_count > 0:
        recommendation = "review_suggested"
    else:
        recommendation = "safe_to_proceed"
    
    return {
        "validated_inputs": validated_inputs,
        "issues": issues,
        "issue_count": len(issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "summary": f"Validated {len(smiles_list) + len(smirks_list)} inputs: {error_count} errors, {warning_count} warnings",
        "recommendation": recommendation,
        "allow_proceeding": True,  # Always allow in suggestive mode
    }



# ============================================================================
# VALIDATOR FUNCTIONS FOR INTEGRATION
# ============================================================================

_validator_instance = None


def get_rdkit_validator() -> dict[str, Any]:
    """Get or create the global rdkit validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = create_simple_validator()
    return _validator_instance


def validate_smiles_batch(smiles_list: list[str] | str) -> dict[str, Any]:
    """
    Quick validation of SMILES batch.
    
    Args:
        smiles_list: Single SMILES or list of SMILES
    
    Returns:
        Validation report
    """
    validator = get_rdkit_validator()
    return validator["validate"]({"smiles": smiles_list, "request_type": "validate"})


def validate_and_repair_smiles_batch(smiles_list: list[str] | str) -> dict[str, Any]:
    """
    Validate SMILES and suggest repairs.
    
    Args:
        smiles_list: Single SMILES or list of SMILES
    
    Returns:
        Validation report with repair suggestions
    """
    validator = get_rdkit_validator()
    return validator["validate"]({"smiles": smiles_list, "request_type": "validate_and_repair"})
