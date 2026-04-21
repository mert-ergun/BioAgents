"""RdkitValidator Agent - Pre-flight chemistry validation in suggestive mode.

Validates chemical inputs (SMILES, SMIRKS) and flags warnings without blocking.
This agent can be inserted into workflows to catch chemistry issues early.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_openai import ChatOpenAI

from bioagents.llms.llm_provider import get_llm
from bioagents.tools.rdkit_tools import (
    repair_invalid_smiles,
    validate_smiles,
    validate_smirks,
    compute_molecular_descriptors,
    analyze_ring_systems,
    detect_functional_groups_tool,
    convert_chemical_notation,
)
from bioagents.tools.rdkit_wrapper import convert_notation

logger = logging.getLogger(__name__)


def _get_common_molecule_smiles(molecule_name: str) -> str | None:
    """Get SMILES for common molecules by name."""
    # Common molecules mapping (lowercase name -> SMILES)
    common_molecules = {
        'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'ethanol': 'CCO',
        'benzene': 'c1ccccc1',
        'toluene': 'Cc1ccccc1',
        'acetone': 'CC(=O)C',
        'glucose': 'C([C@@H]1[C@H]([C@@H]([C@H](O1)O)O)O)O',
        'methane': 'C',
        'ethane': 'CC',
        'propane': 'CCC',
        'butane': 'CCCC',
        'isobutane': 'CC(C)C',
        'indole': 'c1ccc2c(c1)cccn2',
    }
    
    # Try to find the molecule by name
    name_lower = molecule_name.lower().strip()
    return common_molecules.get(name_lower)



    """Convert SMILES to InChI/InChIKey if user requests format conversion."""
    conversion_keywords = ['convert', 'representation', 'format', 'inchi', 'inchikey', 'show multiple']
    
    # Check if user is asking for format conversions
    request_lower = request_text.lower()
    if not any(keyword in request_lower for keyword in conversion_keywords):
        return None
    
    # Try to convert to InChI and InChIKey
    conversions = {}
    
    try:
        # Convert to InChI
        inchi_result = convert_chemical_notation.invoke({
            "input_str": smiles,
            "from_format": "smiles",
            "to_format": "inchi"
        })
        inchi_dict = json.loads(inchi_result) if isinstance(inchi_result, str) else inchi_result
        if inchi_dict.get("success") and inchi_dict.get("data", {}).get("success"):
            conversions["InChI"] = inchi_dict.get("data", {}).get("output", "N/A")
    except Exception as e:
        logger.debug(f"Failed to convert SMILES to InChI for {smiles}: {e}")
    
    try:
        # Convert to InChIKey
        inchikey_result = convert_chemical_notation.invoke({
            "input_str": smiles,
            "from_format": "smiles",
            "to_format": "inchikey"
        })
        inchikey_dict = json.loads(inchikey_result) if isinstance(inchikey_result, str) else inchikey_result
        if inchikey_dict.get("success") and inchikey_dict.get("data", {}).get("success"):
            conversions["InChIKey"] = inchikey_dict.get("data", {}).get("output", "N/A")
    except Exception as e:
        logger.debug(f"Failed to convert SMILES to InChIKey for {smiles}: {e}")
    
    if conversions:
        output = f"\n  Format Conversions:\n"
        output += f"    SMILES: {smiles}\n"
        for fmt, value in conversions.items():
            output += f"    {fmt}: {value}\n"
        return output
    
    return None





def _extract_smiles_patterns(text: str) -> list[str]:
    """
    Extract SMILES/SMIRKS patterns from text.
    Uses intelligent pattern matching to avoid false positives like 'caffeine'.
    """
    import re
    
    # Common words that contain chemical notation but aren't SMILES
    # (lower case for comparison)
    common_words = {
        'caffeine', 'methane', 'ethane', 'propane', 'benzene',
        'the', 'cafe', 'sulfur', 'oxygen', 'nitrogen',
        'phosphorus', 'chlorine', 'bromine', 'iodine', 'fluorine',
        'coffee', 'science', 'council', 'office', 'space',
    }
    
    smiles_list = []
    seen = set()  # Track already extracted SMILES to avoid duplicates
    
    # Pattern 1: Text in parentheses like (CCO) or (c1ccccc1)
    # These are almost always SMILES in chemistry contexts
    paren_matches = re.findall(r'\(([A-Za-z0-9\[\]\(\)=\-#\\/\@\+]+)\)', text)
    for match in paren_matches:
        if match.lower() not in common_words and len(match) > 1:
            if match not in seen:
                smiles_list.append(match)
                seen.add(match)
    
    # Pattern 2: Words with explicit SMILES markers (digits for rings, branches, etc.)
    # Examples: CC(C)C, c1ccccc1, [NH+], CCc1ccccc1
    words = text.split()
    for word in words:
        # Clean the word (remove outer punctuation only, keep structure)
        cleaned = word.strip(".,;:!?\"'`").strip()
        
        # Skip if it's a common word or very short
        if cleaned.lower() in common_words or len(cleaned) < 2:
            continue
        
        # Skip if it's just a parenthesis-wrapped version of an already extracted SMILES
        if cleaned in seen or cleaned.strip('()') in seen:
            continue
        
        # Check if word has SMILES-specific patterns
        has_digit = any(c.isdigit() for c in cleaned)  # Ring numbers 1-9
        has_bracket = '[' in cleaned or ']' in cleaned  # Atom specifications
        has_bond = any(c in cleaned for c in ['=', '#', '\\', '/', '@'])  # Bond types
        has_paren = '(' in cleaned or ')' in cleaned  # Branches
        
        # A real SMILES typically has at least one of these markers
        # OR is a simple organic compound (just element symbols and parens/digits)
        if has_digit or has_bracket or has_bond or has_paren:
            if cleaned not in seen:
                smiles_list.append(cleaned)
                seen.add(cleaned)
        # Also check for simple element sequences (CC, CCO, etc.)
        # but only if they don't look like regular words
        elif _looks_like_simple_smiles(cleaned):
            if cleaned not in seen:
                smiles_list.append(cleaned)
                seen.add(cleaned)
    
    return smiles_list


def _looks_like_simple_smiles(word: str) -> bool:
    """Check if a word looks like a simple SMILES (element symbols only)."""
    import re
    
    # Valid SMILES element patterns (C, N, O, P, S, F, Cl, Br, I, etc.)
    # But avoid common words
    if word.lower() in {'the', 'cafe', 'office', 'science', 'council', 'space', 'coffee'}:
        return False
    
    # Pattern: one or more element symbols (not just any letters)
    # Valid: C, c, N, n, O, S, P, F, Cl, Br, I, S, etc.
    # Examples: CC, CCO, c1ccccc1 (without the digit, just checking element pattern)
    
    # Remove brackets and parentheses for this check
    test_word = re.sub(r'[\[\]()]', '', word)
    
    # Must start with a capital letter or lowercase letter that's an element
    if not test_word or not test_word[0].isalpha():
        return False
    
    # Simple heuristic: if it's all letters (no digits/bonds yet) but looks like element symbols
    # Check that it only contains valid SMILES element characters
    # But be strict - a string like "caffeine" has too many non-element-like sequences
    
    # For simple SMILES like "CCO", check that it's only 1-2 character element symbols
    # or proper aromatic notation (lowercase c, n, o)
    valid_chars = set('CNOPSFclnopsiBr')  # Valid element chars in SMILES
    
    # If ALL characters are valid SMILES chars AND not more than 8 chars AND
    # contains at least one digit OR multiple Cs OR aromatic markers
    if all(c in valid_chars or c.isdigit() for c in test_word):
        # Additional heuristic: simple SMILES are usually short and don't have
        # many vowels (except 'o' for oxygen)
        num_vowels = sum(1 for c in test_word.lower() if c in 'aeiu')
        if num_vowels <= 1 and len(test_word) <= 8:
            return True
        # Or has digit/aromatic markers
        if any(c.isdigit() for c in test_word) or any(c in 'cnops' for c in test_word):
            return True
    
    return False


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
            text = ""  # Keep track of user request for format conversion detection
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    raw = msg.content
                    text = raw if isinstance(raw, str) else str(raw)
                    # Parse SMILES from message using pattern matching
                    smiles_list = _extract_smiles_patterns(text)
                    
                    # Also try to find molecule names and look up their SMILES
                    for mol_name in ['caffeine', 'aspirin', 'ethanol', 'benzene', 'indole', 'toluene']:
                        if mol_name.lower() in text.lower() and mol_name not in [s.lower() for s in smiles_list]:
                            mol_smiles = _get_common_molecule_smiles(mol_name)
                            if mol_smiles and mol_smiles not in smiles_list:
                                smiles_list.append(mol_smiles)
                    
                    break

            validation_results = _validate_chemistry_inputs(
                {"smiles": smiles_list, "smirks": [], "request_type": "validate"}
            )

            # Format output message with clear structure for supervisor
            output_text = "Chemistry Validation & Molecule Identification Report:\n"
            output_text += f"Total inputs: {len(smiles_list)}\n"
            output_text += f"Valid: {len(validation_results['validated_inputs'])}\n"
            output_text += f"Issues: {validation_results['issue_count']}\n"
            output_text += f"Recommendation: {validation_results['recommendation']}\n\n"

            if validation_results["issues"]:
                output_text += "Issues found:\n"
                for issue in validation_results["issues"]:
                    output_text += (
                        f"  - {issue['input']}: {issue['severity']} - {issue['reason']}\n"
                    )
                    if issue["repair_suggestion"]:
                        output_text += f"    Suggestion: {issue['repair_suggestion']}\n"

            # Analyze and identify validated molecules
            if validation_results['validated_inputs']:
                output_text += "\n" + "="*50 + "\n"
                output_text += "MOLECULE IDENTIFICATION\n"
                output_text += "="*50 + "\n"
                for smiles in validation_results['validated_inputs']:
                    output_text += _analyze_molecule(smiles, llm, text)

            # Store in memory
            memory = state.get("memory", {})
            memory["rdkit_validator"] = {
                "status": "success",
                "data": validation_results,
                "output": output_text,
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
                "output": f"Validation failed: {e!s}",
            }
            return {
                "messages": [AIMessage(content=f"Validation error: {e!s}")],
                "memory": memory,
            }

    return rdkit_validator_node


def _analyze_molecule(smiles: str, llm: ChatOpenAI, request_text: str = "") -> str:
    """Analyze a validated SMILES and identify the molecule."""
    analysis = f"\n  SMILES: {smiles}\n"
    
    try:
        # Get molecular descriptors
        desc_result = compute_molecular_descriptors.invoke({"smiles_list": smiles})
        desc_dict = json.loads(desc_result)
        if desc_dict.get("success") and desc_dict.get("data", {}).get("molecules"):
            mol = desc_dict["data"]["molecules"][0]
            mw = mol.get("MW", "N/A")
            analysis += f"  Molecular Weight: {mw} g/mol\n"
    except Exception as e:
        logger.debug(f"Failed to get descriptors for {smiles}: {e}")
    
    try:
        # Analyze rings and get aromaticity
        rings_result = analyze_ring_systems.invoke({"smiles": smiles})
        rings_dict = json.loads(rings_result)
        if rings_dict.get("success"):
            ring_count = rings_dict.get("data", {}).get("ring_count", 0)
            analysis += f"  Rings: {ring_count}\n"
            
            # Get detailed aromaticity information
            aromaticity_info = _get_ring_aromaticity(smiles)
            if aromaticity_info:
                analysis += f"  Ring Aromaticity: {aromaticity_info}\n"
    except Exception as e:
        logger.debug(f"Failed to analyze rings for {smiles}: {e}")
    
    try:
        # Detect functional groups
        fg_result = detect_functional_groups_tool.invoke({"smiles": smiles})
        fg_dict = json.loads(fg_result)
        if fg_dict.get("success"):
            fgs = fg_dict.get("data", {}).get("functional_groups", [])
            if fgs:
                analysis += f"  Functional groups: {', '.join(fgs)}\n"
    except Exception as e:
        logger.debug(f"Failed to detect functional groups for {smiles}: {e}")
    
    # Check if user requested format conversions
    if request_text:
        format_conversions = _convert_smiles_to_formats(smiles, request_text)
        if format_conversions:
            analysis += format_conversions
    
    # Use RDKit direct access to identify the molecule
    mol_name = _identify_molecule_from_smiles(smiles)
    if mol_name:
        analysis += f"  Molecule: {mol_name}\n"
    
    return analysis


def _convert_smiles_to_formats(smiles: str, request_text: str) -> str:
    """Convert SMILES to other chemical formats if requested."""
    output = ""
    
    # Check if user requested format conversions
    format_keywords = {
        "inchi": ("SMILES", "InChI"),
        "inchikey": ("SMILES", "InChIKey"),
        "representation": ("SMILES", "InChI"),
        "convert": ("SMILES", "InChI"),
        "formats": ("SMILES", "InChI"),
    }
    
    should_convert = any(keyword in request_text.lower() for keyword in format_keywords.keys())
    
    if should_convert:
        output += "  Chemical Format Conversions:\n"
        try:
            # Convert to InChI
            inchi_result = convert_notation(smiles, "smiles", "inchi")
            if inchi_result.get("success"):
                inchi = inchi_result.get("output", "N/A")
                output += f"    InChI: {inchi}\n"
            
            # Convert to InChIKey
            inchikey_result = convert_notation(smiles, "smiles", "inchikey")
            if inchikey_result.get("success"):
                inchikey = inchikey_result.get("output", "N/A")
                output += f"    InChIKey: {inchikey}\n"
        except Exception as e:
            logger.debug(f"Error converting formats for {smiles}: {e}")
            output += f"    Format conversion failed: {e}\n"
    
    return output


def _get_ring_aromaticity(smiles: str) -> str | None:
    """Analyze aromaticity of each ring in a molecule."""
    try:
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        ring_info = mol.GetRingInfo()
        if not ring_info.AtomRings():
            return "No rings found"
        
        aromaticity_desc = []
        for ring_idx, ring_atoms in enumerate(ring_info.AtomRings(), 1):
            # Check if all atoms in ring are aromatic
            is_aromatic = all(
                mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
                for atom_idx in ring_atoms
            )
            ring_size = len(ring_atoms)
            aromatic_status = "aromatic" if is_aromatic else "non-aromatic"
            aromaticity_desc.append(f"Ring {ring_idx} ({ring_size} atoms): {aromatic_status}")
        
        return "; ".join(aromaticity_desc) if aromaticity_desc else None
    except Exception as e:
        logger.debug(f"Failed to analyze aromaticity for {smiles}: {e}")
        return None


def _identify_molecule_from_smiles(smiles: str) -> str | None:
    """Identify molecule from SMILES using RDKit and common name mapping."""
    # Common molecule mapping - canonical vs common names
    common_molecules = {
        "C1CCCCC1": "cyclohexane (6-membered alicyclic ring)",
        "c1ccccc1": "benzene (aromatic ring)",
        "CC": "ethane",
        "CCO": "ethanol",
        "CC(C)C": "isobutane (2-methylpropane)",
        "CCc1ccccc1": "ethylbenzene",
        "CC(=O)O": "acetic acid",
        "c1ccc(O)cc1": "phenol (hydroxybenzene)",
        "CC(C)(C)c1ccc(O)cc1": "4-tert-butylphenol",
    }
    
    # Try direct mapping first
    if smiles in common_molecules:
        return common_molecules[smiles]
    
    # Try RDKit to extract formula as fallback
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            num_atoms = mol.GetNumAtoms()
            
            # Identify atom types
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            unique_atoms = set(atom_symbols)
            
            # Classify by atom composition
            if unique_atoms == {'C', 'H'} or unique_atoms == {'C'}:
                ring_count = rdMolDescriptors.CalcNumRings(mol)
                if ring_count > 0:
                    return f"Cyclic hydrocarbon - {formula} ({ring_count} ring{'s' if ring_count > 1 else ''})"
                else:
                    return f"Alkane - {formula}"
            elif 'O' in unique_atoms and 'C' in unique_atoms:
                return f"Organic compound with oxygen - {formula}"
            elif 'N' in unique_atoms and 'C' in unique_atoms:
                return f"Organic compound with nitrogen - {formula}"
            else:
                return f"Organic compound - {formula}"
    except Exception as e:
        logger.debug(f"RDKit identification failed for {smiles}: {e}")
    
    return None


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
                repairs_applied = data.get("repairs_applied", [])

                if is_valid:
                    # Check if it was auto-repaired (ring closures)
                    if repairs_applied and any("close_ring" in r for r in repairs_applied):
                        # SMILES was repaired (had unclosed rings that were closed)
                        corrected = data.get("canonical_smiles", smiles)
                        validated_inputs.append(corrected)
                        issues.append(
                            {
                                "input": smiles,
                                "type": "SMILES",
                                "severity": "warning",
                                "reason": f"Unclosed ring markers - auto-repaired to {corrected}",
                                "is_valid": True,
                                "repair_suggestion": corrected,
                                "repairs_applied": repairs_applied,
                            }
                        )
                    else:
                        # Valid SMILES, no repairs needed
                        validated_inputs.append(smiles)
                        issues.append(
                            {
                                "input": smiles,
                                "type": "SMILES",
                                "severity": "info",
                                "reason": "Valid SMILES",
                                "is_valid": True,
                                "repair_suggestion": None,
                            }
                        )
                else:
                    # Try to get repair suggestion
                    repair_suggestion = None
                    if request_type == "validate_and_repair":
                        try:
                            repair_result = repair_invalid_smiles.invoke({"input_str": smiles})
                            repair_dict = json.loads(repair_result)
                            if repair_dict.get("success"):
                                repair_suggestion = repair_dict.get("data", {}).get(
                                    "canonical_smiles"
                                )
                        except Exception as exc:
                            logger.debug("Repair suggestion failed for %s: %s", smiles, exc)

                    issues.append(
                        {
                            "input": smiles,
                            "type": "SMILES",
                            "severity": "warning",
                            "reason": data.get("summary", "Invalid SMILES"),
                            "is_valid": False,
                            "repair_suggestion": repair_suggestion,
                        }
                    )
            else:
                issues.append(
                    {
                        "input": smiles,
                        "type": "SMILES",
                        "severity": "error",
                        "reason": result_dict.get("error", {}).get("message", "Validation error"),
                        "is_valid": False,
                        "repair_suggestion": None,
                    }
                )
        except Exception as e:
            logger.error(f"Error validating SMILES '{smiles}': {e}")
            issues.append(
                {
                    "input": smiles,
                    "type": "SMILES",
                    "severity": "error",
                    "reason": str(e),
                    "is_valid": False,
                    "repair_suggestion": None,
                }
            )

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
                    issues.append(
                        {
                            "input": smirks,
                            "type": "SMIRKS",
                            "severity": "info",
                            "reason": "Valid SMIRKS",
                            "is_valid": True,
                            "repair_suggestion": None,
                        }
                    )
                else:
                    issues.append(
                        {
                            "input": smirks,
                            "type": "SMIRKS",
                            "severity": "warning",
                            "reason": data.get("summary", "Invalid SMIRKS"),
                            "is_valid": False,
                            "repair_suggestion": None,
                        }
                    )
            else:
                issues.append(
                    {
                        "input": smirks,
                        "type": "SMIRKS",
                        "severity": "error",
                        "reason": result_dict.get("error", {}).get("message", "Validation error"),
                        "is_valid": False,
                        "repair_suggestion": None,
                    }
                )
        except Exception as e:
            logger.error(f"Error validating SMIRKS '{smirks}': {e}")
            issues.append(
                {
                    "input": smirks,
                    "type": "SMIRKS",
                    "severity": "error",
                    "reason": str(e),
                    "is_valid": False,
                    "repair_suggestion": None,
                }
            )

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

_validator_instance: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None


def create_simple_validator() -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Build the batch validation API used by :func:`get_rdkit_validator`."""
    return {"validate": _validate_chemistry_inputs}


def get_rdkit_validator() -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
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
