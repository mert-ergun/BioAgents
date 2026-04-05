"""Integration tests for rdkit-agent with BioAgents system."""

import json
import pytest

from bioagents.agents.rdkit_validator_agent import (
    validate_and_repair_smiles_batch,
    validate_smiles_batch,
)
from bioagents.tools.rdkit_tools import (
    compute_molecular_descriptors,
    filter_by_descriptor_constraints,
    get_all_rdkit_tools,
    validate_smiles,
)


class TestValidatorAgent:
    """Test RdkitValidator agent."""
    
    def test_validator_valid_smiles(self):
        """Test validator with valid SMILES."""
        report = validate_smiles_batch(["CCO", "c1ccccc1"])
        assert report["allow_proceeding"] is True
        assert report["error_count"] == 0
    
    def test_validator_invalid_smiles(self):
        """Test validator with invalid SMILES."""
        report = validate_smiles_batch(["InvalidSMILES"])
        assert report["allow_proceeding"] is True  # Suggestive mode
        assert report["warning_count"] > 0
    
    def test_validator_mixed_input(self):
        """Test validator with mix of valid and invalid."""
        report = validate_smiles_batch(["CCO", "Invalid"])
        assert report["allow_proceeding"] is True
    
    def test_validator_with_repair(self):
        """Test validator suggesting repairs."""
        report = validate_and_repair_smiles_batch(["C1CC"])  # Incomplete ring
        assert report["allow_proceeding"] is True
        issues = [i for i in report["issues"] if i["severity"] != "info"]
        if issues:
            # Should have repair suggestion
            assert issues[0]["repair_suggestion"] is not None or issues[0]["is_valid"] is True


class TestWorkflowIntegration:
    """Test integration with typical workflows."""
    
    def test_protein_design_workflow_with_validation(self):
        """Simulate protein design workflow with chemistry validation."""
        # User provides compounds
        compounds = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
        
        # Step 1: Validate
        validation = validate_smiles_batch(compounds)
        assert validation["allow_proceeding"] is True
        
        # Step 2: Compute descriptors (drug design filtering)
        valid_smiles = ",".join(validation["validated_inputs"])
        descriptors = compute_molecular_descriptors.invoke({"smiles_list": valid_smiles})
        data = json.loads(descriptors)
        assert data["success"] is True
    
    def test_drug_discovery_workflow_lipinski_filter(self):
        """Simulate drug discovery with Lipinski filtering."""
        # Drug-like molecules dataset
        compounds = "CCO,CC(=O)Oc1ccccc1C(=O)O,c1ccccc1,CC(C)Cc1ccc(cc1)C(C)C(O)=O"
        
        # Step 1: Filter by Lipinski's Rule of Five
        filtered = filter_by_descriptor_constraints.invoke({
            "smiles_list": compounds,
            "apply_lipinski": True
        })
        data = json.loads(filtered)
        assert data["success"] is True
        
        # Step 2: Validate remaining compounds
        if data["data"]["filtered_smiles"]:
            valid = validate_smiles_batch(data["data"]["filtered_smiles"])
            assert len(valid["validated_inputs"]) > 0


class TestToolCollection:
    """Test tool collection and retrieval."""
    
    def test_all_rdkit_tools(self):
        """Test that all rdkit tools are accessible."""
        tools = get_all_rdkit_tools()
        assert len(tools) > 0
        # Should have tools from multiple categories
        assert any("validate" in str(t.name) for t in tools)
        assert any("descriptor" in str(t.name).lower() for t in tools)
        assert any("filter" in str(t.name).lower() for t in tools)


class TestBatchProcessing:
    """Test batch processing capability."""
    
    def test_descriptor_batch_processing(self):
        """Test batch descriptor computation for medium-sized dataset."""
        # Medium batch: 50 molecules
        smiles_list = [
            "CCO",
            "c1ccccc1",
            "CC(C)C",
            "CC(=O)O",
            "C1CCCCC1",
        ] * 10  # 50 molecules
        
        smiles_str = ",".join(smiles_list)
        result = compute_molecular_descriptors.invoke({"smiles_list": smiles_str})
        data = json.loads(result)
        
        assert data["success"] is True
        assert len(data["data"]["molecules"]) == 50
    
    def test_filter_batch_processing(self):
        """Test batch filtering."""
        smiles_list = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
            "CC(=O)Oc1ccccc1C(=O)O",
        ] * 10  # 40 molecules
        
        smiles_str = ",".join(smiles_list)
        result = filter_by_descriptor_constraints.invoke({"smiles_list": smiles_str, "mw_max": 300})
        data = json.loads(result)
        
        assert data["success"] is True


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_malformed_smiles_handling(self):
        """Test handling of malformed SMILES."""
        result = validate_smiles.invoke({"smiles": ""}) # Empty SMILES
        data = json.loads(result)
        # Should return error gracefully, not crash
        assert "success" in data
    
    def test_invalid_filter_constraints(self):
        """Test invalid constraint combinations."""
        # Invalid: min > max
        result = filter_by_descriptor_constraints.invoke({
            "smiles_list": "CCO",
            "mw_min": 500,
            "mw_max": 100  # Invalid
        })
        data = json.loads(result)
        # Should handle gracefully
        assert "success" in data or "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
