"""Unit tests for rdkit-agent wrapper and LangChain tools."""

import json
import pytest

from bioagents.tools.rdkit_tools import (
    add_atom_maps,
    analyze_ring_systems,
    analyze_stereochemistry,
    compute_molecular_descriptors,
    compute_molecular_fingerprint,
    convert_chemical_notation,
    detect_functional_groups_tool,
    draw_molecule_svg,
    edit_molecule_structure,
    filter_by_descriptor_constraints,
    remove_atom_maps,
    repair_invalid_smiles,
    search_similar_molecules_tool,
    search_substructure,
    validate_smiles,
    validate_smirks,
)


class TestValidationTools:
    """Test validation tools."""
    
    def test_validate_smiles_valid(self):
        """Test validating a valid SMILES."""
        result = validate_smiles.invoke({"smiles": "CCO"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["data"]["overall_pass"] is True
    
    def test_validate_smiles_invalid(self):
        """Test validating an invalid SMILES."""
        result = validate_smiles.invoke({"smiles": "InvalidSMILES"})
        data = json.loads(result)
        assert data["success"] is True
        # Should have validation result (may be invalid)
        assert "data" in data
    
    def test_validate_smirks(self):
        """Test validating a SMIRKS string."""
        result = validate_smirks.invoke({"smirks": "[C:1][OH]>>[C:1]Br"})
        data = json.loads(result)
        assert data["success"] is True


class TestRepairTools:
    """Test repair tools."""
    
    def test_repair_smiles(self):
        """Test repairing invalid SMILES."""
        result = repair_invalid_smiles.invoke({"input_str": "C1CC"})
        data = json.loads(result)
        assert data["success"] is True
        # Should have repair strategy
        assert "data" in data


class TestDescriptorTools:
    """Test descriptor computation tools."""
    
    def test_compute_descriptors_single(self):
        """Test computing descriptors for single molecule."""
        result = compute_molecular_descriptors.invoke({"smiles_list": "CCO"})
        data = json.loads(result)
        assert data["success"] is True
        assert "molecules" in data["data"]
        assert len(data["data"]["molecules"]) == 1
        mol = data["data"]["molecules"][0]
        assert "MW" in mol
        assert "logP" in mol
    
    def test_compute_descriptors_multiple(self):
        """Test computing descriptors for multiple molecules."""
        result = compute_molecular_descriptors.invoke({"smiles_list": "CCO,c1ccccc1,C1CCCCC1"})
        data = json.loads(result)
        assert data["success"] is True
        assert len(data["data"]["molecules"]) == 3
    
    def test_compute_descriptors_with_fields(self):
        """Test computing specific descriptor fields."""
        result = compute_molecular_descriptors.invoke({"smiles_list": "CCO", "fields": "MW,logP"})
        data = json.loads(result)
        assert data["success"] is True


class TestSimilarityAndFilterTools:
    """Test similarity and filtering tools."""
    
    def test_similarity_search(self):
        """Test similarity search."""
        result = search_similar_molecules_tool.invoke({
            "query": "c1ccccc1",
            "targets": "Cc1ccccc1,CCO,c1ccc2ccccc2c1",
            "threshold": 0.5,
        })
        data = json.loads(result)
        assert data["success"] is True
        # rdkit-agent returns all_results, not hits
        assert "all_results" in data["data"] or "results" in data["data"]
    
    def test_filter_lipinski(self):
        """Test filtering by Lipinski's Rule of Five."""
        result = filter_by_descriptor_constraints.invoke({
            "smiles_list": "CCO,CC(=O)Oc1ccccc1C(=O)O",
            "apply_lipinski": True,
        })
        data = json.loads(result)
        assert data["success"] is True
        assert "filtered_smiles" in data["data"]


class TestFunctionalGroupTools:
    """Test functional group and substructure tools."""
    
    def test_detect_functional_groups(self):
        """Test functional group detection."""
        result = detect_functional_groups_tool.invoke({"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
        data = json.loads(result)
        assert data["success"] is True
        assert "functional_groups" in data["data"]
    
    def test_substructure_search(self):
        """Test substructure search."""
        result = search_substructure.invoke({
            "smiles": "c1ccccc1CC(=O)O",
            "smarts_pattern": "[cR1][cR1]",
        })
        data = json.loads(result)
        assert data["success"] is True


class TestRingAndStereoTools:
    """Test ring and stereochemistry tools."""
    
    def test_analyze_rings(self):
        """Test ring analysis."""
        result = analyze_ring_systems.invoke({"smiles": "c1ccc2c(c1)cccn2"})
        data = json.loads(result)
        assert data["success"] is True
        assert "ring_count" in data["data"]
    
    def test_analyze_stereo(self):
        """Test stereochemistry analysis."""
        result = analyze_stereochemistry.invoke({"smiles": "CC(O)C(N)C"})
        data = json.loads(result)
        assert data["success"] is True
        assert "stereo_center_count" in data["data"]


class TestAtomMappingTools:
    """Test atom mapping tools."""
    
    def test_add_atom_maps(self):
        """Test adding atom maps."""
        result = add_atom_maps.invoke({"smiles": "CCO"})
        data = json.loads(result)
        assert data["success"] is True
        assert "mapped_smiles" in data["data"]
    
    def test_remove_atom_maps(self):
        """Test removing atom maps."""
        result = remove_atom_maps.invoke({"smiles": "[CH3:1][CH2:2][OH:3]"})
        data = json.loads(result)
        assert data["success"] is True


class TestConversionTools:
    """Test conversion tools."""
    
    def test_convert_smiles_to_inchi(self):
        """Test converting SMILES to InChI."""
        result = convert_chemical_notation.invoke({
            "input_str": "CCO",
            "from_format": "smiles",
            "to_format": "inchi"
        })
        data = json.loads(result)
        assert data["success"] is True


class TestEditingTools:
    """Test molecule editing tools."""
    
    def test_neutralize_charges(self):
        """Test neutralizing charges."""
        result = edit_molecule_structure.invoke({
            "smiles": "[NH4+].[OH-]",
            "operation": "neutralize"
        })
        data = json.loads(result)
        assert data["success"] is True


class TestFingerprintTools:
    """Test fingerprinting tools."""
    
    def test_compute_morgan_fingerprint(self):
        """Test computing Morgan fingerprint."""
        result = compute_molecular_fingerprint.invoke({
            "smiles": "CCO",
            "fp_type": "Morgan"
        })
        data = json.loads(result)
        assert data["success"] is True


class TestVisualizationTools:
    """Test visualization tools."""
    
    def test_draw_molecule(self):
        """Test drawing molecule to SVG."""
        result = draw_molecule_svg.invoke({"smiles": "c1ccccc1"})
        data = json.loads(result)
        assert data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
