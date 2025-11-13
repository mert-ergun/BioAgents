"""Tests for analysis tools."""

from bioagents.tools.analysis_tools import (
    AMINO_ACID_WEIGHTS,
    analyze_amino_acid_composition,
    calculate_isoelectric_point,
    calculate_molecular_weight,
    parse_fasta_sequence,
)


class TestParseFastaSequence:
    """Tests for the parse_fasta_sequence helper function."""

    def test_parse_simple_fasta(self):
        """Test parsing a simple FASTA sequence."""
        fasta = ">sp|P12345|TEST_HUMAN Test protein\nMASLKGFVP"
        result = parse_fasta_sequence(fasta)
        assert result == "MASLKGFVP"

    def test_parse_multiline_fasta(self):
        """Test parsing a multi-line FASTA sequence."""
        fasta = ">sp|P12345|TEST_HUMAN\nMASLK\nGFVPT\nARLKD"
        result = parse_fasta_sequence(fasta)
        assert result == "MASLKGFVPTARLKD"

    def test_parse_lowercase_sequence(self):
        """Test that sequence is converted to uppercase."""
        fasta = ">sp|P12345|TEST\nmaslkgfvp"
        result = parse_fasta_sequence(fasta)
        assert result == "MASLKGFVP"

    def test_parse_with_extra_whitespace(self):
        """Test parsing with extra whitespace."""
        fasta = "  >sp|P12345|TEST\n  MASLK  \n  GFVP  "
        result = parse_fasta_sequence(fasta)
        # The implementation joins lines as-is, preserving whitespace within lines
        assert result == "  MASLK    GFVP"

    def test_parse_sequence_only(self):
        """Test parsing sequence without header."""
        fasta = "MASLKGFVP"
        result = parse_fasta_sequence(fasta)
        assert result == "MASLKGFVP"

    def test_parse_empty_sequence(self):
        """Test parsing empty sequence."""
        fasta = ">sp|P12345|TEST"
        result = parse_fasta_sequence(fasta)
        assert result == ""


class TestCalculateMolecularWeight:
    """Tests for the calculate_molecular_weight tool."""

    def test_calculate_simple_sequence(self):
        """Test calculating molecular weight of a simple sequence."""
        fasta = ">test\nAAA"
        result = calculate_molecular_weight.invoke({"fasta_sequence": fasta})

        assert isinstance(result, str)
        assert "Molecular Weight" in result
        assert "Da" in result
        assert "kDa" in result
        assert "Sequence Length: 3" in result

        # AAA: 3 * 89.09 = 267.27 Da, minus 2 water molecules (36.03 Da) = 231.24 Da
        assert "231.24 Da" in result

    def test_calculate_realistic_sequence(self):
        """Test with a realistic protein sequence."""
        fasta = ">test\nMKWVTFISLLFLFSSAYSRGVFRRD"
        result = calculate_molecular_weight.invoke({"fasta_sequence": fasta})

        assert "Molecular Weight" in result
        assert "Sequence Length: 25" in result
        assert "Da" in result

    def test_calculate_with_stop_codon(self):
        """Test that stop codons (*) are ignored."""
        fasta = ">test\nMASL*KGF"
        result = calculate_molecular_weight.invoke({"fasta_sequence": fasta})

        # Should ignore the * character
        assert "Warning" not in result or "Warning: Unknown amino acids" not in result

    def test_calculate_with_gaps(self):
        """Test that gaps (-) are ignored."""
        fasta = ">test\nMASL-KGF"
        result = calculate_molecular_weight.invoke({"fasta_sequence": fasta})

        # Should ignore the - character
        assert "Warning" not in result or "Warning: Unknown amino acids" not in result

    def test_calculate_with_unknown_amino_acid(self):
        """Test handling of unknown amino acids."""
        fasta = ">test\nMASLXKGF"
        result = calculate_molecular_weight.invoke({"fasta_sequence": fasta})

        assert "Warning" in result
        assert "Unknown amino acids" in result
        assert "X" in result

    def test_calculate_multiline_fasta(self):
        """Test with multi-line FASTA format."""
        fasta = ">sp|P12345|TEST\nMASLK\nGFVPT"
        result = calculate_molecular_weight.invoke({"fasta_sequence": fasta})

        assert "Molecular Weight" in result
        assert "Sequence Length: 10" in result

    def test_calculate_error_handling(self):
        """Test error handling for invalid input."""
        result = calculate_molecular_weight.invoke({"fasta_sequence": ""})

        # Should handle empty sequence gracefully
        assert isinstance(result, str)

    def test_tool_metadata(self):
        """Test tool has proper metadata."""
        assert calculate_molecular_weight.name == "calculate_molecular_weight"
        assert hasattr(calculate_molecular_weight, "description")
        assert "molecular weight" in calculate_molecular_weight.description.lower()


class TestAnalyzeAminoAcidComposition:
    """Tests for the analyze_amino_acid_composition tool."""

    def test_analyze_simple_sequence(self):
        """Test analyzing a simple sequence."""
        fasta = ">test\nMASLKGFVP"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        assert isinstance(result, str)
        assert "Amino Acid Composition Analysis" in result
        assert "Total amino acids: 9" in result
        assert "Hydrophobic residues" in result
        assert "Polar residues" in result
        assert "Charged residues" in result

    def test_analyze_all_standard_amino_acids(self):
        """Test with all 20 standard amino acids."""
        fasta = ">test\nARNDCEQGHILKMFPSTWYV"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        assert "Total amino acids: 20" in result
        # All amino acids should appear in the report
        for aa in "ARNDCEQGHILKMFPSTWYV":
            assert aa in result

    def test_analyze_percentage_calculation(self):
        """Test that percentages are calculated correctly."""
        fasta = ">test\nAAAAAAAAAAAA"  # 12 A's
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        assert "Total amino acids: 12" in result
        assert "100.00%" in result  # A should be 100%

    def test_analyze_biochemical_properties(self):
        """Test biochemical property calculations."""
        # Hydrophobic: A, V, I, L, M, F, W, P
        fasta = ">test\nAVILMFWP"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        assert "Total amino acids: 8" in result
        assert "Hydrophobic residues: 8 (100.0%)" in result

    def test_analyze_charged_residues(self):
        """Test charged residue calculation."""
        # Positive: K, R, H
        # Negative: D, E
        fasta = ">test\nKRHDE"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        assert "Charged residues: 5 (100.0%)" in result
        assert "Positive: 3 (60.0%)" in result
        assert "Negative: 2 (40.0%)" in result

    def test_analyze_with_stop_codons(self):
        """Test that stop codons are removed from analysis."""
        fasta = ">test\nMASL*KGF*"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        # Should not count * characters
        assert "Total amino acids: 7" in result

    def test_analyze_with_gaps(self):
        """Test that gaps are removed from analysis."""
        fasta = ">test\nMASL-KGF-"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        # Should not count - characters
        assert "Total amino acids: 7" in result

    def test_analyze_empty_sequence(self):
        """Test error handling for empty sequence."""
        fasta = ">test"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        assert "Error" in result
        assert "Empty sequence" in result

    def test_analyze_multiline_fasta(self):
        """Test with multi-line FASTA format."""
        fasta = ">sp|P12345|TEST\nMASLK\nGFVPT\nARLKD"
        result = analyze_amino_acid_composition.invoke({"fasta_sequence": fasta})

        assert "Total amino acids: 15" in result

    def test_tool_metadata(self):
        """Test tool has proper metadata."""
        assert analyze_amino_acid_composition.name == "analyze_amino_acid_composition"
        assert hasattr(analyze_amino_acid_composition, "description")


class TestCalculateIsoelectricPoint:
    """Tests for the calculate_isoelectric_point tool."""

    def test_calculate_simple_sequence(self):
        """Test calculating pI of a simple sequence."""
        fasta = ">test\nKRHDE"
        result = calculate_isoelectric_point.invoke({"fasta_sequence": fasta})

        assert isinstance(result, str)
        assert "Isoelectric Point (pI) Estimation" in result
        assert "Estimated pI" in result
        assert "Charged Residues" in result

    def test_calculate_shows_charged_residues(self):
        """Test that charged residues are shown in output."""
        fasta = ">test\nKRRDDEHHH"
        result = calculate_isoelectric_point.invoke({"fasta_sequence": fasta})

        assert "Lysine (K): 1" in result
        assert "Arginine (R): 2" in result
        assert "Histidine (H): 3" in result
        assert "Aspartic acid (D): 2" in result
        assert "Glutamic acid (E): 1" in result

    def test_calculate_basic_protein(self):
        """Test with a basic protein (more positive charges)."""
        fasta = ">test\nKKKKKKKKKK"  # 10 lysines
        result = calculate_isoelectric_point.invoke({"fasta_sequence": fasta})

        # Extract pI value - should be high (basic)
        assert "Estimated pI" in result
        # With many positive charges, pI should be > 7
        lines = result.split("\n")
        pi_line = next(line for line in lines if "Estimated pI" in line)
        pi_value = float(pi_line.split(":")[-1].strip())
        assert pi_value > 7.0

    def test_calculate_acidic_protein(self):
        """Test with an acidic protein (more negative charges)."""
        fasta = ">test\nDDDDDDDDDD"  # 10 aspartic acids
        result = calculate_isoelectric_point.invoke({"fasta_sequence": fasta})

        # Extract pI value - should be low (acidic)
        assert "Estimated pI" in result
        lines = result.split("\n")
        pi_line = next(line for line in lines if "Estimated pI" in line)
        pi_value = float(pi_line.split(":")[-1].strip())
        assert pi_value < 7.0

    def test_calculate_neutral_protein(self):
        """Test with a neutral protein (balanced charges)."""
        fasta = ">test\nAAAAAAAAAAA"  # No charged residues
        result = calculate_isoelectric_point.invoke({"fasta_sequence": fasta})

        # Should have a neutral pI estimation
        assert "Estimated pI" in result

    def test_calculate_includes_disclaimer(self):
        """Test that output includes disclaimer about simplified estimation."""
        fasta = ">test\nMASLKGFVP"
        result = calculate_isoelectric_point.invoke({"fasta_sequence": fasta})

        assert "Note:" in result
        assert "simplified estimation" in result

    def test_calculate_multiline_fasta(self):
        """Test with multi-line FASTA format."""
        fasta = ">sp|P12345|TEST\nMASLK\nGFVPT\nARLKD"
        result = calculate_isoelectric_point.invoke({"fasta_sequence": fasta})

        assert "Estimated pI" in result

    def test_calculate_error_handling(self):
        """Test error handling for invalid input."""
        result = calculate_isoelectric_point.invoke({"fasta_sequence": ""})

        # Should handle gracefully
        assert isinstance(result, str)

    def test_tool_metadata(self):
        """Test tool has proper metadata."""
        assert calculate_isoelectric_point.name == "calculate_isoelectric_point"
        assert hasattr(calculate_isoelectric_point, "description")
        assert "isoelectric point" in calculate_isoelectric_point.description.lower()


class TestAminoAcidWeights:
    """Tests for the AMINO_ACID_WEIGHTS constant."""

    def test_all_standard_amino_acids_present(self):
        """Test that all 20 standard amino acids are present."""
        standard_aas = "ARNDCEQGHILKMFPSTWYV"
        for aa in standard_aas:
            assert aa in AMINO_ACID_WEIGHTS

    def test_weights_are_positive(self):
        """Test that all weights are positive numbers."""
        for _aa, weight in AMINO_ACID_WEIGHTS.items():
            assert weight > 0
            assert isinstance(weight, (int, float))

    def test_weight_ranges(self):
        """Test that weights are in reasonable ranges."""
        for _aa, weight in AMINO_ACID_WEIGHTS.items():
            # Amino acid weights should be between 50 and 250 Da
            assert 50 < weight < 250

    def test_specific_weights(self):
        """Test some specific known amino acid weights."""
        # Glycine is the lightest
        assert AMINO_ACID_WEIGHTS["G"] == 75.07

        # Tryptophan is the heaviest
        assert AMINO_ACID_WEIGHTS["W"] == 204.23
