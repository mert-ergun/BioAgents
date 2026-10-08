"""Tests for the native RDKit backend.

Regression context: the 22 chemistry tools used to shell out to a Node.js
`rdkit-agent` CLI that was not installed. Every one of them was unusable, and some
reported the missing CLI as a *chemical* verdict — `validate_smiles` answered
`{"success": true, "valid": false}` for perfectly good molecules, which would make a
caller discard them. Separately, a benchmark run had an agent fake a synthetic
accessibility score with `sa = 3.0 + np.random.normal(0, 0.5)` because it could not
find a real one, while RDKit's SA_Score shipped in the installed distribution all along.
"""

from __future__ import annotations

import pytest

from bioagents.tools.rdkit_native import (
    analyze_rings,
    analyze_stereo,
    apply_reaction,
    atom_map_check,
    check_reaction,
    check_smiles,
    compute_descriptors,
    compute_fingerprint,
    convert_notation,
    dataset_statistics,
    detect_functional_groups,
    draw_molecule,
    edit_molecule,
    filter_molecules,
    repair_smiles,
    search_similar_molecules,
    substructure_search,
)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
L_ALANINE = "C[C@H](N)C(=O)O"


class TestValidation:
    def test_valid_smiles_is_accepted_and_canonicalised(self) -> None:
        result = check_smiles(ETHANOL)

        assert result["valid"] is True
        assert result["canonical_smiles"] == "CCO"
        assert result["confidence"] == 1.0

    def test_structurally_impossible_smiles_is_rejected(self) -> None:
        """An unclosed ring is genuinely invalid and must be reported as such."""
        result = check_smiles("C1CC")

        assert result["valid"] is False
        assert "could not parse" in result["summary"].lower()

    def test_valence_violation_is_rejected(self) -> None:
        result = check_smiles("C(C)(C)(C)(C)C")

        assert result["valid"] is False

    def test_invalid_verdict_is_never_an_infrastructure_failure(self) -> None:
        """A chemical verdict must not be produced by a tooling problem.

        The old backend returned valid=False when its CLI was absent. If RDKit itself
        is missing the backend raises instead, so a caller can tell the two apart.
        """
        result = check_smiles(BENZENE)

        assert result["valid"] is True, (
            "a well-formed molecule reported invalid usually means the backend failed"
        )


class TestRepair:
    def test_lowercase_halogen_is_repaired(self) -> None:
        """'cl' parses as aromatic carbon + L, which is never what the author meant."""
        result = repair_smiles("clc1ccccc1")

        assert result["success"] is True
        assert result["canonical_smiles"] == "Clc1ccccc1"
        assert result["strategy"] == "uppercase_halogens"

    def test_valid_input_is_reported_as_unmodified(self) -> None:
        result = repair_smiles(ASPIRIN)

        assert result["strategy"] == "as_written"
        assert result["confidence"] == 1.0

    def test_unbalanced_parentheses_are_closed(self) -> None:
        result = repair_smiles("CC(=O")

        assert result["success"] is True
        assert result["strategy"] == "close_unbalanced_parens"

    def test_unrepairable_input_reports_failure_not_a_guess(self) -> None:
        result = repair_smiles("this is not a molecule")

        assert result["success"] is False
        assert result["best_candidate"]["valid"] is False

    def test_empty_input_is_rejected(self) -> None:
        result = repair_smiles("   ")

        assert result["success"] is False
        assert result["attempts"] == 0


class TestDescriptors:
    def test_known_descriptor_values(self) -> None:
        molecules = compute_descriptors([ETHANOL])["molecules"]
        ethanol = molecules[0]

        assert ethanol["MW"] == pytest.approx(46.07, abs=0.05)
        assert ethanol["HBD"] == 1
        assert ethanol["HBA"] == 1
        assert ethanol["Rings"] == 0

    def test_sa_score_is_a_real_computed_value(self) -> None:
        """The scorer an agent previously replaced with a random number."""
        simple = compute_descriptors([ETHANOL])["molecules"][0]["SA_Score"]
        complex_mol = compute_descriptors(
            ["C[C@@H]1CC[C@H]2[C@@H](C)[C@H](O)C[C@]3(C)[C@@H]2C1=CC3=O"]
        )["molecules"][0]["SA_Score"]

        assert simple is not None, "SA_Score must be computed, not reported as missing"
        assert 1.0 <= simple <= 10.0
        assert complex_mol > simple, (
            "a fused polycyclic stereochemistry-rich molecule must score as harder to "
            "make than ethanol; equal scores suggest a stubbed implementation"
        )

    def test_invalid_member_of_a_batch_does_not_poison_the_rest(self) -> None:
        molecules = compute_descriptors([ETHANOL, "C1CC", BENZENE])["molecules"]

        assert molecules[0]["valid"] is True
        assert molecules[1]["valid"] is False
        assert molecules[2]["valid"] is True

    def test_field_selection_is_honoured(self) -> None:
        molecules = compute_descriptors([ETHANOL], fields=["MW", "logP"])["molecules"]

        assert set(molecules[0]) == {"smiles", "valid", "MW", "logP"}

    def test_unknown_field_is_rejected_rather_than_ignored(self) -> None:
        result = compute_descriptors([ETHANOL], fields=["NotADescriptor"])

        assert "error" in result
        assert result["molecules"] == []


class TestFiltering:
    def test_lipinski_rejects_an_overly_lipophilic_molecule(self) -> None:
        long_alkane = "C" * 30
        result = filter_molecules([ETHANOL, ASPIRIN, long_alkane], lipinski=True)

        assert result["filtered_smiles"] == [ETHANOL, ASPIRIN]
        assert result["filtered_count"] == 2
        assert any("logP" in f for r in result["rejected"] for f in r["failed"])

    def test_explicit_bounds_are_applied(self) -> None:
        result = filter_molecules([ETHANOL, ASPIRIN], mw_max=100)

        assert result["filtered_smiles"] == [ETHANOL]

    def test_rejections_explain_which_constraint_failed(self) -> None:
        result = filter_molecules([ASPIRIN], mw_max=100)

        assert result["rejected"][0]["failed"], "a rejection must say why"


class TestSimilarityAndFingerprints:
    def test_identical_molecules_score_one(self) -> None:
        result = search_similar_molecules(BENZENE, [BENZENE], threshold=0.0)

        assert result["hits"][0]["similarity"] == pytest.approx(1.0)

    def test_ranking_is_descending(self) -> None:
        result = search_similar_molecules(
            BENZENE, ["Cc1ccccc1", ETHANOL, "c1ccc2ccccc2c1"], threshold=0.0, top=5
        )
        scores = [h["similarity"] for h in result["hits"]]

        assert scores == sorted(scores, reverse=True)

    def test_unparseable_targets_are_reported_not_silently_dropped(self) -> None:
        result = search_similar_molecules(BENZENE, [BENZENE, "C1CC"], threshold=0.0)

        assert result["unparseable_targets"] == ["C1CC"]

    def test_fingerprint_has_set_bits(self) -> None:
        result = compute_fingerprint(CAFFEINE)

        assert result["num_on_bits"] > 0
        assert len(result["fingerprint"]) == result["nbits"]


class TestStructureAnalysis:
    def test_aspirin_functional_groups(self) -> None:
        groups = detect_functional_groups(ASPIRIN)["functional_groups"]

        assert "carboxylic_acid" in groups
        assert "ester" in groups

    def test_ring_counts_for_a_fused_system(self) -> None:
        result = analyze_rings("c1ccc2ccccc2c1")

        assert result["ring_count"] == 2
        assert result["aromatic_rings"] == 2

    def test_caffeine_is_recognised_as_heterocyclic(self) -> None:
        result = analyze_rings(CAFFEINE)

        assert result["ring_count"] == 2
        assert any(ring["is_heterocycle"] for ring in result["rings"])

    def test_stereocentre_cip_code(self) -> None:
        result = analyze_stereo(L_ALANINE)

        assert result["stereo_center_count"] == 1
        assert result["stereo_centers"][0]["cip_code"] == "S"
        assert result["has_unspecified_stereo"] is False

    def test_unspecified_stereocentre_is_flagged(self) -> None:
        result = analyze_stereo("CC(N)C(=O)O")

        assert result["has_unspecified_stereo"] is True

    def test_substructure_match_counts(self) -> None:
        result = substructure_search("c1ccc2ccccc2c1", BENZENE)

        assert result["matched"] is True
        assert result["match_count"] >= 1

    def test_malformed_smarts_is_reported(self) -> None:
        result = substructure_search(BENZENE, "[[[")

        assert result["matched"] is False
        assert "error" in result


class TestConversionAndEditing:
    def test_smiles_to_inchikey(self) -> None:
        result = convert_notation(ETHANOL, "smiles", "inchikey")

        assert result["success"] is True
        assert result["output"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"

    def test_round_trip_through_inchi(self) -> None:
        inchi = convert_notation(ASPIRIN, "smiles", "inchi")["output"]
        back = convert_notation(inchi, "inchi", "smiles")["output"]

        assert (
            convert_notation(back, "smiles", "inchikey")["output"]
            == (convert_notation(ASPIRIN, "smiles", "inchikey")["output"])
        )

    def test_inchikey_cannot_be_reversed(self) -> None:
        """InChIKey is a hash; claiming to decode it would be fabrication."""
        result = convert_notation("LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "inchikey", "smiles")

        assert result["success"] is False

    def test_unsupported_format_is_rejected(self) -> None:
        result = convert_notation(ETHANOL, "smiles", "pdf")

        assert result["success"] is False
        assert "Supported formats" in result["error"]

    def test_neutralise_removes_formal_charge(self) -> None:
        result = edit_molecule("CC(=O)[O-]", "neutralize")

        assert result["success"] is True
        assert result["output_smiles"] == "CC(=O)O"

    def test_unknown_operation_is_rejected(self) -> None:
        result = edit_molecule(ETHANOL, "transmute")

        assert result["success"] is False


class TestReactions:
    def test_reaction_runs(self) -> None:
        """The previous WASM-based backend could not run reactions at all."""
        result = apply_reaction("[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[Cl]", ["CC(=O)O"])

        assert result["products"] == [["CC(=O)Cl"]]

    def test_wrong_reactant_count_is_rejected(self) -> None:
        result = apply_reaction("[C:1]=[C:2].[H][H]>>[C:1][C:2]", ["C=C"])

        assert result["products"] == []
        assert "reactant" in result["error"]

    def test_balanced_reaction_is_recognised(self) -> None:
        result = check_reaction([ETHANOL, "CC(=O)O"], ["CC(=O)OCC", "O"])

        assert result["balanced"] is True

    def test_unbalanced_reaction_reports_the_difference(self) -> None:
        result = check_reaction([ETHANOL], ["CC(=O)OCC"])

        assert result["balanced"] is False
        assert result["atom_differences"]

    def test_atom_map_imbalance_is_detected(self) -> None:
        result = atom_map_check("[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[Cl]")

        assert result["balanced"] is False
        assert 3 in result["reactant_maps_not_in_products"]


class TestDatasetStatistics:
    def test_statistics_summarise_the_set(self) -> None:
        result = dataset_statistics([ETHANOL, BENZENE, "CC(=O)O"])

        assert result["molecule_count"] == 3
        mw = result["descriptors_stats"]["MW"]
        assert mw["min"] < mw["mean"] < mw["max"]
        assert mw["n"] == 3

    def test_unparseable_entries_are_excluded_and_reported(self) -> None:
        result = dataset_statistics([ETHANOL, "C1CC"])

        assert result["molecule_count"] == 1
        assert result["unparseable"] == ["C1CC"]


class TestDrawing:
    def test_svg_is_rendered(self) -> None:
        result = draw_molecule(ASPIRIN)

        assert "<svg" in result["svg"]

    def test_file_output_is_written_to_a_portable_path(self, tmp_path) -> None:
        target = tmp_path / "mol.svg"
        result = draw_molecule(ETHANOL, output_file=str(target))

        assert target.exists()
        assert result["bytes_written"] > 0
