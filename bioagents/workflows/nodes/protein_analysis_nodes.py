"""Protein property nodes built on ``protein_analysis_core``."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, ClassVar

from bioagents.tools.protein_analysis_core import (
    aliphatic_index_percent,
    composition_report,
    gravy_score,
    instability_index_rough,
    isoelectric_report,
    molecular_weight_daltons,
    molecular_weight_report,
    normalize_protein_sequence,
)
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata


class MolecularWeightNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "molecular_weight"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Molecular weight",
            description="Compute Da and kDa from one-letter sequence.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"mw_daltons": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        w, _ln = molecular_weight_daltons(inputs["sequence"])
        rep = molecular_weight_report(inputs["sequence"])
        return {"mw_daltons": round(w, 4), "report": rep}


class AminoAcidCompositionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "amino_acid_composition"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Amino acid composition",
            description="Full composition table and biochemical split.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"report": composition_report(inputs["sequence"])}


class IsoelectricPointNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "isoelectric_point"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Isoelectric point (estimate)",
            description="Rough pI from charged residue counts.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"report": isoelectric_report(inputs["sequence"])}


class ReverseSequenceNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "reverse_sequence"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Reverse sequence",
            description="Reverse one-letter protein sequence.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"reversed_sequence": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        return {"reversed_sequence": seq[::-1]}


class KmerProfileNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "kmer_profile"

    def __init__(self, k: int = 3) -> None:
        if k < 1 or k > 12:
            raise ValueError("k must be 1..12")
        self._k = k

    @property
    def params(self) -> dict[str, Any]:
        return {"k": self._k}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="K-mer profile",
            description=f"Top {self._k}-mer frequencies in the sequence.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        if len(seq) < self._k:
            return {"report": f"Sequence shorter than k={self._k}"}
        kmers = [seq[i : i + self._k] for i in range(len(seq) - self._k + 1)]
        ctr = Counter(kmers)
        top = ctr.most_common(15)
        lines = [f"Top {self._k}-mers (of {len(ctr)} unique):", "-" * 40]
        for mer, c in top:
            lines.append(f"{mer}\t{c}")
        return {"report": "\n".join(lines)}


class HydrophobicFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "hydrophobic_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Hydrophobic fraction",
            description="Fraction of AVILMFYWP residues.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"fraction": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        if not seq:
            return {"fraction": 0.0, "report": "Empty sequence"}
        hyd = "AVILMFYWP"
        n = sum(1 for aa in seq if aa in hyd)
        f = n / len(seq)
        return {
            "fraction": round(f, 6),
            "report": f"Hydrophobic fraction: {f:.4f} ({n}/{len(seq)})",
        }


class ChargedFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "charged_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Charged fraction",
            description="Fraction of KREDH (charged / polar charged).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"fraction": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        if not seq:
            return {"fraction": 0.0, "report": "Empty sequence"}
        ch = "KREDH"
        n = sum(1 for aa in seq if aa in ch)
        f = n / len(seq)
        return {
            "fraction": round(f, 6),
            "report": f"Charged/polar-charged fraction: {f:.4f} ({n}/{len(seq)})",
        }


class AromaticFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "aromatic_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Aromatic fraction",
            description="Fraction of F, Y, W.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"fraction": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        if not seq:
            return {"fraction": 0.0, "report": "Empty sequence"}
        ar = "FYW"
        n = sum(1 for aa in seq if aa in ar)
        f = n / len(seq)
        return {
            "fraction": round(f, 6),
            "report": f"Aromatic fraction: {f:.4f} ({n}/{len(seq)})",
        }


class SequenceDigestNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "sequence_digest"

    def __init__(self, algorithm: str = "md5") -> None:
        if algorithm not in ("md5", "sha256"):
            raise ValueError("algorithm must be md5 or sha256")
        self._algorithm = algorithm

    @property
    def params(self) -> dict[str, Any]:
        return {"algorithm": self._algorithm}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Sequence digest",
            description=f"Hex digest of sequence bytes ({self._algorithm}).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"hex_digest": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        raw = seq.encode("utf-8")
        if self._algorithm == "md5":
            h = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        else:
            h = hashlib.sha256(raw).hexdigest()
        return {"hex_digest": h, "report": f"{self._algorithm.upper()}: {h}"}


class GravyScoreNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "gravy_score"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="GRAVY score",
            description="Grand average of hydropathy (Kyte-Doolittle).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"gravy": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        g = gravy_score(inputs["sequence"])
        return {"gravy": round(g, 4), "report": f"GRAVY (Kyte-Doolittle): {g:.4f}"}


class AliphaticIndexNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "aliphatic_index_proxy"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Aliphatic content",
            description="Percent of A+V+I+L residues (proxy).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"aliphatic_percent": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        p = aliphatic_index_percent(inputs["sequence"])
        return {
            "aliphatic_percent": round(p, 4),
            "report": f"Aliphatic (AVIL) content: {p:.2f}%",
        }


class InstabilityIndexNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "instability_index_rough"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Instability heuristic",
            description="Rough instability score (0-100, higher = less stable).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"score": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        s = instability_index_rough(inputs["sequence"])
        return {"score": round(s, 4), "report": f"Instability heuristic: {s:.2f}"}


class SingleResidueCountNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "single_residue_count"

    def __init__(self, letter: str = "C") -> None:
        u = letter.strip().upper()
        if len(u) != 1 or u not in "ACDEFGHIKLMNPQRSTVWY":
            raise ValueError("letter must be one standard amino acid code")
        self._letter = u

    @property
    def params(self) -> dict[str, Any]:
        return {"letter": self._letter}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name=f"Count residue {self._letter}",
            description=f"Occurrences and fraction of {self._letter}.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"count": "int", "fraction": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        if not seq:
            return {"count": 0, "fraction": 0.0, "report": "Empty sequence"}
        c = seq.count(self._letter)
        f = c / len(seq)
        return {
            "count": c,
            "fraction": round(f, 6),
            "report": f"{self._letter}: {c} ({100 * f:.2f}%)",
        }


class TerminalResidueNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "terminal_residue"

    def __init__(self, end: str = "n") -> None:
        e = end.lower().strip()
        if e not in ("n", "c"):
            raise ValueError("end must be 'n' or 'c'")
        self._end = e

    @property
    def params(self) -> dict[str, Any]:
        return {"end": self._end}

    @property
    def metadata(self) -> NodeMetadata:
        lab = "N-terminal" if self._end == "n" else "C-terminal"
        return NodeMetadata(
            name=f"{lab} residue",
            description="First or last amino acid in the cleaned sequence.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"residue": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        if not seq:
            return {"residue": "", "report": "Empty sequence"}
        r = seq[0] if self._end == "n" else seq[-1]
        lab = "N-term" if self._end == "n" else "C-term"
        return {"residue": r, "report": f"{lab} residue: {r}"}


class SequenceLengthNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "sequence_length"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Sequence length",
            description="Residue count after normalization.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"length": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        ln = len(seq)
        return {"length": ln, "report": f"Length: {ln} residues"}


class CompactBiochemRecordNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "compact_biochem_record"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Compact biochem record",
            description="MW, length, GRAVY, aliphatic%, instability heuristic.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"record": "dict"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        w, ln = molecular_weight_daltons(seq)
        rec = {
            "length": ln,
            "mw_daltons": round(w, 4),
            "gravy": round(gravy_score(seq), 4),
            "aliphatic_percent": round(aliphatic_index_percent(seq), 4),
            "instability_heuristic": round(instability_index_rough(seq), 4),
        }
        return {"record": rec}
