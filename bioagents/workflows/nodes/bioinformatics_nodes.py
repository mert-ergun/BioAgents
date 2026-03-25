"""Broad catalog of bioinformatics-oriented workflow nodes (mostly pure computation)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, ClassVar

from bioagents.tools.protein_analysis_core import normalize_protein_sequence
from bioagents.workflows.node import WorkflowNode
from bioagents.workflows.schemas import NodeMetadata

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _clean_nt(raw: str, *, rna: bool) -> str:
    s = "".join(raw.upper().split())
    out: list[str] = []
    for c in s:
        if c == "T" and rna:
            c = "U"
        elif c == "U" and not rna:
            c = "T"
        if c in ("A", "C", "G"):
            out.append(c)
        elif c == "T" and not rna:
            out.append("T")
        elif c == "U" and rna:
            out.append("U")
        elif c == "N":
            out.append("N")
        # skip invalid (e.g. FASTA header residue if slipped)
    return "".join(out)


def _reverse_comp_dna(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


def _reverse_comp_rna(seq: str) -> str:
    comp = {"A": "U", "U": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


_STOPS = {"TAA", "TAG", "TGA"}
_START = "ATG"


class NtNormalizeNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "nt_normalize"

    def __init__(self, kind: str = "dna") -> None:
        k = kind.lower().strip()
        if k not in ("dna", "rna"):
            raise ValueError("kind must be 'dna' or 'rna'")
        self._rna = k == "rna"

    @property
    def params(self) -> dict[str, Any]:
        return {"kind": "rna" if self._rna else "dna"}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Normalize nucleotide sequence",
            description="Uppercase, strip; DNA uses T, RNA uses U.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str", "length": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = _clean_nt(str(inputs["nt_sequence"]), rna=self._rna)
        lab = "RNA" if self._rna else "DNA"
        return {
            "nt_sequence": seq,
            "length": len(seq),
            "report": f"{lab} length {len(seq)} (non-ACGT/U removed)",
        }


class NtGcContentNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "nt_gc_content"

    def __init__(self, kind: str = "dna") -> None:
        k = kind.lower().strip()
        if k not in ("dna", "rna"):
            raise ValueError("kind must be 'dna' or 'rna'")
        self._rna = k == "rna"

    @property
    def params(self) -> dict[str, Any]:
        return {"kind": "rna" if self._rna else "dna"}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="GC content (nucleotide)",
            description="GC fraction over ACGT (or ACGU) plus length.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"gc_fraction": "float", "length": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = _clean_nt(str(inputs["nt_sequence"]), rna=self._rna)
        if not seq:
            return {"gc_fraction": 0.0, "length": 0, "report": "Empty sequence"}
        gc = sum(1 for b in seq if b in "GC")
        atu = sum(1 for b in seq if b in ("A", "T", "U"))
        denom = gc + atu + sum(1 for b in seq if b == "N")
        f = gc / denom if denom else 0.0
        return {
            "gc_fraction": round(f, 6),
            "length": len(seq),
            "report": f"GC: {100 * f:.2f}% ({gc}/{denom} informative bases)",
        }


class NtReverseComplementNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "nt_reverse_complement"

    def __init__(self, kind: str = "dna") -> None:
        k = kind.lower().strip()
        if k not in ("dna", "rna"):
            raise ValueError("kind must be 'dna' or 'rna'")
        self._rna = k == "rna"

    @property
    def params(self) -> dict[str, Any]:
        return {"kind": "rna" if self._rna else "dna"}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Reverse complement",
            description="Reverse complement for DNA or RNA.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = _clean_nt(str(inputs["nt_sequence"]), rna=self._rna)
        rc = _reverse_comp_rna(seq) if self._rna else _reverse_comp_dna(seq)
        return {"nt_sequence": rc, "report": f"Rev-comp length {len(rc)}"}


class TranscribeDnaToRnaNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "transcribe_dna_to_rna"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Transcribe DNA → RNA",
            description="T → U on cleaned DNA alphabet.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        dna = _clean_nt(str(inputs["nt_sequence"]), rna=False)
        rna = dna.replace("T", "U")
        return {"nt_sequence": rna, "report": f"Transcribed length {len(rna)}"}


class ReverseTranscribeRnaToDnaNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "reverse_transcribe_rna_to_dna"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Reverse transcribe RNA → DNA",
            description="U → T on cleaned RNA alphabet.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rna = _clean_nt(str(inputs["nt_sequence"]), rna=True)
        dna = rna.replace("U", "T")
        return {"nt_sequence": dna, "report": f"cDNA length {len(dna)}"}


class NtCompositionReportNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "nt_composition_report"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Nucleotide composition",
            description="Per-base counts and percentages.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = (
            str(inputs["nt_sequence"]).upper().replace(" ", "").replace("\n", "").replace("\t", "")
        )
        if not seq:
            return {"report": "Empty sequence"}
        ctr = Counter(seq)
        total = sum(ctr.values())
        lines = ["Nucleotide composition", "=" * 40]
        for b in sorted(ctr.keys()):
            lines.append(f"{b}: {ctr[b]} ({100 * ctr[b] / total:.2f}%)")
        return {"report": "\n".join(lines)}


class NtMotifCountNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "nt_motif_count"

    def __init__(self, motif: str = "GAATTC", case_insensitive: bool = True) -> None:
        self._motif = motif.upper().strip()
        self._ci = bool(case_insensitive)

    @property
    def params(self) -> dict[str, Any]:
        return {"motif": self._motif, "case_insensitive": self._ci}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Motif count (nucleotide)",
            description="Count non-overlapping occurrences of a short motif.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"count": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = str(inputs["nt_sequence"]).upper() if self._ci else str(inputs["nt_sequence"])
        if self._ci:
            seq = seq.upper()
        m = self._motif if not self._ci else self._motif.upper()
        if not m:
            return {"count": 0, "report": "Empty motif"}
        c = 0
        i = 0
        while i <= len(seq) - len(m):
            if seq[i : i + len(m)] == m:
                c += 1
                i += len(m)
            else:
                i += 1
        return {"count": c, "report": f"Motif {m!r}: {c} non-overlapping hits"}


class CodonFrequencyReportNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "codon_frequency_report"

    def __init__(self, frame: int = 0) -> None:
        if frame not in (0, 1, 2):
            raise ValueError("frame must be 0, 1, or 2")
        self._frame = frame

    @property
    def params(self) -> dict[str, Any]:
        return {"frame": self._frame}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Codon frequencies",
            description="DNA codon counts on one reading frame (after cleaning to DNA).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        dna = _clean_nt(str(inputs["nt_sequence"]), rna=False)
        dna = dna.replace("U", "T")
        if len(dna) < self._frame + 3:
            return {"report": "Sequence too short for codons"}
        chunk = dna[self._frame :]
        chunk = chunk[: len(chunk) - (len(chunk) % 3)]
        codons = [chunk[i : i + 3] for i in range(0, len(chunk), 3)]
        ctr = Counter(codons)
        top = ctr.most_common(12)
        lines = [f"Codons (frame {self._frame}, {len(codons)} triplets)", "-" * 36]
        for cod, n in top:
            lines.append(f"{cod}\t{n}")
        return {"report": "\n".join(lines)}


class LongestOrfLengthNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "longest_orf_length_aa"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Longest ORF length (aa)",
            description="Scan six frames on DNA for ATG…stop; report longest aa length.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"max_aa_length": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        dna = _clean_nt(str(inputs["nt_sequence"]), rna=False).replace("U", "T")
        best = 0
        for frame in range(3):
            s = dna[frame:]
            s = s[: len(s) - (len(s) % 3)]
            i = 0
            while i + 3 <= len(s):
                if s[i : i + 3] != _START:
                    i += 3
                    continue
                j = i + 3
                aa = 1
                while j + 3 <= len(s):
                    trip = s[j : j + 3]
                    if trip in _STOPS:
                        best = max(best, aa)
                        i = j + 3
                        break
                    aa += 1
                    j += 3
                else:
                    break
        return {
            "max_aa_length": best,
            "report": f"Longest ORF (simple ATG→stop): {best} aa",
        }


class FastaFirstHeaderNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "fasta_first_header"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="FASTA first header",
            description="Extract the first > line from FASTA text.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"fasta": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"header": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        for ln in str(inputs["fasta"]).splitlines():
            t = ln.strip()
            if t.startswith(">"):
                h = t[1:].strip()
                return {"header": h, "report": f"Header: {h}"}
        return {"header": "", "report": "No header found"}


class FastaRecordCountNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "fasta_record_count"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="FASTA record count",
            description="Count lines starting with >.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"fasta": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"count": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        n = sum(1 for ln in str(inputs["fasta"]).splitlines() if ln.strip().startswith(">"))
        return {"count": n, "report": f"FASTA records: {n}"}


class ProteinPolarFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_polar_uncharged_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Polar uncharged fraction",
            description="STNQ (excludes charged KREDH).",
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
            return {"fraction": 0.0, "report": "Empty"}
        polar = "STNQ"
        n = sum(1 for a in seq if a in polar)
        f = n / len(seq)
        return {"fraction": round(f, 6), "report": f"Polar uncharged: {f:.4f}"}


class ProteinTinyFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_tiny_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Tiny residue fraction",
            description="A, G, S, T.",
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
            return {"fraction": 0.0, "report": "Empty"}
        tiny = "AGST"
        n = sum(1 for a in seq if a in tiny)
        f = n / len(seq)
        return {"fraction": round(f, 6), "report": f"Tiny (AGST): {f:.4f}"}


class ProteinNetChargeEstimateNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_net_charge_estimate"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Net charge estimate",
            description="(K+R+H) - (D+E) at neutral pH heuristic.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"net_charge": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        pos = sum(seq.count(x) for x in "KRH")
        neg = sum(seq.count(x) for x in "DE")
        net = pos - neg
        return {"net_charge": net, "report": f"Approx net charge: +{pos} - {neg} = {net:+d}"}


class ProteinHelixFormerFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_helix_former_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Helix-former fraction",
            description="E, K, A, L, M, R (Chou-Fasman style heuristic).",
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
            return {"fraction": 0.0, "report": "Empty"}
        h = "EKALMR"
        n = sum(1 for a in seq if a in h)
        f = n / len(seq)
        return {"fraction": round(f, 6), "report": f"Helix-favoring heuristic: {f:.4f}"}


class ProteinBetaBranchedFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_beta_branched_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="β-branched fraction",
            description="Ile, Val, Thr.",
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
            return {"fraction": 0.0, "report": "Empty"}
        b = "IVT"
        n = sum(1 for a in seq if a in b)
        f = n / len(seq)
        return {"fraction": round(f, 6), "report": f"β-branched (IVT): {f:.4f}"}


class ProteinShannonEntropyNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_sequence_entropy"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Sequence Shannon entropy",
            description="Bits per residue from AA frequency (complexity proxy).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"entropy_bits": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        if not seq:
            return {"entropy_bits": 0.0, "report": "Empty"}
        ctr = Counter(a for a in seq if a in _STANDARD_AA)
        tot = sum(ctr.values())
        h = 0.0
        for c in ctr.values():
            p = c / tot
            h -= p * math.log2(p)
        return {
            "entropy_bits": round(h, 6),
            "report": f"Shannon H ≈ {h:.3f} bits (unique AA: {len(ctr)})",
        }


class PairHammingDistanceNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "pair_hamming_distance"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Hamming distance",
            description="Mismatches at aligned positions (sequences must match length).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence_a": "str", "sequence_b": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"distance": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        a = normalize_protein_sequence(inputs["sequence_a"])
        b = normalize_protein_sequence(inputs["sequence_b"])
        if len(a) != len(b):
            return {
                "distance": -1,
                "report": f"Length mismatch: {len(a)} vs {len(b)}",
            }
        d = sum(1 for x, y in zip(a, b, strict=True) if x != y)
        return {"distance": d, "report": f"Hamming: {d} / {len(a)}"}


class PairIdentityPercentNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "pair_sequence_identity_percent"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Pairwise identity %",
            description="Identical positions / min(length) after normalization.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence_a": "str", "sequence_b": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"identity_percent": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        a = normalize_protein_sequence(inputs["sequence_a"])
        b = normalize_protein_sequence(inputs["sequence_b"])
        if not a or not b:
            return {"identity_percent": 0.0, "report": "Empty sequence"}
        n = min(len(a), len(b))
        matches = sum(1 for i in range(n) if a[i] == b[i])
        pct = 100.0 * matches / n
        return {
            "identity_percent": round(pct, 4),
            "report": f"Identity over first {n} positions: {pct:.2f}%",
        }


class RecordToJsonStringNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "record_to_json_string"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Record → JSON string",
            description="Serialize a dict output (e.g. biochem record) to JSON text.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"record": "dict"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"json": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rec = inputs["record"]
        if not isinstance(rec, dict):
            raise TypeError("record must be dict")
        return {"json": json.dumps(rec, indent=2, default=str)}


class TextTruncateNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "text_truncate"

    def __init__(self, max_chars: int = 500) -> None:
        self._max = max(1, int(max_chars))

    @property
    def params(self) -> dict[str, Any]:
        return {"max_chars": self._max}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Truncate text",
            description="Shorten long strings for display.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"text": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"text": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        t = str(inputs["text"])
        if len(t) <= self._max:
            return {"text": t, "report": f"Length {len(t)} (no truncation)"}
        out = t[: self._max] + "…"
        return {"text": out, "report": f"Truncated to {self._max} chars"}


class SmilesBasicStatsNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "smiles_basic_stats"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="SMILES basic stats",
            description="Length, bracket fragments, ring digit count (heuristic).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"smiles": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"length": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        s = str(inputs["smiles"]).strip()
        rings = len(re.findall(r"[0-9]", s))
        brackets = s.count("[")
        return {
            "length": len(s),
            "report": f"SMILES len={len(s)}, ~ring digits={rings}, '[' count={brackets}",
        }


class OligoTmWallaceNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "oligo_tm_wallace"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Oligo Tm (Wallace rule)",
            description="2*(A+T) + 4*(G+C) for short DNA primers (°C, rough).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"tm_celsius": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        dna = _clean_nt(str(inputs["nt_sequence"]), rna=False).replace("U", "T")
        if len(dna) > 50:
            return {
                "tm_celsius": 0.0,
                "report": "Wallace rule unreliable for len>50; use dedicated tools.",
            }
        a = dna.count("A")
        t = dna.count("T")
        g = dna.count("G")
        c = dna.count("C")
        tm = 2 * (a + t) + 4 * (g + c)
        return {"tm_celsius": float(tm), "report": f"Wallace Tm ≈ {tm} °C (len {len(dna)})"}


class NtKmerProfileNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "nt_kmer_profile"

    def __init__(self, k: int = 4) -> None:
        if k < 2 or k > 12:
            raise ValueError("k must be 2..12")
        self._k = k

    @property
    def params(self) -> dict[str, Any]:
        return {"k": self._k}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Nucleotide k-mer profile",
            description="Top k-mer counts on cleaned DNA.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"nt_sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        dna = _clean_nt(str(inputs["nt_sequence"]), rna=False).replace("U", "T")
        if len(dna) < self._k:
            return {"report": "Sequence shorter than k"}
        kmers = [dna[i : i + self._k] for i in range(len(dna) - self._k + 1)]
        top = Counter(kmers).most_common(12)
        lines = [f"Top {self._k}-mers", "-" * 32]
        for mer, n in top:
            lines.append(f"{mer}\t{n}")
        return {"report": "\n".join(lines)}


class ProteinSerThrFractionNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_ser_thr_fraction"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Ser + Thr fraction",
            description="Proxy for O-linked / phosphorylation-prone density.",
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
            return {"fraction": 0.0, "report": "Empty"}
        n = sum(1 for a in seq if a in "ST")
        f = n / len(seq)
        return {"fraction": round(f, 6), "report": f"Ser+Thr: {f:.4f}"}


class ProteinLysArgRatioNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_lys_arg_ratio"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Lys / Arg ratio",
            description="Count ratio K:R (0 if no R).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"ratio": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        k = seq.count("K")
        r = seq.count("R")
        ratio = (k / r) if r else float(k)
        return {
            "ratio": round(ratio, 6),
            "report": f"Lys {k}, Arg {r}, K/R ratio ≈ {ratio:.3f}",
        }


class ProteinCysteineCountNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_cysteine_count"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Cysteine count",
            description="C residues and fraction (disulfide capacity proxy).",
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
            return {"count": 0, "fraction": 0.0, "report": "Empty"}
        c = seq.count("C")
        f = c / len(seq)
        return {"count": c, "fraction": round(f, 6), "report": f"Cys: {c} ({100 * f:.2f}%)"}


class ProteinAliphaticAromaticRatioNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "protein_aliphatic_aromatic_ratio"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Aliphatic / aromatic ratio",
            description="(AVIL) vs (FYW) counts.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"sequence": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"ratio": "float", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        seq = normalize_protein_sequence(inputs["sequence"])
        ali = sum(seq.count(x) for x in "AVIL")
        aro = sum(seq.count(x) for x in "FYW")
        ratio = (ali / aro) if aro else float(ali)
        return {
            "ratio": round(ratio, 6),
            "report": f"Aliphatic {ali}, Aromatic {aro}, ratio ≈ {ratio:.3f}",
        }


class MergeFourTextNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "merge_four_text"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Merge four text blocks",
            description="Stack four reports with section headers.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"text_a": "str", "text_b": "str", "text_c": "str", "text_d": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"merged": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        labels = ["A", "B", "C", "D"]
        parts = [inputs["text_a"], inputs["text_b"], inputs["text_c"], inputs["text_d"]]
        blocks = [f"=== {labels[i]} ===\n{parts[i]}" for i in range(4)]
        return {"merged": "\n\n".join(blocks)}


class RegexFindInTextNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "regex_find_in_text"

    def __init__(self, pattern: str = r"[A-Z]{6,10}") -> None:
        self._pattern = pattern

    @property
    def params(self) -> dict[str, Any]:
        return {"pattern": self._pattern}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Regex find (text)",
            description="First match of pattern in text (Python regex).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"text": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"match": "str", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        t = str(inputs["text"])
        try:
            m = re.search(self._pattern, t)
        except re.error as e:
            return {"match": "", "report": f"Bad regex: {e}"}
        if not m:
            return {"match": "", "report": "No match"}
        return {"match": m.group(0), "report": f"Matched: {m.group(0)!r}"}


class LineCountNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "line_count"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Line count",
            description="Non-empty lines in a text block.",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"text": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"lines": "int", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        n = sum(1 for ln in str(inputs["text"]).splitlines() if ln.strip())
        return {"lines": n, "report": f"Non-empty lines: {n}"}


class JsonParseToRecordNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "json_parse_to_record"

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Parse JSON → record",
            description="Parse a JSON object string to a dict (single object root).",
            version="1.0.0",
            category="tool",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {"json": "str"}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"record": "dict", "report": "str"}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = str(inputs["json"]).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return {"record": {}, "report": f"Parse error: {e}"}
        if not isinstance(obj, dict):
            return {"record": {}, "report": "JSON root must be an object"}
        return {"record": obj, "report": f"Parsed {len(obj)} keys"}


class ConstantStringNode(WorkflowNode):
    """Useful for labels, organism names, or static accession fragments in demos."""

    workflow_type_id: ClassVar[str] = "constant_string"

    def __init__(self, value: str = "") -> None:
        self._value = str(value)

    @property
    def params(self) -> dict[str, Any]:
        return {"value": self._value}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Constant string",
            description="Emit a fixed string (params.value) — no inputs.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"text": "str"}

    def run(self, _inputs: dict[str, Any]) -> dict[str, Any]:
        return {"text": self._value}


class ConstantProteinIdNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "constant_protein_id"

    def __init__(self, protein_id: str = "P04637") -> None:
        self._pid = str(protein_id).strip()

    @property
    def params(self) -> dict[str, Any]:
        return {"protein_id": self._pid}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Constant UniProt ID",
            description="Emit protein_id for UniProt fetch nodes.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"protein_id": "str"}

    def run(self, _inputs: dict[str, Any]) -> dict[str, Any]:
        return {"protein_id": self._pid}


class ConstantPdbIdNode(WorkflowNode):
    workflow_type_id: ClassVar[str] = "constant_pdb_id"

    def __init__(self, pdb_id: str = "1YCR") -> None:
        self._pdb = str(pdb_id).strip().upper()[:4]

    @property
    def params(self) -> dict[str, Any]:
        return {"pdb_id": self._pdb}

    @property
    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name="Constant PDB ID",
            description="Emit pdb_id for structure summary nodes.",
            version="1.0.0",
            category="data",
        )

    @property
    def input_schema(self) -> dict[str, str]:
        return {}

    @property
    def output_schema(self) -> dict[str, str]:
        return {"pdb_id": "str"}

    def run(self, _inputs: dict[str, Any]) -> dict[str, Any]:
        return {"pdb_id": self._pdb}


# Export list for serialization registry
BIOINFORMATICS_COMPUTE_NODES: list[type[WorkflowNode]] = [
    NtNormalizeNode,
    NtGcContentNode,
    NtReverseComplementNode,
    TranscribeDnaToRnaNode,
    ReverseTranscribeRnaToDnaNode,
    NtCompositionReportNode,
    NtMotifCountNode,
    CodonFrequencyReportNode,
    LongestOrfLengthNode,
    FastaFirstHeaderNode,
    FastaRecordCountNode,
    ProteinPolarFractionNode,
    ProteinTinyFractionNode,
    ProteinNetChargeEstimateNode,
    ProteinHelixFormerFractionNode,
    ProteinBetaBranchedFractionNode,
    ProteinShannonEntropyNode,
    PairHammingDistanceNode,
    PairIdentityPercentNode,
    RecordToJsonStringNode,
    TextTruncateNode,
    SmilesBasicStatsNode,
    OligoTmWallaceNode,
    NtKmerProfileNode,
    ProteinSerThrFractionNode,
    ProteinLysArgRatioNode,
    ProteinCysteineCountNode,
    ProteinAliphaticAromaticRatioNode,
    MergeFourTextNode,
    RegexFindInTextNode,
    LineCountNode,
    JsonParseToRecordNode,
    ConstantStringNode,
    ConstantProteinIdNode,
    ConstantPdbIdNode,
]
