"""Pure protein sequence metrics for workflow nodes and tools (no LangChain)."""

from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise

from bioagents.tools.analysis_tools import AMINO_ACID_WEIGHTS

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def normalize_protein_sequence(raw: str) -> str:
    s = raw.upper().strip()
    s = re.sub(r"\s+", "", s)
    return "".join(c if c in _STANDARD_AA else "X" for c in s)


def molecular_weight_daltons(sequence: str) -> tuple[float, int]:
    """Return (weight Da, length) for a plain one-letter sequence."""
    seq = normalize_protein_sequence(sequence)
    if not seq:
        return 0.0, 0
    weight = 0.0
    for aa in seq:
        if aa in AMINO_ACID_WEIGHTS:
            weight += AMINO_ACID_WEIGHTS[aa]
    water = 18.015 * (len(seq) - 1)
    weight -= water
    return weight, len(seq)


def molecular_weight_report(sequence: str) -> str:
    w, ln = molecular_weight_daltons(sequence)
    return f"Molecular Weight: {w:.2f} Da ({w / 1000:.2f} kDa)\nSequence Length: {ln} amino acids"


def composition_report(sequence: str) -> str:
    """Same text format as analyze_amino_acid_composition tool output."""
    seq = normalize_protein_sequence(sequence)
    if not seq:
        return "Error: Empty sequence provided"
    aa_counts = Counter(seq)
    for char in ["*", "-", "X"]:
        aa_counts.pop(char, None)
    total = sum(aa_counts.values())
    if total == 0:
        return "Error: No standard residues"
    lines = ["Amino Acid Composition Analysis", "=" * 50, f"Total amino acids: {total}\n"]
    sorted_aas = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)
    lines.append(f"{'AA':<4} {'Count':<8} {'Percentage':<12}\n{'-' * 50}")
    for aa, count in sorted_aas:
        pct = (count / total) * 100
        lines.append(f"{aa:<4} {count:<8} {pct:>6.2f}%")
    hydrophobic = sum(aa_counts.get(aa, 0) for aa in ["A", "V", "I", "L", "M", "F", "W", "P"])
    polar = sum(aa_counts.get(aa, 0) for aa in ["S", "T", "N", "Q", "Y", "C"])
    charged = sum(aa_counts.get(aa, 0) for aa in ["K", "R", "H", "D", "E"])
    positive = sum(aa_counts.get(aa, 0) for aa in ["K", "R", "H"])
    negative = sum(aa_counts.get(aa, 0) for aa in ["D", "E"])
    lines.extend(
        [
            "",
            "=" * 50,
            "Biochemical Properties:",
            f"  Hydrophobic residues: {hydrophobic} ({(hydrophobic / total) * 100:.1f}%)",
            f"  Polar residues: {polar} ({(polar / total) * 100:.1f}%)",
            f"  Charged residues: {charged} ({(charged / total) * 100:.1f}%)",
            f"    - Positive: {positive} ({(positive / total) * 100:.1f}%)",
            f"    - Negative: {negative} ({(negative / total) * 100:.1f}%)",
        ]
    )
    return "\n".join(lines)


def isoelectric_report(sequence: str) -> str:
    """Simplified pI text (matches tool spirit)."""
    seq = normalize_protein_sequence(sequence)
    aa_counts = Counter(seq)
    n_lys = aa_counts.get("K", 0)
    n_arg = aa_counts.get("R", 0)
    n_his = aa_counts.get("H", 0)
    n_asp = aa_counts.get("D", 0)
    n_glu = aa_counts.get("E", 0)
    n_cys = aa_counts.get("C", 0)
    n_tyr = aa_counts.get("Y", 0)
    positive_groups = n_lys + n_arg + n_his + 1
    negative_groups = n_asp + n_glu + n_cys + n_tyr + 1
    if positive_groups + negative_groups == 0:
        estimated_pi = 7.0
    else:
        estimated_pi = (
            6.5 + (positive_groups - negative_groups) / (positive_groups + negative_groups) * 4.5
        )
    lines = [
        "Isoelectric Point (pI) Estimation",
        "=" * 50,
        f"Estimated pI: {estimated_pi:.2f}\n",
        "Charged Residues:",
        "  Basic residues (+):",
        f"    Lysine (K): {n_lys}",
        f"    Arginine (R): {n_arg}",
        f"    Histidine (H): {n_his}",
        "  Acidic residues (-):",
        f"    Aspartic acid (D): {n_asp}",
        f"    Glutamic acid (E): {n_glu}",
        "\nNote: Simplified estimation for screening.",
    ]
    return "\n".join(lines)


# Kyte-Doolittle (short scale)
_KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "X": 0.0,
}


def gravy_score(sequence: str) -> float:
    seq = normalize_protein_sequence(sequence)
    if not seq:
        return 0.0
    return sum(_KD.get(aa, 0.0) for aa in seq) / len(seq)


def aliphatic_index_percent(sequence: str) -> float:
    """Very rough aliphatic index proxy (% A+V+I+L)."""
    seq = normalize_protein_sequence(sequence)
    if not seq:
        return 0.0
    aliphatic = sum(1 for aa in seq if aa in "AVIL")
    return 100.0 * aliphatic / len(seq)


def instability_index_rough(sequence: str) -> float:
    """Heuristic 0-100 scale (higher = less stable); not DIWV."""
    seq = normalize_protein_sequence(sequence)
    if len(seq) < 2:
        return 0.0
    # Count dipeptides with charged neighbors as proxy
    unstable = 0
    for a, b in pairwise(seq):
        if a in "DE" and b in "KRH":
            unstable += 1
        if a in "KRH" and b in "DE":
            unstable += 1
    return min(100.0, 40.0 + 20.0 * unstable / max(1, len(seq) - 1))
