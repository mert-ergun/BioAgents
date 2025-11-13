"""Analysis tools for protein sequence analysis."""

from collections import Counter

from langchain_core.tools import tool

# Standard amino acid molecular weights (average isotopic composition)
AMINO_ACID_WEIGHTS = {
    "A": 89.09,
    "R": 174.20,
    "N": 132.12,
    "D": 133.10,
    "C": 121.15,
    "E": 147.13,
    "Q": 146.15,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "L": 131.17,
    "K": 146.19,
    "M": 149.21,
    "F": 165.19,
    "P": 115.13,
    "S": 105.09,
    "T": 119.12,
    "W": 204.23,
    "Y": 181.19,
    "V": 117.15,
}


def parse_fasta_sequence(fasta_text: str) -> str:
    """
    Extract the amino acid sequence from FASTA format.

    Args:
        fasta_text: FASTA formatted text

    Returns:
        The amino acid sequence (without header)
    """
    lines = fasta_text.strip().split("\n")
    # Skip the header line (starts with >)
    sequence_lines = [line for line in lines if not line.startswith(">")]
    return "".join(sequence_lines).upper()


@tool
def calculate_molecular_weight(fasta_sequence: str) -> str:
    """
    Calculate the molecular weight of a protein from its FASTA sequence.

    Args:
        fasta_sequence: The protein sequence in FASTA format

    Returns:
        A string describing the molecular weight in Daltons (Da)
    """
    try:
        sequence = parse_fasta_sequence(fasta_sequence)

        weight = 0.0
        unknown_aas = []

        for aa in sequence:
            if aa in AMINO_ACID_WEIGHTS:
                weight += AMINO_ACID_WEIGHTS[aa]
            elif aa not in ["*", "-"]:  # Ignore stop codons and gaps
                unknown_aas.append(aa)

        # Subtract water molecules (peptide bonds)
        # Each peptide bond releases one water molecule (18.015 Da)
        water_weight = 18.015 * (len(sequence) - 1)
        weight -= water_weight

        result = f"Molecular Weight: {weight:.2f} Da ({weight / 1000:.2f} kDa)\n"
        result += f"Sequence Length: {len(sequence)} amino acids"

        if unknown_aas:
            result += f"\nWarning: Unknown amino acids encountered: {set(unknown_aas)}"

        return result

    except Exception as e:
        return f"Error calculating molecular weight: {e!s}"


@tool
def analyze_amino_acid_composition(fasta_sequence: str) -> str:
    """
    Analyze the amino acid composition of a protein sequence.

    Args:
        fasta_sequence: The protein sequence in FASTA format

    Returns:
        A detailed report of amino acid composition and properties
    """
    try:
        sequence = parse_fasta_sequence(fasta_sequence)

        if not sequence:
            return "Error: Empty sequence provided"

        aa_counts = Counter(sequence)

        # Remove non-standard characters
        for char in ["*", "-"]:
            aa_counts.pop(char, None)

        total = sum(aa_counts.values())

        result = "Amino Acid Composition Analysis\n"
        result += f"{'=' * 50}\n"
        result += f"Total amino acids: {total}\n\n"

        sorted_aas = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)

        result += f"{'AA':<4} {'Count':<8} {'Percentage':<12}\n"
        result += f"{'-' * 50}\n"

        for aa, count in sorted_aas:
            percentage = (count / total) * 100
            result += f"{aa:<4} {count:<8} {percentage:>6.2f}%\n"

        # Calculate biochemical properties
        hydrophobic = sum(aa_counts.get(aa, 0) for aa in ["A", "V", "I", "L", "M", "F", "W", "P"])
        polar = sum(aa_counts.get(aa, 0) for aa in ["S", "T", "N", "Q", "Y", "C"])
        charged = sum(aa_counts.get(aa, 0) for aa in ["K", "R", "H", "D", "E"])
        positive = sum(aa_counts.get(aa, 0) for aa in ["K", "R", "H"])
        negative = sum(aa_counts.get(aa, 0) for aa in ["D", "E"])

        result += f"\n{'=' * 50}\n"
        result += "Biochemical Properties:\n"
        result += f"  Hydrophobic residues: {hydrophobic} ({(hydrophobic / total) * 100:.1f}%)\n"
        result += f"  Polar residues: {polar} ({(polar / total) * 100:.1f}%)\n"
        result += f"  Charged residues: {charged} ({(charged / total) * 100:.1f}%)\n"
        result += f"    - Positive: {positive} ({(positive / total) * 100:.1f}%)\n"
        result += f"    - Negative: {negative} ({(negative / total) * 100:.1f}%)\n"

        return result

    except Exception as e:
        return f"Error analyzing composition: {e!s}"


@tool
def calculate_isoelectric_point(fasta_sequence: str) -> str:
    """
    Estimate the isoelectric point (pI) of a protein.

    The pI is the pH at which the protein has no net charge.

    Args:
        fasta_sequence: The protein sequence in FASTA format

    Returns:
        The estimated isoelectric point
    """
    try:
        sequence = parse_fasta_sequence(fasta_sequence)

        aa_counts = Counter(sequence)

        n_lys = aa_counts.get("K", 0)  # pKa ~10.5
        n_arg = aa_counts.get("R", 0)  # pKa ~12.5
        n_his = aa_counts.get("H", 0)  # pKa ~6.0
        n_asp = aa_counts.get("D", 0)  # pKa ~3.9
        n_glu = aa_counts.get("E", 0)  # pKa ~4.2
        n_cys = aa_counts.get("C", 0)  # pKa ~8.3
        n_tyr = aa_counts.get("Y", 0)  # pKa ~10.1

        # Simple estimation (this is a simplified calculation)
        positive_groups = n_lys + n_arg + n_his + 1  # +1 for N-terminus
        negative_groups = n_asp + n_glu + n_cys + n_tyr + 1  # +1 for C-terminus

        # Rough estimate based on charged residues
        if positive_groups + negative_groups == 0:
            estimated_pi = 7.0
        else:
            # Weighted average approach (simplified)
            estimated_pi = (
                6.5
                + (positive_groups - negative_groups) / (positive_groups + negative_groups) * 4.5
            )

        result = "Isoelectric Point (pI) Estimation\n"
        result += f"{'=' * 50}\n"
        result += f"Estimated pI: {estimated_pi:.2f}\n\n"
        result += "Charged Residues:\n"
        result += "  Basic residues (+):\n"
        result += f"    Lysine (K): {n_lys}\n"
        result += f"    Arginine (R): {n_arg}\n"
        result += f"    Histidine (H): {n_his}\n"
        result += "  Acidic residues (-):\n"
        result += f"    Aspartic acid (D): {n_asp}\n"
        result += f"    Glutamic acid (E): {n_glu}\n"
        result += "\nNote: This is a simplified estimation. For accurate pI calculation, "
        result += "use specialized tools that account for all ionizable groups and their microenvironments."

        return result

    except Exception as e:
        return f"Error calculating isoelectric point: {e!s}"
