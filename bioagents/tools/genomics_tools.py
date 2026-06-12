"""Genomics tools for sequence analysis and BLAST searches."""

import contextlib
import json
import time
import xml.etree.ElementTree as ET  # nosec B405

import requests
from langchain_core.tools import tool

HTTP_TIMEOUT = 30
BLAST_API = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

COMPLEMENT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "a": "t",
    "t": "a",
    "g": "c",
    "c": "g",
    "N": "N",
    "n": "n",
}


@tool
def run_blast_search(
    sequence: str,
    database: str = "nr",
    program: str = "blastp",
    max_hits: int = 10,
) -> str:
    """Run a BLAST search against NCBI databases via the REST API.

    Submits the query and polls for results. This may take 30s-5min depending
    on server load.

    Args:
        sequence: The query sequence (protein or nucleotide).
        database: BLAST database to search (default 'nr'). Options include
                  'nr', 'nt', 'swissprot', 'pdb', 'refseq_protein'.
        program: BLAST program (default 'blastp'). Options: 'blastp', 'blastn',
                 'blastx', 'tblastn', 'tblastx'.
        max_hits: Maximum number of hits to return (default 10, max 50).

    Returns:
        JSON string with hit descriptions, E-values, identities, and accessions.
    """
    try:
        max_hits = min(max(max_hits, 1), 50)
        seq_clean = sequence.strip().replace(" ", "").replace("\n", "")

        put_params = {
            "CMD": "Put",
            "PROGRAM": program,
            "DATABASE": database,
            "QUERY": seq_clean,
            "HITLIST_SIZE": max_hits,
            "FORMAT_TYPE": "XML",
        }
        put_resp = requests.post(BLAST_API, data=put_params, timeout=60)
        put_resp.raise_for_status()

        rid = None
        for line in put_resp.text.split("\n"):
            if "RID = " in line:
                rid = line.split("=")[1].strip()
                break
        if not rid:
            return "Error: Could not submit BLAST job — no RID returned."

        for _attempt in range(60):
            time.sleep(10)
            check_params = {"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"}
            check_resp = requests.get(BLAST_API, params=check_params, timeout=HTTP_TIMEOUT)
            if "Status=READY" in check_resp.text:
                break
            if "Status=FAILED" in check_resp.text:
                return "Error: BLAST search failed on the server."
        else:
            return "Error: BLAST search timed out after 10 minutes."

        get_params = {"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"}
        get_resp = requests.get(BLAST_API, params=get_params, timeout=60)
        get_resp.raise_for_status()

        root = ET.fromstring(get_resp.text)  # nosec B314
        hits = root.findall(".//Hit")
        results = []
        for hit in hits[:max_hits]:
            hit_def = hit.find("Hit_def")
            hit_acc = hit.find("Hit_accession")
            hsp = hit.find(".//Hsp")
            evalue = hsp.find("Hsp_evalue") if hsp is not None else None
            identity = hsp.find("Hsp_identity") if hsp is not None else None
            align_len = hsp.find("Hsp_align-len") if hsp is not None else None

            pct_identity = ""
            if identity is not None and align_len is not None:
                with contextlib.suppress(ValueError, ZeroDivisionError):
                    pct_identity = (
                        f"{(int(identity.text or '0') / int(align_len.text or '1')) * 100:.1f}%"
                    )

            results.append(
                {
                    "description": hit_def.text if hit_def is not None else "",
                    "accession": hit_acc.text if hit_acc is not None else "",
                    "evalue": evalue.text if evalue is not None else "",
                    "identity": pct_identity,
                }
            )

        if not results:
            return "No BLAST hits found for the given sequence."
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error running BLAST search: {e}"


@tool
def parse_fasta_file(file_path: str) -> str:
    """Parse a FASTA file and return sequence information.

    Args:
        file_path: Path to the FASTA file (in sandbox or absolute).

    Returns:
        JSON string with sequence headers, lengths, and first 100 characters.
    """
    try:
        from bioagents.sandbox.sandbox_manager import get_sandbox

        sandbox = get_sandbox()
        content = sandbox.read_file(file_path)

        sequences = []
        current_header = ""
        current_seq: list[str] = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    seq = "".join(current_seq)
                    sequences.append(
                        {
                            "header": current_header,
                            "length": len(seq),
                            "preview": seq[:100],
                        }
                    )
                current_header = line[1:].strip()
                current_seq = []
            elif line:
                current_seq.append(line)

        if current_header:
            seq = "".join(current_seq)
            sequences.append(
                {
                    "header": current_header,
                    "length": len(seq),
                    "preview": seq[:100],
                }
            )

        if not sequences:
            return "No sequences found in the file. Ensure it is in FASTA format."
        return json.dumps(
            {
                "num_sequences": len(sequences),
                "sequences": sequences[:100],
            },
            indent=2,
        )
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error parsing FASTA file: {e}"


@tool
def reverse_complement(sequence: str) -> str:
    """Compute the reverse complement of a DNA sequence.

    Args:
        sequence: DNA sequence (A, T, G, C characters). Case insensitive.

    Returns:
        The reverse complement sequence, or an error message.
    """
    try:
        seq = sequence.strip().replace(" ", "").replace("\n", "")
        invalid = set(seq.upper()) - {"A", "T", "G", "C", "N"}
        if invalid:
            return (
                f"Error: Invalid DNA characters found: {invalid}. Only A, T, G, C, N are allowed."
            )
        rc = "".join(COMPLEMENT.get(base, "N") for base in reversed(seq))
        return rc
    except Exception as e:
        return f"Error computing reverse complement: {e}"


@tool
def translate_dna(dna_sequence: str) -> str:
    """Translate a DNA sequence into a protein sequence using the standard codon table.

    The sequence is translated in reading frame 1 (from the first nucleotide).

    Args:
        dna_sequence: DNA sequence to translate. Must be A/T/G/C characters.

    Returns:
        The translated protein sequence (single-letter amino acids). Stop codons
        are shown as '*'.
    """
    try:
        seq = dna_sequence.strip().replace(" ", "").replace("\n", "").upper()
        invalid = set(seq) - {"A", "T", "G", "C", "N"}
        if invalid:
            return f"Error: Invalid DNA characters: {invalid}"

        protein: list[str] = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if "N" in codon:
                protein.append("X")
            else:
                protein.append(CODON_TABLE.get(codon, "X"))

        return "".join(protein)
    except Exception as e:
        return f"Error translating DNA: {e}"


@tool
def calculate_gc_content(sequence: str) -> str:
    """Calculate the GC content of a DNA or RNA sequence.

    Args:
        sequence: DNA or RNA sequence (A, T/U, G, C characters).

    Returns:
        GC content as a percentage with sequence statistics.
    """
    try:
        seq = sequence.strip().replace(" ", "").replace("\n", "").upper()
        seq = seq.replace("U", "T")
        valid = {"A", "T", "G", "C", "N"}
        invalid = set(seq) - valid
        if invalid:
            return f"Error: Invalid characters found: {invalid}"

        if not seq:
            return "Error: Empty sequence provided."

        gc_count = seq.count("G") + seq.count("C")
        total = len(seq)
        gc_pct = (gc_count / total) * 100

        return json.dumps(
            {
                "gc_content": f"{gc_pct:.2f}%",
                "gc_count": gc_count,
                "at_count": seq.count("A") + seq.count("T"),
                "n_count": seq.count("N"),
                "total_length": total,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error calculating GC content: {e}"


def get_genomics_tools() -> list:
    """Return all genomics tools."""
    return [
        run_blast_search,
        parse_fasta_file,
        reverse_complement,
        translate_dna,
        calculate_gc_content,
    ]
