"""Structural biology tools for protein structure retrieval and analysis.

This module provides tools for:
- Fetching structures from PDB and AlphaFold DB
- Analyzing protein-protein interfaces
- Extracting interaction motifs from complexes
- Computing interface quality metrics
"""

from __future__ import annotations

import json
import logging

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Constants
ALPHAFOLD_DB_API = "https://alphafold.ebi.ac.uk/api"
PDB_RCSB_API = "https://data.rcsb.org/rest/v1"
PDB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"


@tool
def fetch_alphafold_structure(uniprot_id: str) -> str:
    """
    Fetch AlphaFold predicted structure for a UniProt accession.

    Retrieves the AlphaFold DB structure prediction including:
    - PDB format structure
    - Per-residue confidence (pLDDT) scores
    - Predicted Aligned Error (PAE) matrix URL

    Args:
        uniprot_id: UniProt accession (e.g., 'P04637' for human p53)

    Returns:
        JSON string with structure data, PDB URL, and confidence metrics
    """
    try:
        # Get AlphaFold DB entry
        url = f"{ALPHAFOLD_DB_API}/prediction/{uniprot_id}"
        response = requests.get(url, timeout=30)

        if response.status_code == 404:
            return json.dumps(
                {"status": "error", "message": f"No AlphaFold prediction found for {uniprot_id}"}
            )

        response.raise_for_status()
        entries = response.json()

        if not entries:
            return json.dumps(
                {"status": "error", "message": f"No AlphaFold prediction found for {uniprot_id}"}
            )

        # Get the latest entry (v4 preferred)
        entry = entries[0] if isinstance(entries, list) else entries

        result = {
            "status": "success",
            "uniprot_id": uniprot_id,
            "entry_id": entry.get("entryId"),
            "gene": entry.get("gene"),
            "organism": entry.get("organismScientificName"),
            "uniprot_description": entry.get("uniprotDescription"),
            "sequence_length": entry.get("uniprotEnd", 0) - entry.get("uniprotStart", 0) + 1,
            "model_created_date": entry.get("modelCreatedDate"),
            "pdb_url": entry.get("pdbUrl"),
            "cif_url": entry.get("cifUrl"),
            "pae_image_url": entry.get("paeImageUrl"),
            "pae_doc_url": entry.get("paeDocUrl"),
            "confidence_version": entry.get("latestVersion"),
            "global_metric_plddt": entry.get("globalMetricValue"),
        }

        return json.dumps(result, indent=2)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching AlphaFold structure: {e}")
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def fetch_pdb_structure(pdb_id: str) -> str:
    """
    Fetch structure information from RCSB PDB.

    Retrieves comprehensive structure information including:
    - Resolution and experimental method
    - Polymer entities (chains)
    - Ligands and cofactors
    - Associated UniProt IDs

    Args:
        pdb_id: PDB identifier (e.g., '1TUP' for p53 DNA complex)

    Returns:
        JSON string with structure metadata and download URLs
    """
    try:
        # Get structure summary
        url = f"{PDB_RCSB_API}/core/entry/{pdb_id}"
        response = requests.get(url, timeout=30)

        if response.status_code == 404:
            return json.dumps({"status": "error", "message": f"PDB entry {pdb_id} not found"})

        response.raise_for_status()
        entry = response.json()

        # Extract key information
        struct_info = entry.get("struct", {})
        exptl = entry.get("exptl", [{}])[0] if entry.get("exptl") else {}
        refine = entry.get("refine", [{}])[0] if entry.get("refine") else {}

        result = {
            "status": "success",
            "pdb_id": pdb_id.upper(),
            "title": struct_info.get("title"),
            "experimental_method": exptl.get("method"),
            "resolution": refine.get("ls_d_res_high"),
            "r_work": refine.get("ls_R_factor_R_work"),
            "r_free": refine.get("ls_R_factor_R_free"),
            "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date"),
            "release_date": entry.get("rcsb_accession_info", {}).get("initial_release_date"),
            "pdb_url": f"https://files.rcsb.org/download/{pdb_id}.pdb",
            "cif_url": f"https://files.rcsb.org/download/{pdb_id}.cif",
            "fasta_url": f"https://www.rcsb.org/fasta/entry/{pdb_id}",
        }

        return json.dumps(result, indent=2)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching PDB structure: {e}")
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def search_pdb_complexes(uniprot_id: str, limit: int = 10) -> str:
    """
    Search PDB for protein-protein complexes containing the target protein.

    Finds experimentally resolved structures of the target protein
    bound to other proteins, useful for identifying native binding interfaces.

    Args:
        uniprot_id: UniProt accession of the target protein
        limit: Maximum number of results to return

    Returns:
        JSON string with list of PDB complexes containing the target
    """
    try:
        # RCSB search query for complexes
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                            "operator": "exact_match",
                            "value": uniprot_id,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "greater",
                            "value": 1,
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": limit},
                "sort": [{"sort_by": "rcsb_accession_info.deposit_date", "direction": "desc"}],
            },
        }

        response = requests.post(
            PDB_SEARCH_API, json=query, headers={"Content-Type": "application/json"}, timeout=30
        )

        if response.status_code != 200:
            return json.dumps({"status": "error", "message": f"Search failed: {response.text}"})

        search_results = response.json()
        total = search_results.get("total_count", 0)
        entries = search_results.get("result_set", [])

        complexes = []
        for entry in entries:
            pdb_id = entry.get("identifier")
            complexes.append(
                {
                    "pdb_id": pdb_id,
                    "pdb_url": f"https://www.rcsb.org/structure/{pdb_id}",
                    "download_pdb": f"https://files.rcsb.org/download/{pdb_id}.pdb",
                    "download_cif": f"https://files.rcsb.org/download/{pdb_id}.cif",
                }
            )

        return json.dumps(
            {
                "status": "success",
                "uniprot_id": uniprot_id,
                "total_complexes_found": total,
                "returned": len(complexes),
                "complexes": complexes,
            },
            indent=2,
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching PDB: {e}")
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def get_pdb_polymer_entities(pdb_id: str) -> str:
    """
    Get detailed information about all polymer entities (chains) in a PDB structure.

    Retrieves information about each protein/nucleic acid chain including:
    - Chain identifiers
    - Sequence
    - UniProt mappings
    - Molecular weight

    Args:
        pdb_id: PDB identifier

    Returns:
        JSON string with polymer entity details
    """
    try:
        url = f"{PDB_RCSB_API}/core/entry/{pdb_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        entry = response.json()

        # Get polymer entities
        polymer_entities = []
        entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])

        for entity_id in entity_ids:
            entity_url = f"{PDB_RCSB_API}/core/polymer_entity/{pdb_id}/{entity_id}"
            entity_resp = requests.get(entity_url, timeout=30)

            if entity_resp.status_code == 200:
                entity_data = entity_resp.json()

                # Get UniProt references
                uniprot_refs = []
                ref_ids = entity_data.get("rcsb_polymer_entity_container_identifiers", {}).get(
                    "reference_sequence_identifiers", []
                )
                for ref in ref_ids:
                    if ref.get("database_name") == "UniProt":
                        uniprot_refs.append(ref.get("database_accession"))

                entity_info = entity_data.get("entity_poly", {})
                rcsb_entity = entity_data.get("rcsb_polymer_entity", {})

                polymer_entities.append(
                    {
                        "entity_id": entity_id,
                        "type": entity_info.get("type"),
                        "pdbx_strand_id": entity_info.get("pdbx_strand_id"),  # Chain IDs
                        "sequence": entity_info.get("pdbx_seq_one_letter_code_can"),
                        "sequence_length": len(entity_info.get("pdbx_seq_one_letter_code_can", "")),
                        "uniprot_accessions": uniprot_refs,
                        "name": rcsb_entity.get("pdbx_description"),
                        "molecular_weight": rcsb_entity.get("formula_weight"),
                    }
                )

        return json.dumps(
            {
                "status": "success",
                "pdb_id": pdb_id.upper(),
                "polymer_count": len(polymer_entities),
                "entities": polymer_entities,
            },
            indent=2,
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching polymer entities: {e}")
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def download_structure_file(
    pdb_id: str | None = None,
    uniprot_id: str | None = None,
    url: str | None = None,
    output_dir: str = "data",
    file_format: str = "pdb",
) -> str:
    """
    Download a structure file (PDB or CIF format) from PDB or AlphaFold DB.

    Provide EITHER pdb_id, uniprot_id, or url. The file will be saved to output_dir.

    Args:
        pdb_id: PDB ID (e.g., '1YCR') to download from RCSB PDB
        uniprot_id: UniProt ID (e.g., 'P04637') to download from AlphaFold DB
        url: Direct URL to download structure file from
        output_dir: Directory to save the file (default: 'data')
        file_format: File format - 'pdb' or 'cif' (default: 'pdb')

    Returns:
        JSON string with download status and absolute file path
    """
    try:
        from pathlib import Path

        # Determine source and construct URL
        if pdb_id:
            if file_format == "pdb":
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                filename = f"{pdb_id}.pdb"
            else:
                url = f"https://files.rcsb.org/download/{pdb_id}.cif"
                filename = f"{pdb_id}.cif"
            source = "RCSB PDB"
        elif uniprot_id:
            if file_format == "pdb":
                url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
                filename = f"AF-{uniprot_id}-F1-model_v6.pdb"
            else:
                url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.cif"
                filename = f"AF-{uniprot_id}-F1-model_v6.cif"
            source = "AlphaFold DB"
        elif url:
            filename = url.split("/")[-1]
            source = "Custom URL"
        else:
            return json.dumps(
                {"status": "error", "message": "Must provide either pdb_id, uniprot_id, or url"}
            )

        # Download the file
        response = requests.get(url, timeout=60)

        # If PDB fails with 404, try CIF format
        if response.status_code == 404 and file_format == "pdb":
            logger.info("PDB not found, trying CIF format")
            if pdb_id:
                url = f"https://files.rcsb.org/download/{pdb_id}.cif"
                filename = f"{pdb_id}.cif"
            elif uniprot_id:
                url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.cif"
                filename = f"AF-{uniprot_id}-F1-model_v6.cif"
            response = requests.get(url, timeout=60)

        response.raise_for_status()

        # Save to output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / filename
        output_file.write_bytes(response.content)

        return json.dumps(
            {
                "status": "success",
                "file_path": str(output_file.absolute()),
                "filename": filename,
                "file_size_bytes": len(response.content),
                "source": source,
                "url": url,
            },
            indent=2,
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading structure: {e}")
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def analyze_interface_contacts(
    structure_path: str, chain1: str, chain2: str, distance_cutoff: float = 8.0
) -> str:
    """
    Analyze the interface between two chains in a PDB structure.

    Identifies interface residues based on distance cutoff between
    atoms of different chains. Useful for defining binding hotspots.

    Note: Requires BioPython to be installed.

    Args:
        structure_path: Path to PDB/CIF structure file
        chain1: First chain identifier
        chain2: Second chain identifier
        distance_cutoff: Distance cutoff in Angstroms for interface contacts

    Returns:
        JSON string with interface residues and contact analysis
    """
    try:
        from pathlib import Path

        # Check if BioPython is available
        try:
            from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser
        except ImportError:
            return json.dumps(
                {
                    "status": "error",
                    "message": "BioPython is required for interface analysis. Install with: pip install biopython",
                }
            )

        path = Path(structure_path)
        if not path.exists():
            return json.dumps(
                {"status": "error", "message": f"Structure file not found: {structure_path}"}
            )

        # Parse structure
        if path.suffix.lower() in [".cif", ".mmcif"]:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        structure = parser.get_structure("structure", str(path))
        model = structure[0]

        if chain1 not in model or chain2 not in model:
            available = [c.id for c in model.get_chains()]
            return json.dumps(
                {"status": "error", "message": f"Chain(s) not found. Available chains: {available}"}
            )

        # Get atoms from both chains
        chain1_atoms = list(model[chain1].get_atoms())
        chain2_atoms = list(model[chain2].get_atoms())

        # Build neighbor search for chain2
        ns = NeighborSearch(chain2_atoms)

        # Find interface residues
        interface_residues_chain1 = set()
        interface_residues_chain2 = set()
        contacts = []

        for atom1 in chain1_atoms:
            nearby = ns.search(atom1.coord, distance_cutoff)
            for atom2 in nearby:
                res1 = atom1.get_parent()
                res2 = atom2.get_parent()

                interface_residues_chain1.add((res1.get_resname(), res1.id[1]))
                interface_residues_chain2.add((res2.get_resname(), res2.id[1]))

                # Record close contacts (< 4 Å)
                dist = atom1 - atom2
                if dist < 4.0:
                    contacts.append(
                        {
                            "atom1": f"{chain1}:{res1.get_resname()}{res1.id[1]}:{atom1.name}",
                            "atom2": f"{chain2}:{res2.get_resname()}{res2.id[1]}:{atom2.name}",
                            "distance": round(
                                float(dist), 2
                            ),  # Convert numpy float to Python float
                        }
                    )

        # Format interface residues
        chain1_interface = sorted(
            [{"residue": r[0], "position": r[1]} for r in interface_residues_chain1],
            key=lambda x: x["position"],
        )

        chain2_interface = sorted(
            [{"residue": r[0], "position": r[1]} for r in interface_residues_chain2],
            key=lambda x: x["position"],
        )

        return json.dumps(
            {
                "status": "success",
                "structure_file": structure_path,
                "chain1": chain1,
                "chain2": chain2,
                "distance_cutoff": distance_cutoff,
                "interface_residues_chain1": chain1_interface,
                "interface_residues_chain2": chain2_interface,
                "interface_size_chain1": len(chain1_interface),
                "interface_size_chain2": len(chain2_interface),
                "close_contacts": contacts[:50],  # Limit output
                "total_close_contacts": len(contacts),
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error analyzing interface: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def compute_interface_metrics(pae_json_path: str, chain1_range: str, chain2_range: str) -> str:
    """
    Compute interface quality metrics from AlphaFold PAE matrix.

    Calculates interface predicted aligned error (ipAE) and related metrics
    for evaluating the confidence of a predicted protein-protein interface.

    Args:
        pae_json_path: Path to AlphaFold PAE JSON file
        chain1_range: Residue range for chain 1 (e.g., "1-100")
        chain2_range: Residue range for chain 2 (e.g., "101-200")

    Returns:
        JSON string with interface quality metrics including ipAE
    """
    try:
        from pathlib import Path

        import numpy as np

        path = Path(pae_json_path)
        if not path.exists():
            return json.dumps(
                {"status": "error", "message": f"PAE file not found: {pae_json_path}"}
            )

        # Load PAE matrix
        pae_data = json.loads(path.read_text())

        # Handle different PAE formats
        if isinstance(pae_data, list):
            pae_matrix = np.array(pae_data[0].get("predicted_aligned_error", []))
        elif "predicted_aligned_error" in pae_data:
            pae_matrix = np.array(pae_data["predicted_aligned_error"])
        elif "pae" in pae_data:
            pae_matrix = np.array(pae_data["pae"])
        else:
            return json.dumps({"status": "error", "message": "Could not find PAE data in file"})

        # Parse residue ranges
        def parse_range(range_str: str) -> tuple[int, int]:
            parts = range_str.split("-")
            return int(parts[0]) - 1, int(parts[1])  # 0-indexed

        c1_start, c1_end = parse_range(chain1_range)
        c2_start, c2_end = parse_range(chain2_range)

        # Extract interface PAE blocks
        # ipAE: PAE between chain1 and chain2
        interface_pae_1to2 = pae_matrix[c1_start:c1_end, c2_start:c2_end]
        interface_pae_2to1 = pae_matrix[c2_start:c2_end, c1_start:c1_end]

        # Compute metrics
        ipae = (np.mean(interface_pae_1to2) + np.mean(interface_pae_2to1)) / 2
        ipae_min = min(np.min(interface_pae_1to2), np.min(interface_pae_2to1))
        ipae_max = max(np.max(interface_pae_1to2), np.max(interface_pae_2to1))

        # Intra-chain PAE for reference
        chain1_pae = pae_matrix[c1_start:c1_end, c1_start:c1_end]
        chain2_pae = pae_matrix[c2_start:c2_end, c2_start:c2_end]

        return json.dumps(
            {
                "status": "success",
                "chain1_range": chain1_range,
                "chain2_range": chain2_range,
                "metrics": {
                    "ipAE": round(float(ipae), 3),
                    "ipAE_min": round(float(ipae_min), 3),
                    "ipAE_max": round(float(ipae_max), 3),
                    "chain1_mean_pAE": round(float(np.mean(chain1_pae)), 3),
                    "chain2_mean_pAE": round(float(np.mean(chain2_pae)), 3),
                },
                "interpretation": {
                    "ipAE < 5": "High confidence interface",
                    "ipAE 5-10": "Medium confidence interface",
                    "ipAE > 10": "Low confidence interface",
                },
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error computing interface metrics: {e}")
        return json.dumps({"status": "error", "message": str(e)})


def get_structural_tools():
    """Return list of all structural biology tools."""
    return [
        fetch_alphafold_structure,
        fetch_pdb_structure,
        search_pdb_complexes,
        get_pdb_polymer_entities,
        download_structure_file,
        analyze_interface_contacts,
        compute_interface_metrics,
    ]
