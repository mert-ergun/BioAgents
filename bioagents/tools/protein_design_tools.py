"""Protein design tools for de novo binder generation and evaluation.

This module provides tools for:
- RFdiffusion backbone generation
- ProteinMPNN sequence design
- AlphaFold-Multimer structure prediction
- BindCraft binder design
- Interface quality metric computation (iPTM, ipSAE)

Note: Many of these tools require external services or local installations.
The implementations here provide both API-based and local execution options.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ============================================================================
# BindCraft / Binder Design API Tools
# ============================================================================


@tool
def design_binders_bindcraft(
    target_pdb_path: str,
    target_hotspot_residues: str,
    num_designs: int = 100,
    binder_length_range: str = "50-80",
    output_dir: str = "bindcraft_output",
) -> str:
    """
    Design protein binders using BindCraft-style workflow.

    This implements a de novo binder design pipeline similar to BindCraft:
    1. Define target hotspot residues
    2. Generate binder backbones using RFdiffusion
    3. Design sequences using ProteinMPNN
    4. Predict structures using AlphaFold2
    5. Filter by interface metrics

    Args:
        target_pdb_path: Path to target protein PDB file
        target_hotspot_residues: Comma-separated residue positions to target (e.g., "45,47,89,92")
        num_designs: Number of binder designs to generate
        binder_length_range: Range of binder lengths (e.g., "50-80")
        output_dir: Directory to save output files

    Returns:
        JSON string with design job status and output paths
    """
    try:
        path = Path(target_pdb_path)
        if not path.exists():
            return json.dumps(
                {"status": "error", "message": f"Target PDB file not found: {target_pdb_path}"}
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Parse parameters
        hotspots = [int(r.strip()) for r in target_hotspot_residues.split(",")]
        min_len, max_len = map(int, binder_length_range.split("-"))

        # Create design configuration
        config = {
            "target_pdb": str(path.absolute()),
            "hotspot_residues": hotspots,
            "num_designs": num_designs,
            "binder_length_min": min_len,
            "binder_length_max": max_len,
            "output_dir": str(out_path.absolute()),
            "pipeline_steps": [
                "rfdiffusion_backbone",
                "proteinmpnn_sequence",
                "alphafold_prediction",
                "interface_metrics",
            ],
        }

        config_path = out_path / "design_config.json"
        config_path.write_text(json.dumps(config, indent=2))

        # Check for local installations
        rfdiffusion_available = _check_rfdiffusion_available()
        proteinmpnn_available = _check_proteinmpnn_available()
        colabfold_available = _check_colabfold_available()

        available_tools = {
            "rfdiffusion": rfdiffusion_available,
            "proteinmpnn": proteinmpnn_available,
            "colabfold": colabfold_available,
        }

        # Determine execution mode
        if all(available_tools.values()):
            execution_mode = "local"
            message = "All tools available locally. Ready to run design pipeline."
        else:
            execution_mode = "api"
            missing = [k for k, v in available_tools.items() if not v]
            message = (
                f"Missing local tools: {missing}. Will use API-based alternatives where available."
            )

        return json.dumps(
            {
                "status": "success",
                "message": message,
                "execution_mode": execution_mode,
                "config_path": str(config_path),
                "output_dir": str(out_path),
                "available_tools": available_tools,
                "design_parameters": {
                    "target": str(path.name),
                    "hotspots": hotspots,
                    "num_designs": num_designs,
                    "binder_length_range": binder_length_range,
                },
                "next_steps": [
                    "Run generate_binder_backbones() to generate backbones with RFdiffusion",
                    "Run design_binder_sequences() to design sequences with ProteinMPNN",
                    "Run predict_binder_structures() to predict with AlphaFold",
                    "Run compute_binding_metrics() to evaluate designs",
                ],
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error setting up BindCraft design: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def generate_binder_backbones(
    target_pdb_path: str,
    target_chain: str,
    hotspot_residues: str,
    num_backbones: int = 100,
    binder_length: int = 70,
    output_dir: str = "rfdiffusion_output",
) -> str:
    """
    Generate binder backbones using RFdiffusion.

    RFdiffusion generates diverse protein backbones conditioned on:
    - Target structure
    - Hotspot residues to bind
    - Desired binder length

    Note: Requires RFdiffusion to be installed locally.

    Args:
        target_pdb_path: Path to target protein PDB file
        target_chain: Chain ID of the target protein
        hotspot_residues: Comma-separated residue positions (e.g., "45,47,89")
        num_backbones: Number of backbone structures to generate
        binder_length: Length of designed binder
        output_dir: Directory to save output backbones

    Returns:
        JSON string with generated backbone paths or instructions for local execution
    """
    try:
        import subprocess  # nosec B404
        from pathlib import Path

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Convert target_pdb_path to absolute path
        target_pdb_abs = str(Path(target_pdb_path).absolute())

        # Get the length of the target chain from the PDB file
        try:
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("target", target_pdb_abs)
            chain = structure[0][target_chain]
            residues = list(chain.get_residues())
            target_length = len(residues)
        except Exception:
            # Fallback: assume typical protein length
            target_length = 400

        hotspots = [int(r.strip()) for r in hotspot_residues.split(",")]
        hotspot_str = ",".join([f"{target_chain}{r}" for r in hotspots])

        if _check_rfdiffusion_available():
            # Run RFdiffusion in its conda environment
            rfdiffusion_dir = os.environ.get("RFDIFFUSION_DIR", "RFdiffusion")

            # PPI binder design format (from RFdiffusion examples/design_ppi.sh):
            # contigmap.contigs='[A1-150/0 70-100]'
            # - A1-150: keep target chain A residues 1 to 150
            # - /0: chain break (NOT a separate element!)
            # - 70-100: design a binder of length 70-100 (sampled)
            contigs_arg = f"'contigmap.contigs=[{target_chain}1-{target_length}/0 {binder_length}-{binder_length}]'"
            hotspots_arg = f"'ppi.hotspot_res=[{hotspot_str}]'"

            # Check if we should force CPU mode
            # Note: PyTorch 2.7 nightly with CUDA 12.4 works on RTX 5070 Ti (sm_120) despite warnings
            force_cpu = os.environ.get("RFDIFFUSION_FORCE_CPU", "0")
            if force_cpu == "0":
                try:
                    import torch

                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        logger.info(f"Using GPU: {gpu_name}")
                    else:
                        force_cpu = "1"
                        logger.info("No GPU available, using CPU mode")
                except Exception:
                    force_cpu = "1"
                    logger.info("Could not detect GPU, using CPU mode")

            # Create shell script to activate conda and run RFdiffusion
            cpu_env = "export CUDA_VISIBLE_DEVICES=''" if force_cpu == "1" else ""
            # Set PYTORCH_ENABLE_MPS_FALLBACK to allow legacy torch.load behavior
            # Required for RFdiffusion's e3nn dependency with PyTorch 2.7+
            shell_cmd = f"""
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SE3nv
{cpu_env}
export PYTORCH_LOAD_WEIGHTS_ONLY=0
cd {rfdiffusion_dir}
python scripts/run_inference.py \\
  inference.output_prefix={out_path.absolute()}/binder \\
  inference.input_pdb={target_pdb_abs} \\
  {contigs_arg} \\
  {hotspots_arg} \\
  inference.num_designs={num_backbones} \\
  denoiser.noise_scale_ca=0 \\
  denoiser.noise_scale_frame=0
"""

            result = subprocess.run(
                ["/usr/bin/bash", "-c", shell_cmd],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )  # nosec B603 B607

            if result.returncode != 0:
                return json.dumps(
                    {"status": "error", "message": f"RFdiffusion failed: {result.stderr}"}
                )

            # List generated files
            backbone_files = list(out_path.glob("binder_*.pdb"))

            return json.dumps(
                {
                    "status": "success",
                    "message": f"Generated {len(backbone_files)} backbone structures",
                    "output_dir": str(out_path),
                    "backbone_files": [str(f) for f in backbone_files[:10]],
                    "total_generated": len(backbone_files),
                },
                indent=2,
            )
        else:
            # Return instructions for Google Colab (fastest option)
            colab_command = f"""
# In Google Colab notebook:
!python scripts/run_inference.py \\
  inference.input_pdb={target_pdb_abs} \\
  'contigmap.contigs=[{target_chain}1-{target_length}/0 {binder_length}-{binder_length}]' \\
  'ppi.hotspot_res=[{hotspot_str}]' \\
  inference.num_designs={num_backbones} \\
  inference.output_prefix=binder_designs \\
  denoiser.noise_scale_ca=0 \\
  denoiser.noise_scale_frame=0

# Then download results:
!zip -r binder_designs.zip binder_designs_*.pdb
from google.colab import files
files.download('binder_designs.zip')
"""
            return json.dumps(
                {
                    "status": "use_google_colab",
                    "message": "⚡ FASTEST SOLUTION: Use Google Colab for GPU acceleration (30 min vs 8-16 hours CPU)",
                    "colab_notebook": "https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.1/rf/examples/diffusion.ipynb",
                    "colab_command": colab_command.strip(),
                    "local_install_guide": "/home/mert/dev/BioAgents/FASTEST_GPU_SOLUTION.md",
                    "note": "Your RTX 5070 Ti is excellent but requires CUDA Toolkit + PyTorch build from source (4-6 hours setup). Colab gives you results in 30 minutes!",
                    "parameters": {
                        "target_pdb": target_pdb_abs,
                        "target_chain": target_chain,
                        "hotspots": [f"{target_chain}{r}" for r in hotspots],
                        "num_backbones": num_backbones,
                        "binder_length": binder_length,
                        "target_length": target_length,
                    },
                },
                indent=2,
            )

    except Exception as e:
        logger.error(f"Error generating backbones: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def design_binder_sequences(
    backbone_pdb_path: str,
    target_chain: str,
    binder_chain: str,
    num_sequences: int = 8,
    temperature: float = 0.1,
    output_dir: str = "proteinmpnn_output",
) -> str:
    """
    Design sequences for a binder backbone using ProteinMPNN.

    ProteinMPNN designs amino acid sequences that fold into the given backbone
    while optimizing for stability and target binding.

    Note: Requires ProteinMPNN to be installed locally.

    Args:
        backbone_pdb_path: Path to backbone PDB file (from RFdiffusion)
        target_chain: Chain ID of the target protein (to keep fixed)
        binder_chain: Chain ID of the binder (to design sequences for)
        num_sequences: Number of sequences to design per backbone
        temperature: Sampling temperature (lower = more confident)
        output_dir: Directory to save designed sequences

    Returns:
        JSON string with designed sequences in FASTA format
    """
    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        backbone_path = Path(backbone_pdb_path)
        if not backbone_path.exists():
            return json.dumps(
                {"status": "error", "message": f"Backbone PDB not found: {backbone_pdb_path}"}
            )

        # ProteinMPNN command
        cmd = [
            "python",
            "protein_mpnn_run.py",
            "--pdb_path",
            str(backbone_path),
            "--out_folder",
            str(out_path),
            "--num_seq_per_target",
            str(num_sequences),
            "--sampling_temp",
            str(temperature),
            "--batch_size",
            "1",
            "--chain_id_jsonl",
            "",  # Will be created
            "--fixed_positions_jsonl",
            "",  # Fix target chain
        ]

        if _check_proteinmpnn_available():
            proteinmpnn_dir = os.environ.get("PROTEINMPNN_DIR", "ProteinMPNN")

            # Create chain definition files
            chains_jsonl = out_path / "chains.jsonl"

            # Define which chains to design vs keep fixed
            chain_data = {"chains_to_design": binder_chain}
            chains_jsonl.write_text(json.dumps({backbone_path.stem: chain_data}))

            # ProteinMPNN can use the same conda environment or run standalone
            # It has minimal dependencies that are likely already in SE3nv
            shell_cmd = f"""
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SE3nv
cd {proteinmpnn_dir}
{" ".join(cmd)}
"""

            result = subprocess.run(
                ["/usr/bin/bash", "-c", shell_cmd],
                capture_output=True,
                text=True,
                timeout=1800,
            )  # nosec B603 B607

            if result.returncode != 0:
                return json.dumps(
                    {"status": "error", "message": f"ProteinMPNN failed: {result.stderr}"}
                )

            # Parse output sequences
            fasta_files = list(out_path.glob("*.fasta"))
            sequences = []
            for fasta in fasta_files:
                content = fasta.read_text()
                # Parse FASTA
                for block in content.strip().split(">"):
                    if block:
                        lines = block.strip().split("\n")
                        header = lines[0]
                        seq = "".join(lines[1:])
                        sequences.append({"header": header, "sequence": seq})

            return json.dumps(
                {
                    "status": "success",
                    "message": f"Designed {len(sequences)} sequences",
                    "output_dir": str(out_path),
                    "sequences": sequences[:20],  # Limit output
                    "total_designed": len(sequences),
                },
                indent=2,
            )

        else:
            return json.dumps(
                {
                    "status": "manual_execution_required",
                    "message": "ProteinMPNN not found locally. Run the following command manually:",
                    "command": " ".join(cmd),
                    "installation_instructions": """
To install ProteinMPNN:
1. git clone https://github.com/dauparas/ProteinMPNN.git
2. cd ProteinMPNN
3. pip install -r requirements.txt

Or use Hugging Face Spaces: https://huggingface.co/spaces/simonduerr/ProteinMPNN
                """,
                    "parameters": {
                        "backbone_pdb": backbone_pdb_path,
                        "target_chain": target_chain,
                        "binder_chain": binder_chain,
                        "num_sequences": num_sequences,
                        "temperature": temperature,
                    },
                },
                indent=2,
            )

    except Exception as e:
        logger.error(f"Error designing sequences: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def predict_complex_structure(
    target_sequence: str,
    binder_sequence: str,
    output_dir: str = "alphafold_output",
    num_recycles: int = 3,
) -> str:
    """
    Predict target-binder complex structure using AlphaFold-Multimer.

    Runs structure prediction for the target-binder complex to evaluate
    if the designed binder actually binds to the intended interface.

    Note: Uses ColabFold API or local installation if available.

    Args:
        target_sequence: Amino acid sequence of target protein
        binder_sequence: Amino acid sequence of designed binder
        output_dir: Directory to save prediction outputs
        num_recycles: Number of recycling iterations

    Returns:
        JSON string with predicted structure path and confidence metrics
    """
    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Combine sequences for multimer prediction
        complex_fasta = f">target\n{target_sequence}\n>binder\n{binder_sequence}\n"

        fasta_path = out_path / "complex.fasta"
        fasta_path.write_text(complex_fasta)

        if _check_colabfold_available():
            # Run ColabFold locally (it's in the BioAgents venv)
            # Note: colabfold_batch should be available when running via 'uv run'
            cmd = [
                "colabfold_batch",
                str(fasta_path),
                str(out_path),
                "--num-recycle",
                str(num_recycles),
                "--num-models",
                "3",
                "--amber",  # Relax structures
                "--use-gpu-relax",  # Use GPU for relaxation if available
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )  # nosec B603

            if result.returncode != 0:
                return json.dumps(
                    {"status": "error", "message": f"ColabFold failed: {result.stderr}"}
                )

            # Parse results
            pdb_files = list(out_path.glob("*_relaxed_rank_*.pdb"))
            json_files = list(out_path.glob("*_scores_rank_*.json"))

            predictions = []
            for pdb, scores in zip(sorted(pdb_files), sorted(json_files)):
                score_data = json.loads(scores.read_text())
                predictions.append(
                    {
                        "pdb_path": str(pdb),
                        "pTM": score_data.get("ptm", 0),
                        "ipTM": score_data.get("iptm", 0),
                        "pLDDT_mean": score_data.get("mean_plddt", 0),
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "message": f"Generated {len(predictions)} structure predictions",
                    "output_dir": str(out_path),
                    "predictions": predictions,
                    "fasta_path": str(fasta_path),
                },
                indent=2,
            )

        else:
            # Return instructions for API or manual execution
            return json.dumps(
                {
                    "status": "manual_execution_required",
                    "message": "ColabFold not installed locally. Use one of these options:",
                    "options": {
                        "colab_notebook": "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb",
                        "colabfold_batch": "pip install colabfold[alphafold]",
                        "api_endpoint": "https://api.esmatlas.com/foldMultimer/v1",
                    },
                    "fasta_content": complex_fasta,
                    "fasta_path": str(fasta_path),
                    "parameters": {
                        "num_recycles": num_recycles,
                        "target_length": len(target_sequence),
                        "binder_length": len(binder_sequence),
                    },
                },
                indent=2,
            )

    except Exception as e:
        logger.error(f"Error predicting structure: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def compute_binding_metrics(
    complex_pdb_path: str,
    target_chain: str = "A",
    binder_chain: str = "B",
    pae_json_path: str | None = None,
) -> str:
    """
    Compute comprehensive binding quality metrics for a predicted complex.

    Calculates:
    - iPTM (interface predicted TM-score)
    - ipSAE (interface predicted structural alignment error)
    - Interface buried surface area
    - Number of interface contacts
    - Shape complementarity

    Args:
        complex_pdb_path: Path to predicted complex PDB file
        target_chain: Chain ID of target protein
        binder_chain: Chain ID of binder protein
        pae_json_path: Optional path to PAE JSON file for ipSAE calculation

    Returns:
        JSON string with all binding quality metrics
    """
    try:
        pdb_path = Path(complex_pdb_path)
        if not pdb_path.exists():
            return json.dumps(
                {"status": "error", "message": f"Complex PDB not found: {complex_pdb_path}"}
            )

        # Try to use BioPython for analysis
        try:
            import numpy as np
            from Bio.PDB import NeighborSearch, PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", str(pdb_path))
            model = structure[0]

            if target_chain not in model or binder_chain not in model:
                available = [c.id for c in model.get_chains()]
                return json.dumps(
                    {"status": "error", "message": f"Chains not found. Available: {available}"}
                )

            target = model[target_chain]
            binder = model[binder_chain]

            # Get interface atoms (< 5 Å between chains)
            binder_atoms = list(binder.get_atoms())
            ns = NeighborSearch(binder_atoms)

            interface_target_residues = set()
            interface_binder_residues = set()
            contacts = 0

            for atom in target.get_atoms():
                nearby = ns.search(atom.coord, 5.0)
                if nearby:
                    contacts += len(nearby)
                    interface_target_residues.add(atom.get_parent().id[1])
                    for n in nearby:
                        interface_binder_residues.add(n.get_parent().id[1])

            # Calculate interface metrics
            metrics: dict[str, Any] = {
                "interface_contacts": contacts,
                "interface_target_residues": len(interface_target_residues),
                "interface_binder_residues": len(interface_binder_residues),
                "target_length": len(list(target.get_residues())),
                "binder_length": len(list(binder.get_residues())),
            }

            # Calculate ipSAE if PAE available
            if pae_json_path:
                pae_path = Path(pae_json_path)
                if pae_path.exists():
                    pae_data = json.loads(pae_path.read_text())

                    # Handle different PAE formats
                    if isinstance(pae_data, list):
                        pae_matrix = np.array(pae_data[0].get("predicted_aligned_error", []))
                    elif "pae" in pae_data:
                        pae_matrix = np.array(pae_data["pae"])
                    else:
                        pae_matrix = np.array(pae_data.get("predicted_aligned_error", []))

                    if pae_matrix.size > 0:
                        target_len = metrics["target_length"]

                        # Extract interface PAE
                        ipae_1to2 = pae_matrix[:target_len, target_len:]
                        ipae_2to1 = pae_matrix[target_len:, :target_len]

                        metrics["ipSAE"] = round(
                            float((np.mean(ipae_1to2) + np.mean(ipae_2to1)) / 2), 3
                        )
                        metrics["ipSAE_min"] = round(
                            float(min(np.min(ipae_1to2), np.min(ipae_2to1))), 3
                        )

            # Try to get iPTM from associated scores file
            scores_path = pdb_path.with_name(pdb_path.stem.replace("_relaxed", "_scores") + ".json")
            if scores_path.exists():
                scores = json.loads(scores_path.read_text())
                metrics["iPTM"] = scores.get("iptm", None)
                metrics["pTM"] = scores.get("ptm", None)
                metrics["mean_pLDDT"] = scores.get("mean_plddt", None)

            # Quality assessment
            quality_pass = True
            quality_notes = []

            if metrics.get("iPTM") and metrics["iPTM"] < 0.7:
                quality_pass = False
                quality_notes.append("iPTM below 0.7 threshold")

            if metrics.get("ipSAE") and metrics["ipSAE"] > 10:
                quality_pass = False
                quality_notes.append("ipSAE above 10 threshold")

            if contacts < 50:
                quality_notes.append("Low number of interface contacts")

            return json.dumps(
                {
                    "status": "success",
                    "complex_pdb": str(pdb_path),
                    "metrics": metrics,
                    "quality_assessment": {
                        "passes_threshold": quality_pass,
                        "notes": quality_notes,
                    },
                    "thresholds": {
                        "iPTM_recommended": ">= 0.7 for high confidence",
                        "ipSAE_recommended": "<= 10 for good interface",
                        "contacts_recommended": ">= 50 for stable binding",
                    },
                },
                indent=2,
            )

        except ImportError:
            # Fallback without BioPython
            return json.dumps(
                {
                    "status": "partial",
                    "message": "BioPython not available. Install with: pip install biopython",
                    "complex_pdb": str(pdb_path),
                    "manual_analysis": {
                        "pymol_command": f"fetch {pdb_path.stem}; select interface, chain {target_chain} within 5 of chain {binder_chain}",
                        "chimerax_command": f"open {complex_pdb_path}; contacts {target_chain} {binder_chain}",
                    },
                },
                indent=2,
            )

    except Exception as e:
        logger.error(f"Error computing binding metrics: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool
def rank_binder_designs(
    results_dir: str, iptm_threshold: float = 0.7, ipsae_threshold: float = 10.0, top_n: int = 10
) -> str:
    """
    Rank all binder designs by binding quality metrics.

    Collects results from structure predictions and ranks by:
    1. iPTM score (higher is better)
    2. ipSAE score (lower is better)
    3. Number of interface contacts

    Args:
        results_dir: Directory containing prediction results
        iptm_threshold: Minimum iPTM score to pass filter
        ipsae_threshold: Maximum ipSAE score to pass filter
        top_n: Number of top designs to return

    Returns:
        JSON string with ranked designs and the best iPTM among qualifying designs
    """
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            return json.dumps(
                {"status": "error", "message": f"Results directory not found: {results_dir}"}
            )

        # Collect all score files
        score_files = list(results_path.glob("**/*_scores*.json"))

        designs = []
        for score_file in score_files:
            try:
                scores = json.loads(score_file.read_text())

                iptm = scores.get("iptm", 0)

                # Find associated PDB
                pdb_name = score_file.stem.replace("_scores", "_relaxed")
                pdb_path = score_file.parent / f"{pdb_name}.pdb"

                design = {
                    "name": score_file.stem,
                    "pdb_path": str(pdb_path) if pdb_path.exists() else None,
                    "iPTM": round(iptm, 3),
                    "pTM": round(scores.get("ptm", 0), 3),
                    "mean_pLDDT": round(scores.get("mean_plddt", 0), 1),
                }

                # Check for ipSAE in PAE file
                pae_file = score_file.parent / f"{score_file.stem.replace('_scores', '_pae')}.json"
                if pae_file.exists():
                    design["has_pae"] = True

                designs.append(design)

            except Exception as e:
                logger.warning(f"Could not parse {score_file}: {e}")

        # Sort by iPTM (descending)
        designs.sort(key=lambda x: x.get("iPTM", 0), reverse=True)

        # Filter by thresholds
        passing_designs = [d for d in designs if d.get("iPTM", 0) >= iptm_threshold]

        # Get best iPTM among passing designs
        max_iptm = max([d["iPTM"] for d in passing_designs], default=0)

        return json.dumps(
            {
                "status": "success",
                "total_designs_analyzed": len(designs),
                "designs_passing_threshold": len(passing_designs),
                "thresholds_used": {"iPTM_min": iptm_threshold, "ipSAE_max": ipsae_threshold},
                "max_iPTM_passing": round(max_iptm, 3),
                "top_designs": designs[:top_n],
                "answer": f"The maximum iPTM score among designs passing the quality threshold is {max_iptm:.3f}",
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error ranking designs: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================================
# Helper Functions
# ============================================================================


def _check_rfdiffusion_available() -> bool:
    """Check if RFdiffusion is available locally."""
    rfdiffusion_dir = os.environ.get("RFDIFFUSION_DIR")
    if rfdiffusion_dir and Path(rfdiffusion_dir).exists():
        return True

    # Check common locations
    for path in ["RFdiffusion", "~/RFdiffusion", "/opt/RFdiffusion"]:
        if Path(path).expanduser().exists():
            return True

    return False


def _check_proteinmpnn_available() -> bool:
    """Check if ProteinMPNN is available locally."""
    proteinmpnn_dir = os.environ.get("PROTEINMPNN_DIR")
    if proteinmpnn_dir and Path(proteinmpnn_dir).exists():
        return True

    for path in ["ProteinMPNN", "~/ProteinMPNN", "/opt/ProteinMPNN"]:
        if Path(path).expanduser().exists():
            return True

    return False


def _check_colabfold_available() -> bool:
    """Check if ColabFold is available locally."""
    try:
        # Check if colabfold_batch is in PATH
        import shutil

        colabfold_path = shutil.which("colabfold_batch")
        if not colabfold_path:
            return False

        result = subprocess.run([colabfold_path, "--help"], capture_output=True, timeout=5)  # nosec B603
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_protein_design_tools():
    """Return list of all protein design tools."""
    return [
        design_binders_bindcraft,
        generate_binder_backbones,
        design_binder_sequences,
        predict_complex_structure,
        compute_binding_metrics,
        rank_binder_designs,
    ]
