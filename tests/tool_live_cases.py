"""Declarative live-invocation table for every agent-callable tool.

Each entry says how to really call a tool through ``tool.invoke(...)`` -- the
exact interface an agent uses -- and how to tell a real result from a lie.

Every tool in ``tests.tool_inventory.ALL_TOOLS`` must appear here, either with a
runnable case or with an explicit ``skip_reason``. ``test_tool_callability.py``
enforces that, so a tool can never be silently untested.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Markers that mean "this tool did not actually do the work", regardless of how
# cheerful the rest of the string is.
GENERIC_FAILURE_MARKERS: tuple[str, ...] = (
    "Traceback (most recent call last)",
    "No module named",
    "CLI not found",
    "command not found",
    "is not installed",
    "ENGAGEMENT_PENDING",
)


#: Tools whose live invocation currently fails for a real, diagnosed reason.
#: These are FINDINGS, not accepted behaviour: the live test stays red on
#: purpose so the defect cannot be forgotten. Remove an entry once it is fixed.
KNOWN_LIVE_FAILURES: dict[str, str] = {}
#: Empty, and it should stay that way. The three original entries are fixed:
#:  - create_heatmap: seaborn is now a declared dependency, and the tool accepts the
#:    matrix/label key spellings an agent actually produces instead of raising KeyError.
#:  - run_differential_expression: statsmodels is now a declared dependency.
#:  - run_gene_set_enrichment: the default is the real Enrichr library name, and an
#:    unknown library is rejected up front rather than returning a misleading empty list.


@dataclass
class LiveContext:
    """Scratch resources shared by the live cases."""

    workspace: Path
    sandbox_dir: Path
    _cache: dict = field(default_factory=dict)

    def path(self, name: str) -> str:
        return str(self.workspace / name)

    def pdb_file(self, pdb_id: str = "1CRN") -> str:
        """Download a PDB file once per session and return its local path."""
        key = f"pdb:{pdb_id}"
        if key not in self._cache:
            import requests

            target = self.workspace / f"{pdb_id}.pdb"
            response = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60)
            response.raise_for_status()
            target.write_text(response.text)
            self._cache[key] = str(target)
        return self._cache[key]

    def text_file(self, name: str, content: str) -> str:
        target = self.workspace / name
        target.write_text(content)
        return str(target)

    def sandbox_file(self, name: str, content: str) -> str:
        """Write a file into the sandbox workdir and return its sandbox path."""
        target = self.sandbox_dir / "default" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return str(target)


Checker = Callable[[object], None]


@dataclass(frozen=True)
class LiveCase:
    """One live invocation of one tool."""

    args: dict | Callable[[LiveContext], dict] | None = None
    check: Checker | None = None
    skip_reason: str | None = None
    slow: bool = False

    def resolve_args(self, ctx: LiveContext) -> dict:
        return self.args(ctx) if callable(self.args) else dict(self.args or {})


def expect(
    *,
    contains: tuple[str, ...] = (),
    contains_any: tuple[str, ...] = (),
    not_contains: tuple[str, ...] = (),
    min_length: int = 1,
    json_keys: tuple[str, ...] = (),
    predicate: Callable[[str], bool] | None = None,
    predicate_desc: str = "custom predicate",
) -> Checker:
    """Build an assertion over a tool result.

    The assertions are deliberately strict: a tool that returns an error string,
    an empty payload, or a cheerful message with no data fails.
    """

    def _check(result: object) -> None:
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        assert len(text.strip()) >= min_length, (
            f"result too short ({len(text.strip())} chars, need {min_length}): {text!r}"
        )
        for marker in (*GENERIC_FAILURE_MARKERS, *not_contains):
            assert marker.lower() not in text.lower(), (
                f"result contains failure marker {marker!r}: {text[:400]!r}"
            )
        for needle in contains:
            assert needle.lower() in text.lower(), (
                f"result is missing expected content {needle!r}: {text[:400]!r}"
            )
        if contains_any:
            assert any(n.lower() in text.lower() for n in contains_any), (
                f"result contains none of {contains_any}: {text[:400]!r}"
            )
        if json_keys:
            payload = json.loads(text)
            for key in json_keys:
                assert key in payload, f"JSON result is missing key {key!r}: {text[:400]!r}"
        if predicate is not None:
            assert predicate(text), f"result failed {predicate_desc}: {text[:400]!r}"

    return _check


def _file_written(*, min_bytes: int = 1) -> Checker:
    """Assert the tool reported a path and that the file really exists on disk.

    This is the anti-lie check for every download/write tool: a JSON
    ``"status": "success"`` means nothing if no bytes landed anywhere.
    """

    def _check(result: object) -> None:
        text = str(result)
        path = _extract_path(text)
        assert path, f"tool reported no output path: {text[:300]!r}"
        resolved = Path(path)
        assert resolved.exists(), f"tool reported success but {path} does not exist: {text[:300]!r}"
        size = resolved.stat().st_size
        assert size >= min_bytes, f"tool wrote {path} but it is only {size} bytes: {text[:300]!r}"

    return _check


_PATH_KEYS = ("file_path", "output_path", "path", "saved_to", "output_file")


def _extract_path(text: str) -> str:
    """Pull the written file path out of a tool result (JSON or prose)."""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        for key in _PATH_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    for token in reversed(text.replace(":", " ").split()):
        cleaned = token.strip("\"'`,.")
        if cleaned.startswith("/") and Path(cleaned).exists():
            return cleaned
    return ""


# ---------------------------------------------------------------------------
# Skip reasons, grouped so the audit report explains every untested tool.
# ---------------------------------------------------------------------------

SKIP_PHANTOM = (
    "known phantom tool: returns a hardcoded success string without doing work "
    "(see test_no_phantom_tools); a live assertion here would validate a lie"
)
SKIP_GPU = "requires GPU and multi-GB model weights not available in CI"
SKIP_EXTERNAL_BINARY = "requires an external binary/pipeline not installed in CI"
SKIP_API_KEY = "requires a paid API key (LLM / hosted provider)"
SKIP_MUTATES_ENV = "destructive: mutates the host Python environment"
SKIP_MUTATES_REGISTRY = "destructive: writes to the persistent custom-tool registry"
SKIP_VERY_SLOW = "external job queue, minutes-long runtime"
SKIP_NEEDS_PRIOR_RUN = "needs output produced by a prior long-running tool"


def _skip(reason: str) -> LiveCase:
    return LiveCase(skip_reason=reason)


def _hosted_model_skip(tool_name: str, label: str) -> LiveCase:
    """Skip reason for a hosted-model tool, derived from its live source."""
    from tests.tool_inventory import ALL_TOOLS, is_phantom_tool

    tool = ALL_TOOLS.get(tool_name)
    if tool is not None and is_phantom_tool(tool):
        return _skip(SKIP_PHANTOM)
    return _skip(f"{SKIP_API_KEY} (hosted {label} provider)")


def _rdkit_case(args: dict, check: Checker) -> LiveCase:
    """RDKit tools shell out to the `rdkit-agent` Node CLI (optional install)."""
    if shutil.which("rdkit-agent") is None:
        return _skip(
            "rdkit-agent CLI not installed (npm install -g rdkit-agent); note the "
            "tool still returns success=true in this state"
        )
    return LiveCase(args=args, check=check)


_PROBE_PEPTIDE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
_FASTA = ">test\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP\n"
_COUNTS_CSV = (
    "gene,s1,s2,s3,s4\n"
    "TP53,100,110,250,260\n"
    "BRCA1,50,55,52,48\n"
    "EGFR,10,12,80,85\n"
    "MYC,200,190,205,210\n"
)
_META_CSV = "sample,condition\ns1,ctrl\ns2,ctrl\ns3,treat\ns4,treat\n"


LIVE_CASES: dict[str, LiveCase] = {
    # --- web tools -------------------------------------------------------
    "fetch_url_content": LiveCase(
        args={"url": "https://example.com"},
        check=expect(contains=("example domain",), min_length=50),
    ),
    "search_google_scholar": LiveCase(
        args={"query": "CRISPR gene editing", "num_results": 3},
        check=expect(contains_any=("title", "paper", "crispr"), min_length=50),
    ),
    "download_file_from_url": LiveCase(
        args=lambda ctx: {
            "url": "https://files.rcsb.org/download/1CRN.pdb",
            "output_path": ctx.path("downloaded_1crn.pdb"),
        },
        check=_file_written(min_bytes=100),
    ),
    # --- file tools ------------------------------------------------------
    "write_local_file": LiveCase(
        args=lambda ctx: {
            "file_path": ctx.path("written.txt"),
            "content": "bioagents probe payload",
        },
        check=_file_written(),
    ),
    "read_local_file": LiveCase(
        args=lambda ctx: {"file_path": ctx.text_file("to_read.txt", "bioagents probe payload")},
        check=expect(contains=("bioagents probe payload",)),
    ),
    "list_local_directory": LiveCase(
        args=lambda ctx: {"path": str(ctx.workspace)},
        check=expect(min_length=2),
    ),
    "get_file_info": LiveCase(
        args=lambda ctx: {"file_path": ctx.text_file("info.txt", "x" * 100)},
        check=expect(contains_any=("size", "bytes"), min_length=10),
    ),
    # --- literature tools -------------------------------------------------
    "search_pubmed": LiveCase(
        args={"query": "p53 tumor suppressor", "max_results": 3},
        check=expect(contains_any=("pmid", "title"), min_length=80),
    ),
    "search_arxiv": LiveCase(
        args={"query": "protein structure prediction", "max_results": 3},
        check=expect(contains_any=("title", "arxiv"), min_length=80),
    ),
    "search_biorxiv": LiveCase(
        args={"query": "protein", "max_results": 3},
        check=expect(min_length=40),
    ),
    "fetch_paper_metadata": LiveCase(
        args={"doi": "10.1038/nature12373"},
        check=expect(contains_any=("title", "author"), min_length=40),
    ),
    # --- structural / protein design (real data paths) --------------------
    "fetch_pdb_structure": LiveCase(
        args={"pdb_id": "1CRN"},
        check=expect(contains_any=("crambin", "1crn"), min_length=100),
    ),
    "fetch_alphafold_structure": LiveCase(
        args={"uniprot_id": "P04637"},
        check=expect(contains_any=("alphafold", "af-p04637", "pdb"), min_length=100),
    ),
    "download_structure_file": LiveCase(
        args=lambda ctx: {"pdb_id": "1CRN", "output_dir": str(ctx.workspace)},
        check=_file_written(min_bytes=100),
    ),
    "search_pdb_complexes": LiveCase(
        args={"uniprot_id": "P04637", "limit": 5},
        check=expect(min_length=20),
    ),
    "get_pdb_polymer_entities": LiveCase(
        args={"pdb_id": "1CRN"},
        check=expect(min_length=20),
    ),
    "analyze_interface_contacts": LiveCase(
        args=lambda ctx: {
            "structure_path": ctx.pdb_file("1BRS"),
            "chain1": "A",
            "chain2": "D",
        },
        check=expect(contains_any=("contact", "residue"), min_length=20),
    ),
    "compute_interface_metrics": _skip(SKIP_NEEDS_PRIOR_RUN + " (AlphaFold PAE JSON)"),
    "design_binders_bindcraft": _skip(SKIP_GPU),
    "generate_binder_backbones": _skip(SKIP_GPU),
    "design_binder_sequences": _skip(SKIP_GPU),
    "predict_complex_structure": _skip(SKIP_GPU),
    "compute_binding_metrics": _skip(SKIP_NEEDS_PRIOR_RUN + " (predicted complex PDB)"),
    "rank_binder_designs": _skip(SKIP_NEEDS_PRIOR_RUN + " (binder design results dir)"),
    # --- hosted-model tools (phantom today, real once implemented) --------
    # Skip reasons are computed from the live source: while a tool is still a
    # stub the reason says so, and once it is implemented the reason becomes the
    # honest "needs a hosted provider key".
    **{
        name: _hosted_model_skip(name, label)
        for name, label in (
            ("run_alphafold2", "AlphaFold 2"),
            ("run_boltz", "Boltz-2 / BoltzGen"),
            ("run_abodybuilder3", "ABodyBuilder3"),
            ("run_unimol", "Uni-Mol"),
            ("run_esm3", "ESM-3"),
            ("run_saprot", "SaProt"),
        )
    },
    # --- locally-executed model tools -------------------------------------
    # These run real computation on this machine, so they are exercised live rather
    # than skipped as "hosted". Each must either produce a real result or report an
    # honest failure; neither may return a success string for work that did not run.
    "run_esm2": LiveCase(
        args={"sequence": _PROBE_PEPTIDE},
        check=expect(
            contains=("embedding_dim", "mean_embedding"),
            not_contains=("embeddings generated successfully",),
        ),
    ),
    "run_proteinmpnn": LiveCase(
        args=lambda ctx: {
            "structure_pdb": ctx.pdb_file("1CRN"),
            "chains_to_design": "A",
            "num_sequences": 1,
            "output_dir": ctx.path("proteinmpnn_out"),
        },
        slow=True,
        check=expect(
            contains=("ProteinMPNN",),
            not_contains=("designed successfully with ProteinMPNN using",),
        ),
    ),
    "run_rfdiffusion": LiveCase(
        args=lambda ctx: {"target_pdb": ctx.pdb_file("1CRN")},
        # RFdiffusion's GPU dependencies are not installed here, so the honest
        # outcome is a named capability_unavailable — never a fabricated backbone.
        check=expect(
            contains=("capability_unavailable", "performed_any_computation"),
            not_contains=("Backbone generated successfully",),
        ),
    ),
    "run_aggrescan3d": LiveCase(
        args=lambda ctx: {"structure_pdb": ctx.pdb_file("1CRN")},
        check=expect(
            not_contains=("analysis completed successfully using",),
        ),
    ),
    "run_esmfold": _skip(SKIP_GPU + " (facebook/esmfold_v1 is ~2.6GB and GPU-bound)"),
    # --- ESM protein language models (local, tiny checkpoint) -------------
    "run_esm_embedding": LiveCase(
        args={"sequence": _PROBE_PEPTIDE, "model": "esm2-tiny"},
        check=expect(
            contains=("embedding_dim", "mean_embedding"),
            json_keys=("status", "mean_embedding"),
            predicate=lambda t: json.loads(t)["status"] == "success"
            and len(json.loads(t)["mean_embedding"]) > 0,
            predicate_desc="returned a non-empty embedding vector",
        ),
        slow=True,
    ),
    "score_mutations_esm": LiveCase(
        args={
            "sequence": _PROBE_PEPTIDE,
            "mutations": "Y5A",
            "model": "esm2-tiny",
        },
        check=expect(
            contains=("masked-marginal",),
            json_keys=("status",),
            predicate=lambda t: json.loads(t)["status"] == "success",
            predicate_desc="scored the mutation successfully",
        ),
        slow=True,
    ),
    "saturation_scan_esm": LiveCase(
        args={
            "sequence": _PROBE_PEPTIDE,
            "positions": "5",
            "model": "esm2-tiny",
            "top_n": 3,
        },
        check=expect(
            contains=("positions_scanned",),
            json_keys=("status",),
            predicate=lambda t: json.loads(t)["status"] == "success",
            predicate_desc="scanned the requested position",
        ),
        slow=True,
    ),
    # --- proteomics / analysis -------------------------------------------
    "fetch_uniprot_fasta": LiveCase(
        args={"protein_id": "P04637"},
        check=expect(contains=(">",), contains_any=("p53", "p04637"), min_length=100),
    ),
    "download_uniprot_flat_file": LiveCase(
        args=lambda ctx: {
            "accession": "P04637",
            "output_path": ctx.path("P04637.txt"),
        },
        check=_file_written(min_bytes=100),
    ),
    "calculate_molecular_weight": LiveCase(
        args={"fasta_sequence": _FASTA},
        check=expect(contains_any=("da", "weight", "6"), min_length=5),
    ),
    "analyze_amino_acid_composition": LiveCase(
        args={"fasta_sequence": _FASTA},
        check=expect(min_length=20),
    ),
    "calculate_isoelectric_point": LiveCase(
        args={"fasta_sequence": _FASTA},
        check=expect(min_length=3),
    ),
    # --- genomics ---------------------------------------------------------
    "reverse_complement": LiveCase(
        args={"sequence": "ATGCATGC"},
        check=expect(contains=("GCATGCAT",)),
    ),
    "translate_dna": LiveCase(
        args={"dna_sequence": "ATGGCCATTGTAATGGGCCGCTGA"},
        check=expect(contains=("MAIVMGR",)),
    ),
    "calculate_gc_content": LiveCase(
        args={"sequence": "GGGGCCCCATAT"},
        check=expect(contains=("66",), min_length=2),
    ),
    "parse_fasta_file": LiveCase(
        args=lambda ctx: {"file_path": ctx.text_file("seqs.fasta", _FASTA)},
        check=expect(contains=("test",), min_length=10),
    ),
    "run_blast_search": _skip(SKIP_VERY_SLOW + " (NCBI BLAST queue)"),
    # --- transcriptomics --------------------------------------------------
    "run_differential_expression": LiveCase(
        args=lambda ctx: {
            "counts_file": ctx.sandbox_file("probe_counts.csv", _COUNTS_CSV),
            "metadata_file": ctx.sandbox_file("probe_meta.csv", _META_CSV),
        },
        check=expect(contains=("genes tested",), min_length=20),
        slow=True,
    ),
    "normalize_expression_data": LiveCase(
        args=lambda ctx: {
            "counts_file": ctx.sandbox_file("probe_counts_norm.csv", _COUNTS_CSV),
            "method": "TPM",
        },
        check=expect(min_length=20),
        slow=True,
    ),
    "run_gene_set_enrichment": LiveCase(
        args={"gene_list": "TP53,BRCA1,EGFR,MYC,CDKN1A"},
        check=expect(min_length=20),
        slow=True,
    ),
    # --- visualization ----------------------------------------------------
    "create_bar_chart": LiveCase(
        args={
            "data_json": '{"labels": ["a", "b"], "values": [1, 2]}',
            "output_path": "probe_bar.png",
        },
        check=expect(contains=("probe_bar.png",), min_length=10),
        slow=True,
    ),
    "create_heatmap": LiveCase(
        args={
            "data_json": '{"rows": ["r1", "r2"], "columns": ["c1", "c2"], '
            '"values": [[1, 2], [3, 4]]}',
            "output_path": "probe_heat.png",
        },
        check=expect(contains=("probe_heat.png",), min_length=10),
        slow=True,
    ),
    "create_scatter_plot": LiveCase(
        args={
            "data_json": '{"x": [1, 2, 3], "y": [2, 4, 6]}',
            "output_path": "probe_scatter.png",
        },
        check=expect(contains=("probe_scatter.png",), min_length=10),
        slow=True,
    ),
    "create_volcano_plot": LiveCase(
        args={
            "data_json": '[{"gene": "A", "log2FoldChange": 2.0, "pvalue": 0.001}, '
            '{"gene": "B", "log2FoldChange": -1.5, "pvalue": 0.02}]',
            "output_path": "probe_volcano.png",
        },
        check=expect(contains=("probe_volcano.png",), min_length=10),
        slow=True,
    ),
    # --- docking ----------------------------------------------------------
    "prepare_receptor": LiveCase(
        args=lambda ctx: {
            "pdb_path": ctx.pdb_file("1CRN"),
            "output_dir": str(ctx.workspace),
        },
        check=expect(contains=("pdbqt",), min_length=10),
        slow=True,
    ),
    "prepare_ligand": LiveCase(
        args=lambda ctx: {
            "smiles": "CCO",
            "ligand_name": "ethanol",
            "output_dir": str(ctx.workspace),
        },
        check=expect(contains=("pdbqt",), min_length=10),
        slow=True,
    ),
    "identify_binding_site": LiveCase(
        args=lambda ctx: {"pdb_path": ctx.pdb_file("1CRN")},
        check=expect(contains_any=("center", "center_x"), min_length=10),
    ),
    "run_docking": LiveCase(
        args=lambda ctx: _docking_args(ctx),
        check=expect(contains_any=("affinity", "pose", "kcal"), min_length=20),
        slow=True,
    ),
    "analyze_docking_results": _skip(SKIP_NEEDS_PRIOR_RUN + " (docked pose file)"),
    # --- shell / environment ---------------------------------------------
    "run_shell_command": LiveCase(
        args={"command": "echo bioagents_probe_ok"},
        check=expect(contains=("bioagents_probe_ok",)),
    ),
    "check_installed_packages": LiveCase(args={}, check=expect(min_length=20)),
    "install_python_package": _skip(SKIP_MUTATES_ENV),
    "check_gpu_available": LiveCase(
        args={}, check=expect(contains=("gpu_available",), min_length=10)
    ),
    "get_system_info": LiveCase(args={}, check=expect(contains=("python_version",), min_length=10)),
    "create_virtual_environment": _skip(SKIP_MUTATES_ENV + " (creates a venv on disk)"),
    "install_requirements": _skip(SKIP_MUTATES_ENV),
    # --- git --------------------------------------------------------------
    "git_clone_repo": LiveCase(
        args=lambda ctx: {
            "repo_url": "https://github.com/octocat/Hello-World.git",
            "target_dir": str(ctx.workspace / "hello_world"),
        },
        check=expect(contains_any=("cloned", "hello_world"), min_length=10),
        slow=True,
    ),
    "list_repo_files": LiveCase(
        args=lambda ctx: {"repo_path": str(ctx.workspace), "pattern": "*"},
        check=expect(min_length=2),
    ),
    "read_repo_file": LiveCase(
        args=lambda ctx: {"file_path": ctx.text_file("repo_doc.md", "# probe readme")},
        check=expect(contains=("probe readme",)),
    ),
    "git_checkout_branch": _skip(SKIP_NEEDS_PRIOR_RUN + " and mutates a git worktree"),
    # --- PDF / paperqa ----------------------------------------------------
    "fetch_webpage_as_pdf_text": LiveCase(
        args={"url": "https://example.com"},
        check=expect(contains_any=("example", "domain"), min_length=30),
        slow=True,
    ),
    "extract_pdf_text_spacy_layout": LiveCase(
        args=lambda ctx: {"local_pdf_path": _make_pdf(ctx)},
        check=expect(contains=("bioagents probe pdf",), min_length=10),
    ),
    "search_local_papers_with_paperqa": _skip(SKIP_API_KEY + " (paper-qa needs an LLM)"),
    # --- tool universe ----------------------------------------------------
    "tool_universe_find_tools": LiveCase(
        args={"description": "protein structure", "limit": 3},
        check=expect(contains=("result",), min_length=100),
        slow=True,
    ),
    "tool_universe_call_tool": _skip(
        "needs a concrete ToolUniverse tool name + valid provider arguments; "
        "discovery path is covered by tool_universe_find_tools"
    ),
    # --- tool builder -----------------------------------------------------
    "extract_tools_from_text": _skip(
        SKIP_API_KEY + " (LLM-backed extraction; note it raises ValueError out of invoke() "
        "instead of returning an error string when the key is missing)"
    ),
    "list_custom_tools": LiveCase(args={}, check=expect(min_length=5)),
    "search_custom_tools": LiveCase(args={"query": "pdb", "limit": 3}, check=expect(min_length=5)),
    "get_tool_code": _skip(SKIP_NEEDS_PRIOR_RUN + " (a registered custom tool name)"),
    "register_custom_tool": _skip(SKIP_MUTATES_REGISTRY),
    "validate_custom_tool": _skip(SKIP_MUTATES_REGISTRY),
    "execute_custom_tool": _skip("executes arbitrary registered code"),
    "generate_tool_wrapper": _skip(SKIP_API_KEY),
    "research_tool_documentation": _skip(SKIP_API_KEY + " / live web research"),
}


def _docking_args(ctx: LiveContext) -> dict:
    """Prepare a receptor + ligand, then dock -- the real agent sequence."""
    from bioagents.tools.docking_tools import get_docking_tools

    tools = {t.name: t for t in get_docking_tools()}
    receptor = json.loads(
        tools["prepare_receptor"].invoke(
            {"pdb_path": ctx.pdb_file("1CRN"), "output_dir": str(ctx.workspace)}
        )
    )
    ligand = json.loads(
        tools["prepare_ligand"].invoke(
            {"smiles": "CCO", "ligand_name": "ethanol", "output_dir": str(ctx.workspace)}
        )
    )
    site = json.loads(tools["identify_binding_site"].invoke({"pdb_path": ctx.pdb_file("1CRN")}))
    center = site.get("grid_center", {})
    return {
        "receptor_pdbqt_path": receptor["file_path"],
        "ligand_pdbqt_path": ligand["file_path"],
        "center_x": center.get("x", 0.0),
        "center_y": center.get("y", 0.0),
        "center_z": center.get("z", 0.0),
        "box_size_x": 20.0,
        "box_size_y": 20.0,
        "box_size_z": 20.0,
        "exhaustiveness": 2,
        "num_poses": 1,
        "output_dir": str(ctx.workspace),
    }


def _make_pdf(ctx: LiveContext) -> str:
    import pymupdf

    target = ctx.workspace / "probe.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "bioagents probe pdf")
    doc.save(str(target))
    doc.close()
    return str(target)


def _rdkit_cases() -> dict[str, LiveCase]:
    """RDKit tools all shell out to the same optional Node CLI."""
    simple = expect(min_length=10)
    cases = {
        "validate_smiles": ({"smiles": "CCO"}, expect(contains=("valid",), min_length=10)),
        "validate_smirks": ({"smirks": "[C:1][O:2]>>[C:1]=[O:2]"}, simple),
        "validate_reaction": ({"reactants": ["CCO"], "products": ["CC=O"]}, simple),
        "repair_invalid_smiles": ({"input_str": "C1CC"}, simple),
        "compute_molecular_descriptors": ({"smiles_list": "CCO,c1ccccc1"}, simple),
        "analyze_ring_systems": ({"smiles": "c1ccccc1"}, simple),
        "analyze_stereochemistry": ({"smiles": "C[C@H](N)C(=O)O"}, simple),
        "detect_functional_groups_tool": ({"smiles": "CC(=O)O"}, simple),
        "compute_molecular_fingerprint": ({"smiles": "CCO"}, simple),
        "search_similar_molecules_tool": ({"query": "CCO", "targets": "CCC,CCN"}, simple),
        "filter_by_descriptor_constraints": ({"smiles_list": "CCO,c1ccccc1"}, simple),
        "search_substructure": ({"smiles": "c1ccccc1O", "smarts_pattern": "c1ccccc1"}, simple),
        "convert_chemical_notation": (
            {"input_str": "CCO", "from_format": "smiles", "to_format": "inchi"},
            simple,
        ),
        "edit_molecule_structure": ({"smiles": "CCO", "operation": "canonicalize"}, simple),
        "apply_chemical_reaction": (
            {"smirks": "[C:1][O:2]>>[C:1]=[O:2]", "reactants": "CCO"},
            simple,
        ),
        "validate_atom_mapping": ({"smirks": "[C:1][O:2]>>[C:1]=[O:2]"}, simple),
        "list_atom_maps": ({"smiles": "[CH3:1][OH:2]"}, simple),
        "add_atom_maps": ({"smiles": "CCO"}, simple),
        "remove_atom_maps": ({"smiles": "[CH3:1][OH:2]"}, simple),
        "draw_molecule_svg": ({"smiles": "CCO"}, simple),
        "compute_dataset_statistics": ({"smiles_list": "CCO,c1ccccc1"}, simple),
    }
    return {name: _rdkit_case(args, check) for name, (args, check) in cases.items()}


LIVE_CASES.update(_rdkit_cases())
