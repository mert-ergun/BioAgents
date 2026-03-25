"""Catalog of runnable workflow presets (metadata + graph builders)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bioagents.workflows.esm_models import ALLOWED_ESM2_MODEL_NAMES, DEFAULT_ESM2_MODEL_NAME
from bioagents.workflows.graph import WorkflowGraph
from bioagents.workflows.nodes.dummy_embedder import DummyEmbedderNode
from bioagents.workflows.nodes.esm2_embedder import Esm2EmbedderNode
from bioagents.workflows.nodes.export_extra import (
    ExportDictJsonNode,
    ExportFastaJsonNode,
    ExportSequenceJsonNode,
    ExportTextJsonNode,
)
from bioagents.workflows.nodes.export_json import ExportJsonNode
from bioagents.workflows.nodes.fasta_preprocess import FastaPreprocessorNode
from bioagents.workflows.nodes.merge_text import MergeThreeTextNode, MergeTwoTextNode
from bioagents.workflows.nodes.protein_analysis_nodes import (
    AliphaticIndexNode,
    AminoAcidCompositionNode,
    AromaticFractionNode,
    ChargedFractionNode,
    CompactBiochemRecordNode,
    GravyScoreNode,
    HydrophobicFractionNode,
    InstabilityIndexNode,
    IsoelectricPointNode,
    KmerProfileNode,
    MolecularWeightNode,
    ReverseSequenceNode,
    SequenceDigestNode,
    SequenceLengthNode,
    SingleResidueCountNode,
    TerminalResidueNode,
)
from bioagents.workflows.nodes.rcsb_summary import RcsbEntrySummaryNode
from bioagents.workflows.nodes.uniprot_fasta import UniprotFastaNode


def _add_fetch_clean(g: WorkflowGraph) -> None:
    g.add_node("fetch", UniprotFastaNode())
    g.add_node("clean", FastaPreprocessorNode())
    g.add_edge("fetch", "clean")


def _merge_kw(
    *,
    embedding_dim: int,
    esm2_model_name: str,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    o = dict(options or {})
    o.setdefault("embedding_dim", embedding_dim)
    o.setdefault("esm2_model_name", esm2_model_name)
    return o


def _build_uniprot_fasta_json(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    g.add_node("fetch", UniprotFastaNode())
    g.add_node("ex", ExportFastaJsonNode())
    g.add_edge("fetch", "ex")
    return g


def _build_uniprot_clean_sequence_json(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("ex", ExportSequenceJsonNode())
    g.add_edge("clean", "ex")
    return g


def _build_uniprot_molecular_weight(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("mw", MolecularWeightNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "mw")
    g.add_edge("mw", "ex", {"report": "text"})
    return g


def _build_uniprot_aa_composition(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("cmp", AminoAcidCompositionNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "cmp")
    g.add_edge("cmp", "ex", {"report": "text"})
    return g


def _build_uniprot_isoelectric(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("pi", IsoelectricPointNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "pi")
    g.add_edge("pi", "ex", {"report": "text"})
    return g


def _build_uniprot_reverse(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("rev", ReverseSequenceNode())
    g.add_node("ex", ExportSequenceJsonNode())
    g.add_edge("clean", "rev")
    g.add_edge("rev", "ex", {"reversed_sequence": "sequence"})
    g.add_edge("clean", "ex", {"residue_count": "residue_count"})
    return g


def _build_uniprot_kmer(opts: dict[str, Any]) -> WorkflowGraph:
    k = max(1, min(12, int(opts.get("kmer_k", 3))))
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("km", KmerProfileNode(k=k))
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "km")
    g.add_edge("km", "ex", {"report": "text"})
    return g


def _build_uniprot_hydrophobic(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("h", HydrophobicFractionNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "h")
    g.add_edge("h", "ex", {"report": "text"})
    return g


def _build_uniprot_charged(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("c", ChargedFractionNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "c")
    g.add_edge("c", "ex", {"report": "text"})
    return g


def _build_uniprot_aromatic(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("a", AromaticFractionNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "a")
    g.add_edge("a", "ex", {"report": "text"})
    return g


def _build_uniprot_digest_md5(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("dg", SequenceDigestNode(algorithm="md5"))
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "dg")
    g.add_edge("dg", "ex", {"report": "text"})
    return g


def _build_uniprot_digest_sha256(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("dg", SequenceDigestNode(algorithm="sha256"))
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "dg")
    g.add_edge("dg", "ex", {"report": "text"})
    return g


def _build_uniprot_compact_biochem(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("cb", CompactBiochemRecordNode())
    g.add_node("ex", ExportDictJsonNode())
    g.add_edge("clean", "cb")
    g.add_edge("cb", "ex", {"record": "record"})
    return g


def _build_uniprot_parallel_mw_pi(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("mw", MolecularWeightNode())
    g.add_node("pi", IsoelectricPointNode())
    g.add_node("mg", MergeTwoTextNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "mw")
    g.add_edge("clean", "pi")
    g.add_edge("mw", "mg", {"report": "text_a"})
    g.add_edge("pi", "mg", {"report": "text_b"})
    g.add_edge("mg", "ex", {"merged": "text"})
    return g


def _build_uniprot_parallel_mw_composition(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("mw", MolecularWeightNode())
    g.add_node("cmp", AminoAcidCompositionNode())
    g.add_node("mg", MergeTwoTextNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "mw")
    g.add_edge("clean", "cmp")
    g.add_edge("mw", "mg", {"report": "text_a"})
    g.add_edge("cmp", "mg", {"report": "text_b"})
    g.add_edge("mg", "ex", {"merged": "text"})
    return g


def _build_uniprot_triple_report(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("mw", MolecularWeightNode())
    g.add_node("cmp", AminoAcidCompositionNode())
    g.add_node("pi", IsoelectricPointNode())
    g.add_node("mg", MergeThreeTextNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "mw")
    g.add_edge("clean", "cmp")
    g.add_edge("clean", "pi")
    g.add_edge("mw", "mg", {"report": "text_a"})
    g.add_edge("cmp", "mg", {"report": "text_b"})
    g.add_edge("pi", "mg", {"report": "text_c"})
    g.add_edge("mg", "ex", {"merged": "text"})
    return g


def _build_uniprot_gravy(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("gv", GravyScoreNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "gv")
    g.add_edge("gv", "ex", {"report": "text"})
    return g


def _build_uniprot_aliphatic(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("al", AliphaticIndexNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "al")
    g.add_edge("al", "ex", {"report": "text"})
    return g


def _build_uniprot_instability(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("ins", InstabilityIndexNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "ins")
    g.add_edge("ins", "ex", {"report": "text"})
    return g


def _build_uniprot_cys(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("cnt", SingleResidueCountNode("C"))
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "cnt")
    g.add_edge("cnt", "ex", {"report": "text"})
    return g


def _build_uniprot_trp(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("cnt", SingleResidueCountNode("W"))
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "cnt")
    g.add_edge("cnt", "ex", {"report": "text"})
    return g


def _build_uniprot_pro(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("cnt", SingleResidueCountNode("P"))
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "cnt")
    g.add_edge("cnt", "ex", {"report": "text"})
    return g


def _build_uniprot_hydro_charge_merge(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("h", HydrophobicFractionNode())
    g.add_node("c", ChargedFractionNode())
    g.add_node("mg", MergeTwoTextNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "h")
    g.add_edge("clean", "c")
    g.add_edge("h", "mg", {"report": "text_a"})
    g.add_edge("c", "mg", {"report": "text_b"})
    g.add_edge("mg", "ex", {"merged": "text"})
    return g


def _build_uniprot_charge_aromatic_merge(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("c", ChargedFractionNode())
    g.add_node("a", AromaticFractionNode())
    g.add_node("mg", MergeTwoTextNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "c")
    g.add_edge("clean", "a")
    g.add_edge("c", "mg", {"report": "text_a"})
    g.add_edge("a", "mg", {"report": "text_b"})
    g.add_edge("mg", "ex", {"merged": "text"})
    return g


def _build_uniprot_terminals_merge(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("n", TerminalResidueNode("n"))
    g.add_node("c", TerminalResidueNode("c"))
    g.add_node("mg", MergeTwoTextNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "n")
    g.add_edge("clean", "c")
    g.add_edge("n", "mg", {"report": "text_a"})
    g.add_edge("c", "mg", {"report": "text_b"})
    g.add_edge("mg", "ex", {"merged": "text"})
    return g


def _build_uniprot_length_report(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("ln", SequenceLengthNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("clean", "ln")
    g.add_edge("ln", "ex", {"report": "text"})
    return g


def _build_pdb_rcsb_summary(_opts: dict[str, Any]) -> WorkflowGraph:
    g = WorkflowGraph()
    g.add_node("pdb", RcsbEntrySummaryNode())
    g.add_node("ex", ExportTextJsonNode())
    g.add_edge("pdb", "ex", {"summary": "text"})
    return g


def _build_uniprot_kmer2(opts: dict[str, Any]) -> WorkflowGraph:
    return _build_uniprot_kmer({**opts, "kmer_k": 2})


def _build_uniprot_kmer5(opts: dict[str, Any]) -> WorkflowGraph:
    return _build_uniprot_kmer({**opts, "kmer_k": 5})


def _build_uniprot_esm_embedding_export(_opts: dict[str, Any]) -> WorkflowGraph:
    """Fetch → clean → ESM-2 → merge with residue count → JSON (explicit graph)."""
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node(
        "embed",
        Esm2EmbedderNode(model_name=str(_opts.get("esm2_model_name", DEFAULT_ESM2_MODEL_NAME))),
    )
    g.add_node("ex", ExportJsonNode())
    g.add_edge("clean", "embed")
    g.add_edge("embed", "ex", {"embedding": "embedding"})
    g.add_edge("clean", "ex", {"residue_count": "residue_count"})
    return g


def _build_uniprot_dummy_embedding_export(_opts: dict[str, Any]) -> WorkflowGraph:
    dim = max(1, min(1280, int(_opts.get("embedding_dim", 8))))
    g = WorkflowGraph()
    _add_fetch_clean(g)
    g.add_node("embed", DummyEmbedderNode(dim=dim))
    g.add_node("ex", ExportJsonNode())
    g.add_edge("clean", "embed")
    g.add_edge("embed", "ex", {"embedding": "embedding"})
    g.add_edge("clean", "ex", {"residue_count": "residue_count"})
    return g


PresetBuilder = Callable[[dict[str, Any]], WorkflowGraph]


class PresetEntry:
    __slots__ = (
        "builder",
        "description",
        "id",
        "name",
        "options_help",
        "source_input_key",
        "source_node_id",
    )

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        source_node_id: str,
        source_input_key: str,
        options_help: dict[str, Any],
        builder: PresetBuilder,
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        self.source_node_id = source_node_id
        self.source_input_key = source_input_key
        self.options_help = options_help
        self.builder = builder

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_node_id": self.source_node_id,
            "inputs": {
                self.source_input_key: (
                    "UniProt accession (e.g. P04637)"
                    if self.source_input_key == "protein_id"
                    else "PDB ID (e.g. 1YCR or 1ycr)"
                ),
            },
            "options": self.options_help,
        }


WORKFLOW_PRESETS: tuple[PresetEntry, ...] = (
    PresetEntry(
        "protein_embedding",
        "Protein embedding (ESM-2)",
        "UniProt → clean sequence → ESM-2 embedding + residue count → JSON. Requires uv sync --extra esm.",
        "fetch",
        "protein_id",
        {
            "esm2_model_name": {
                "default": DEFAULT_ESM2_MODEL_NAME,
                "choices": sorted(ALLOWED_ESM2_MODEL_NAMES),
            },
        },
        _build_uniprot_esm_embedding_export,
    ),
    PresetEntry(
        "protein_embedding_dummy",
        "Protein embedding (placeholder)",
        "Same as ESM pipeline but zero vector; no PyTorch.",
        "fetch",
        "protein_id",
        {"embedding_dim": {"default": 8, "min": 1, "max": 1280}},
        _build_uniprot_dummy_embedding_export,
    ),
    PresetEntry(
        "uniprot_fasta_json",
        "UniProt → FASTA JSON",
        "Download raw FASTA and wrap in JSON.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_fasta_json,
    ),
    PresetEntry(
        "uniprot_clean_sequence_json",
        "UniProt → cleaned sequence JSON",
        "Strip headers; export sequence + length.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_clean_sequence_json,
    ),
    PresetEntry(
        "uniprot_molecular_weight",
        "UniProt → molecular weight",
        "Estimated mass (Da) from sequence.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_molecular_weight,
    ),
    PresetEntry(
        "uniprot_amino_acid_composition",
        "UniProt → AA composition",
        "Full composition table + hydrophobic/polar/charged split.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_aa_composition,
    ),
    PresetEntry(
        "uniprot_isoelectric_point",
        "UniProt → pI estimate",
        "Rough isoelectric point from charged residues.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_isoelectric,
    ),
    PresetEntry(
        "uniprot_reverse_sequence",
        "UniProt → reverse sequence",
        "Cleaned sequence reversed; JSON with length.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_reverse,
    ),
    PresetEntry(
        "uniprot_kmer_profile",
        "UniProt → k-mer profile",
        "Top k-mers (default k=3); set options.kmer_k 1-12.",
        "fetch",
        "protein_id",
        {"kmer_k": {"default": 3, "min": 1, "max": 12}},
        _build_uniprot_kmer,
    ),
    PresetEntry(
        "uniprot_kmer2_profile",
        "UniProt → dipeptide (k=2) profile",
        "2-mer frequency snapshot.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_kmer2,
    ),
    PresetEntry(
        "uniprot_kmer5_profile",
        "UniProt → 5-mer profile",
        "5-mer frequency snapshot.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_kmer5,
    ),
    PresetEntry(
        "uniprot_hydrophobic_fraction",
        "UniProt → hydrophobic fraction",
        "Fraction of AVILMFYWP.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_hydrophobic,
    ),
    PresetEntry(
        "uniprot_charged_fraction",
        "UniProt → charged fraction",
        "Fraction of KREDH.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_charged,
    ),
    PresetEntry(
        "uniprot_aromatic_fraction",
        "UniProt → aromatic fraction",
        "Fraction of F, Y, W.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_aromatic,
    ),
    PresetEntry(
        "uniprot_digest_md5",
        "UniProt → MD5 digest",
        "MD5 hex digest of cleaned sequence.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_digest_md5,
    ),
    PresetEntry(
        "uniprot_digest_sha256",
        "UniProt → SHA-256 digest",
        "SHA-256 hex digest of cleaned sequence.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_digest_sha256,
    ),
    PresetEntry(
        "uniprot_compact_biochem_json",
        "UniProt → compact biochem JSON",
        "Length, MW, GRAVY, aliphatic%, instability heuristic as JSON object.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_compact_biochem,
    ),
    PresetEntry(
        "uniprot_parallel_mw_pi",
        "UniProt → MW ∥ pI → merged report",
        "Parallel molecular weight and pI sections in one text export.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_parallel_mw_pi,
    ),
    PresetEntry(
        "uniprot_parallel_mw_composition",
        "UniProt → MW ∥ composition → merged",
        "Parallel MW and AA composition reports merged.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_parallel_mw_composition,
    ),
    PresetEntry(
        "uniprot_triple_biochemical_report",
        "UniProt → MW ∥ composition ∥ pI",
        "All three reports concatenated.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_triple_report,
    ),
    PresetEntry(
        "uniprot_gravy_score",
        "UniProt → GRAVY",
        "Kyte-Doolittle grand average of hydropathy.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_gravy,
    ),
    PresetEntry(
        "uniprot_aliphatic_content",
        "UniProt → aliphatic content",
        "Percent of A+V+I+L (proxy).",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_aliphatic,
    ),
    PresetEntry(
        "uniprot_instability_heuristic",
        "UniProt → instability heuristic",
        "Rough 0-100 instability score.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_instability,
    ),
    PresetEntry(
        "uniprot_cysteine_content",
        "UniProt → cysteine count",
        "C residue count and fraction.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_cys,
    ),
    PresetEntry(
        "uniprot_tryptophan_content",
        "UniProt → tryptophan count",
        "W residue count and fraction.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_trp,
    ),
    PresetEntry(
        "uniprot_proline_content",
        "UniProt → proline count",
        "P residue count and fraction.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_pro,
    ),
    PresetEntry(
        "uniprot_hydrophobic_and_charged",
        "UniProt → hydrophobic + charged",
        "Two parallel fraction summaries merged.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_hydro_charge_merge,
    ),
    PresetEntry(
        "uniprot_charged_and_aromatic",
        "UniProt → charged + aromatic",
        "Charged fraction and aromatic fraction merged.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_charge_aromatic_merge,
    ),
    PresetEntry(
        "uniprot_n_c_terminal_report",
        "UniProt → N- and C-terminal residues",
        "First and last amino acid reports merged.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_terminals_merge,
    ),
    PresetEntry(
        "uniprot_length_report",
        "UniProt → length only",
        "Residue count after normalization.",
        "fetch",
        "protein_id",
        {},
        _build_uniprot_length_report,
    ),
    PresetEntry(
        "pdb_rcsb_entry_summary",
        "PDB → RCSB metadata",
        "Title and experimental method from data.rcsb.org (enter 4-letter PDB ID).",
        "pdb",
        "pdb_id",
        {},
        _build_pdb_rcsb_summary,
    ),
)


PRESET_BY_ID: dict[str, PresetEntry] = {p.id: p for p in WORKFLOW_PRESETS}


def list_preset_api_payloads() -> list[dict[str, Any]]:
    return [p.to_api_dict() for p in WORKFLOW_PRESETS]


def build_graph_for_preset(
    preset_id: str,
    *,
    embedding_dim: int,
    esm2_model_name: str,
    options: dict[str, Any] | None,
) -> WorkflowGraph:
    entry = PRESET_BY_ID[preset_id]
    opts = _merge_kw(
        embedding_dim=embedding_dim,
        esm2_model_name=esm2_model_name,
        options=options,
    )
    return entry.builder(opts)


def initial_inputs_for_preset(preset_id: str, primary_id: str) -> dict[str, dict[str, Any]]:
    entry = PRESET_BY_ID[preset_id]
    key = entry.source_input_key
    sid = entry.source_node_id
    val = primary_id.strip()
    if key == "pdb_id":
        val = val.upper().replace(" ", "")
        if len(val) >= 4:
            val = val[:4]
    return {sid: {key: val}}
