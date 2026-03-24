"""Named workflow graphs for demos, API, and examples."""

from __future__ import annotations

from bioagents.workflows.esm_models import DEFAULT_ESM2_MODEL_NAME
from bioagents.workflows.graph import WorkflowGraph
from bioagents.workflows.nodes.dummy_embedder import DummyEmbedderNode
from bioagents.workflows.nodes.esm2_embedder import Esm2EmbedderNode
from bioagents.workflows.nodes.export_json import ExportJsonNode
from bioagents.workflows.nodes.fasta_preprocess import FastaPreprocessorNode
from bioagents.workflows.nodes.uniprot_fasta import UniprotFastaNode


def _build_protein_embedding_core(embed: DummyEmbedderNode | Esm2EmbedderNode) -> WorkflowGraph:
    g = WorkflowGraph()
    g.add_node("fetch", UniprotFastaNode())
    g.add_node("clean", FastaPreprocessorNode())
    g.add_node("embed", embed)
    g.add_node("export", ExportJsonNode())
    g.add_edge("fetch", "clean")
    g.add_edge("clean", "embed")
    g.add_edge("clean", "export")
    g.add_edge("embed", "export")
    return g


def build_protein_embedding_pipeline_graph(
    *,
    embedding_dim: int = 8,
    use_esm2: bool = True,
    esm2_model_name: str = DEFAULT_ESM2_MODEL_NAME,
) -> WorkflowGraph:
    """
    UniProt FASTA → preprocess → embed → JSON export.

    With ``use_esm2=True`` (default), runs **ESM-2** via ``fair-esm`` (optional extra).
    With ``use_esm2=False``, uses :class:`DummyEmbedderNode` with ``embedding_dim``.

    Source node id ``fetch`` expects initial input ``protein_id: str``.
    """
    if use_esm2:
        return _build_protein_embedding_core(Esm2EmbedderNode(model_name=esm2_model_name))
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    return _build_protein_embedding_core(DummyEmbedderNode(dim=embedding_dim))
