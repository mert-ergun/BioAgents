#!/usr/bin/env python3
"""
Demo: protein FASTA -> preprocess -> ESM-2 (or dummy) embedding -> JSON export.

Run from repo root:
  uv sync --extra esm   # first-time: PyTorch + fair-esm for real embeddings
  uv run python examples/scientific_workflow_demo.py P04637
  uv run python examples/scientific_workflow_demo.py P04637 --dummy
"""

from __future__ import annotations

import argparse
import json
import sys

from bioagents.workflows.esm_models import DEFAULT_ESM2_MODEL_NAME
from bioagents.workflows.executor import WorkflowExecutor
from bioagents.workflows.presets import build_protein_embedding_pipeline_graph
from bioagents.workflows.serialization import graph_to_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "protein_id",
        nargs="?",
        default="P04637",
        help="UniProt accession (default: P04637)",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print workflow definition JSON and exit (no execution).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=8,
        help="With --dummy: zero-vector length (default: 8).",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use placeholder zeros instead of ESM-2 (no PyTorch).",
    )
    parser.add_argument(
        "--esm-model",
        default=DEFAULT_ESM2_MODEL_NAME,
        help=f"ESM-2 checkpoint when not --dummy (default: {DEFAULT_ESM2_MODEL_NAME}).",
    )
    args = parser.parse_args()

    graph = build_protein_embedding_pipeline_graph(
        embedding_dim=args.dim,
        use_esm2=not args.dummy,
        esm2_model_name=args.esm_model,
    )
    if args.dump_json:
        print(graph_to_json(graph))
        return 0

    executor = WorkflowExecutor(graph)
    result = executor.run({"fetch": {"protein_id": args.protein_id}})
    payload = result.sink_outputs["export"]["json"]
    print(json.dumps(json.loads(payload), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
