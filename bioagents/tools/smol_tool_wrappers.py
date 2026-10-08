"""Wrappers for ToolUniverse tools to be used with smolagents."""

from typing import Any, ClassVar

from smolagents import Tool


class ToolUniverseSearchTool(Tool):
    """Tool for searching bioinformatics tools in ToolUniverse."""

    name = "tool_universe_find_tools"
    description = "Search for bioinformatics tools in ToolUniverse."
    inputs: ClassVar[dict[str, Any]] = {
        "description": {"type": "string", "description": "Description of the capability needed."},
        "limit": {
            "type": "integer",
            "description": "Max number of tools to return (default 5).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, description: str, limit: int = 5) -> str:
        from bioagents.tools.tool_universe import DEFAULT_WRAPPER

        return DEFAULT_WRAPPER.find_tools(description, limit=limit)


class ToolUniverseExecuteTool(Tool):
    """Tool for executing specific ToolUniverse tools."""

    name = "tool_universe_call_tool"
    description = "Execute a specific ToolUniverse tool."
    inputs: ClassVar[dict[str, Any]] = {
        "tool_name": {"type": "string", "description": "Exact name of the tool."},
        "arguments_json": {
            "type": "string",
            "description": "JSON string of arguments.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, tool_name: str, arguments_json: str = "{}") -> str:
        from bioagents.tools.tool_universe import DEFAULT_WRAPPER

        return DEFAULT_WRAPPER.execute_tool(tool_name, arguments_json)


class EsmMutationScoringTool(Tool):
    """Score point mutations with a real ESM protein language model."""

    name = "score_mutations_esm"
    description = (
        "Score protein point mutations for fitness / variant effect using a real ESM "
        "protein language model (ESM-1b by default, ESM-2 also supported). Runs the "
        "model locally and returns the masked-marginal score log P(mutant) - log P(wild-type) "
        "for each mutation; more negative means more deleterious. Returns JSON with a "
        "per-mutation breakdown and a ranking of the worst mutations. Use this instead of "
        "writing your own ESM code."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "sequence": {
            "type": "string",
            "description": "Wild-type protein sequence in single-letter amino acid code.",
        },
        "mutations": {
            "type": "string",
            "description": (
                "Comma-separated mutations as wild-type residue + 1-based position + "
                "mutant residue, e.g. 'G11A,K33M,D145N'."
            ),
        },
        "model": {
            "type": "string",
            "description": "Model alias: 'esm1b' (default), 'esm2', 'esm2-small', 'esm2-tiny'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, sequence: str, mutations: str, model: str = "esm1b") -> str:
        from bioagents.tools.esm_tools import score_mutations_esm

        return str(
            score_mutations_esm.invoke(
                {"sequence": sequence, "mutations": mutations, "model": model}
            )
        )


class EsmSaturationScanTool(Tool):
    """Run a real in-silico saturation mutagenesis scan with ESM."""

    name = "saturation_scan_esm"
    description = (
        "Run full in-silico saturation mutagenesis at the given positions using a real ESM "
        "model: substitutes all 19 alternative amino acids at each position, scores every "
        "variant with the masked-marginal log-likelihood ratio, and returns the variants "
        "ranked from most to least deleterious plus per-position summary statistics. "
        "Use this for 'mutate these residues to the other 19 amino acids and rank by fitness'."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "sequence": {
            "type": "string",
            "description": "Wild-type protein sequence in single-letter amino acid code.",
        },
        "positions": {
            "type": "string",
            "description": "Comma-separated 1-based positions to scan; ranges allowed, e.g. '80-86,145'.",
        },
        "model": {
            "type": "string",
            "description": "Model alias: 'esm1b' (default), 'esm2', 'esm2-small', 'esm2-tiny'.",
            "nullable": True,
        },
        "top_n": {
            "type": "integer",
            "description": "How many of the most deleterious variants to highlight (default 10).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, sequence: str, positions: str, model: str = "esm1b", top_n: int = 10) -> str:
        from bioagents.tools.esm_tools import saturation_scan_esm

        return str(
            saturation_scan_esm.invoke(
                {
                    "sequence": sequence,
                    "positions": positions,
                    "model": model,
                    "top_n": top_n,
                }
            )
        )


class EsmEmbeddingTool(Tool):
    """Compute a real ESM protein embedding."""

    name = "run_esm_embedding"
    description = (
        "Compute a real ESM protein language-model embedding for a sequence. Runs locally "
        "and returns the actual mean-pooled embedding vector (and optionally the per-residue "
        "matrix) as JSON. Use for similarity, clustering, or as features for a downstream model."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "sequence": {
            "type": "string",
            "description": "Protein sequence in single-letter amino acid code.",
        },
        "model": {
            "type": "string",
            "description": "Model alias: 'esm2' (default), 'esm2-small', 'esm2-tiny', 'esm1b'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, sequence: str, model: str = "esm2") -> str:
        from bioagents.tools.esm_tools import run_esm_embedding

        return str(run_esm_embedding.invoke({"sequence": sequence, "model": model}))


def get_esm_smol_tools() -> list:
    """Return smolagents-compatible ESM tools for the code-writing agents."""
    return [EsmMutationScoringTool(), EsmSaturationScanTool(), EsmEmbeddingTool()]
