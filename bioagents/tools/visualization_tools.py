"""Visualization tools for creating charts and plots."""

import textwrap

from langchain_core.tools import tool

from bioagents.sandbox.sandbox_manager import get_sandbox


def _run_plot_script(script: str, output_path: str) -> str:
    """Helper to run a matplotlib plotting script in the sandbox."""
    sandbox = get_sandbox()
    script_path = "plot_script.py"
    sandbox.write_file(script_path, script)
    result = sandbox.run_command(f"python {script_path}", timeout=60)

    if result["success"]:
        full_path = sandbox.workdir / output_path
        if full_path.exists():
            return f"Plot saved to: {full_path}"
        return f"Script ran but output file not found at {output_path}.\n{result['stdout']}"
    return f"Plot generation failed:\n{result['stderr']}\n{result['stdout']}"


@tool
def create_bar_chart(
    data_json: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    output_path: str = "chart.png",
) -> str:
    """Create a bar chart from JSON data.

    Args:
        data_json: JSON string. Accepts either:
                   - {"labels": [...], "values": [...]}
                   - [{"name": "A", "value": 10}, ...]
                   - {"A": 10, "B": 20, ...}
        title: Chart title (optional).
        x_label: X-axis label (optional).
        y_label: Y-axis label (optional).
        output_path: Output file path in sandbox (default 'chart.png').

    Returns:
        Path to the saved chart image, or an error message.
    """
    try:
        script = textwrap.dedent(f"""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import json

            data = json.loads('''{data_json}''')

            if isinstance(data, dict) and 'labels' in data and 'values' in data:
                labels, values = data['labels'], data['values']
            elif isinstance(data, list):
                labels = [d.get('name', d.get('label', str(i))) for i, d in enumerate(data)]
                values = [d.get('value', d.get('count', 0)) for d in data]
            elif isinstance(data, dict):
                labels, values = list(data.keys()), list(data.values())
            else:
                raise ValueError("Unsupported data format")

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(labels)), values, color='steelblue', edgecolor='white')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            if {title!r}: ax.set_title({title!r}, fontsize=14, fontweight='bold')
            if {x_label!r}: ax.set_xlabel({x_label!r})
            if {y_label!r}: ax.set_ylabel({y_label!r})
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig("{output_path}", dpi=150, bbox_inches='tight')
            plt.close()
            print("Chart created successfully.")
        """)
        return _run_plot_script(script, output_path)
    except Exception as e:
        return f"Error creating bar chart: {e}"


@tool
def create_heatmap(
    data_json: str,
    title: str = "",
    output_path: str = "heatmap.png",
) -> str:
    """Create a heatmap from a 2D data matrix.

    Args:
        data_json: JSON string representing a 2D matrix. Accepts either a bare matrix
                   ``[[1,2,3], [4,5,6]]`` or an object holding one. For the object form
                   the matrix key may be any of "data", "values", "matrix" or "z", the
                   row-label key any of "row_labels", "rows" or "index", and the
                   column-label key any of "col_labels", "columns" or "cols". Example:
                   ``{"values": [[1,2],[3,4]], "rows": ["r1","r2"], "columns": ["c1","c2"]}``
        title: Chart title (optional).
        output_path: Output file path in sandbox (default 'heatmap.png').

    Returns:
        Path to the saved heatmap image. On bad input returns an error message naming the
        keys that were found and the keys that are accepted, rather than a traceback.
    """
    try:
        script = textwrap.dedent(f"""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            import json

            raw = json.loads('''{data_json}''')

            def _first(mapping, keys):
                for key in keys:
                    if key in mapping:
                        return mapping[key]
                return None

            if isinstance(raw, dict):
                # Accept the shapes an agent naturally produces, not one exact spelling.
                values = _first(raw, ('data', 'values', 'matrix', 'z'))
                if values is None:
                    raise SystemExit(
                        "create_heatmap: no matrix found in the input object. Got keys "
                        f"{{sorted(raw)}}; expected one of 'data', 'values', 'matrix', 'z' "
                        "to hold a 2D array."
                    )
                matrix = np.array(values, dtype=float)
                row_labels = _first(raw, ('row_labels', 'rows', 'index'))
                col_labels = _first(raw, ('col_labels', 'columns', 'cols'))
            else:
                matrix = np.array(raw, dtype=float)
                row_labels = None
                col_labels = None

            if matrix.ndim != 2:
                raise SystemExit(
                    f"create_heatmap: expected a 2D matrix, got {{matrix.ndim}}D with "
                    f"shape {{matrix.shape}}."
                )

            fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1]*0.8), max(6, matrix.shape[0]*0.5)))
            sns.heatmap(matrix, annot=matrix.size <= 100, fmt='.2f' if matrix.size <= 100 else '',
                        cmap='RdBu_r', center=0, ax=ax,
                        xticklabels=col_labels if col_labels else False,
                        yticklabels=row_labels if row_labels else False)
            if {title!r}: ax.set_title({title!r}, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig("{output_path}", dpi=150, bbox_inches='tight')
            plt.close()
            print("Heatmap created successfully.")
        """)
        return _run_plot_script(script, output_path)
    except Exception as e:
        return f"Error creating heatmap: {e}"


@tool
def create_scatter_plot(
    data_json: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    output_path: str = "scatter.png",
) -> str:
    """Create a scatter plot from paired x/y data.

    Args:
        data_json: JSON string. Accepts:
                   - {"x": [...], "y": [...]}
                   - [{"x": 1, "y": 2}, ...]
                   Optionally include "color" or "size" arrays/fields.
        title: Chart title (optional).
        x_label: X-axis label (optional).
        y_label: Y-axis label (optional).
        output_path: Output file path in sandbox (default 'scatter.png').

    Returns:
        Path to the saved scatter plot, or an error message.
    """
    try:
        script = textwrap.dedent(f"""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import json

            data = json.loads('''{data_json}''')

            if isinstance(data, dict):
                x, y = data['x'], data['y']
                colors = data.get('color')
                sizes = data.get('size')
            elif isinstance(data, list):
                x = [d['x'] for d in data]
                y = [d['y'] for d in data]
                colors = [d.get('color') for d in data] if 'color' in data[0] else None
                sizes = [d.get('size') for d in data] if 'size' in data[0] else None
            else:
                raise ValueError("Unsupported data format")

            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(x, y, c=colors, s=sizes or 50, alpha=0.7, edgecolors='white', linewidth=0.5)
            if colors and not isinstance(colors[0], str):
                plt.colorbar(scatter, ax=ax)
            if {title!r}: ax.set_title({title!r}, fontsize=14, fontweight='bold')
            if {x_label!r}: ax.set_xlabel({x_label!r})
            if {y_label!r}: ax.set_ylabel({y_label!r})
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig("{output_path}", dpi=150, bbox_inches='tight')
            plt.close()
            print("Scatter plot created successfully.")
        """)
        return _run_plot_script(script, output_path)
    except Exception as e:
        return f"Error creating scatter plot: {e}"


@tool
def create_volcano_plot(
    data_json: str,
    title: str = "",
    output_path: str = "volcano.png",
) -> str:
    """Create a volcano plot for differential expression results.

    Args:
        data_json: JSON string with DE results. Accepts:
                   - {"log2fc": [...], "pvalue": [...]} or
                   - {"log2fc": [...], "padj": [...]}
                   - [{"gene": "X", "log2fc": 1.2, "pvalue": 0.01}, ...]
                   Optionally include "gene" names for labeling top hits.
        title: Chart title (optional, default 'Volcano Plot').
        output_path: Output file path in sandbox (default 'volcano.png').

    Returns:
        Path to the saved volcano plot, or an error message.
    """
    try:
        plot_title = title or "Volcano Plot"
        script = textwrap.dedent(f"""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import json

            data = json.loads('''{data_json}''')

            if isinstance(data, dict):
                log2fc = np.array(data.get('log2fc', data.get('log2FoldChange', [])), dtype=float)
                pvals = np.array(data.get('pvalue', data.get('padj', [])), dtype=float)
                genes = data.get('gene', data.get('genes', []))
            elif isinstance(data, list):
                log2fc = np.array([d.get('log2fc', d.get('log2FoldChange', 0)) for d in data], dtype=float)
                pvals = np.array([d.get('pvalue', d.get('padj', 1)) for d in data], dtype=float)
                genes = [d.get('gene', '') for d in data]
            else:
                raise ValueError("Unsupported data format")

            pvals = np.clip(pvals, 1e-300, 1)
            neg_log10_p = -np.log10(pvals)

            fc_thresh = 1.0
            p_thresh = 0.05
            sig_up = (log2fc > fc_thresh) & (pvals < p_thresh)
            sig_down = (log2fc < -fc_thresh) & (pvals < p_thresh)
            ns = ~sig_up & ~sig_down

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(log2fc[ns], neg_log10_p[ns], c='grey', alpha=0.4, s=20, label='Not significant')
            ax.scatter(log2fc[sig_up], neg_log10_p[sig_up], c='firebrick', alpha=0.6, s=30, label='Up-regulated')
            ax.scatter(log2fc[sig_down], neg_log10_p[sig_down], c='steelblue', alpha=0.6, s=30, label='Down-regulated')

            ax.axhline(-np.log10(p_thresh), color='grey', linestyle='--', linewidth=0.8)
            ax.axvline(fc_thresh, color='grey', linestyle='--', linewidth=0.8)
            ax.axvline(-fc_thresh, color='grey', linestyle='--', linewidth=0.8)

            if genes:
                top_idx = np.argsort(pvals)[:10]
                for i in top_idx:
                    if i < len(genes) and genes[i]:
                        ax.annotate(genes[i], (log2fc[i], neg_log10_p[i]),
                                    fontsize=7, alpha=0.8,
                                    arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

            ax.set_xlabel('log2 Fold Change', fontsize=12)
            ax.set_ylabel('-log10(p-value)', fontsize=12)
            ax.set_title({plot_title!r}, fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig("{output_path}", dpi=150, bbox_inches='tight')
            plt.close()
            print("Volcano plot created successfully.")
        """)
        return _run_plot_script(script, output_path)
    except Exception as e:
        return f"Error creating volcano plot: {e}"


def get_visualization_tools() -> list:
    """Return all visualization tools."""
    return [create_bar_chart, create_heatmap, create_scatter_plot, create_volcano_plot]
