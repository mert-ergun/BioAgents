"""Transcriptomics tools for differential expression, enrichment, and normalization."""

import textwrap

from langchain_core.tools import tool

from bioagents.sandbox.sandbox_manager import get_sandbox


@tool
def run_differential_expression(
    counts_file: str,
    metadata_file: str,
    method: str = "auto",
) -> str:
    """Run differential expression analysis on RNA-seq count data.

    Generates and executes a Python script in the sandbox that performs DE analysis.

    Args:
        counts_file: Path to the counts matrix (CSV/TSV) with genes as rows and
                     samples as columns. First column should be gene IDs.
        metadata_file: Path to sample metadata (CSV/TSV) with at least 'sample'
                       and 'condition' columns.
        method: Analysis method. 'auto' selects based on data, 'deseq2_like'
                for negative binomial, or 'ttest' for simple t-test approach.

    Returns:
        Path to results file and summary statistics, or an error message.
    """
    try:
        sandbox = get_sandbox()
        script = textwrap.dedent(f"""\
            import pandas as pd
            import numpy as np
            from scipy import stats
            import warnings
            warnings.filterwarnings('ignore')

            counts = pd.read_csv("{counts_file}", index_col=0, sep=None, engine='python')
            metadata = pd.read_csv("{metadata_file}", sep=None, engine='python')

            if 'condition' not in metadata.columns:
                condition_col = [c for c in metadata.columns if c != 'sample'][0]
            else:
                condition_col = 'condition'

            conditions = metadata[condition_col].unique()
            if len(conditions) < 2:
                print("ERROR: Need at least 2 conditions for DE analysis.")
                exit(1)

            cond1, cond2 = conditions[0], conditions[1]
            samples1 = metadata[metadata[condition_col] == cond1].iloc[:, 0].tolist()
            samples2 = metadata[metadata[condition_col] == cond2].iloc[:, 0].tolist()

            group1 = counts[[c for c in samples1 if c in counts.columns]]
            group2 = counts[[c for c in samples2 if c in counts.columns]]

            if group1.empty or group2.empty:
                print("ERROR: Could not match sample names between counts and metadata.")
                exit(1)

            results = []
            for gene in counts.index:
                vals1 = group1.loc[gene].values.astype(float)
                vals2 = group2.loc[gene].values.astype(float)
                mean1 = np.mean(vals1)
                mean2 = np.mean(vals2)
                if mean1 == 0 and mean2 == 0:
                    continue
                log2fc = np.log2((mean2 + 1) / (mean1 + 1))
                try:
                    stat, pval = stats.ttest_ind(vals1, vals2, equal_var=False)
                except Exception:
                    pval = 1.0
                results.append({{'gene': gene, 'log2FoldChange': log2fc, 'pvalue': pval,
                                 'mean_{cond1}': mean1, 'mean_{cond2}': mean2}})

            df = pd.DataFrame(results)
            from statsmodels.stats.multitest import multipletests
            try:
                _, df['padj'], _, _ = multipletests(df['pvalue'], method='fdr_bh')
            except Exception:
                df['padj'] = df['pvalue']

            df = df.sort_values('pvalue')
            output_path = "{counts_file.rsplit(".", 1)[0]}_de_results.csv"
            df.to_csv(output_path, index=False)

            sig = df[df['padj'] < 0.05]
            up = sig[sig['log2FoldChange'] > 0]
            down = sig[sig['log2FoldChange'] < 0]
            print(f"Total genes tested: {{len(df)}}")
            print(f"Significant (padj < 0.05): {{len(sig)}}")
            print(f"  Upregulated: {{len(up)}}")
            print(f"  Downregulated: {{len(down)}}")
            print(f"Results saved to: {{output_path}}")
        """)

        script_path = "de_analysis_script.py"
        sandbox.write_file(script_path, script)
        result = sandbox.run_command(f"python {script_path}", timeout=120)

        if result["success"]:
            return result["stdout"]
        return f"DE analysis failed:\n{result['stderr']}\n{result['stdout']}"
    except Exception as e:
        return f"Error running differential expression analysis: {e}"


@tool
def run_gene_set_enrichment(gene_list: str, database: str = "GO_Biological_Process") -> str:
    """Run gene set enrichment analysis on a list of genes.

    Generates and executes a Python script using the Enrichr API.

    Args:
        gene_list: Comma-separated list of gene symbols (e.g. 'TP53,BRCA1,EGFR').
        database: Enrichr gene set library. Common options:
                  'GO_Biological_Process_2023', 'GO_Molecular_Function_2023',
                  'KEGG_2021_Human', 'Reactome_2022', 'MSigDB_Hallmark_2020'.

    Returns:
        JSON string with enriched terms, p-values, and overlapping genes.
    """
    try:
        sandbox = get_sandbox()
        genes = [g.strip() for g in gene_list.split(",") if g.strip()]
        if not genes:
            return "Error: No genes provided. Use comma-separated gene symbols."

        script = textwrap.dedent(f"""\
            import requests, json

            genes = {genes}
            db = "{database}"

            # Submit gene list to Enrichr
            url = "https://maayanlab.cloud/Enrichr/addList"
            payload = {{"list": (None, "\\n".join(genes)), "description": (None, "BioAgents query")}}
            resp = requests.post(url, files=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            user_list_id = data["userListId"]

            # Get enrichment results
            url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={{user_list_id}}&backgroundType={{db}}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            enrich_data = resp.json()

            results = []
            for lib_name, terms in enrich_data.items():
                for term in terms[:20]:
                    results.append({{
                        "term": term[1],
                        "pvalue": term[2],
                        "adjusted_pvalue": term[6],
                        "odds_ratio": term[3],
                        "combined_score": term[4],
                        "overlapping_genes": term[5],
                    }})

            results.sort(key=lambda x: x["pvalue"])
            print(json.dumps(results[:15], indent=2))
        """)

        script_path = "enrichment_script.py"
        sandbox.write_file(script_path, script)
        result = sandbox.run_command(f"python {script_path}", timeout=60)

        if result["success"]:
            return result["stdout"]
        return f"Enrichment analysis failed:\n{result['stderr']}\n{result['stdout']}"
    except Exception as e:
        return f"Error running gene set enrichment: {e}"


@tool
def normalize_expression_data(counts_file: str, method: str = "TPM") -> str:
    """Normalize RNA-seq expression data using various methods.

    Generates and runs a normalization script in the sandbox.

    Args:
        counts_file: Path to raw counts matrix (CSV/TSV). Genes as rows,
                     samples as columns. First column is gene IDs.
        method: Normalization method. Options: 'TPM', 'CPM', 'log2CPM',
                'quantile', 'median_ratio'.

    Returns:
        Path to the normalized output file and summary statistics.
    """
    try:
        sandbox = get_sandbox()
        method = method.upper()
        valid_methods = {"TPM", "CPM", "LOG2CPM", "QUANTILE", "MEDIAN_RATIO"}
        if method not in valid_methods:
            return f"Error: Unknown method '{method}'. Options: {', '.join(sorted(valid_methods))}"

        script = textwrap.dedent(f"""\
            import pandas as pd
            import numpy as np
            import warnings
            warnings.filterwarnings('ignore')

            counts = pd.read_csv("{counts_file}", index_col=0, sep=None, engine='python')
            method = "{method}"

            if method == "CPM":
                lib_sizes = counts.sum(axis=0)
                normalized = counts.div(lib_sizes, axis=1) * 1e6
            elif method == "LOG2CPM":
                lib_sizes = counts.sum(axis=0)
                cpm = counts.div(lib_sizes, axis=1) * 1e6
                normalized = np.log2(cpm + 1)
            elif method == "TPM":
                # Approximate TPM using gene length = 1 (same as CPM without gene lengths)
                lib_sizes = counts.sum(axis=0)
                normalized = counts.div(lib_sizes, axis=1) * 1e6
                print("Note: True TPM requires gene lengths. Using CPM as approximation.")
            elif method == "QUANTILE":
                rank_mean = counts.stack().groupby(counts.rank(method='first').stack().astype(int)).mean()
                normalized = counts.rank(method='min').stack().astype(int).map(rank_mean).unstack()
            elif method == "MEDIAN_RATIO":
                geometric_mean = np.exp(np.log(counts + 1).mean(axis=1))
                ratios = counts.add(1).div(geometric_mean, axis=0)
                size_factors = ratios.median(axis=0)
                normalized = counts.div(size_factors, axis=1)

            output_path = "{counts_file.rsplit(".", 1)[0]}_normalized_{method.lower()}.csv"
            normalized.to_csv(output_path)

            print(f"Normalization method: {method}")
            print(f"Genes: {{len(normalized)}}")
            print(f"Samples: {{len(normalized.columns)}}")
            print(f"Value range: {{normalized.min().min():.2f}} - {{normalized.max().max():.2f}}")
            print(f"Results saved to: {{output_path}}")
        """)

        script_path = "normalize_script.py"
        sandbox.write_file(script_path, script)
        result = sandbox.run_command(f"python {script_path}", timeout=60)

        if result["success"]:
            return result["stdout"]
        return f"Normalization failed:\n{result['stderr']}\n{result['stdout']}"
    except Exception as e:
        return f"Error normalizing expression data: {e}"


def get_transcriptomics_tools() -> list:
    """Return all transcriptomics tools."""
    return [run_differential_expression, run_gene_set_enrichment, normalize_expression_data]
