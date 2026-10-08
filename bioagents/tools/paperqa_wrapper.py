"""PaperQA wrapper tool for searching local PDF papers."""

import subprocess  # nosec B404
import sys
from pathlib import Path

from langchain_core.tools import tool


@tool
def search_local_papers_with_paperqa(pdf_folder_path: str, query: str) -> str:
    """Answer a question by reading the PDF papers in a local folder (RAG over PDFs).

    Use this to extract methods, summarise findings, or answer a specific question from
    papers already downloaded to disk. It searches ONLY local files — to search published
    literature online use the literature agent's search_pubmed/search_arxiv tools.

    Args:
        pdf_folder_path: Folder containing the PDF files, relative to the project root.
        query: The question to answer or research topic to look for in those papers.

    Returns:
        A prose answer grounded in the PDFs, with the supporting passages cited. If the
        folder has no PDFs, or no passage answers the question, it says so — treat that
        as "not found in these papers", not as evidence of absence, and never fill the
        gap with recalled knowledge presented as a finding from the papers.
    """
    import os

    # Clean up path if LLM incorrectly prepends 'bioagents\'
    if pdf_folder_path.startswith(("bioagents\\", "bioagents/", "bioagents")):
        pdf_folder_path = pdf_folder_path.replace("bioagents\\", "").replace("bioagents/", "")

    print(
        f"\n[Research Agent triggered PaperQA Tool: Folder='{pdf_folder_path}', Query='{query}']\n"
    )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        script_path = Path(__file__).resolve().parent / "paperqa_tool.py"

        result = subprocess.run(  # nosec B603
            [
                sys.executable,
                str(script_path),
                "--pdf_dir",
                pdf_folder_path,
                "--query",
                query,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=120,
        )

        output = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode == 0:
            start_marker = "--- OUTPUT START ---"
            end_marker = "--- OUTPUT END ---"

            if start_marker in output and end_marker in output:
                clean_output = output.split(start_marker)[1].split(end_marker)[0].strip()
                return clean_output
            else:
                return (
                    f"Tool executed but did not produce the expected format. Raw output:\n{output}"
                )
        else:
            return f"PaperQA Tool Error:\n{stderr}\nConsole Output:\n{output}"

    except Exception as e:
        return f"Subprocess execution error: {e!s}"
