"""PaperQA wrapper tool for searching local PDF papers."""

import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool


@tool
def search_local_papers_with_paperqa(pdf_folder_path: str, query: str) -> str:
    """
    Answers user questions by reading local PDF papers in the specified directory.
    Use this tool to extract information, summarize literature, or answer specific
    questions based on PDFs.

    Args:
        pdf_folder_path: The folder path where the PDF files are located.
        query: The question or research topic to search for in the papers.
    """
    import os

    # Clean up path if LLM incorrectly prepends 'bioagents\'
    if pdf_folder_path.startswith(("bioagents\\", "bioagents/", "bioagents")):
        pdf_folder_path = pdf_folder_path.replace("bioagents\\", "").replace("bioagents/", "")

    print(
        f"\n[Research Agent triggered PaperQA Tool: "
        f"Folder='{pdf_folder_path}', Query='{query}']\n"
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
                    f"Tool executed but did not produce the expected format. "
                    f"Raw output:\n{output}"
                )
        else:
            return f"PaperQA Tool Error:\n{stderr}\nConsole Output:\n{output}"

    except Exception as e:
        return f"Subprocess execution error: {str(e)}"
