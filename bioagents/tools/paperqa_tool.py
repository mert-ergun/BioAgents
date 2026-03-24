# Increase recursion limit to prevent PDF parsing crashes
# sys.setrecursionlimit(10000)

import argparse
import os

from dotenv import load_dotenv
from paperqa import Settings, ask

# Force load .env
load_dotenv(override=True)


def run_paperqa(pdf_dir, query):
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found. Please check your .env file.")
        return

    print(f"Analyzing PDFs in [{pdf_dir}] using Gemini models in plain text mode...")

    settings = Settings()
    settings.temperature = 0.0
    settings.llm = "gemini/gemini-2.5-flash"
    settings.summary_llm = "gemini/gemini-2.5-flash"
    settings.embedding = "gemini/gemini-embedding-001"

    settings.agent.agent_llm = "gemini/gemini-2.5-flash"
    settings.agent.index.paper_directory = pdf_dir

    # CRITICAL: Disable multimodal to prevent complex PDF crashes
    settings.parsing.multimodal = False

    try:
        response = ask(query, settings=settings)

        print("\n--- OUTPUT START ---\n")
        answer_text = getattr(response, "answer", str(response))
        print(answer_text)
        print("\n--- OUTPUT END ---")

    except Exception as e:
        print(f"ERROR: An issue occurred while running PaperQA: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone PaperQA2 tool for BioAgents")
    parser.add_argument(
        "--pdf_dir", type=str, required=True, help="Absolute or relative path to PDF folder"
    )
    parser.add_argument("--query", type=str, required=True, help="Query to ask over the PDFs")

    args = parser.parse_args()
    run_paperqa(args.pdf_dir, args.query)
