"""Mock and integration tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# === DEBUG OUTPUT SETUP ===
DEBUG_OUTPUT_DIR = Path("debug_tool_outputs")
DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def debug_write(name, content):
    """Write debug output to a file under debug_tool_outputs/."""
    path = DEBUG_OUTPUT_DIR / name
    with path.open("w", encoding="utf-8") as f:
        if isinstance(content, (dict, list)):
            f.write(json.dumps(content, indent=2))
        else:
            f.write(str(content))
    print(f"✓ Debug written: {path}")


# ===== MOCK TESTS: Fast, no dependencies =====
class TestPdfToolsMocked:
    """Unit tests - using Mock."""

    @patch("bioagents.tools.pdf_tools.DEFAULT_WRAPPER")
    def test_fetch_webpage_mock(self, mock_wrapper):
        """Test if the web fetch function works correctly."""
        from bioagents.tools.pdf_tools import fetch_webpage_as_pdf_text

        mock_wrapper.execute_tool.return_value = "Mocked webpage content"

        result = fetch_webpage_as_pdf_text.invoke({"url": "https://biomni.stanford.edu/"})

        debug_write("mock_webpage_test.txt", result)

        assert result == "Mocked webpage content"
        mock_wrapper.execute_tool.assert_called_once()

    @patch("bioagents.tools.pdf_tools.DEFAULT_WRAPPER")
    def test_fetch_webpage_error_mock(self, mock_wrapper):
        """Error condition test."""
        from bioagents.tools.pdf_tools import fetch_webpage_as_pdf_text

        mock_wrapper.execute_tool.side_effect = Exception("Connection timeout")

        result = fetch_webpage_as_pdf_text.invoke({"url": "https://biomni.stanford.edu/"})

        debug_write("mock_webpage_error.txt", result)

        assert "Error" in result
        assert "Connection timeout" in result


# ===== INTEGRATION TESTS: Real reading =====
@pytest.mark.integration
class TestPdfToolsIntegration:
    """Integration tests - Real reading."""

    def test_fetch_real_webpage(self):
        """Real web page reading."""
        from bioagents.tools.pdf_tools import fetch_webpage_as_pdf_text

        print("\n🌐 Reading real web page...")

        try:
            result = fetch_webpage_as_pdf_text.invoke(
                {"url": "https://biomni.stanford.edu/", "timeout": 30}
            )

            debug_write("integration_real_webpage.txt", result)

            # Basic checks
            assert len(result) > 0, "Web page returned empty"
            assert "Error" not in result or "example" in result.lower(), (
                f"Unexpected error: {result[:200]}"
            )

            print(f"✓ Web content successfully retrieved: {len(result)} characters")

        except Exception as e:
            error_msg = f"Web reading error: {type(e).__name__}: {e!s}"
            debug_write("integration_webpage_error.txt", error_msg)
            pytest.fail(error_msg)

    def test_extract_real_pdf(self):
        """Real PDF reading."""
        from bioagents.tools.pdf_tools import (
            HAS_PDF_LIBRARIES,
            extract_pdf_text_spacy_layout,
        )

        if not HAS_PDF_LIBRARIES:
            pytest.skip("Required libraries not installed ('spacy', 'spacy_layout')")

        print("\n📄 Searching for PDF file...")

        # Try multiple possible paths
        test_paths = [
            "tests/test_files/sample.pdf",
            "BioAgents/tests/test_files/sample.pdf",
            str(Path("tests") / "test_files" / "sample.pdf"),
            str(Path("BioAgents") / "tests" / "test_files" / "sample.pdf"),
            str((Path(__file__).parent / "sample.pdf").resolve()),
        ]

        test_pdf = None
        for path in test_paths:
            if Path(path).exists():
                test_pdf = path
                print(f"✓ PDF found: {path}")
                break

        if not test_pdf:
            skip_msg = "Test PDF not found. Paths tried:\n  - " + "\n  - ".join(test_paths)
            print(f"⚠️  {skip_msg}")
            pytest.skip(skip_msg)

        try:
            result = extract_pdf_text_spacy_layout.invoke({"local_pdf_path": test_pdf})

            debug_write("integration_real_pdf.md", result)

            # Basic checks
            assert len(result) > 0, "PDF text returned empty"
            assert "Error" not in result, f"PDF reading error: {result[:200]}"

            print(f"✓ PDF successfully read: {len(result)} characters")

        except Exception as e:
            error_msg = f"PDF reading error: {type(e).__name__}: {e!s}"
            debug_write("integration_pdf_error.txt", error_msg)
            pytest.fail(error_msg)


# ===== DEBUG TEST: Always runs =====
class TestDebugOutput:
    """Verify that the debug system works."""

    def test_debug_write_works(self):
        """Test if the debug write function works."""
        test_content = {"test": "value", "timestamp": "2024-01-01", "items": [1, 2, 3]}

        debug_write("test_debug_output.json", test_content)

        path = DEBUG_OUTPUT_DIR / "test_debug_output.json"
        assert path.exists(), f"Debug file not created: {path}"

        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == test_content
        print(f"✓ Debug test successful: {path}")
