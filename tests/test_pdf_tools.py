"""Hem mock hem de integration testler."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock

# === DEBUG OUTPUT SETUP ===
DEBUG_OUTPUT_DIR = "debug_tool_outputs"
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

def debug_write(name, content):
    """Write debug output to a file under debug_tool_outputs/."""
    path = os.path.join(DEBUG_OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(content, (dict, list)):
            f.write(json.dumps(content, indent=2))
        else:
            f.write(str(content))
    print(f"✓ Debug yazıldı: {path}")


# ===== MOCK TESTLER: Hızlı, bağımlılık yok =====
class TestPdfToolsMocked:
    """Birim testler - Mock kullanarak."""
    
    @patch('bioagents.tools.pdf_tools.DEFAULT_WRAPPER')
    def test_fetch_webpage_mock(self, mock_wrapper):
        """Web fetch fonksiyonunun doğru çalıştığını test et."""
        from bioagents.tools.pdf_tools import fetch_webpage_as_pdf_text
        
        mock_wrapper.execute_tool.return_value = "Mocked webpage content"
        
        result = fetch_webpage_as_pdf_text.invoke({"url": "https://biomni.stanford.edu/"})
        
        debug_write("mock_webpage_test.txt", result)
        
        assert result == "Mocked webpage content"
        mock_wrapper.execute_tool.assert_called_once()
    
    @patch('bioagents.tools.pdf_tools.DEFAULT_WRAPPER')
    def test_fetch_webpage_error_mock(self, mock_wrapper):
        """Hata durumu testi."""
        from bioagents.tools.pdf_tools import fetch_webpage_as_pdf_text
        
        mock_wrapper.execute_tool.side_effect = Exception("Connection timeout")
        
        result = fetch_webpage_as_pdf_text.invoke({"url": "https://biomni.stanford.edu/"})
        
        debug_write("mock_webpage_error.txt", result)
        
        assert "Error" in result
        assert "Connection timeout" in result


# ===== INTEGRATION TESTLER: Gerçek okuma =====
@pytest.mark.integration
class TestPdfToolsIntegration:
    """Entegrasyon testleri - Gerçek okuma."""
    
    def test_fetch_real_webpage(self):
        """Gerçek web sayfası okuması."""
        from bioagents.tools.pdf_tools import fetch_webpage_as_pdf_text
        
        print("\n🌐 Gerçek web sayfası okunuyor...")
        
        try:
            result = fetch_webpage_as_pdf_text.invoke({
                "url": "https://biomni.stanford.edu/",
                "timeout": 30
            })
            
            debug_write("integration_real_webpage.txt", result)
            
            # Temel kontroller
            assert len(result) > 0, "Web sayfası boş döndü"
            assert "Error" not in result or "example" in result.lower(), f"Beklenmeyen hata: {result[:200]}"
            
            print(f"✓ Web içeriği başarıyla alındı: {len(result)} karakter")
            
        except Exception as e:
            error_msg = f"Web okuma hatası: {type(e).__name__}: {str(e)}"
            debug_write("integration_webpage_error.txt", error_msg)
            pytest.fail(error_msg)
    
    def test_extract_real_pdf(self):
        """Gerçek PDF okuması."""
        from bioagents.tools.pdf_tools import extract_pdf_text_spacy_layout
        
        print("\n📄 PDF dosyası aranıyor...")
        
        # Birden fazla olası yol dene
        test_paths = [
            "tests/test_files/sample.pdf",
            "BioAgents/tests/test_files/sample.pdf",
            os.path.join("tests","test_files", "sample.pdf"),
            os.path.join("BioAgents", "tests", "test_files", "sample.pdf"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "sample.pdf")),
        ]
        
        test_pdf = None
        for path in test_paths:
            if os.path.exists(path):
                test_pdf = path
                print(f"✓ PDF bulundu: {path}")
                break
        
        if not test_pdf:
            skip_msg = f"Test PDF bulunamadı. Denenen yollar:\n  - " + "\n  - ".join(test_paths)
            print(f"⚠️  {skip_msg}")
            pytest.skip(skip_msg)
        
        try:
            result = extract_pdf_text_spacy_layout.invoke({
                "local_pdf_path": test_pdf
            })
            
            debug_write("integration_real_pdf.md", result)
            
            # Temel kontroller
            assert len(result) > 0, "PDF metni boş döndü"
            assert "Error" not in result, f"PDF okuma hatası: {result[:200]}"
            
            print(f"✓ PDF başarıyla okundu: {len(result)} karakter")
            
        except Exception as e:
            error_msg = f"PDF okuma hatası: {type(e).__name__}: {str(e)}"
            debug_write("integration_pdf_error.txt", error_msg)
            pytest.fail(error_msg)


# ===== DEBUG TEST: Her zaman çalışır =====
class TestDebugOutput:
    """Debug sisteminin çalıştığını doğrula."""
    
    def test_debug_write_works(self):
        """Debug yazma fonksiyonunun çalıştığını test et."""
        test_content = {
            "test": "value",
            "timestamp": "2024-01-01",
            "items": [1, 2, 3]
        }
        
        debug_write("test_debug_output.json", test_content)
        
        path = os.path.join(DEBUG_OUTPUT_DIR, "test_debug_output.json")
        assert os.path.exists(path), f"Debug dosyası oluşturulmadı: {path}"
        
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert loaded == test_content
        print(f"✓ Debug test başarılı: {path}")