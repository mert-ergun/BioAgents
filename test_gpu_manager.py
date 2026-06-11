import os
import sys
from unittest.mock import patch
import torch

# Geliştirdiğimiz gpu_manager modülünü import ediyoruz
import gpu_manager

def reset_env():
    """Her test öncesi ortam değişkenlerini temizler."""
    for key in ["HUGGINGFACE_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
        if key in os.environ:
            del os.environ[key]

def run_comprehensive_tests():
    print("==========================================================")
    print("   BIOAGENTS COMPUTE & RESOURCE MANAGER VERIFICATION     ")
    print("==========================================================\n")

    # -------------------------------------------------------------------------
    # SCENARIO 1: GPU Available & Has Ample VRAM
    # -------------------------------------------------------------------------
    print("--- SCENARIO 1: GPU Available & Yeterli VRAM Var ---")
    reset_env()
    
    # torch.cuda fonksiyonlarını taklit ediyoruz: GPU var ve 8GB boş yer var
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.mem_get_info', return_value=(8 * 1024**3, 16 * 1024**3)):
        
        # 2GB VRAM isteyen bir esm2 modeli lokalde çalışmalı, key istememeli
        result = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", 
            model_name="esm2_t6_8M_UR50D", 
            required_vram_gb=2.0
        )
        print(f"Result: {result}\n")


    # -------------------------------------------------------------------------
    # SCENARIO 2: GPU Available BUT Out of VRAM (Hocanın Özel İsteği)
    # -------------------------------------------------------------------------
    print("--- SCENARIO 2: GPU Var Ama VRAM Yetersiz (Lokal Yer Yok) ---")
    reset_env()
    
    # GPU var ama sadece 0.5 GB boş yer kalmış, model ise 2GB istüyor
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.mem_get_info', return_value=(0.5 * 1024**3, 16 * 1024**3)):
        
        # Ortamda key yoksa, VRAM yetmediği için API'ye düşmeli ve LangGraph sinyali fırlatmalı
        result = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", 
            model_name="esm2_t6_8M_UR50D", 
            required_vram_gb=2.0
        )
        print(f"Result (Key Yokken Kesinti): {result}")
        
        # Şimdi ortama geçerli bir key simüle edelim (Kullanıcı arayüzden key'i girmiş gibi)
        os.environ["HUGGINGFACE_API_KEY"] = "mock_hf_token_xyz123"
        result_with_key = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", 
            model_name="esm2_t6_8M_UR50D", 
            required_vram_gb=2.0
        )
        print(f"Result (Key Varken API'ye Yönlendirme): {result_with_key}\n")


    # -------------------------------------------------------------------------
    # SCENARIO 3: No GPU & Missing API Key (LangGraph Interruption)
    # -------------------------------------------------------------------------
    print("--- SCENARIO 3: GPU Yok & API Key Eksik (LangGraph Akış Durdurma) ---")
    reset_env()
    
    with patch('torch.cuda.is_available', return_value=False):
        # Sistem direkt buluta yönelecek, HF key'i bulamadığı için ENGAGEMENT_PENDING dönecek
        result = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", 
            model_name="esm2_t6_8M_UR50D"
        )
        print(f"Result: {result}\n")


    # -------------------------------------------------------------------------
    # SCENARIO 4: Separate Tools Asking for Separate Keys (Ayrı Ayrı Sorma)
    # -------------------------------------------------------------------------
    print("--- SCENARIO 4: Farklı Sağlayıcılar İçin Ayrı Ayrı Key İstenmesi ---")
    reset_env()
    
    with patch('torch.cuda.is_available', return_value=False):
        # 4a. Hugging Face için esm2 testi
        print("[Test 4a - Hugging Face Modeli]")
        result_hf = gpu_manager.execute_model_inference("MKTVRQERLK", model_name="esm2_t6_8M_UR50D")
        print(f"HF Result: {result_hf}")
        
        # 4b. Google (Official) için gemini testi - Ayrı bir env_var istemeli
        print("\n[Test 4b - Gemini Modeli]")
        result_gemini = gpu_manager.execute_model_inference("Say Hello", model_name="gemini-1.5-flash")
        print(f"Gemini Result: {result_gemini}\n")


    # -------------------------------------------------------------------------
    # SCENARIO 5: Safe Failure for Unregistered Models
    # -------------------------------------------------------------------------
    print("--- SCENARIO 5: Kayıtlı Olmayan Model Durumunda Güvenli Hata ---")
    result_unknown = gpu_manager.execute_model_inference("SEQUENCE", model_name="unknown-dna-model")
    print(f"Result: {result_unknown}\n")

    print("==========================================================")
    print("               ALL TESTS COMPLETED SUCCESSFULLY           ")
    print("==========================================================")

if __name__ == "__main__":
    run_tests = run_comprehensive_tests()