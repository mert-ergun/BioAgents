import os
import torch
import requests

from bioagents.tools.provider_utils import get_provider_key_or_ask

# 1. DICTIONARY (REGISTRY) - PROVIDER İSİMLERİ provider_utils.py İLE BİREBİR AYNI OLMALIDIR
MODEL_REGISTRY = {
    "esm2_t6_8M_UR50D": {
        "provider": "Hugging Face", # provider_utils.py içindeki key ile aynı yapıldı
        "api_url": "https://api-inference.huggingface.co/models/facebook/esm2_t6_8M_UR50D"
    },
    "gpt-4-science": {
        "provider": "OpenAI", 
        "api_url": "https://api.openai.com/v1/chat/completions"
    },
    "gemini-1.5-flash": {
        "provider": "Google (Official)", # provider_utils.py içindeki key ile aynı yapıldı
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    }
}

def is_gpu_available_and_has_memory(required_vram_gb=4.0) -> bool:
    """
    Hem GPU'nun varlığını hem de yeterli boş VRAM (bellek) olup olmadığını kontrol eder.
    Hocanın istediği 'yer yoksa API'ye düşsün' mantığını sağlar.
    """
    if not torch.cuda.is_available():
        return False
        
    try:
        # free_memory ve total_memory byte cinsinden döner
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_gb = free_memory / (1024 ** 3)
        
        if free_gb < required_vram_gb:
            print(f"⚠️ GPU bulundu ancak yeterli VRAM yok (Boş: {free_gb:.1f}GB, İstenen: {required_vram_gb}GB). İşlem API'ye taşınıyor...")
            return False
            
        return True
    except Exception:
        # Eğer mem_get_info desteklenmeyen bir CUDA versiyonuysa en azından is_available true dönsün
        return True 

def execute_model_inference(sequence, model_name="esm2_t6_8M_UR50D", required_vram_gb=2.0):
    if model_name not in MODEL_REGISTRY:
        return {"status": "error", "message": f"Model '{model_name}' is not registered!"}

    # Artık sadece GPU var mı diye değil, yer var mı diye de bakıyoruz
    if is_gpu_available_and_has_memory(required_vram_gb):
        return _run_locally(sequence, model_name)
    else:
        model_info = MODEL_REGISTRY[model_name]
        provider = model_info["provider"]
        
        # LangGraph uyumlu, akışı kesen fonksiyonunuz devrede
        api_key_or_signal = get_provider_key_or_ask(provider, model_name)
        
        # DÜZELTME: provider_utils.py çıktısı "Error:" ile başladığı için kontrolü esnetiyoruz
        if isinstance(api_key_or_signal, str) and (api_key_or_signal.startswith("Error:") or "[ENGAGEMENT_PENDING]" in api_key_or_signal):
             # Bu bir key değil, kesinti sinyalidir; aynen LangGraph ajanına fırlatıyoruz
             return {"status": "pending", "signal": api_key_or_signal} 
             
        # Eğer yukarıdaki if'e girmediyse, elimizde gerçek bir key var demektir
        return _run_via_api(sequence, model_info, api_key_or_signal)

def _run_locally(sequence, model_name):
    """Your local execution code (e.g., fair-esm) goes here."""
    return {"status": "success", "source": "local_gpu", "result": f"{model_name} successfully ran locally."}

def _run_via_api(sequence, model_info, api_key):
    """Executes the model via the configured API provider."""
    provider = model_info["provider"]
    url = model_info["api_url"]
    
    headers = {}
    
    if provider == "Google (Official)":
        url = f"{url}?key={api_key}" # Google key'i genelde url query parametresinde ister
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": sequence}]}]}
    elif provider == "Hugging Face":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"inputs": sequence}
    elif provider == "OpenAI":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": sequence}]}
    else:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"data": sequence}
        
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result_data = response.json()
        if provider == "Google (Official)":
            extracted_text = result_data["candidates"][0]["content"]["parts"][0]["text"]
            return {"status": "success", "source": f"api_{provider}", "result": extracted_text}
            
        return {"status": "success", "source": f"api_{provider}", "result": result_data}
    except Exception as e:
        return {"status": "error", "message": str(e), "details": response.text if 'response' in locals() else ""}

        