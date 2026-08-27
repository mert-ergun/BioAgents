import os
import torch
import requests

from bioagents.tools.provider_utils import get_provider_key_or_ask

# 1. DICTIONARY (REGISTRY) - PROVIDER NAMES MUST BE EXACTLY THE SAME AS IN provider_utils.py
MODEL_REGISTRY = {
    "esm2_t6_8M_UR50D": {
        "provider": "Hugging Face", # Made the same as the key in provider_utils.py
        "api_url": "https://api-inference.huggingface.co/models/facebook/esm2_t6_8M_UR50D"
    },
    "gpt-4-science": {
        "provider": "OpenAI", 
        "api_url": "https://api.openai.com/v1/chat/completions"
    },
    "gemini-1.5-flash": {
        "provider": "Google (Official)", # Made the same as the key in provider_utils.py
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    }
}

def is_gpu_available_and_has_memory(required_vram_gb=4.0) -> bool:
    """
    Checks both the availability of the GPU and whether there is enough free VRAM (memory).
    Provides the 'fallback to API if no space' logic requested by the professor.
    """
    if not torch.cuda.is_available():
        return False
        
    try:
        # free_memory and total_memory return in bytes
        free_memory, _total_memory = torch.cuda.mem_get_info()
        free_gb = free_memory / (1024 ** 3)
        
        if free_gb < required_vram_gb:
            print(f"⚠️ GPU found but not enough VRAM (Free: {free_gb:.1f}GB, Required: {required_vram_gb}GB). Moving process to API...")
            return False
            
        return True
    except Exception:
        # If mem_get_info is an unsupported CUDA version, at least let is_available return true
        return True 

def execute_model_inference(sequence, model_name="esm2_t6_8M_UR50D", required_vram_gb=2.0):
    if model_name not in MODEL_REGISTRY:
        return {"status": "error", "message": f"Model '{model_name}' is not registered!"}

    # Now we are checking not only if there is a GPU, but also if there is space
    if is_gpu_available_and_has_memory(required_vram_gb):
        return _run_locally(sequence, model_name)
    else:
        model_info = MODEL_REGISTRY[model_name]
        provider = model_info["provider"]
        
        # Your LangGraph compatible, flow-interrupting function is activated
        api_key_or_signal = get_provider_key_or_ask(provider, model_name)
        
        # FIX: We are loosening the check because the output of provider_utils.py starts with "Error:"
        if isinstance(api_key_or_signal, str) and (api_key_or_signal.startswith("Error:") or "[ENGAGEMENT_PENDING]" in api_key_or_signal):
             # This is not a key, it's an interruption signal; throwing it exactly to the LangGraph agent
             return {"status": "pending", "signal": api_key_or_signal} 
             
        # If it didn't enter the if statement above, it means we have a real key
        return _run_via_api(sequence, model_info, api_key_or_signal)

def _run_locally(_sequence, model_name):
    """Your local execution code (e.g., fair-esm) goes here."""
    return {"status": "success", "source": "local_gpu", "result": f"{model_name} successfully ran locally."}

def _run_via_api(sequence, model_info, api_key):
    """Executes the model via the configured API provider."""
    provider = model_info["provider"]
    url = model_info["api_url"]
    
    headers = {}
    
    if provider == "Google (Official)":
        url = f"{url}?key={api_key}" # Google usually asks for the key in the url query parameter
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
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result_data = response.json()
        if provider == "Google (Official)":
            extracted_text = result_data["candidates"][0]["content"]["parts"][0]["text"]
            return {"status": "success", "source": f"api_{provider}", "result": extracted_text}
            
        return {"status": "success", "source": f"api_{provider}", "result": result_data}
    except Exception as e:
        return {"status": "error", "message": str(e), "details": response.text if 'response' in locals() else ""}