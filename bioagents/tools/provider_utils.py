import os

def get_provider_key_or_ask(provider: str, tool_name: str) -> str:
    """
    Checks if the required API key for the given provider is in the environment.
    Returns the API key if present.
    Returns an [ENGAGEMENT_PENDING] string to interrupt the LangGraph stream if missing.
    """
    env_map = {
        "Tamarind Bio": "TAMARIND_API_KEY",
        "NVIDIA BioNeMo": "NVIDIA_API_KEY",
        "Vertex AI": "VERTEX_API_KEY",
        "Neurosnap": "NEUROSNAP_API_KEY",
        "Levitate Bio": "LEVITATE_API_KEY",
        "Hugging Face": "HUGGINGFACE_API_KEY",
        "Hugging Face (Weights)": "HUGGINGFACE_API_KEY",
        "EvolutionaryScale Forge": "EVOSCALE_API_KEY",
        "AWS SageMaker": "AWS_API_KEY",
        "Google (Official)": "GOOGLE_API_KEY"
    }

    req_env = env_map.get(provider)
    if not req_env:
        return "NO_KEY_REQUIRED"

    api_key = os.environ.get(req_env)
    
    if not api_key:
        return (
            f"Error: Missing API key for {provider}. "
            f"You must ask the user for this key. Your output MUST ONLY be in this format:\n"
            f'[ENGAGEMENT_PENDING] {{"type": "api_key_request", "env_var": "{req_env}", "question": "Please enter your API key for {provider} to use {tool_name}:"}}'
        )
        
    return api_key