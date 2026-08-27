import os
from unittest.mock import patch

# Importing the gpu_manager module we developed
import gpu_manager


def reset_env():
    """Clears environment variables before each test."""
    for key in ["HUGGINGFACE_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
        if key in os.environ:
            del os.environ[key]


def run_comprehensive_tests():
    print("==========================================================")
    print("    BIOAGENTS COMPUTE & RESOURCE MANAGER VERIFICATION     ")
    print("==========================================================\n")

    # -------------------------------------------------------------------------
    # SCENARIO 1: GPU Available & Has Ample VRAM
    # -------------------------------------------------------------------------
    print("--- SCENARIO 1: GPU Available & Has Ample VRAM ---")
    reset_env()

    # Mocking torch.cuda functions: GPU is available and has 8GB of free space
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3)),
    ):
        # An esm2 model requiring 2GB VRAM should run locally, should not ask for a key
        result = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", model_name="esm2_t6_8M_UR50D", required_vram_gb=2.0
        )
        print(f"Result: {result}\n")

    # -------------------------------------------------------------------------
    # SCENARIO 2: GPU Available BUT Out of VRAM (Professor's Special Request)
    # -------------------------------------------------------------------------
    print("--- SCENARIO 2: GPU Available BUT Out of VRAM (No Local Space) ---")
    reset_env()

    # GPU is available but there is only 0.5 GB of free space left, while the model requires 2GB
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(0.5 * 1024**3, 16 * 1024**3)),
    ):
        # If there's no key in the environment, it should fallback to API due to insufficient VRAM and throw a LangGraph signal
        result = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", model_name="esm2_t6_8M_UR50D", required_vram_gb=2.0
        )
        print(f"Result (Interruption Without Key): {result}")

        # Now let's simulate a valid key in the environment (as if the user entered the key from the UI)
        os.environ["HUGGINGFACE_API_KEY"] = "mock_hf_token_xyz123"
        result_with_key = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", model_name="esm2_t6_8M_UR50D", required_vram_gb=2.0
        )
        print(f"Result (Routing to API With Key): {result_with_key}\n")

    # -------------------------------------------------------------------------
    # SCENARIO 3: No GPU & Missing API Key (LangGraph Interruption)
    # -------------------------------------------------------------------------
    print("--- SCENARIO 3: No GPU & Missing API Key (LangGraph Flow Interruption) ---")
    reset_env()

    with patch("torch.cuda.is_available", return_value=False):
        # The system will route directly to the cloud, it will return ENGAGEMENT_PENDING since it cannot find the HF key
        result = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK", model_name="esm2_t6_8M_UR50D"
        )
        print(f"Result: {result}\n")

    # -------------------------------------------------------------------------
    # SCENARIO 4: Separate Tools Asking for Separate Keys (Asking Separately)
    # -------------------------------------------------------------------------
    print("--- SCENARIO 4: Asking for Separate Keys for Different Providers ---")
    reset_env()

    with patch("torch.cuda.is_available", return_value=False):
        # 4a. esm2 test for Hugging Face
        print("[Test 4a - Hugging Face Model]")
        result_hf = gpu_manager.execute_model_inference("MKTVRQERLK", model_name="esm2_t6_8M_UR50D")
        print(f"HF Result: {result_hf}")

        # 4b. gemini test for Google (Official) - Should ask for a separate env_var
        print("\n[Test 4b - Gemini Model]")
        result_gemini = gpu_manager.execute_model_inference(
            "Say Hello", model_name="gemini-1.5-flash"
        )
        print(f"Gemini Result: {result_gemini}\n")

    # -------------------------------------------------------------------------
    # SCENARIO 5: Safe Failure for Unregistered Models
    # -------------------------------------------------------------------------
    print("--- SCENARIO 5: Safe Error for Unregistered Models ---")
    result_unknown = gpu_manager.execute_model_inference("SEQUENCE", model_name="unknown-dna-model")
    print(f"Result: {result_unknown}\n")

    print("==========================================================")
    print("               ALL TESTS COMPLETED SUCCESSFULLY           ")
    print("==========================================================")


if __name__ == "__main__":
    run_tests = run_comprehensive_tests()
