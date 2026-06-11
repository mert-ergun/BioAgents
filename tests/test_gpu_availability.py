"""Tests for GPU availability detection and fallback behavior."""

import json
import os
from unittest.mock import Mock, patch

import gpu_manager
from bioagents.tools.environment_tools import check_gpu_available


class TestIsGpuAvailableAndHasMemory:
    """Tests for gpu_manager.is_gpu_available_and_has_memory."""

    @patch("gpu_manager.torch.cuda.mem_get_info")
    @patch("gpu_manager.torch.cuda.is_available", return_value=True)
    def test_returns_true_when_sufficient_vram(self, _mock_available, mock_mem_get_info):
        """Test that sufficient free VRAM returns True."""
        mock_mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)

        assert gpu_manager.is_gpu_available_and_has_memory(required_vram_gb=4.0) is True

    @patch("gpu_manager.torch.cuda.mem_get_info")
    @patch("gpu_manager.torch.cuda.is_available", return_value=True)
    def test_returns_false_when_vram_insufficient(self, _mock_available, mock_mem_get_info):
        """Test that insufficient free VRAM returns False."""
        mock_mem_get_info.return_value = (0.5 * 1024**3, 16 * 1024**3)

        assert gpu_manager.is_gpu_available_and_has_memory(required_vram_gb=2.0) is False

    @patch("gpu_manager.torch.cuda.is_available", return_value=False)
    def test_returns_false_when_no_cuda(self, _mock_available):
        """Test that missing CUDA returns False without checking memory."""
        assert gpu_manager.is_gpu_available_and_has_memory() is False

    @patch("gpu_manager.torch.cuda.mem_get_info", side_effect=RuntimeError("unsupported"))
    @patch("gpu_manager.torch.cuda.is_available", return_value=True)
    def test_mem_get_info_failure_falls_back_to_true(self, _mock_available, _mock_mem):
        """Test fallback when mem_get_info is unsupported."""
        assert gpu_manager.is_gpu_available_and_has_memory() is True


class TestExecuteModelInferenceGpuFallback:
    """Tests for gpu_manager.execute_model_inference GPU/API routing."""

    @patch("gpu_manager.torch.cuda.mem_get_info")
    @patch("gpu_manager.torch.cuda.is_available", return_value=True)
    def test_runs_locally_when_gpu_has_memory(self, _mock_available, mock_mem_get_info):
        """Test local GPU execution when VRAM is sufficient."""
        mock_mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)

        result = gpu_manager.execute_model_inference(
            sequence="MKTVRQERLK",
            model_name="esm2_t6_8M_UR50D",
            required_vram_gb=2.0,
        )

        assert result["status"] == "success"
        assert result["source"] == "local_gpu"
        assert "esm2_t6_8M_UR50D" in result["result"]

    @patch("gpu_manager.torch.cuda.mem_get_info")
    @patch("gpu_manager.torch.cuda.is_available", return_value=True)
    def test_low_vram_without_key_returns_pending(self, _mock_available, mock_mem_get_info):
        """Test API fallback with engagement signal when VRAM is low and key is missing."""
        mock_mem_get_info.return_value = (0.5 * 1024**3, 16 * 1024**3)

        with patch.dict(os.environ, {}, clear=True):
            result = gpu_manager.execute_model_inference(
                sequence="MKTVRQERLK",
                model_name="esm2_t6_8M_UR50D",
                required_vram_gb=2.0,
            )

        assert result["status"] == "pending"
        assert "[ENGAGEMENT_PENDING]" in result["signal"]
        assert "HUGGINGFACE_API_KEY" in result["signal"]

    @patch("gpu_manager.requests.post")
    @patch("gpu_manager.torch.cuda.mem_get_info")
    @patch("gpu_manager.torch.cuda.is_available", return_value=True)
    def test_low_vram_with_key_routes_to_api(
        self, _mock_available, mock_mem_get_info, mock_post
    ):
        """Test API execution when VRAM is insufficient but API key is present."""
        mock_mem_get_info.return_value = (0.5 * 1024**3, 16 * 1024**3)
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"prediction": "mock_result"}
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "mock_hf_token"}, clear=True):
            result = gpu_manager.execute_model_inference(
                sequence="MKTVRQERLK",
                model_name="esm2_t6_8M_UR50D",
                required_vram_gb=2.0,
            )

        assert result["status"] == "success"
        assert result["source"] == "api_Hugging Face"
        mock_post.assert_called_once()

    @patch("gpu_manager.torch.cuda.is_available", return_value=False)
    def test_no_gpu_without_key_returns_pending(self, _mock_available):
        """Test engagement signal when no GPU and API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = gpu_manager.execute_model_inference(
                sequence="MKTVRQERLK",
                model_name="esm2_t6_8M_UR50D",
            )

        assert result["status"] == "pending"
        assert "[ENGAGEMENT_PENDING]" in result["signal"]

    @patch("gpu_manager.torch.cuda.is_available", return_value=False)
    def test_different_providers_request_different_keys(self, _mock_available):
        """Test that different models request keys for their respective providers."""
        with patch.dict(os.environ, {}, clear=True):
            hf_result = gpu_manager.execute_model_inference(
                sequence="MKTVRQERLK",
                model_name="esm2_t6_8M_UR50D",
            )
            gemini_result = gpu_manager.execute_model_inference(
                sequence="Say Hello",
                model_name="gemini-1.5-flash",
            )

        assert "HUGGINGFACE_API_KEY" in hf_result["signal"]
        assert "GOOGLE_API_KEY" in gemini_result["signal"]

    def test_unregistered_model_returns_error(self):
        """Test safe failure for models not in the registry."""
        result = gpu_manager.execute_model_inference(
            sequence="SEQUENCE",
            model_name="unknown-dna-model",
        )

        assert result["status"] == "error"
        assert "not registered" in result["message"]


class TestCheckGpuAvailable:
    """Tests for environment_tools.check_gpu_available."""

    @patch("bioagents.tools.environment_tools.get_sandbox")
    def test_nvidia_smi_detects_gpu(self, mock_get_sandbox):
        """Test GPU detection via nvidia-smi output."""
        mock_sandbox = Mock()
        mock_sandbox.run_command.side_effect = [
            {
                "success": True,
                "stdout": "NVIDIA A100, 40960 MiB, 38000 MiB, 535.54\n",
                "stderr": "",
            },
            {
                "success": True,
                "stdout": "Cuda compilation tools, release 12.1, V12.1.105",
                "stderr": "",
            },
        ]
        mock_get_sandbox.return_value = mock_sandbox

        result = json.loads(check_gpu_available.invoke({}))

        assert result["gpu_available"] is True
        assert result["num_gpus"] == 1
        assert result["gpus"][0]["name"] == "NVIDIA A100"
        assert "cuda_info" in result

    @patch("bioagents.tools.environment_tools.get_sandbox")
    def test_pytorch_fallback_when_nvidia_smi_unavailable(self, mock_get_sandbox):
        """Test PyTorch fallback detection when nvidia-smi fails."""
        mock_sandbox = Mock()
        mock_sandbox.run_command.side_effect = [
            {"success": False, "stdout": "", "stderr": "nvidia-smi not found"},
            {
                "success": True,
                "stdout": "True\nNVIDIA GeForce RTX 3090",
                "stderr": "",
            },
        ]
        mock_get_sandbox.return_value = mock_sandbox

        result = json.loads(check_gpu_available.invoke({}))

        assert result["gpu_available"] is True
        assert result["detected_via"] == "pytorch"
        assert result["device_name"] == "NVIDIA GeForce RTX 3090"

    @patch("bioagents.tools.environment_tools.get_sandbox")
    def test_no_gpu_detected(self, mock_get_sandbox):
        """Test response when no GPU is detected."""
        mock_sandbox = Mock()
        mock_sandbox.run_command.side_effect = [
            {"success": False, "stdout": "", "stderr": ""},
            {"success": True, "stdout": "False\nN/A", "stderr": ""},
        ]
        mock_get_sandbox.return_value = mock_sandbox

        result = json.loads(check_gpu_available.invoke({}))

        assert result["gpu_available"] is False
        assert "No GPU detected" in result["message"]

    @patch("bioagents.tools.environment_tools.get_sandbox")
    def test_sandbox_exception_returns_error_string(self, mock_get_sandbox):
        """Test error handling when sandbox operations fail."""
        mock_get_sandbox.side_effect = RuntimeError("Sandbox unavailable")

        result = check_gpu_available.invoke({})

        assert isinstance(result, str)
        assert "Error checking GPU" in result
