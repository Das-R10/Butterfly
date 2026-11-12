"""
tests/test_llm_diagnostics.py

Unit and integration tests for LLM Model Selectors, Ollama Integration,
Hardware GPU Diagnostics, and No-Silent-Mock Enforcement.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app
from src.core.llm_client import LLMClient, LLMInferenceError
from src.core.gpu_diagnostics import get_gpu_diagnostics
from src.core.ollama_service import (
    check_ollama_health,
    get_installed_ollama_models,
    is_model_installed,
    unload_model_from_vram,
)
from src.core.schemas import SimulationConfig
from src.core.simulation_engine import SimulationEngine

client = TestClient(app)


def test_llm_health_endpoint():
    """GET /api/llm/health returns complete status and never leaks API keys or secrets."""
    res = client.get("/api/llm/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert "ollama" in data
    assert "gpu" in data
    assert "cloud_providers" in data
    assert "mock" in data

    # Verify no secret values are returned
    text_data = str(data)
    assert "sk-" not in text_data
    assert "api_key" not in data.get("cloud_providers", {}).get("groq", {})
    assert "api_key" not in data.get("cloud_providers", {}).get("gemini", {})


def test_llm_models_endpoint_when_ollama_unavailable():
    """When Ollama is not running, /api/llm/models reports connected=false and models not installed."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = client.get("/api/llm/models")
        assert res.status_code == 200
        data = res.json()

        assert data["ollama_connected"] is False
        assert len(data["installed_models"]) == 0
        assert "Ollama is not available" in data["error"]

        # Recommended models must be present but NOT marked as installed
        rec_models = data["recommended_models"]
        assert len(rec_models) >= 2
        for m in rec_models:
            assert m["is_installed"] is False


def test_llm_models_endpoint_when_ollama_connected():
    """When Ollama responds, tags are mapped and recommended models verified accurately."""
    mock_tags_response = {
        "models": [
            {
                "name": "qwen3.5:4b",
                "model": "qwen3.5:4b:latest",
                "size": 2800000000,
                "digest": "abc123456789",
                "modified_at": "2026-11-01T12:00:00Z",
            }
        ]
    }

    with patch("requests.get") as mock_get:
        # First call is /api/version (health), second is /api/tags (models)
        health_resp = MagicMock(status_code=200)
        health_resp.json.return_value = {"version": "0.5.1"}

        tags_resp = MagicMock(status_code=200)
        tags_resp.json.return_value = mock_tags_response

        mock_get.side_effect = [health_resp, tags_resp]

        res = client.get("/api/llm/models")
        assert res.status_code == 200
        data = res.json()

        assert data["ollama_connected"] is True
        assert len(data["installed_models"]) == 1
        assert data["installed_models"][0]["name"] == "qwen3.5:4b"

        # qwen3.5:4b is confirmed installed; gemma3:4b remains not installed
        recs = {m["model"]: m["is_installed"] for m in data["recommended_models"]}
        assert recs["qwen3.5:4b"] is True
        assert recs["gemma3:4b"] is False


def test_llm_providers_endpoint():
    """GET /api/llm/providers returns the 4 supported categories with recommended local config."""
    res = client.get("/api/llm/providers")
    assert res.status_code == 200
    data = res.json()

    providers = data["providers"]
    provider_ids = [p["id"] for p in providers]
    assert "ollama" in provider_ids
    assert "groq" in provider_ids
    assert "gemini" in provider_ids
    assert "mock" in provider_ids

    # Cloud providers require API key
    groq_p = next(p for p in providers if p["id"] == "groq")
    assert groq_p["requires_api_key"] is True

    gemini_p = next(p for p in providers if p["id"] == "gemini")
    assert gemini_p["requires_api_key"] is True

    # Mock is categorized as mock (not production)
    mock_p = next(p for p in providers if p["id"] == "mock")
    assert mock_p["category"] == "mock"
    assert "Development / Testing Only" in mock_p["name"]

    # Recommended local dual-model configuration
    rec_cfg = data["recommended_local_config"]
    assert rec_cfg["model_a"]["provider"] == "ollama"
    assert rec_cfg["model_a"]["model"] == "qwen3.5:4b"
    assert rec_cfg["model_b"]["provider"] == "ollama"
    assert rec_cfg["model_b"]["model"] == "gemma3:4b"


def test_ollama_fails_without_silent_mock_fallback():
    """Real Ollama inference MUST raise LLMInferenceError when Ollama is unavailable, never silently falling back to mock."""
    ollama_client = LLMClient(provider="ollama", model_name="qwen3.5:4b")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(LLMInferenceError) as exc_info:
            ollama_client.generate_json("Test prompt", "System instruction")

        assert "Ollama is not available" in str(exc_info.value)


def test_ollama_missing_model_raises_error():
    """Ollama HTTP 404 must raise LLMInferenceError instructing the user to pull the model."""
    ollama_client = LLMClient(provider="ollama", model_name="nonexistent:model")

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock(status_code=404, text="model not found")
        mock_post.return_value = mock_resp

        with pytest.raises(LLMInferenceError) as exc_info:
            ollama_client.generate_json("Test prompt")

        assert "is not installed" in str(exc_info.value)
        assert "ollama pull" in str(exc_info.value)


def test_simulation_engine_rejects_unavailable_ollama():
    """SimulationEngine pre-flight validation aborts early when Ollama is requested but offline."""
    cfg = SimulationConfig(
        signal="Test economic signal",
        provider_a="ollama",
        model_a="qwen3.5:4b",
        provider_b="mock",
        model_b="mock-beta",
        max_iterations=3,
    )
    engine = SimulationEngine()
    with patch("src.core.simulation_engine.check_ollama_health", return_value={"running": False}):
        with pytest.raises(RuntimeError) as exc_info:
            engine.run_simulation(cfg)

        assert "Ollama is not available" in str(exc_info.value)


def test_simulation_engine_rejects_missing_model():
    """SimulationEngine pre-flight validation aborts when model is missing from Ollama."""
    cfg = SimulationConfig(
        signal="Test economic signal",
        provider_a="ollama",
        model_a="missing_model:latest",
        provider_b="mock",
        model_b="mock-beta",
        max_iterations=3,
    )
    engine = SimulationEngine()
    with patch("src.core.simulation_engine.check_ollama_health", return_value={"running": True}):
        with patch("src.core.simulation_engine.is_model_installed", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                engine.run_simulation(cfg)

            assert "is not installed" in str(exc_info.value)
            assert "ollama pull" in str(exc_info.value)


def test_gpu_diagnostics_structure():
    """get_gpu_diagnostics returns verified status without claiming false acceleration."""
    diag = get_gpu_diagnostics()
    assert "available" in diag
    assert "verified" in diag
    assert "vram_total_mb" in diag
    assert "sequential_recommended" in diag
    if diag["available"]:
        assert diag["verified"] is True
        assert "5070" in diag["name"] or "NVIDIA" in diag["name"]
        assert diag["vram_total_mb"] > 0


def test_vram_unloading_on_sequential_switch():
    """unload_model_from_vram sends keep_alive:0 to Ollama API."""
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        res = unload_model_from_vram("qwen3.5:4b")
        assert res is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["keep_alive"] == 0
        assert call_kwargs["json"]["model"] == "qwen3.5:4b"


def test_ollama_timeout_separated_and_default_300s():
    """Ollama requests separate connect (10s) and read/inference (300s) timeouts."""
    ollama_client = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"response": '{"stance": 0.5, "virality_estimate": 0.5, "perceived_credibility": 0.5, "emotional_charge": 0.5, "primary_narrative": "test", "key_arguments": [], "narrative_tags": []}'}
        mock_post.return_value = mock_resp

        res = ollama_client.generate_json("Test signal prompt")
        assert res["stance"] == 0.5
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "timeout" in call_kwargs
        connect_t, read_t = call_kwargs["timeout"]
        assert connect_t == 10.0
        assert read_t == 300.0


def test_ollama_read_timeout_raises_clear_error():
    """ReadTimeout raises LLMInferenceError with explicit duration information."""
    ollama_client = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    with patch("requests.post", side_effect=requests.exceptions.ReadTimeout("Read timed out")):
        with pytest.raises(LLMInferenceError) as exc_info:
            ollama_client.generate_json("Prompt")
        assert "inference timed out after 300" in str(exc_info.value)
        assert "qwen3.5:4b" in str(exc_info.value)


def test_ollama_connect_timeout_raises_clear_error():
    """ConnectTimeout raises LLMInferenceError with connect duration."""
    ollama_client = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    with patch("requests.post", side_effect=requests.exceptions.ConnectTimeout("Connect timed out")):
        with pytest.raises(LLMInferenceError) as exc_info:
            ollama_client.generate_json("Prompt")
        assert "connection timed out after 10" in str(exc_info.value)
