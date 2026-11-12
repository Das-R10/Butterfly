"""
src/core/ollama_service.py

Ollama Service & Diagnostics Utility.
Queries local Ollama instance (http://localhost:11434) for:
  - Binary installation status on host
  - Service connectivity (health & version)
  - Installed local model tags
  - Verification of recommended models (qwen3.5:4b, gemma3:4b)
  - Explicit VRAM model unloading for sequential execution
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional
import requests
from loguru import logger

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

RECOMMENDED_LOCAL_MODELS = [
    {
        "model": "qwen3.5:4b",
        "display_name": "Qwen 3.5 4B",
        "role": "Primary Analyst (Model A)",
        "parameter_size": "4B",
        "approx_vram_gb": 2.8,
        "pull_command": "ollama pull qwen3.5:4b",
    },
    {
        "model": "gemma3:4b",
        "display_name": "Gemma 3 4B",
        "role": "Independent Reviewer (Model B)",
        "parameter_size": "4B",
        "approx_vram_gb": 3.1,
        "pull_command": "ollama pull gemma3:4b",
    },
]


def check_ollama_installation() -> Dict[str, Any]:
    """Checks whether the Ollama binary is installed on the host system."""
    binary_path = shutil.which("ollama")
    
    # Common Windows installation fallback path
    if not binary_path:
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidate = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
            if os.path.isfile(candidate):
                binary_path = candidate

    return {
        "installed": binary_path is not None,
        "binary_path": binary_path,
        "install_command": "winget install Ollama.Ollama",
    }


def check_ollama_health(base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Pings the Ollama HTTP service to verify it is actively listening.
    Does not assume Ollama is running just because binary is installed.
    """
    url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    install_info = check_ollama_installation()

    result: Dict[str, Any] = {
        "installed": install_info["installed"],
        "binary_path": install_info["binary_path"],
        "running": False,
        "base_url": url,
        "version": None,
        "error": None,
    }

    try:
        resp = requests.get(f"{url}/api/version", timeout=1.5)
        if resp.status_code == 200:
            result["running"] = True
            result["version"] = resp.json().get("version", "unknown")
            return result
        else:
            result["error"] = f"Ollama HTTP returned status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        result["error"] = "Ollama is not available. Install Ollama and start its local service before running local inference."
    except requests.exceptions.Timeout:
        result["error"] = f"Ollama connection timed out at {url}"
    except Exception as e:
        result["error"] = f"Ollama connectivity check failed: {str(e)}"

    return result


def get_installed_ollama_models(base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Queries GET /api/tags from Ollama to retrieve actually installed models.
    Strictly reports installed status; recommended models are only marked installed
    if confirmed present by Ollama.
    """
    health = check_ollama_health(base_url)
    url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")

    if not health["running"]:
        # Do not invent models.
        recommended_annotated = [
            {
                **m,
                "is_installed": False,
            }
            for m in RECOMMENDED_LOCAL_MODELS
        ]
        return {
            "ollama_connected": False,
            "base_url": url,
            "installed_models": [],
            "recommended_models": recommended_annotated,
            "error": health["error"] or "Ollama is not available. Install Ollama and start its local service before running local inference.",
        }

    try:
        resp = requests.get(f"{url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            raw_models = data.get("models", [])
            
            installed_list = []
            installed_names = set()

            for item in raw_models:
                m_name = item.get("name", "")
                installed_names.add(m_name.lower())
                # Also index bare name without :latest or tag
                if ":" in m_name:
                    installed_names.add(m_name.split(":")[0].lower())

                installed_list.append({
                    "name": m_name,
                    "model": item.get("model", m_name),
                    "size_bytes": item.get("size", 0),
                    "size_human": f"{item.get('size', 0) / (1024**3):.1f} GB" if item.get("size") else "N/A",
                    "modified_at": item.get("modified_at"),
                    "digest": item.get("digest", "")[:12] if item.get("digest") else "",
                })

            # Check recommended models strictly
            def is_match(rec_model: str) -> bool:
                rec_lower = rec_model.lower()
                rec_bare = rec_lower.split(":")[0]
                return (
                    rec_lower in installed_names
                    or f"{rec_lower}:latest" in installed_names
                    or any(name.startswith(rec_bare) for name in installed_names)
                )

            recommended_annotated = [
                {
                    **m,
                    "is_installed": is_match(m["model"]),
                }
                for m in RECOMMENDED_LOCAL_MODELS
            ]

            return {
                "ollama_connected": True,
                "base_url": url,
                "version": health["version"],
                "installed_models": installed_list,
                "recommended_models": recommended_annotated,
                "error": None,
            }
        else:
            return {
                "ollama_connected": False,
                "base_url": url,
                "installed_models": [],
                "recommended_models": [{**m, "is_installed": False} for m in RECOMMENDED_LOCAL_MODELS],
                "error": f"Ollama returned HTTP {resp.status_code} when listing models.",
            }
    except Exception as e:
        logger.warning(f"Failed to query Ollama models: {e}")
        return {
            "ollama_connected": False,
            "base_url": url,
            "installed_models": [],
            "recommended_models": [{**m, "is_installed": False} for m in RECOMMENDED_LOCAL_MODELS],
            "error": str(e),
        }


def is_model_installed(model_name: str, base_url: Optional[str] = None) -> bool:
    """Checks if a specific model tag is installed in local Ollama."""
    info = get_installed_ollama_models(base_url)
    if not info.get("ollama_connected"):
        return False

    name_lower = model_name.strip().lower()
    for m in info.get("installed_models", []):
        m_tag = m["name"].lower()
        if m_tag == name_lower or m_tag == f"{name_lower}:latest" or name_lower == f"{m_tag}:latest":
            return True
        if ":" in name_lower and m_tag == name_lower:
            return True
    return False


def unload_model_from_vram(model_name: str, base_url: Optional[str] = None) -> bool:
    """
    Sends an unload request to Ollama by invoking /api/generate with keep_alive=0.
    Freed VRAM prevents memory contention on 8 GB VRAM GPUs when switching models.
    """
    if not model_name:
        return False
    url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=2.0,
        )
        if resp.status_code == 200:
            logger.info(f"[Ollama] Unloaded '{model_name}' from VRAM (keep_alive=0).")
            return True
    except Exception as e:
        logger.debug(f"[Ollama] Unload request for '{model_name}' skipped: {e}")
    return False

