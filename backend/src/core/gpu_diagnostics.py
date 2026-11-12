"""
src/core/gpu_diagnostics.py

Hardware and GPU diagnostics utility.
Detects NVIDIA GPUs (e.g., RTX 5070 Laptop GPU, 8 GB VRAM), queries memory usage,
and verifies hardware acceleration status without false claims.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, Optional
from loguru import logger


def get_gpu_diagnostics() -> Dict[str, Any]:
    """
    Queries the host system for verified NVIDIA GPU hardware details and VRAM metrics.
    Only reports verified=True when GPU presence and memory are directly confirmed.
    """
    diagnostics: Dict[str, Any] = {
        "available": False,
        "name": None,
        "vram_total_mb": None,
        "vram_used_mb": None,
        "vram_free_mb": None,
        "driver_version": None,
        "verified": False,
        "sequential_recommended": True,
        "details": "No GPU detected.",
    }

    nvsmi_path = shutil.which("nvidia-smi")
    if nvsmi_path:
        try:
            res = subprocess.run(
                [
                    nvsmi_path,
                    "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                # Line format: "Name, Total, Used, Free, Driver"
                first_line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                if len(parts) >= 4:
                    name = parts[0]
                    total_mb = float(parts[1])
                    used_mb = float(parts[2])
                    free_mb = float(parts[3])
                    driver = parts[4] if len(parts) > 4 else None

                    diagnostics["available"] = True
                    diagnostics["name"] = name
                    diagnostics["vram_total_mb"] = total_mb
                    diagnostics["vram_used_mb"] = used_mb
                    diagnostics["vram_free_mb"] = free_mb
                    diagnostics["driver_version"] = driver
                    diagnostics["verified"] = True
                    diagnostics["sequential_recommended"] = total_mb <= 12288.0  # 12 GB or lower
                    diagnostics["details"] = (
                        f"{name} ({total_mb:.0f} MB VRAM, {free_mb:.0f} MB free). "
                        f"Sequential execution {'recommended' if diagnostics['sequential_recommended'] else 'optional'}."
                    )
                    return diagnostics
        except Exception as e:
            logger.warning(f"Failed to query nvidia-smi: {e}")

    # Fallback to PyTorch CUDA check if available
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "CUDA Device"
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if device_count > 0 else 0.0

            diagnostics["available"] = True
            diagnostics["name"] = device_name
            diagnostics["vram_total_mb"] = round(total_mem, 1)
            diagnostics["verified"] = True
            diagnostics["sequential_recommended"] = total_mem <= 12288.0
            diagnostics["details"] = f"{device_name} detected via PyTorch CUDA."
            return diagnostics
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"PyTorch CUDA check skipped: {e}")

    diagnostics["details"] = "No verified NVIDIA GPU hardware detected via nvidia-smi or CUDA."
    return diagnostics

