"""
src/core/storage_hygiene.py

Explicit one-time backup, forensic audit, and clean reset utilities for
The Butterfly Algorithm session archives.
NEVER executed automatically on application startup; must be invoked explicitly.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


KNOWN_TEST_SIGNALS = {
    "Autonomous drone swarm deployed in exercise",
    "Border skirmish reported",
    "Diplomatic sanctions introduced",
    "Critical semiconductor export embargo announced",
    "Supply chain cyberattack",
    "New currency union formed",
    "Central bank evaluates digital currency infrastructure.",
    "Test economic signal",
    "A major city announces congestion pricing starting next month.",
    "Global chipmaker faces unexpected packaging bottleneck delaying AI accelerators by two quarters.",
    "Major quantum decryption milestone claimed in unverified preprint.",
}


def categorize_session_data(filepath: str, data: Dict[str, Any]) -> str:
    """Classifies a persisted simulation JSON record by origin."""
    cfg = data.get("config") or {}
    diag_models = (data.get("diagnostics") or {}).get("selected_models") or {}
    sig = str(data.get("signal_text") or cfg.get("signal") or "").strip()

    pa = cfg.get("provider_a") or (diag_models.get("model_a") or {}).get("provider")
    pb = cfg.get("provider_b") or (diag_models.get("model_b") or {}).get("provider")

    if not pa or not pb:
        iters = data.get("iterations") or []
        if iters and isinstance(iters, list) and iters[0].get("micro_interpretations"):
            provs = list(dict.fromkeys(str(m.get("provider", "unknown")) for m in iters[0]["micro_interpretations"]))
            pa = pa or (provs[0] if provs else "unknown")
            pb = pb or (provs[1] if len(provs) > 1 else pa)

    pa_low = str(pa or "unknown").lower()
    pb_low = str(pb or "unknown").lower()

    if sig in KNOWN_TEST_SIGNALS or sig.startswith("Synthetic ") or "pytest" in os.path.basename(filepath).lower():
        return "automated_test_mock"
    if pa_low == "mock" and pb_low == "mock":
        return "mock_development_run"
    if any(p in ("ollama", "groq", "gemini", "google") for p in (pa_low, pb_low)):
        return "real_llm_simulation"
    return "legacy_or_other"


def audit_sessions_directory(sessions_dir: str) -> Dict[str, Any]:
    """Scans a sessions directory and returns a category breakdown without mutating files."""
    if not os.path.exists(sessions_dir):
        return {
            "sessions_dir": sessions_dir,
            "total_records": 0,
            "categories": {},
            "files": [],
        }

    json_files = sorted(glob.glob(os.path.join(sessions_dir, "*.json")))
    categories: collections.Counter[str] = collections.Counter()
    details: List[Dict[str, Any]] = []

    for fpath in json_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            cat = categorize_session_data(fpath, data)
            categories[cat] += 1
            cfg = data.get("config") or {}
            details.append({
                "filename": fname,
                "simulation_id": data.get("simulation_id") or fname[:-5],
                "category": cat,
                "provider_a": cfg.get("provider_a"),
                "provider_b": cfg.get("provider_b"),
                "model_a": cfg.get("model_a"),
                "model_b": cfg.get("model_b"),
                "timestamp": data.get("timestamp"),
                "signal_preview": str(data.get("signal_text") or cfg.get("signal") or "")[:90],
            })
        except Exception as exc:
            categories["corrupt_json"] += 1
            details.append({
                "filename": fname,
                "simulation_id": fname[:-5],
                "category": "corrupt_json",
                "error": str(exc),
            })

    return {
        "sessions_dir": sessions_dir,
        "total_records": len(json_files),
        "categories": dict(categories),
        "files": details,
    }


def backup_and_reset_sessions(
    sessions_dir: str,
    backups_dir: str,
    outputs_dir: Optional[str] = None,
    reports_dir: Optional[str] = None,
    timestamp_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Explicitly backs up all session records to `backups_dir/sessions_backup_<timestamp>/`
    and then performs a clean one-time reset of the active session storage (and stale generated
    reports/outputs if specified). Never invoked automatically at app startup.
    """
    audit = audit_sessions_directory(sessions_dir)
    stamp = timestamp_tag or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(backups_dir, f"sessions_backup_{stamp}")
    os.makedirs(backup_folder, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(sessions_dir, "*.json"))) if os.path.exists(sessions_dir) else []
    copied_count = 0
    for src_path in json_files:
        dst_path = os.path.join(backup_folder, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied_count += 1

    if copied_count != len(json_files):
        raise RuntimeError(
            f"Backup verification mismatch: expected {len(json_files)} files, copied {copied_count}. Aborting reset."
        )

    manifest = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_sessions_dir": os.path.abspath(sessions_dir),
        "backup_dir": os.path.abspath(backup_folder),
        "total_backed_up": copied_count,
        "categories": audit["categories"],
        "records": audit["files"],
    }
    manifest_path = os.path.join(backup_folder, "backup_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    # Verify all backed-up files exist before removing originals
    for src_path in json_files:
        if not os.path.exists(os.path.join(backup_folder, os.path.basename(src_path))):
            raise RuntimeError(f"Missing backup file for {src_path}. Aborting reset.")

    removed_sessions = 0
    for src_path in json_files:
        os.remove(src_path)
        removed_sessions += 1

    # Also remove any stray non-backup index files inside sessions_dir if present
    if os.path.exists(sessions_dir):
        for leftover in os.listdir(sessions_dir):
            leftover_path = os.path.join(sessions_dir, leftover)
            if os.path.isfile(leftover_path):
                os.remove(leftover_path)

    removed_reports = 0
    if reports_dir and os.path.exists(reports_dir):
        for fname in os.listdir(reports_dir):
            fpath = os.path.join(reports_dir, fname)
            if os.path.isfile(fpath) and (fname.endswith(".pdf") or fname.endswith(".meta.json")):
                os.remove(fpath)
                removed_reports += 1

    removed_outputs = 0
    if outputs_dir and os.path.exists(outputs_dir):
        for fname in os.listdir(outputs_dir):
            fpath = os.path.join(outputs_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                removed_outputs += 1

    remaining_after = len(glob.glob(os.path.join(sessions_dir, "*.json"))) if os.path.exists(sessions_dir) else 0

    return {
        "backup_dir": os.path.abspath(backup_folder),
        "manifest_path": os.path.abspath(manifest_path),
        "total_backed_up": copied_count,
        "categories": audit["categories"],
        "removed_sessions": removed_sessions,
        "removed_reports": removed_reports,
        "removed_outputs": removed_outputs,
        "active_sessions_remaining": remaining_after,
    }

