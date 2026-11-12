"""
ba/tests/conftest.py

Global pytest storage isolation and test hygiene configuration for The Butterfly Algorithm.

Guarantees:
  - Every test automatically receives isolated temporary directories for
    `SESSIONS_DIR`, `OUTPUTS_DIR`, and `REPORTS_DIR` via pytest's `tmp_path`.
  - Automated test runs NEVER write mock or fixture files into production
    `data/sessions/`, `data/outputs/`, or `data/reports/`.
  - FastAPI in-memory async job store (`_JOBS`) is cleared before and after each test.
  - Verifies at session teardown that production `data/sessions/` gained zero files.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Set

import pytest

_BA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BA_ROOT, ".."))
if _BA_ROOT not in sys.path:
    sys.path.insert(0, _BA_ROOT)

import src.core.forecast_tracking as forecast_tracking_mod
import src.core.report_generator as report_generator_mod
import src.core.simulation_engine as simulation_engine_mod
from src.api.main import _JOBS


def _snapshot_production_dirs() -> dict[str, Set[str]]:
    candidates: Set[str] = set()
    for rel in (simulation_engine_mod.PRODUCTION_SESSIONS_DIR, forecast_tracking_mod.PRODUCTION_FORECASTS_DIR):
        candidates.add(os.path.abspath(os.path.join(_REPO_ROOT, rel) if not os.path.isabs(rel) else rel))
        candidates.add(os.path.abspath(rel))
    snap: dict[str, Set[str]] = {}
    for prod_dir in candidates:
        if os.path.exists(prod_dir):
            snap[prod_dir] = {
                os.path.basename(p)
                for p in glob.glob(os.path.join(prod_dir, "*"))
                if os.path.isfile(p)
            }
        else:
            snap[prod_dir] = set()
    return snap


_INITIAL_PROD_SNAP = _snapshot_production_dirs()


@pytest.fixture(autouse=True)
def isolated_storage_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Automatically isolates every test from production storage directories
    and resets in-memory async job state.
    """
    sessions_dir = tmp_path / "sessions"
    outputs_dir = tmp_path / "outputs"
    reports_dir = tmp_path / "reports"
    forecasts_dir = tmp_path / "forecasts"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(simulation_engine_mod, "SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setattr(simulation_engine_mod, "OUTPUTS_DIR", str(outputs_dir))
    monkeypatch.setattr(simulation_engine_mod, "REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(report_generator_mod, "REPORTS_DIR", str(reports_dir))
    monkeypatch.setattr(forecast_tracking_mod, "FORECASTS_DIR", str(forecasts_dir))

    _JOBS.clear()
    yield {
        "sessions_dir": sessions_dir,
        "outputs_dir": outputs_dir,
        "reports_dir": reports_dir,
        "forecasts_dir": forecasts_dir,
    }
    _JOBS.clear()


@pytest.fixture(scope="session", autouse=True)
def verify_no_production_session_leakage():
    """Session-wide guardrail asserting pytest never pollutes production `data/sessions/`."""
    yield
    after_snap = _snapshot_production_dirs()
    for prod_dir, after_files in after_snap.items():
        initial_files = _INITIAL_PROD_SNAP.get(prod_dir, set())
        new_files = sorted(after_files - initial_files)
        assert not new_files, (
            f"Test suite leaked {len(new_files)} file(s) into production directory "
            f"{prod_dir}: {new_files[:5]}"
        )
