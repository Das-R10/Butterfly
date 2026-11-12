"""
ba/tests/test_phase5_hygiene_and_reports.py

Comprehensive Phase 5 verification suite covering:
  Part A–C: Storage Hygiene, Backup & Reset, Demo-Free Startup, and Test/Mock Isolation
  Part D–G: Professional Research PDF Report Generation, API Caching, Identity Isolation & Cleanup
  Part H: All 11 required regression assertions including PyMuPDF (fitz) PDF inspection
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.core.report_generator as report_generator_mod
import src.core.simulation_engine as simulation_engine_mod
from src.api.main import app, _JOBS
from src.core.llm_client import LLMClient
from src.core.schemas import SimulationConfig, SimulationStatus
from src.core.simulation_engine import SimulationEngine
from src.core.storage_hygiene import (
    audit_sessions_directory,
    backup_and_reset_sessions,
    categorize_session_data,
)


def _create_isolated_simulation(
    sessions_dir: str,
    signal: str,
    event_title: str,
    source_url: str = "https://research.example.org/signal",
    max_iterations: int = 3,
    provider_a: str = "ollama",
    model_a: str = "qwen3.5:4b",
    provider_b: str = "ollama",
    model_b: str = "gemma3:4b",
    make_partial: bool = False,
    make_failed: bool = False,
) -> Dict[str, Any]:
    """
    Creates an isolated saved simulation JSON inside `sessions_dir` using mock LLM responses
    under non-mock provider labels when needed (or Ollama failure simulation for partial/failed).
    Never touches production directories.
    """
    client_a = LLMClient(provider="mock", model_name=model_a)
    client_b = LLMClient(provider="mock", model_name=model_b)

    if make_partial or make_failed:
        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.json.return_value = {"response": "", "done": True, "done_reason": "stop"}
        if make_partial:
            client_a = LLMClient(provider="ollama", model_name=model_a)
        if make_failed:
            client_a = LLMClient(provider="ollama", model_name=model_a)
            client_b = LLMClient(provider="ollama", model_name=model_b)

        engine = SimulationEngine(client_a=client_a, client_b=client_b)
        cfg = SimulationConfig(
            signal=signal,
            event_title=event_title,
            source_url=source_url,
            max_iterations=max_iterations,
            provider_a="ollama" if make_partial or make_failed else provider_a,
            model_a=model_a,
            provider_b="ollama" if make_failed else "mock",
            model_b=model_b,
        )
        with patch.dict(os.environ, {"OLLAMA_MAX_RETRIES": "0"}), \
             patch("src.core.simulation_engine.check_ollama_health", return_value={"running": True}), \
             patch("src.core.simulation_engine.is_model_installed", return_value=True), \
             patch("src.core.llm_client.requests.post", return_value=empty_resp), \
             patch("src.core.micro_layer.unload_model_from_vram"):
            result = engine.run_simulation(cfg)
    else:
        engine = SimulationEngine(client_a=client_a, client_b=client_b)
        cfg = SimulationConfig(
            signal=signal,
            event_title=event_title,
            source_url=source_url,
            max_iterations=max_iterations,
            provider_a="mock",
            model_a=model_a,
            provider_b="mock",
            model_b=model_b,
        )
        result = engine.run_simulation(cfg)

    data = result.model_dump()
    if provider_a != "mock":
        data["config"]["provider_a"] = provider_a
    if provider_b != "mock":
        data["config"]["provider_b"] = provider_b

    os.makedirs(sessions_dir, exist_ok=True)
    out_path = os.path.join(sessions_dir, f"{result.simulation_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


# ── 1 & 2. Clean Reset + Backup Contains Original Records & Category Counts ────


def test_backup_and_clean_reset_categories_and_zero_remaining(tmp_path):
    base = tmp_path / "hygiene_workspace"
    sessions_dir = base / "sessions"
    backups_dir = base / "backups"
    outputs_dir = base / "outputs"
    reports_dir = base / "reports"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Seed representative categorized records in temp directory
    # 1. Automated pytest mock signature
    s_test = {
        "simulation_id": "sim_test_001",
        "event_title": "Test Battery Discovery",
        "signal_text": "Autonomous drone swarm deployed in exercise",
        "config": {"provider_a": "mock", "provider_b": "mock", "model_a": "groq-llama-70b", "model_b": "gemini-flash"},
        "timestamp": "2026-11-03T18:00:00Z",
    }
    # 2. Development mock run
    s_dev = {
        "simulation_id": "sim_dev_002",
        "event_title": "Dev Signal",
        "signal_text": "Custom user signal tested with deterministic mock",
        "config": {"provider_a": "mock", "provider_b": "mock", "model_a": "mock-a", "model_b": "mock-b"},
        "timestamp": "2026-11-03T19:00:00Z",
    }
    # 3. Genuine Ollama run
    s_real = {
        "simulation_id": "sim_real_003",
        "event_title": "Genuine Local Ollama Study",
        "signal_text": "OpenAI launched GPT-5.1 across regional enterprise tiers.",
        "config": {"provider_a": "ollama", "provider_b": "ollama", "model_a": "qwen3.5:4b", "model_b": "gemma3:4b"},
        "timestamp": "2026-11-03T23:14:09Z",
    }
    p_test = str(sessions_dir / f"{s_test['simulation_id']}.json")
    p_dev = str(sessions_dir / f"{s_dev['simulation_id']}.json")
    p_real = str(sessions_dir / f"{s_real['simulation_id']}.json")
    for p, s in ((p_test, s_test), (p_dev, s_dev), (p_real, s_real)):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(s, f)

    # Add an extra dummy artifact in reports/outputs
    (reports_dir / "sim_test_001.pdf").write_bytes(b"%PDF-1.4 dummy")
    (outputs_dir / "old_trace.txt").write_text("stale", encoding="utf-8")

    assert categorize_session_data(p_test, s_test) == "automated_test_mock"
    assert categorize_session_data(p_dev, s_dev) == "mock_development_run"
    assert categorize_session_data(p_real, s_real) == "real_llm_simulation"

    audit_before = audit_sessions_directory(str(sessions_dir))
    assert audit_before["total_records"] == 3

    summary = backup_and_reset_sessions(
        sessions_dir=str(sessions_dir),
        backups_dir=str(backups_dir),
        outputs_dir=str(outputs_dir),
        reports_dir=str(reports_dir),
    )

    # Verify active workspace reset cleanly to 0
    assert summary["active_sessions_remaining"] == 0
    assert summary["removed_sessions"] == 3
    assert summary["removed_reports"] == 1
    assert summary["removed_outputs"] == 1
    assert glob.glob(os.path.join(str(sessions_dir), "*.json")) == []

    # Verify timestamped backup contains all 3 originals + accurate manifest
    bdir = summary["backup_dir"]
    assert os.path.isdir(bdir)
    backed_jsons = glob.glob(os.path.join(bdir, "*.json"))
    # 3 session files + 1 backup_manifest.json
    assert len(backed_jsons) == 4
    manifest_path = summary["manifest_path"]
    assert os.path.isfile(manifest_path)
    manifest = json.loads(open(manifest_path, "r", encoding="utf-8").read())
    assert manifest["total_backed_up"] == 3
    assert manifest["categories"]["automated_test_mock"] == 1
    assert manifest["categories"]["mock_development_run"] == 1
    assert manifest["categories"]["real_llm_simulation"] == 1


# ── 3. App Startup / Listing Does Not Recreate Demo Data ──────────────────────


def test_startup_and_endpoints_never_recreate_demo_data(isolated_storage_dirs):
    client = TestClient(app)

    r_list_1 = client.get("/sessions")
    r_list_2 = client.get("/api/simulations")
    assert r_list_1.status_code == 200
    assert r_list_2.status_code == 200
    assert r_list_1.json()["sessions"] == []
    assert r_list_2.json()["sessions"] == []
    assert glob.glob(os.path.join(str(isolated_storage_dirs["sessions_dir"]), "*.json")) == []


# ── 4. Running Tests & Mock Runs Never Pollute Production data/sessions/ ──────


def test_mock_runs_refuse_production_sessions_dir_and_tests_stay_isolated(
    tmp_path, monkeypatch
):
    fake_prod = tmp_path / "prod_sessions"
    fake_prod.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(simulation_engine_mod, "PRODUCTION_SESSIONS_DIR", str(fake_prod))
    monkeypatch.setattr(simulation_engine_mod, "SESSIONS_DIR", str(fake_prod))

    engine = SimulationEngine(
        client_a=LLMClient(provider="mock", model_name="mock-a"),
        client_b=LLMClient(provider="mock", model_name="mock-b"),
    )
    cfg = SimulationConfig(
        signal="Deterministic mock pollution check signal",
        event_title="Mock Guardrail Test",
        max_iterations=3,
        provider_a="mock",
        model_a="mock-a",
        provider_b="mock",
        model_b="mock-b",
    )
    res = engine.run_simulation(cfg)
    assert res.status == SimulationStatus.COMPLETED.value
    assert res.diagnostics["is_mock_only"] is True
    # Must NOT have written anything into fake production directory
    assert list(fake_prod.glob("*.json")) == []

    # Now point SESSIONS_DIR to isolated test dir -> persistence DOES happen there cleanly
    iso_test = tmp_path / "iso_test_sessions"
    monkeypatch.setattr(simulation_engine_mod, "SESSIONS_DIR", str(iso_test))
    res2 = engine.run_simulation(cfg)
    saved = iso_test / f"{res2.simulation_id}.json"
    assert saved.is_file()


# ── 5, 6, 8. Full Academic PDF Verification (Completed, Partial & Failed) ──────


def test_pdf_generation_complete_partial_failed_via_pymupdf(isolated_storage_dirs):
    sessions_dir = str(isolated_storage_dirs["sessions_dir"])
    reports_dir = str(isolated_storage_dirs["reports_dir"])
    client = TestClient(app)

    # A) Completed simulation with rich signal & source URL
    sim_ok = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="Clean hydrogen electrolyzer subsidies expansion signed into regional grid policy <verified & audited>.",
        event_title="Clean Hydrogen Grid Policy <Phase 5>",
        source_url="https://energy.example.gov/bulletins/hydrogen-2026",
        max_iterations=3,
    )
    sim_ok_id = sim_ok["simulation_id"]

    # Pre-status must show report does not exist
    st_before = client.get(f"/api/simulations/{sim_ok_id}/report/status")
    assert st_before.status_code == 200
    assert st_before.json()["report_exists"] is False

    # Download before generate must return honest 404
    dl_before = client.get(f"/api/simulations/{sim_ok_id}/report/download")
    assert dl_before.status_code == 404

    # Generate PDF via endpoint
    gen_resp = client.post(f"/api/simulations/{sim_ok_id}/report")
    assert gen_resp.status_code == 200
    gen_json = gen_resp.json()
    assert gen_json["simulation_id"] == sim_ok_id
    assert gen_json["report_exists"] is True
    assert gen_json["cached"] is False
    assert gen_json["filename"].endswith(".pdf")
    assert gen_json["file_size_bytes"] > 10_000

    pdf_file = os.path.join(reports_dir, f"{sim_ok_id}.pdf")
    assert os.path.isfile(pdf_file)

    # Inspect actual generated PDF using PyMuPDF (fitz)
    with fitz.open(pdf_file) as doc:
        assert len(doc) >= 6
        full_text = "\n".join(page.get_text("text") for page in doc)
        total_images = sum(len(page.get_images(full=True)) for page in doc)

    assert "THE BUTTERFLY ALGORITHM" in full_text
    assert "Clean Hydrogen Grid Policy <Phase 5>" in full_text
    assert sim_ok_id in full_text
    assert "COMPLETED" in full_text
    assert "Micro Layer" in full_text
    assert "Meso Layer" in full_text
    assert "Macro Layer" in full_text
    assert "Feedback Layer" in full_text
    assert "Cross-Layer" in full_text
    assert "Technical Appendix" in full_text
    assert "dN/dt" in full_text
    assert "Page 1 of " in full_text
    # All 4 charts (Micro, Meso, Macro, Feedback) embedded cleanly
    assert total_images >= 4

    # Listing summary reflects has_report == True
    sessions = client.get("/api/simulations").json()["sessions"]
    ok_summary = next(s for s in sessions if s["simulation_id"] == sim_ok_id)
    assert ok_summary["has_report"] is True

    # B) Partially Completed simulation (Model A fails, Model B succeeds -> N/A Agreement)
    sim_part = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="Grid storage tariff reform triggers split commentary across energy utilities.",
        event_title="Grid Tariff Partial Failure Case",
        max_iterations=3,
        make_partial=True,
    )
    part_id = sim_part["simulation_id"]
    r_part = client.post(f"/api/simulations/{part_id}/report")
    assert r_part.status_code == 200
    part_pdf = os.path.join(reports_dir, f"{part_id}.pdf")
    with fitz.open(part_pdf) as doc_part:
        assert len(doc_part) >= 5
        part_text = "\n".join(page.get_text("text") for page in doc_part)
    assert "PARTIALLY COMPLETED" in part_text
    assert part_id in part_text
    assert "Grid Tariff Partial Failure Case" in part_text
    assert "N/A" in part_text
    assert "15/30" in part_text

    # C) Failed simulation (0 valid interpretations across both models)
    sim_fail = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="Unavailable provider test for failed simulation report verification.",
        event_title="All Models Failed Report Case",
        max_iterations=3,
        make_failed=True,
    )
    fail_id = sim_fail["simulation_id"]
    r_fail = client.post(f"/api/simulations/{fail_id}/report")
    assert r_fail.status_code == 200
    fail_pdf = os.path.join(reports_dir, f"{fail_id}.pdf")
    with fitz.open(fail_pdf) as doc_fail:
        fail_text = "\n".join(page.get_text("text") for page in doc_fail)
    assert "FAILED" in fail_text
    assert "0/30" in fail_text
    assert fail_id in fail_text


# ── 7. Missing / Invalid / Corrupted Simulation Error Handling ────────────────


def test_report_endpoints_handle_missing_invalid_and_corrupted_records(
    isolated_storage_dirs,
):
    sessions_dir = str(isolated_storage_dirs["sessions_dir"])
    client = TestClient(app)

    # Nonexistent ID -> 404
    assert client.post("/api/simulations/sim_does_not_exist/report").status_code == 404
    assert client.get("/api/simulations/sim_does_not_exist/report/status").status_code == 404
    assert client.get("/api/simulations/sim_does_not_exist/report/download").status_code == 404

    # Path traversal ID -> 400
    assert client.post("/api/simulations/..%2Fsecret/report").status_code in (400, 404)
    assert client.post("/api/simulations/..secret/report").status_code == 400
    assert client.get("/api/simulations/..secret/report/download").status_code == 400

    # Empty / 0-iteration corrupted record -> 422 with clear error message
    bad_id = "sim_corrupt_empty"
    with open(os.path.join(sessions_dir, f"{bad_id}.json"), "w", encoding="utf-8") as f:
        json.dump({"simulation_id": bad_id, "signal_text": "Broken record", "iterations": []}, f)
    resp = client.post(f"/api/simulations/{bad_id}/report")
    assert resp.status_code == 422
    assert "0 execution iterations" in resp.json()["detail"]


# ── 9. Download & Cached Report Never Triggers Inference or Mutates JSON ──────


def test_download_and_cached_report_never_calls_llm_or_mutates_json(isolated_storage_dirs):
    sessions_dir = str(isolated_storage_dirs["sessions_dir"])
    client = TestClient(app)

    sim = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="Quantum key distribution trial announced across municipal fiber backbones.",
        event_title="Quantum Fiber Backbone Study",
        max_iterations=3,
    )
    sim_id = sim["simulation_id"]
    json_path = os.path.join(sessions_dir, f"{sim_id}.json")
    raw_before = open(json_path, "rb").read()
    mtime_before = os.path.getmtime(json_path)

    with patch("src.core.llm_client.LLMClient.generate_structured", side_effect=AssertionError("LLM must not be called!")), \
         patch("src.core.simulation_engine.SimulationEngine.run_simulation", side_effect=AssertionError("Engine must not be called!")):
        # First POST generates from persisted JSON
        first = client.post(f"/api/simulations/{sim_id}/report")
        assert first.status_code == 200
        assert first.json()["cached"] is False

        # Second POST hits cache cleanly
        second = client.post(f"/api/simulations/{sim_id}/report")
        assert second.status_code == 200
        assert second.json()["cached"] is True

        # Download GET endpoints serve binary PDF cleanly
        dl_1 = client.get(f"/api/simulations/{sim_id}/report/download")
        dl_2 = client.get(f"/api/simulations/{sim_id}/report")
        for dl in (dl_1, dl_2):
            assert dl.status_code == 200
            assert dl.headers["content-type"] == "application/pdf"
            assert ".pdf" in dl.headers.get("content-disposition", "")
            assert dl.content.startswith(b"%PDF-")

    # Source JSON remains bit-for-bit identical and unmutated
    raw_after = open(json_path, "rb").read()
    assert raw_after == raw_before
    assert os.path.getmtime(json_path) == mtime_before


# ── 10. Two Different Simulations Never Mix Content Inside Reports ────────────


def test_two_simulation_reports_are_strictly_isolated(isolated_storage_dirs):
    sessions_dir = str(isolated_storage_dirs["sessions_dir"])
    reports_dir = str(isolated_storage_dirs["reports_dir"])
    client = TestClient(app)

    sim_a = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="UNIQUE_TOKEN_ALPHA: Maritime autonomous cargo corridor opens in North Sea.",
        event_title="Alpha Maritime Freight Report",
        max_iterations=3,
    )
    sim_b = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="UNIQUE_TOKEN_OMEGA: Central alpine geothermal district enters phase two.",
        event_title="Omega Alpine Geothermal Report",
        max_iterations=3,
    )
    id_a, id_b = sim_a["simulation_id"], sim_b["simulation_id"]

    assert client.post(f"/api/simulations/{id_a}/report").status_code == 200
    assert client.post(f"/api/simulations/{id_b}/report").status_code == 200

    with fitz.open(os.path.join(reports_dir, f"{id_a}.pdf")) as doc_a:
        text_a = "\n".join(p.get_text("text") for p in doc_a)
    with fitz.open(os.path.join(reports_dir, f"{id_b}.pdf")) as doc_b:
        text_b = "\n".join(p.get_text("text") for p in doc_b)

    assert "UNIQUE_TOKEN_ALPHA" in text_a
    assert "Alpha Maritime Freight Report" in text_a
    assert id_a in text_a
    assert "UNIQUE_TOKEN_OMEGA" not in text_a
    assert "Omega Alpine Geothermal Report" not in text_a
    assert id_b not in text_a

    assert "UNIQUE_TOKEN_OMEGA" in text_b
    assert "Omega Alpine Geothermal Report" in text_b
    assert id_b in text_b
    assert "UNIQUE_TOKEN_ALPHA" not in text_b
    assert "Alpha Maritime Freight Report" not in text_b
    assert id_a not in text_b


# ── 11. Deleting a Simulation Removes JSON + Report and Preserves Others ──────


def test_delete_simulation_cleans_associated_pdf_and_leaves_siblings(isolated_storage_dirs):
    sessions_dir = str(isolated_storage_dirs["sessions_dir"])
    reports_dir = str(isolated_storage_dirs["reports_dir"])
    client = TestClient(app)

    sim_1 = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="Delete target simulation signal one.",
        event_title="Delete Target One",
        max_iterations=3,
    )
    sim_2 = _create_isolated_simulation(
        sessions_dir=sessions_dir,
        signal="Survivor simulation signal two.",
        event_title="Survivor Record Two",
        max_iterations=3,
    )
    id_1, id_2 = sim_1["simulation_id"], sim_2["simulation_id"]

    assert client.post(f"/api/simulations/{id_1}/report").status_code == 200
    assert client.post(f"/api/simulations/{id_2}/report").status_code == 200

    assert os.path.isfile(os.path.join(reports_dir, f"{id_1}.pdf"))
    assert os.path.isfile(os.path.join(reports_dir, f"{id_1}.meta.json"))
    assert os.path.isfile(os.path.join(reports_dir, f"{id_2}.pdf"))

    del_resp = client.delete(f"/api/simulations/{id_1}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Deleted JSON & report artifacts are gone
    assert not os.path.exists(os.path.join(sessions_dir, f"{id_1}.json"))
    assert not os.path.exists(os.path.join(reports_dir, f"{id_1}.pdf"))
    assert not os.path.exists(os.path.join(reports_dir, f"{id_1}.meta.json"))

    # Sibling JSON & report remain intact
    assert os.path.isfile(os.path.join(sessions_dir, f"{id_2}.json"))
    assert os.path.isfile(os.path.join(reports_dir, f"{id_2}.pdf"))

    # Deleted ID excluded from both listing endpoints and 404 on report routes
    list_1 = [s["simulation_id"] for s in client.get("/sessions").json()["sessions"]]
    list_2 = [s["simulation_id"] for s in client.get("/api/simulations").json()["sessions"]]
    assert id_1 not in list_1 and id_1 not in list_2
    assert id_2 in list_1 and id_2 in list_2
    assert client.get(f"/api/simulations/{id_1}/report/download").status_code == 404
