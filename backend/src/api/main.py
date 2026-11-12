"""
src/api/main.py

FastAPI application — ButterflyAI REST API + WebSocket.

Core 4-Layer Architecture Endpoints:
  POST   /api/simulations/run                    Run 4-layer simulation synchronously
  POST   /api/simulations/run/async              Start async simulation, return job_id & simulation_id
  POST   /api/simulations/perturb                Run counterfactual perturbation experiment
  POST   /api/simulations/evaluate               Evaluate simulation against empirical observations
  GET    /api/simulations                        List past simulations with summary metadata
  GET    /api/simulations/{sim_id}               Retrieve isolated simulation record
  DELETE /api/simulations/{sim_id}               Delete a single simulation record safely
  GET    /api/simulations/{sim_id}/trajectory    Retrieve trajectory time series

Preserved Legacy & Registry Endpoints:
  POST   /signals/analyze                        Run full pipeline (routes to 4-layer engine)
  POST   /signals/analyze/async                  Start async job, return job_id immediately
  GET    /jobs/{job_id}                          Poll async job status
  WS     /ws/{job_id}                            Stream live progress for a job
  GET    /registry/pending                       List predictions awaiting resolution
  POST   /registry/resolve                       Mark ground truth on a locked prediction
  GET    /registry/leaderboard                   Archetype Brier Score leaderboard
  GET    /sessions                               List past simulations (compatible summary shape)
  GET    /sessions/{session_id}                  Retrieve isolated session record
  DELETE /sessions/{session_id}                  Delete a single session record
  GET    /health                                 Heartbeat
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure project paths are in sys.path for direct module resolution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ba_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _ba_root not in sys.path:
    sys.path.insert(0, _ba_root)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from src.core.schemas import SimulationConfig, SimulationResult
import src.core.simulation_engine as simulation_engine_mod
import src.core.report_generator as report_generator_mod
import src.core.forecast_tracking as forecast_tracking_mod
from src.core.simulation_engine import SimulationEngine, derive_event_title
from src.core.forecasting_engine import ensure_forecast_fields
from src.core.forecast_tracking import (
    get_forecast_record,
    list_tracked_forecasts,
    resolve_forecast_outcome,
)
from src.core.report_generator import (
    ReportGenerationError,
    build_download_filename,
    generate_simulation_pdf_report,
    get_report_paths,
    get_report_status,
)
from src.core.perturbation import PerturbationExperiment
from src.core.evaluation import TrajectoryEvaluator
from src.core.ollama_service import check_ollama_health, get_installed_ollama_models
from src.core.gpu_diagnostics import get_gpu_diagnostics

app = FastAPI(
    title="ButterflyAI",
    description="The Butterfly Algorithm — 4-Layer Behavioral Dynamics Engine",
    version="4.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store for async runs
_JOBS: dict[str, dict] = {}
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _get_sessions_dir() -> str:
    return getattr(simulation_engine_mod, "SESSIONS_DIR", "data/sessions")


def _get_reports_dir() -> str:
    return getattr(report_generator_mod, "REPORTS_DIR", "data/reports")


def _validate_safe_id(raw_id: str) -> str:
    if not raw_id or not _SAFE_ID_RE.match(raw_id) or ".." in raw_id:
        raise HTTPException(status_code=400, detail=f"Invalid simulation identifier: '{raw_id}'")
    return raw_id


def _resolve_session_path(sim_id: str) -> str:
    safe_id = _validate_safe_id(sim_id)
    return os.path.join(_get_sessions_dir(), f"{safe_id}.json")


def _build_session_summary(s: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    """Extracts complete, legacy-compatible Past Simulations metadata from a persisted JSON dict."""
    sim_id = s.get("simulation_id") or s.get("session_id") or fallback_id
    cfg = s.get("config") or {}
    diag_models = (s.get("diagnostics") or {}).get("selected_models") or {}
    signal_text = s.get("signal_text") or cfg.get("signal") or ""
    iters = s.get("iterations") if isinstance(s.get("iterations"), list) else []

    # Infer model names cleanly for older sessions lacking selected_models
    inferred_models: List[str] = []
    inferred_providers: List[str] = []
    valid_count = 0
    total_slots = 0

    for it in iters:
        micros = it.get("micro_interpretations") or []
        for idx, m in enumerate(micros):
            total_slots += 1
            m_name = m.get("model_name")
            m_prov = m.get("provider")
            if m_name and m_name not in inferred_models:
                inferred_models.append(m_name)
                inferred_providers.append(m_prov or "unknown")
            exec_status = m.get("execution_status")
            if exec_status is not None:
                if exec_status == "success":
                    valid_count += 1
            else:
                if str(m.get("primary_narrative") or "").strip():
                    valid_count += 1

    model_a = (
        cfg.get("model_a")
        or (diag_models.get("model_a") or {}).get("model")
        or (inferred_models[0] if len(inferred_models) >= 1 else "unknown")
    )
    model_b = (
        cfg.get("model_b")
        or (diag_models.get("model_b") or {}).get("model")
        or (inferred_models[1] if len(inferred_models) >= 2 else inferred_models[0] if inferred_models else "unknown")
    )
    provider_a = (
        cfg.get("provider_a")
        or (diag_models.get("model_a") or {}).get("provider")
        or (inferred_providers[0] if inferred_providers else "unknown")
    )
    provider_b = (
        cfg.get("provider_b")
        or (diag_models.get("model_b") or {}).get("provider")
        or (inferred_providers[1] if len(inferred_providers) >= 2 else inferred_providers[0] if inferred_providers else "unknown")
    )

    expected_count = s.get("expected_interpretations_count")
    if expected_count is None:
        expected_count = total_slots if total_slots > 0 else len(iters) * 10

    successful_count = s.get("successful_interpretations_count")
    if successful_count is None:
        successful_count = valid_count

    status = s.get("status")
    if not status:
        if total_slots == 0:
            status = "completed"
        elif successful_count >= expected_count:
            status = "completed"
        elif successful_count > 0:
            status = "partially_completed"
        else:
            status = "failed"

    short_desc = (signal_text[:137] + "...") if len(signal_text) > 140 else signal_text
    rep_status = get_report_status(sim_id, reports_dir=_get_reports_dir())
    verdict = s.get("final_verdict") or {}
    scen_probs = verdict.get("scenario_probabilities") or {}

    return {
        "session_id": sim_id,
        "simulation_id": sim_id,
        "event_title": s.get("event_title") or cfg.get("event_title") or derive_event_title(signal_text),
        "signal": signal_text,
        "short_description": short_desc,
        "source_url": cfg.get("source_url"),
        "model_a": model_a,
        "model_b": model_b,
        "provider_a": provider_a,
        "provider_b": provider_b,
        "iterations": len(iters) if iters else s.get("n_signals", 0),
        "max_iterations": cfg.get("max_iterations") or len(iters) or 3,
        "status": status,
        "status_details": s.get("status_details"),
        "successful_interpretations_count": successful_count,
        "expected_interpretations_count": expected_count,
        "failed_models": s.get("failed_models") or [],
        "model_agreement_score": s.get("model_agreement_score"),
        "timestamp": s.get("timestamp") or s.get("last_updated"),
        "dominant_narrative": s.get("dominant_narrative", "?"),
        "verdict_headline": verdict.get("headline"),
        "leading_scenario": scen_probs.get("leading_scenario"),
        "verdict_confidence": verdict.get("verdict_confidence"),
        "research_question": s.get("research_question"),
        "has_report": rep_status["report_exists"],
        "report_generated_at": rep_status["generated_at"],
        "report_filename": rep_status["filename"],
    }


# ── Models ────────────────────────────────────────────────────────────────────

class SignalRequest(BaseModel):
    signal: str
    mode: str = "support"          # "support" | "pure"
    max_iterations: int = 3
    run_debate: bool = True
    lock_prediction: bool = True


class PerturbRequest(BaseModel):
    signal: str
    max_iterations: int = 3
    shock_type: str = "fatigue_spike"  # "debunk_injection" | "amplifier_blast" | "fatigue_spike"
    shock_iteration: int = 2
    intensity: float = 0.5


class EvaluateRequest(BaseModel):
    simulation_id: str
    observed_adoption: List[float]
    observed_attention: Optional[List[float]] = None


class ResolveRequest(BaseModel):
    signal_id: str
    ground_truth_bucket: str
    resolution_notes: str = ""


class ForecastResolveRequest(BaseModel):
    actual_outcome_summary: str
    realized_scenario_type: str = "baseline"  # baseline | positive | negative | extreme | unforeseen
    evidence_url: Optional[str] = None
    notes: Optional[str] = None


# ── Health & Diagnostics ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "4.2.0",
        "layers": ["micro", "meso", "macro", "feedback"],
    }


@app.get("/api/llm/health")
def get_llm_health():
    """Verifies connectivity to localhost:11434, checks GPU presence and VRAM status, and reports provider configurations."""
    ollama_info = check_ollama_health()
    gpu_info = get_gpu_diagnostics()

    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    gemini_configured = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ollama": ollama_info,
        "gpu": gpu_info,
        "cloud_providers": {
            "groq": {
                "configured": groq_configured,
                "available": groq_configured,
            },
            "gemini": {
                "configured": gemini_configured,
                "available": gemini_configured,
            },
        },
        "mock": {
            "available": True,
        },
    }


@app.get("/api/llm/models")
def get_llm_models():
    """Queries actual local Ollama API to retrieve installed models and check recommended model availability."""
    return get_installed_ollama_models()


@app.get("/api/llm/providers")
def get_llm_providers():
    """Returns available LLM provider categories, API readiness, and recommended dual-model local configuration."""
    ollama_health = check_ollama_health()
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    gemini_configured = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    providers = [
        {
            "id": "ollama",
            "name": "Local Ollama",
            "category": "local",
            "description": "Local on-device inference via Ollama HTTP service (localhost:11434).",
            "requires_api_key": False,
            "api_key_configured": True,
            "is_available": ollama_health["running"],
            "status_message": (
                f"Online (Ollama v{ollama_health.get('version', '')})"
                if ollama_health["running"]
                else "Ollama is not available. Install Ollama and start its local service before running local inference."
            ),
            "recommended_models": ["qwen3.5:4b", "gemma3:4b"],
        },
        {
            "id": "groq",
            "name": "Groq Cloud (Requires API Key)",
            "category": "cloud",
            "description": "Ultra-fast cloud LPU inference. Requires GROQ_API_KEY.",
            "requires_api_key": True,
            "api_key_configured": groq_configured,
            "is_available": groq_configured,
            "status_message": "Ready" if groq_configured else "GROQ_API_KEY is not configured",
            "recommended_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        },
        {
            "id": "gemini",
            "name": "Google Gemini API (Requires API Key)",
            "category": "cloud",
            "description": "Cloud Gemini inference with structured JSON. Requires GEMINI_API_KEY.",
            "requires_api_key": True,
            "api_key_configured": gemini_configured,
            "is_available": gemini_configured,
            "status_message": "Ready" if gemini_configured else "GEMINI_API_KEY is not configured",
            "recommended_models": ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        },
        {
            "id": "mock",
            "name": "Deterministic Mock (Development / Testing Only)",
            "category": "mock",
            "description": "Offline deterministic archetype interpretation generator for development and testing only.",
            "requires_api_key": False,
            "api_key_configured": True,
            "is_available": True,
            "status_message": "Development / Testing Only",
            "recommended_models": ["mock-analyst-alpha", "mock-debunker-beta"],
        },
    ]

    return {
        "providers": providers,
        "recommended_local_config": {
            "model_a": {
                "provider": "ollama",
                "model": "qwen3.5:4b",
                "role": "Primary Analyst",
            },
            "model_b": {
                "provider": "ollama",
                "model": "gemma3:4b",
                "role": "Independent Reviewer",
            },
        },
    }


# ── 4-Layer Simulation Endpoints ─────────────────────────────────────────────

@app.post("/api/simulations/run")
def run_simulation_sync(config: SimulationConfig):
    """Executes the complete 4-layer simulation synchronously."""
    logger.info(f"POST /api/simulations/run | {config.signal[:60]}")
    try:
        engine = SimulationEngine()
        result = engine.run_simulation(config)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulations/run/async")
def run_simulation_async(config: SimulationConfig, background_tasks: BackgroundTasks):
    """Enqueues 4-layer simulation in background with pre-assigned isolated simulation_id and job_id."""
    job_id = str(uuid.uuid4())[:8]
    sim_id = f"sim_{uuid.uuid4().hex[:10]}_{int(time.time())}"
    _JOBS[job_id] = {
        "job_id": job_id,
        "simulation_id": sim_id,
        "status": "queued",
        "result": None,
        "progress": [f"Job {job_id} queued for isolated simulation {sim_id}."],
        "started": datetime.utcnow().isoformat() + "Z",
    }

    background_tasks.add_task(_execute_simulation_job, job_id, sim_id, config)
    logger.info(f"Async simulation job {job_id} (sim={sim_id}) queued | {config.signal[:60]}")
    return {
        "job_id": job_id,
        "simulation_id": sim_id,
        "status": "queued",
        "message": f"Poll GET /jobs/{job_id} or connect WS /ws/{job_id}",
    }


def _execute_simulation_job(job_id: str, sim_id: str, config: SimulationConfig):
    def on_progress(step: str, iteration: int, data: dict):
        msg = f"[Iteration {iteration}] {step.upper()}: {data}"
        if job_id in _JOBS and _JOBS[job_id].get("simulation_id") == sim_id:
            _JOBS[job_id]["progress"].append(msg)

    try:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "running"
            _JOBS[job_id]["progress"].append(f"Initializing 4-layer simulation engine for {sim_id}...")
        engine = SimulationEngine()
        result = engine.run_simulation(config, progress_callback=on_progress, simulation_id=sim_id)
        dumped = result.model_dump()

        if job_id in _JOBS and _JOBS[job_id].get("simulation_id") == sim_id:
            _JOBS[job_id]["result"] = dumped
            if result.status == "failed":
                _JOBS[job_id]["status"] = "failed"
                _JOBS[job_id]["error"] = result.status_details or "All model inferences failed."
                _JOBS[job_id]["progress"].append(f"Simulation Failed: {_JOBS[job_id]['error']}")
            elif result.status == "partially_completed":
                _JOBS[job_id]["status"] = "partially_completed"
                _JOBS[job_id]["progress"].append(f"Simulation Partially Completed: {result.status_details}")
            else:
                _JOBS[job_id]["status"] = "completed"
                _JOBS[job_id]["progress"].append("Simulation Completed.")
        logger.info(f"Job {job_id} (sim={sim_id}) finished with status={result.status}.")
    except Exception as e:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = str(e)
            _JOBS[job_id]["progress"].append(f"Execution aborted: {e}")
        logger.error(f"Job {job_id} (sim={sim_id}) failed: {e}")


@app.post("/api/simulations/perturb")
def run_perturbation_experiment(req: PerturbRequest):
    """Executes a counterfactual perturbation experiment comparing baseline vs shock."""
    try:
        exp = PerturbationExperiment()
        cfg = SimulationConfig(signal=req.signal, max_iterations=req.max_iterations)
        result = exp.run_perturbation(
            base_config=cfg,
            shock_type=req.shock_type,  # type: ignore
            shock_iteration=req.shock_iteration,
            intensity=req.intensity,
        )
        return result
    except Exception as e:
        logger.error(f"Perturbation experiment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulations/evaluate")
def evaluate_simulation(req: EvaluateRequest):
    """Evaluates an existing simulation trajectory against empirical observations."""
    session_path = _resolve_session_path(req.simulation_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation {req.simulation_id} not found")

    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        sim_result = SimulationResult(**data)
        eval_metrics = TrajectoryEvaluator.evaluate(
            sim_result=sim_result,
            observed_adoption=req.observed_adoption,
            observed_attention=req.observed_attention,
        )
        return eval_metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulations")
def list_simulations():
    """Lists all persisted simulations with Past Simulations metadata."""
    return list_sessions()


@app.get("/api/simulations/{sim_id}")
def get_simulation(sim_id: str):
    """Strictly retrieves only the specified simulation_id JSON record (read-only), ensuring Phase 6.3.1 forecast/grounding fields."""
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    with open(session_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ensure_forecast_fields(raw)


@app.delete("/api/simulations/{sim_id}")
def delete_simulation(sim_id: str):
    """Deletes ONLY the specified simulation JSON file (and its cached report if present)."""
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")

    try:
        os.remove(session_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete simulation '{sim_id}': {e}")

    # Remove associated PDF report & metadata if generated for this simulation
    safe_id = _validate_safe_id(sim_id)
    pdf_path, meta_path = get_report_paths(safe_id, reports_dir=_get_reports_dir())
    forecast_path = os.path.join(
        getattr(forecast_tracking_mod, "FORECASTS_DIR", "data/forecasts"),
        f"{safe_id}.forecast.json",
    )
    for artifact in (pdf_path, meta_path, forecast_path):
        if os.path.isfile(artifact):
            try:
                os.remove(artifact)
            except Exception as e:
                logger.warning(f"Could not delete simulation artifact {artifact}: {e}")

    # Clean up in-memory job references pointing to this simulation
    stale_job_ids = [jid for jid, job in _JOBS.items() if job.get("simulation_id") == sim_id]
    for jid in stale_job_ids:
        _JOBS.pop(jid, None)

    logger.info(f"Deleted simulation {sim_id} at {session_path}")
    return {"deleted": True, "simulation_id": sim_id}


@app.post("/api/simulations/{sim_id}/report")
def generate_simulation_report(
    sim_id: str,
    force: bool = Query(default=False, description="Force PDF regeneration even if cached copy exists"),
):
    """
    Generates (or reuses cached) professional research PDF report for a single saved simulation.
    Never triggers LLM inference and never modifies the source simulation JSON.
    """
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found.")

    try:
        with open(session_path, "r", encoding="utf-8") as f:
            sim_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to read persisted simulation '{sim_id}': {e}")

    try:
        info = generate_simulation_pdf_report(
            sim_data=sim_data,
            session_path=session_path,
            reports_dir=_get_reports_dir(),
            force_regenerate=force,
        )
        return {
            "simulation_id": info["simulation_id"],
            "event_title": info["event_title"],
            "report_exists": True,
            "cached": info["cached"],
            "filename": info["filename"],
            "file_size_bytes": info["file_size_bytes"],
            "generated_at": info["generated_at"],
            "download_url": info["download_url"],
        }
    except ReportGenerationError as rge:
        raise HTTPException(status_code=422, detail=str(rge))
    except Exception as e:
        logger.error(f"PDF report generation failed for {sim_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed for '{sim_id}': {e}")


@app.get("/api/simulations/{sim_id}/report/status")
def simulation_report_status(sim_id: str):
    """Returns whether a generated PDF report already exists for simulation `sim_id`."""
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found.")
    return get_report_status(sim_id, session_path=session_path, reports_dir=_get_reports_dir())


@app.get("/api/simulations/{sim_id}/report/download")
@app.get("/api/simulations/{sim_id}/report")
def download_simulation_report(sim_id: str):
    """
    Downloads a previously generated PDF report without rerunning model inference or mutating history.
    Returns 404 if simulation does not exist or report has not yet been generated.
    """
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found.")

    safe_id = _validate_safe_id(sim_id)
    reports_dir = _get_reports_dir()
    pdf_path, meta_path = get_report_paths(safe_id, reports_dir=reports_dir)
    if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"PDF report has not been generated yet for simulation '{sim_id}'.",
        )

    filename = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                filename = json.load(mf).get("filename")
        except Exception:
            filename = None
    if not filename:
        try:
            with open(session_path, "r", encoding="utf-8") as sf:
                filename = build_download_filename(json.load(sf), safe_id)
        except Exception:
            filename = f"Butterfly_Research_Report_{safe_id}.pdf"

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
    )


@app.get("/api/simulations/{sim_id}/trajectory")
def get_simulation_trajectory(sim_id: str):
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "simulation_id": sim_id,
        "trajectory": data.get("trajectory", {}),
        "model_agreement_score": data.get("model_agreement_score"),
        "narrative_convergence_score": data.get("narrative_convergence_score"),
        "dominant_narrative": data.get("dominant_narrative"),
    }


@app.get("/api/simulations/{sim_id}/forecast")
def get_simulation_forecast(sim_id: str):
    """
    Returns the complete Butterfly-Effect causal pathways, 4 scenarios,
    time-horizon forecast, epistemic breakdown, 8-question Final Verdict,
    and forecast tracking status (computing on-the-fly for older JSONs without mutation).
    """
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")
    with open(session_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    enriched = ensure_forecast_fields(raw_data)
    tracking = get_forecast_record(
        sim_id,
        sim_data=enriched,
        forecasts_dir=getattr(forecast_tracking_mod, "FORECASTS_DIR", "data/forecasts"),
    )
    return {
        "simulation_id": sim_id,
        "event_title": enriched.get("event_title"),
        "status": enriched.get("status", "completed"),
        "research_question": enriched.get("research_question"),
        "causal_pathways": enriched.get("causal_pathways", []),
        "order_effects_summary": enriched.get("order_effects_summary"),
        "stakeholder_reactions": enriched.get("stakeholder_reactions", []),
        "scenarios": enriched.get("scenarios", []),
        "time_horizon_forecast": enriched.get("time_horizon_forecast"),
        "epistemic_breakdown": enriched.get("epistemic_breakdown"),
        "final_verdict": enriched.get("final_verdict"),
        "tracking": tracking.model_dump() if tracking else None,
    }


@app.post("/api/simulations/{sim_id}/forecast/resolve")
def resolve_simulation_forecast(sim_id: str, req: ForecastResolveRequest):
    """Logs the real-world observed outcome against a completed simulation forecast and scores accuracy."""
    session_path = _resolve_session_path(sim_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")
    if not req.actual_outcome_summary or not req.actual_outcome_summary.strip():
        raise HTTPException(status_code=422, detail="actual_outcome_summary must be non-empty.")

    allowed = {"baseline", "positive", "negative", "extreme", "unforeseen"}
    stype = req.realized_scenario_type.lower().strip()
    if stype not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"realized_scenario_type must be one of {sorted(allowed)}",
        )

    with open(session_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    enriched = ensure_forecast_fields(raw_data)
    record = resolve_forecast_outcome(
        sim_id=sim_id,
        sim_data=enriched,
        actual_outcome_summary=req.actual_outcome_summary,
        realized_scenario_type=stype,
        evidence_url=req.evidence_url,
        notes=req.notes,
        forecasts_dir=getattr(forecast_tracking_mod, "FORECASTS_DIR", "data/forecasts"),
    )
    return record.model_dump()


@app.get("/api/forecasts")
def list_forecasts(include_mock: bool = Query(default=False)):
    """Lists tracked simulation forecasts and any resolved post-event evaluations."""
    records = list_tracked_forecasts(
        include_mock=include_mock,
        forecasts_dir=getattr(forecast_tracking_mod, "FORECASTS_DIR", "data/forecasts"),
    )
    return {"forecasts": [r.model_dump() for r in records]}


# ── Preserved & Compatible Endpoints ──────────────────────────────────────────

@app.post("/signals/analyze")
def analyze_signal(req: SignalRequest):
    """
    Synchronous analysis endpoint.
    Routes to the 4-layer engine and returns structured result preserving existing client compatibility.
    """
    logger.info(f"POST /signals/analyze | mode={req.mode} | {req.signal[:60]}")
    try:
        cfg = SimulationConfig(signal=req.signal, max_iterations=req.max_iterations)
        engine = SimulationEngine()
        result = engine.run_simulation(cfg)
        return _serialize_result(result)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/analyze/async")
def analyze_signal_async(req: SignalRequest, background_tasks: BackgroundTasks):
    """Start pipeline in background, return job_id and simulation_id immediately."""
    cfg = SimulationConfig(signal=req.signal, max_iterations=req.max_iterations)
    job_id = str(uuid.uuid4())[:8]
    sim_id = f"sim_{uuid.uuid4().hex[:10]}_{int(time.time())}"
    _JOBS[job_id] = {
        "job_id": job_id,
        "simulation_id": sim_id,
        "status": "queued",
        "result": None,
        "progress": [f"Simulation {sim_id} queued."],
        "started": datetime.utcnow().isoformat() + "Z",
    }
    background_tasks.add_task(_execute_simulation_job, job_id, sim_id, cfg)
    return {
        "job_id": job_id,
        "simulation_id": sim_id,
        "status": "queued",
        "message": f"Poll GET /jobs/{job_id} or connect WS /ws/{job_id}",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _JOBS[job_id]
    out: Dict[str, Any] = {
        "job_id": job_id,
        "simulation_id": job.get("simulation_id"),
        "status": job["status"],
        "progress": job["progress"],
    }
    if job["status"] in ("completed", "complete", "partially_completed") and job.get("result"):
        out["result"] = job["result"]
    if job["status"] == "failed":
        out["error"] = job.get("error")
        if job.get("result"):
            out["result"] = job["result"]
    return out


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """Stream live pipeline progress isolated cleanly to job_id and its simulation_id."""
    await websocket.accept()
    if job_id not in _JOBS:
        await websocket.send_json({"type": "error", "job_id": job_id, "data": f"Job {job_id} not found"})
        await websocket.close()
        return

    sim_id = _JOBS[job_id].get("simulation_id")
    last_len = 0
    try:
        while True:
            job = _JOBS.get(job_id)
            if not job:
                await websocket.send_json({"type": "error", "job_id": job_id, "simulation_id": sim_id, "data": "Job cancelled or deleted."})
                break

            progress = job.get("progress", [])
            for msg in progress[last_len:]:
                await websocket.send_json({
                    "type": "progress",
                    "job_id": job_id,
                    "simulation_id": sim_id,
                    "data": msg,
                })
            last_len = len(progress)

            status = job.get("status")
            if status in ("completed", "complete", "partially_completed"):
                await websocket.send_json({
                    "type": "complete",
                    "job_id": job_id,
                    "simulation_id": sim_id,
                    "status": status,
                    "data": job.get("result", {}),
                })
                break
            if status == "failed":
                await websocket.send_json({
                    "type": "error",
                    "job_id": job_id,
                    "simulation_id": sim_id,
                    "data": job.get("error", "Unknown error"),
                    "result": job.get("result"),
                })
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {job_id}")


# ── Preserved Registry Endpoints ──────────────────────────────────────────────

@app.get("/registry/pending")
def get_pending():
    from src.ensemble.prediction_registry import get_pending_resolutions
    return {"pending": get_pending_resolutions()}


@app.post("/registry/resolve")
def resolve(req: ResolveRequest):
    from src.ensemble.prediction_registry import resolve_prediction
    result = resolve_prediction(
        signal_id=req.signal_id,
        ground_truth_bucket=req.ground_truth_bucket,
        resolution_notes=req.resolution_notes,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Prediction {req.signal_id} not found")
    return result


@app.get("/registry/leaderboard")
def get_leaderboard():
    from src.ensemble.prediction_registry import get_leaderboard_summary, compare_modes
    return {"leaderboard": get_leaderboard_summary(), "mode_comparison": compare_modes()}


@app.get("/registry/{signal_id}")
def get_prediction(signal_id: str):
    safe_id = _validate_safe_id(signal_id)
    path = f"data/registry/locked/{safe_id}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Prediction {signal_id} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Preserved Sessions Endpoints ──────────────────────────────────────────────

@app.get("/sessions")
def list_sessions():
    sessions = []
    session_dir = _get_sessions_dir()
    if not os.path.exists(session_dir):
        return {"sessions": []}
    for fname in os.listdir(session_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(session_dir, fname), "r", encoding="utf-8") as f:
                s = json.load(f)
            sessions.append(_build_session_summary(s, fname[:-5]))
        except Exception:
            pass
    return {"sessions": sorted(sessions, key=lambda x: str(x.get("timestamp") or ""), reverse=True)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session_path = _resolve_session_path(session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    with open(session_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ensure_forecast_fields(raw)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    return delete_simulation(session_id)


# ── Helper Serializer for Legacy Compatibility ────────────────────────────────

def _serialize_result(sim: SimulationResult) -> dict:
    """Serializes a SimulationResult into a payload compatible with legacy clients."""
    last_iter = sim.iterations[-1]
    last_macro = last_iter.macro_state

    clusters = []
    for interp in last_iter.micro_interpretations:
        clusters.append({
            "archetype": interp.archetype.value,
            "label": interp.archetype_name,
            "model": interp.model_name,
            "model_slot": interp.model_slot,
            "execution_status": interp.execution_status,
            "stance": interp.stance,
            "virality": interp.virality,
            "credibility": interp.credibility,
            "probability": round(max(0.01, (interp.stance + 1.0) / 2.0), 3),
            "primary_narrative": interp.primary_narrative,
        })

    return {
        "simulation_id": sim.simulation_id,
        "event_title": sim.event_title,
        "signal_text": sim.signal_text,
        "dominant_narrative": sim.dominant_narrative,
        "model_agreement_score": sim.model_agreement_score,
        "narrative_convergence_score": sim.narrative_convergence_score,
        "total_iterations": len(sim.iterations),
        "trajectory": sim.trajectory,
        "final_macro": last_macro.model_dump(),
        "clusters": clusters,
        "timestamp": sim.timestamp,
        "status": sim.status,
    }
