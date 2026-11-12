"""
src/core/forecast_tracking.py

V4 Forecast Snapshot & Post-Event Outcome Resolution Registry.
Tracks each simulation's forecast snapshot (4 conditional scenario weights + Final Verdict)
and allows researchers to log real-world outcomes later to evaluate directional & Brier accuracy
without ever mutating the historical simulation record itself.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.core.schemas import (
    CORE_RESEARCH_QUESTION,
    ForecastOutcomeResolution,
    ForecastTrackingRecord,
)

PRODUCTION_FORECASTS_DIR = os.getenv("BUTTERFLY_FORECASTS_DIR", "data/forecasts")
FORECASTS_DIR = PRODUCTION_FORECASTS_DIR


def _safe_id(sim_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "", str(sim_id or ""))


def _record_path(sim_id: str, forecasts_dir: Optional[str] = None) -> str:
    base = forecasts_dir or FORECASTS_DIR
    return os.path.join(base, f"{_safe_id(sim_id)}.forecast.json")


def _extract_scenario_weights(sim_data: Dict[str, Any]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for sc in (sim_data.get("scenarios") or []):
        stype = str(sc.get("scenario_type") or "")
        w = sc.get("conditional_weight")
        if stype and w is not None:
            weights[stype] = float(w)
    return weights


def build_tracking_record_from_sim(sim_data: Dict[str, Any]) -> ForecastTrackingRecord:
    verdict = sim_data.get("final_verdict") or {}
    weights = _extract_scenario_weights(sim_data)
    leading = None
    if weights:
        leading = max(weights.items(), key=lambda kv: kv[1])[0]
    elif isinstance(verdict.get("scenario_probabilities"), dict):
        leading = verdict["scenario_probabilities"].get("leading_scenario")

    diag = sim_data.get("diagnostics") or {}
    cfg = sim_data.get("config") or {}
    is_mock = bool(diag.get("is_mock_only")) or (
        str(cfg.get("provider_a") or "mock").lower() == "mock"
        and str(cfg.get("provider_b") or "mock").lower() == "mock"
    )

    return ForecastTrackingRecord(
        simulation_id=str(sim_data.get("simulation_id") or ""),
        event_title=str(sim_data.get("event_title") or cfg.get("event_title") or "Untitled Simulation"),
        research_question=str(sim_data.get("research_question") or CORE_RESEARCH_QUESTION),
        created_at=str(sim_data.get("timestamp") or datetime.utcnow().isoformat() + "Z"),
        simulation_status=str(sim_data.get("status") or "completed"),
        leading_scenario=leading,
        scenario_probabilities=weights,
        verdict_headline=verdict.get("headline"),
        verdict_confidence=verdict.get("verdict_confidence"),
        is_mock_only=is_mock,
        status="open",
        resolution=None,
    )


def save_forecast_snapshot(sim_data: Dict[str, Any], forecasts_dir: Optional[str] = None) -> Optional[str]:
    """
    Persists a forecast snapshot under FORECASTS_DIR.
    Enforces storage hygiene: deterministic mock runs do NOT write into production FORECASTS_DIR.
    """
    record = build_tracking_record_from_sim(sim_data)
    target_dir = forecasts_dir or FORECASTS_DIR
    if record.is_mock_only and os.path.abspath(target_dir) == os.path.abspath(PRODUCTION_FORECASTS_DIR):
        logger.info(
            f"[Forecast Hygiene] Mock forecast {record.simulation_id} skipped writing to production '{PRODUCTION_FORECASTS_DIR}'."
        )
        return None

    os.makedirs(target_dir, exist_ok=True)
    path = _record_path(record.simulation_id, forecasts_dir=target_dir)
    if os.path.isfile(path):
        # Preserve existing resolution if snapshot was already tracked
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("resolution"):
                record.status = existing.get("status", "resolved")
                record.resolution = ForecastOutcomeResolution(**existing["resolution"])
        except Exception:
            pass

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
    return path


def get_forecast_record(
    sim_id: str,
    sim_data: Optional[Dict[str, Any]] = None,
    forecasts_dir: Optional[str] = None,
) -> Optional[ForecastTrackingRecord]:
    path = _record_path(sim_id, forecasts_dir=forecasts_dir)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ForecastTrackingRecord(**data)
        except Exception as e:
            logger.warning(f"Failed loading forecast tracking record {sim_id}: {e}")
    if sim_data:
        return build_tracking_record_from_sim(sim_data)
    return None


def compute_brier_and_match(
    scenario_probabilities: Dict[str, float],
    realized_scenario_type: str,
) -> tuple[Optional[float], Optional[float]]:
    """
    Computes multi-class Brier score sum((p_k - y_k)^2) across {baseline, positive, negative, extreme}
    where lower Brier is better (0.0 perfect, 2.0 worst) and match_score = p(realized) in [0, 1].
    """
    keys = ["baseline", "positive", "negative", "extreme"]
    if not scenario_probabilities or not all(k in scenario_probabilities for k in keys):
        return None, None
    realized = str(realized_scenario_type or "").lower().strip()
    brier = 0.0
    for k in keys:
        y = 1.0 if k == realized else 0.0
        p = float(scenario_probabilities[k])
        brier += (p - y) ** 2
    match_score = float(scenario_probabilities.get(realized, 0.0))
    return round(brier, 4), round(match_score, 4)


def resolve_forecast_outcome(
    sim_id: str,
    sim_data: Dict[str, Any],
    *,
    actual_outcome_summary: str,
    realized_scenario_type: str,
    evidence_url: Optional[str] = None,
    notes: Optional[str] = None,
    forecasts_dir: Optional[str] = None,
) -> ForecastTrackingRecord:
    """
    Records a real-world post-event observation against an existing simulation forecast,
    scoring multi-class Brier error and scenario match weight.
    """
    record = get_forecast_record(sim_id, sim_data=sim_data, forecasts_dir=forecasts_dir)
    if record is None:
        record = build_tracking_record_from_sim(sim_data)

    brier, match_score = compute_brier_and_match(
        record.scenario_probabilities,
        realized_scenario_type,
    )
    resolution = ForecastOutcomeResolution(
        actual_outcome_summary=actual_outcome_summary.strip(),
        realized_scenario_type=realized_scenario_type.lower().strip(),
        outcome_match_score=match_score,
        brier_score=brier,
        resolved_at=datetime.utcnow().isoformat() + "Z",
        evidence_url=evidence_url.strip() if evidence_url else None,
        notes=notes.strip() if notes else None,
    )
    record.status = "resolved"
    record.resolution = resolution

    target_dir = forecasts_dir or FORECASTS_DIR
    if record.is_mock_only and os.path.abspath(target_dir) == os.path.abspath(PRODUCTION_FORECASTS_DIR):
        logger.info(
            f"[Forecast Hygiene] Mock forecast resolution for {sim_id} returned in memory without writing to production '{PRODUCTION_FORECASTS_DIR}'."
        )
        return record

    os.makedirs(target_dir, exist_ok=True)
    path = _record_path(sim_id, forecasts_dir=target_dir)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
    return record


def list_tracked_forecasts(
    include_mock: bool = False,
    forecasts_dir: Optional[str] = None,
) -> List[ForecastTrackingRecord]:
    base = forecasts_dir or FORECASTS_DIR
    if not os.path.isdir(base):
        return []
    records: List[ForecastTrackingRecord] = []
    for fname in os.listdir(base):
        if not fname.endswith(".forecast.json"):
            continue
        fpath = os.path.join(base, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                rec = ForecastTrackingRecord(**json.load(f))
            if not include_mock and rec.is_mock_only:
                continue
            records.append(rec)
        except Exception as e:
            logger.warning(f"Skipping corrupted forecast file {fname}: {e}")
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records
