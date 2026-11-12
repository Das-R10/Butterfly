"""
ba/tests/test_phase6_forecasting.py

Phase 6 verification suite for Butterfly-Effect Future Impact Forecasting:
  1. Archetype causal hypothesis fields + backward compat with legacy 11-key payloads
  2. Signal-aware deterministic mock generator differentiating domains/stakeholders
  3. Feedback systemic state evolution & hypothesis-labeled prompt carryover
  4. Competing multi-order causal pathways & automated weak-link flagging
  5. Four distinct future scenarios summing to 1.0000 with calibration warning
  6. 6-tier epistemic breakdown & 8-question Final Verdict with traceability
  7. Honest FAILED / PARTIALLY COMPLETED forecast & verdict behavior (zero hallucinated certainty)
  8. Forecast tracking API endpoints (/forecast, /forecast/resolve, /api/forecasts) & cleanup
  9. 15-Section Butterfly-Effect Forecast PDF verification via PyMuPDF (fitz)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.core.forecast_tracking as forecast_tracking_mod
from src.api.main import app
from src.core.llm_client import LLMClient
from src.core.forecasting_engine import CALIBRATION_WARNING, _evaluate_weak_link
from src.core.report_generator import generate_simulation_pdf_report
from src.core.schemas import (
    CORE_RESEARCH_QUESTION,
    InterpretationOutput,
    SimulationConfig,
    SimulationStatus,
)
from src.core.simulation_engine import SimulationEngine


def _patch_llm_client_for_isolated_run(monkeypatch: pytest.MonkeyPatch, model_a: str, model_b: str):
    """Uses signal-aware mock responses under realistic Ollama labels without network calls."""
    real_mock_a = LLMClient(provider="mock", model_name=model_a)
    real_mock_b = LLMClient(provider="mock", model_name=model_b)

    def _fake_ollama_raw(
        self,
        prompt: str,
        system_prompt: str = "",
        json_schema=None,
        num_predict_override=None,
    ):
        helper = real_mock_b if "gemma" in (self.model_name or "").lower() else real_mock_a
        payload_dict = helper._generate_mock(prompt, system_prompt)
        raw_text = json.dumps(payload_dict)
        return raw_text, {
            "finish_reason": "stop",
            "response_length": len(raw_text),
            "thinking_length": 0,
        }

    monkeypatch.setattr(LLMClient, "_call_ollama_raw", _fake_ollama_raw)


def test_archetype_causal_hypothesis_fields_and_legacy_11_key_compat():
    # Legacy 11-key JSON (pre-Phase 6) should parse seamlessly with safe defaults
    legacy_payload = {
        "evidence_status": "supported",
        "supporting_evidence": ["Official release notes confirmation"],
        "uncertainty": 0.25,
        "claims": ["New release announced"],
        "stance": 0.4,
        "virality_estimate": 0.55,
        "perceived_credibility": 0.75,
        "emotional_charge": 0.3,
        "primary_narrative": "Measured official announcement reviewed cleanly.",
        "key_arguments": ["Primary data aligns"],
        "narrative_tags": ["policy", "official"],
    }
    parsed_legacy = InterpretationOutput(**legacy_payload)
    assert parsed_legacy.causal_mechanism == ""
    assert parsed_legacy.time_horizon == "short_term_weeks"
    assert parsed_legacy.expected_stakeholder_actions == []

    # Rich v5 causal hypothesis output with normalized time_horizon
    causal_payload = {
        **legacy_payload,
        "causal_mechanism": "Import duties trigger forward inventory pull and contract repricing.",
        "expected_stakeholder_actions": ["Automakers accelerate cell orders", "Customs issues quota FAQ"],
        "behavioural_changes": ["Precautionary buffer stockpiling across tier-2 suppliers"],
        "emergent_consequences": ["Regional EV battery spot prices climb within 4-8 weeks (Hypothesis)"],
        "time_horizon": "medium_term_months",
        "trigger_conditions": ["No phase-in waiver within 14 days"],
        "invalidation_conditions": ["Customs grants immediate 12-month tariff suspension"],
    }
    parsed_causal = InterpretationOutput(**causal_payload)
    assert "Import duties" in parsed_causal.causal_mechanism
    assert parsed_causal.time_horizon == "medium_term_months"
    assert len(parsed_causal.emergent_consequences) == 1

    # Strict schema includes causal keys with concise maxItems constraints
    schema = InterpretationOutput.strict_json_schema()
    for field in (
        "causal_mechanism",
        "expected_stakeholder_actions",
        "behavioural_changes",
        "emergent_consequences",
        "time_horizon",
        "trigger_conditions",
        "invalidation_conditions",
    ):
        assert field in schema["properties"]
        assert field in schema["required"]
    assert schema["properties"]["emergent_consequences"]["maxItems"] == 3


def test_signal_aware_mock_differentiates_domains():
    client = LLMClient(provider="mock", model_name="mock-analyst-alpha")
    schema = InterpretationOutput.strict_json_schema()

    tariff_prompt = (
        "RAW SIGNAL TEXT:\n"
        '"Emergency 25% import tariff imposed on lithium battery packs and port customs fees."\n'
        "Focus strictly on institutional response."
    )
    cyber_prompt = (
        "RAW SIGNAL TEXT:\n"
        '"Major cloud DNS routing outage halts regional bank payment clearing and settlement."\n'
        "Focus strictly on institutional response."
    )

    data_tariff = client.generate_structured(
        tariff_prompt,
        system_prompt="You are analyzing as Institutional Analyst.",
        json_schema=schema,
    )
    data_cyber = client.generate_structured(
        cyber_prompt,
        system_prompt="You are analyzing as Institutional Analyst.",
        json_schema=schema,
    )
    raw_tariff = json.dumps(data_tariff)
    raw_cyber = json.dumps(data_cyber)

    assert data_tariff["primary_narrative"] != data_cyber["primary_narrative"]
    assert data_tariff["causal_mechanism"] != data_cyber["causal_mechanism"]
    assert any(k in raw_tariff.lower() for k in ("tariff", "customs", "inventory", "freight"))
    assert any(k in raw_cyber.lower() for k in ("outage", "incident", "failover", "settlement"))


def test_end_to_end_forecast_pathways_scenarios_systemic_feedback_and_verdict(
    isolated_storage_dirs, monkeypatch: pytest.MonkeyPatch
):
    _patch_llm_client_for_isolated_run(monkeypatch, "qwen3.5:4b", "gemma3:4b")

    engine = SimulationEngine(
        client_a=LLMClient(provider="ollama", model_name="qwen3.5:4b"),
        client_b=LLMClient(provider="ollama", model_name="gemma3:4b"),
    )
    cfg = SimulationConfig(
        event_title="Surprise 25% Battery Semiconductor Tariff",
        signal=(
            "Trade authorities announce an immediate 25% tariff surcharge on imported battery "
            "semiconductors and power electronics, prompting automaker hedging warnings and "
            "speculative consumer price rumors."
        ),
        source_url="https://trade.example.gov/bulletins/tariff-25pct",
        max_iterations=3,
        provider_a="ollama",
        model_a="qwen3.5:4b",
        provider_b="ollama",
        model_b="gemma3:4b",
    )

    with patch("src.core.simulation_engine.check_ollama_health", return_value={"running": True}), \
         patch("src.core.simulation_engine.is_model_installed", return_value=True), \
         patch("src.core.micro_layer.unload_model_from_vram"):
        result = engine.run_simulation(cfg)

    assert result.status == SimulationStatus.COMPLETED.value
    assert result.research_question == CORE_RESEARCH_QUESTION
    assert len(result.iterations) == 3

    # Check Feedback systemic_state and non-factual prompt carryover
    for idx, it in enumerate(result.iterations):
        fb = it.feedback_state
        assert fb.systemic_state
        for key in (
            "institutional_posture",
            "institutional_dampening_index",
            "public_pressure",
            "economic_market_stress",
            "escalation_risk",
            "epistemic_fragility",
        ):
            assert key in fb.systemic_state
        assert len(fb.what_changed) >= 1
        assert fb.why_changed.strip()
        assert len(fb.new_consequences_emerged) >= 1
        assert len(fb.preventive_conditions) >= 1
        if idx < 2:
            assert "[SIMULATED PRIOR SYSTEMIC STATE" in fb.prompt_context_for_next_round
            assert "HYPOTHESIS ONLY, NOT VERIFIED FACT" in fb.prompt_context_for_next_round

    # Check Causal Pathways & automated weak-link flagging
    assert len(result.causal_pathways) >= 4
    weak_steps = [st for pw in result.causal_pathways for st in pw.steps if st.is_weak_link]
    assert len(weak_steps) >= 1
    for st in weak_steps:
        assert st.weak_link_reason
    # Direct unit verification of _evaluate_weak_link rule
    is_w_unsup, reason_unsup = _evaluate_weak_link(
        evidence_status="unsupported",
        uncertainty=0.4,
        assumption_count=1,
        supporting_evidence=["Signal note"],
        order="1st_order",
    )
    assert is_w_unsup and reason_unsup
    is_w_assump, _ = _evaluate_weak_link(
        evidence_status="inferred",
        uncertainty=0.4,
        assumption_count=3,
        supporting_evidence=["Signal note"],
        order="2nd_order",
    )
    assert is_w_assump

    # Check 4 distinct scenarios & heuristic weight sum == 1.0000
    assert len(result.scenarios) == 4
    types_seen = {sc.scenario_type for sc in result.scenarios}
    assert types_seen == {"baseline", "positive", "negative", "extreme"}
    weight_sum = sum(sc.conditional_weight or 0.0 for sc in result.scenarios)
    assert weight_sum == pytest.approx(1.0, abs=1e-3)
    for sc in result.scenarios:
        assert sc.calibration_warning == CALIBRATION_WARNING
        assert len(sc.indicators_to_monitor) >= 1
        assert len(sc.trigger_conditions) >= 1

    # Check 6-tier epistemic breakdown & 8-question Final Verdict
    ep = result.epistemic_breakdown
    assert ep is not None
    assert len(ep.source_confirmed_facts) >= 1
    assert len(ep.simulated_behavioral_outcomes) >= 1
    assert len(ep.hypothetical_future_consequences) >= 1

    fv = result.final_verdict
    assert fv is not None
    assert fv.research_question == CORE_RESEARCH_QUESTION
    assert not fv.is_partial_or_failed
    assert fv.verdict_confidence is not None
    assert 0.0 < fv.verdict_confidence <= 1.0
    assert fv.most_likely_immediate_outcome
    assert fv.most_plausible_secondary_cascade
    assert fv.most_dangerous_downside_risk
    assert fv.most_plausible_stabilizing_outcome
    assert len(fv.key_triggers_and_turning_points) >= 1
    assert len(fv.recommended_monitoring_actions) >= 1
    assert fv.traceability_refs["simulation_id"] == result.simulation_id
    assert "co-occurrence" in fv.correlation_vs_causation_note.lower()

    # Check Stakeholders & Horizons
    assert len(result.stakeholder_reactions) >= 4
    assert result.time_horizon_forecast is not None
    assert len(result.time_horizon_forecast.immediate_days) >= 1

    # Check forecast tracking snapshot created inside isolated forecasts_dir
    f_path = os.path.join(forecast_tracking_mod.FORECASTS_DIR, f"{result.simulation_id}.forecast.json")
    assert os.path.exists(f_path)

    # Check all 15 sections + Butterfly Causal Diagram image in generated PDF via PyMuPDF
    session_file = isolated_storage_dirs["sessions_dir"] / f"{result.simulation_id}.json"
    pdf_meta = generate_simulation_pdf_report(
        result.model_dump(mode="json"),
        session_path=str(session_file),
        reports_dir=str(isolated_storage_dirs["reports_dir"]),
    )
    assert os.path.exists(pdf_meta["pdf_path"])
    with fitz.open(pdf_meta["pdf_path"]) as doc:
        page_count = doc.page_count
        full_text = "\n".join(doc.load_page(i).get_text() for i in range(page_count))
        total_images = sum(len(doc.load_page(i).get_images()) for i in range(page_count))

    assert page_count >= 5
    assert total_images >= 5  # 4 layer charts + Section 9 Butterfly Causal Diagram + optional header/assets
    for expected_heading in (
        "1. Executive Verdict",
        "2. Initial Event & Evidence Baseline",
        "3. Five Behavioural Interpretations",
        "4. Stakeholder Reaction Analysis",
        "5. Social Influence Network",
        "6. First-Order Effects",
        "7. Second-Order Effects",
        "8. Third-Order & Systemic Effects",
        "9. Causal Butterfly-Effect Diagrams",
        "10. Four Future Scenarios",
        "11. Time-Horizon Forecast & Reversal Conditions",
        "12. Stakeholder Impact Assessment",
        "13. Risk & Uncertainty Analysis",
        "14. Key Indicators & Turning Points to Monitor Next",
        "15. Final Verdict & Technical Appendix",
    ):
        assert expected_heading in full_text, f"Missing PDF section heading: {expected_heading}"


def test_failed_execution_produces_honest_unavailable_verdict(isolated_storage_dirs):
    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"response": "", "done": True, "done_reason": "stop"}

    engine = SimulationEngine(
        client_a=LLMClient(provider="ollama", model_name="qwen3.5:4b"),
        client_b=LLMClient(provider="ollama", model_name="gemma3:4b"),
    )
    cfg = SimulationConfig(
        event_title="Failed Connectivity Test",
        signal="Central bank announces emergency overnight liquidity facility adjustment.",
        max_iterations=3,
        provider_a="ollama",
        model_a="qwen3.5:4b",
        provider_b="ollama",
        model_b="gemma3:4b",
    )

    with patch.dict(os.environ, {"OLLAMA_MAX_RETRIES": "0"}), \
         patch("src.core.simulation_engine.check_ollama_health", return_value={"running": True}), \
         patch("src.core.simulation_engine.is_model_installed", return_value=True), \
         patch("src.core.llm_client.requests.post", return_value=empty_resp), \
         patch("src.core.micro_layer.unload_model_from_vram"):
        result = engine.run_simulation(cfg)

    assert result.status == SimulationStatus.FAILED.value
    assert result.final_verdict is not None
    assert result.final_verdict.is_partial_or_failed is True
    assert result.final_verdict.verdict_confidence is None
    assert result.final_verdict.failure_disclosure
    assert all(sc.conditional_weight is None for sc in result.scenarios)
    assert all("N/A" in sc.probability_band for sc in result.scenarios)


def test_forecast_api_endpoints_legacy_non_mutation_resolve_and_delete(
    isolated_storage_dirs, monkeypatch: pytest.MonkeyPatch
):
    client = TestClient(app)

    # 1. Seed a legacy session JSON lacking Phase 6 forecast fields cleanly into isolated sessions_dir
    legacy_id = "sim_legacy_phase5_001"
    sessions_dir = str(isolated_storage_dirs["sessions_dir"])
    legacy_path = os.path.join(sessions_dir, f"{legacy_id}.json")
    legacy_payload = {
        "simulation_id": legacy_id,
        "event_title": "Legacy Port Strike Event",
        "signal_text": "Dockworkers union issues 72-hour strike notice across three coastal container terminals.",
        "config": {
            "signal": "Dockworkers union issues 72-hour strike notice across three coastal container terminals.",
            "max_iterations": 3,
            "model_a": "qwen3.5:4b",
            "model_b": "gemma3:4b",
            "provider_a": "ollama",
            "provider_b": "ollama",
        },
        "iterations": [
            {
                "iteration": 1,
                "micro_interpretations": [
                    {
                        "archetype": "institutional_analyst",
                        "archetype_name": "Institutional Analyst",
                        "model_name": "qwen3.5:4b",
                        "provider": "ollama",
                        "model_slot": "model_a",
                        "execution_status": "success",
                        "stance": 0.2,
                        "virality": 0.55,
                        "credibility": 0.80,
                        "emotional_charge": 0.25,
                        "evidence_status": "supported",
                        "supporting_evidence": ["72-hour strike notice issued"],
                        "claims": ["Container terminal strike notice active"],
                        "primary_narrative": "Port operators prepare diversion contingencies ahead of deadline.",
                        "key_arguments": ["Freight diversion timelines tighten"],
                        "narrative_tags": ["ports", "logistics"],
                        "iteration": 1,
                    }
                ],
                "meso_state": {
                    "nodes": [],
                    "edges": [],
                    "external_influence_score": 0.42,
                    "internal_contagion_score": 0.35,
                    "effective_spread_potential": 0.48,
                    "node_activations": {
                        "external_media": 0.45,
                        "public_mainstream": 0.30,
                        "public_engaged": 0.52,
                    },
                },
                "macro_state": {
                    "iteration": 1,
                    "p_external": 0.04,
                    "q_internal": 0.18,
                    "M_potential": 1.0,
                    "c_churn": 0.08,
                    "cumulative_adoption": 0.18,
                    "active_adoption": 0.16,
                    "adoption_velocity": 0.05,
                    "attention_indicator": 0.54,
                    "attention_vs_adoption_ratio": 3.37,
                },
                "feedback_state": {
                    "iteration": 1,
                    "macro_summary": "Logistics risk monitoring initiated.",
                    "dominant_framing": "Container diversion contingencies active",
                    "polarization_index": 0.22,
                    "prompt_context_for_next_round": "SIMULATION_FINAL_CYCLE",
                    "claim_verification_audit": {"supported": 1, "inferred": 0, "unsupported": 0},
                },
            }
        ],
        "model_agreement_score": 0.72,
        "narrative_convergence_score": 0.65,
        "dominant_narrative": "Container diversion contingencies active",
        "trajectory": {"iterations": [1], "active_adoption": [0.16], "attention_indicator": [0.54], "adoption_velocity": [0.05]},
        "timestamp": "2026-11-11T00:00:00Z",
        "status": "completed",
    }
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(legacy_payload, f, indent=2)
    raw_bytes_before = Path(legacy_path).read_bytes()

    # GET /forecast enriches non-destructively in memory without modifying JSON file
    res_forecast = client.get(f"/api/simulations/{legacy_id}/forecast")
    assert res_forecast.status_code == 200
    body = res_forecast.json()
    assert len(body["scenarios"]) == 4
    assert body["final_verdict"]["headline"]
    assert raw_bytes_before == Path(legacy_path).read_bytes()

    # POST /forecast/resolve logs observed outcome and computes Brier score
    res_resolve = client.post(
        f"/api/simulations/{legacy_id}/forecast/resolve",
        json={
            "actual_outcome_summary": "Mediation extended deadline and freight rerouting contained disruptions cleanly.",
            "realized_scenario_type": "baseline",
            "evidence_url": "https://port.example.org/mediation-settlement",
            "notes": "Baseline hedging assumptions held.",
        },
    )
    assert res_resolve.status_code == 200
    rec = res_resolve.json()
    assert rec["status"] == "resolved"
    assert rec["resolution"]["brier_score"] is not None
    assert 0.0 <= rec["resolution"]["outcome_match_score"] <= 1.0

    # GET /api/forecasts lists the resolved record
    res_list = client.get("/api/forecasts")
    assert res_list.status_code == 200
    ids = [r["simulation_id"] for r in res_list.json()["forecasts"]]
    assert legacy_id in ids

    # DELETE /api/simulations/{sim_id} removes JSON + PDF + forecast record
    res_del = client.delete(f"/api/simulations/{legacy_id}")
    assert res_del.status_code == 200
    assert not os.path.exists(legacy_path)
    assert not os.path.exists(
        os.path.join(str(isolated_storage_dirs["forecasts_dir"]), f"{legacy_id}.forecast.json")
    )
