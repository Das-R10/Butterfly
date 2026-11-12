"""
ba/tests/test_phase6_3_1_relevance.py

Phase 6.3.1 Verification Suite — Event-Domain Grounding, Causal Relevance,
Negation-Aware Trigger Classification, Non-Repeating Monitoring Matrix,
Ungrounded Quantitative Reframing, Horizon Alignment, and Isolated
Non-Destructive Re-Evaluation of sim_a650771db6_1794385571.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app
from src.core.event_grounding import (
    check_event_relevance,
    classify_event_domain,
    classify_trigger_and_condition,
    enforce_horizon_alignment,
    is_completed_premise_action,
    is_reassuring_or_stabilizing,
    reframe_ungrounded_numbers,
    sanitize_prompt_anchor_leaks,
)
from src.core.forecasting_engine import ensure_forecast_fields
from src.core.llm_client import LLMClient
from src.core.report_generator import generate_simulation_pdf_report
from src.core.schemas import SimulationConfig
from src.core.simulation_engine import SimulationEngine


NVIDIA_SIGNAL = (
    "Reports alleging that foreign intelligence agencies targeted NVIDIA CEO Jensen Huang "
    "have triggered an independent board investigation."
)

BANNED_CONTAMINATION = (
    "transit lead-time",
    "customs exemption",
    "freight/input",
    "multi-port",
    "retaliatory",
    "procedural waiver",
    "smooth routing",
    "trade, customs",
    "open-source ecosystems",
    "open-source builders",
    "open-source developer",
    "educators",
    "competing benchmark reverses",
    "board initiates formal inquiry",
    "10-15%",
    "5-10%",
    "-5% to -10%",
    "1-5%",
)


def test_domain_classifier_word_boundaries_no_port_reports_collision():
    ctx = classify_event_domain(NVIDIA_SIGNAL)
    assert ctx["domain"] == "corporate_governance"
    assert ctx["has_explicit_trade"] is False
    assert ctx["has_open_source"] is False
    assert ctx["has_education"] is False
    assert "ai_software" not in ctx["secondary_themes"]
    assert "NVIDIA" in ctx["entities"]
    assert "Jensen Huang" in ctx["entities"]
    assert all("NVIDIA CEO" not in e for e in ctx["entities"])

    alt_ctx = classify_event_domain(
        "The controversy surrounding Jensen Huang, NVIDIA's CEO, and foreign intelligence reports "
        "triggered an independent board investigation. Analyze semiconductor risk and Forecast impacts."
    )
    assert alt_ctx["org_entity"] == "NVIDIA"
    assert alt_ctx["person_entity"] == "Jensen Huang"
    assert "Analyze" not in alt_ctx["entities"]
    assert "Forecast" not in alt_ctx["entities"]

    trade_ctx = classify_event_domain("Dockworkers union strike halts freight containers at major seaports.")
    assert trade_ctx["domain"] == "trade_supply"

    ai_ctx = classify_event_domain("DeepSeek releases R1 open-weight reasoning model for developers.")
    assert ai_ctx["domain"] == "ai_software"


def test_event_relevance_and_explicit_causal_bridge_exception():
    ctx = classify_event_domain(NVIDIA_SIGNAL)

    # Bare off-domain templates must be rejected
    for bare in (
        "Transit lead-time shifts vs retained procurement reallocation",
        "Port customs exemption backlog delays container cargo",
        "Open-source builders to reaffirm reliance on platform stability",
        "Competing benchmark reverses capability advantage",
        "Educators adopt oral exam rubrics across classrooms",
    ):
        ok, reason = check_event_relevance(bare, ctx, allow_bridged=True)
        assert ok is False, f"Expected bare off-domain statement to be rejected: {bare}"
        assert reason

    # Explicit multi-step causal bridge tied to signal entities is allowed
    bridged = (
        "Foreign intelligence allegations targeting NVIDIA trigger enterprise compliance audits "
        "leading to slower cloud GPU contract renewals across major customers."
    )
    ok_b, _ = check_event_relevance(bridged, ctx, allow_bridged=True)
    assert ok_b is True

    bridged_chip = (
        "NVIDIA investigation uncertainty triggers semiconductor export compliance delays "
        "causing customs exemption scrutiny for data-center GPU shipments."
    )
    ok_bc, _ = check_event_relevance(bridged_chip, ctx, allow_bridged=True)
    assert ok_bc is True


def test_premise_action_exoneration_numeric_and_horizon_guards():
    # Completed initial premise repetition must be blocked, while post-trigger findings stay
    assert is_completed_premise_action("Board initiates formal inquiry protocol", NVIDIA_SIGNAL, []) is True
    assert (
        is_completed_premise_action(
            "Independent board releases preliminary audit findings on governance controls",
            NVIDIA_SIGNAL,
            [],
        )
        is False
    )

    # Reassuring / exonerating outcomes must be filtered from downside risk paths
    assert (
        is_reassuring_or_stabilizing(
            "Rapid market rationalization once investigation findings confirm no operational security breach."
        )
        is True
    )
    assert (
        is_reassuring_or_stabilizing(
            "Leaked classified documents contradicting prior 'no breach' assurances trigger regulatory scrutiny."
        )
        is False
    )

    # Ungrounded market percentages are reframed without leaving raw figures behind,
    # while explicit simulation telemetry lines remain untouched
    reframed, note = reframe_ungrounded_numbers(
        "10-15% drop in NVIDIA GPU sales and -5% to -10% stock price decline due to uncertainty.",
        NVIDIA_SIGNAL,
    )
    assert note is not None
    assert "10-15%" not in reframed and "-5% to -10%" not in reframed
    assert "10-15%" not in note and "-5% to -10%" not in note
    assert "magnitude" in reframed.lower()

    reframed_def, note_def = reframe_ungrounded_numbers(
        "Customer defections (5-10%) from large enterprise clients concerned about security vulnerabilities.",
        NVIDIA_SIGNAL,
    )
    assert note_def is not None
    assert "5-10%" not in reframed_def
    assert "customer defections" in reframed_def.lower()

    telemetry_line = (
        "Active Sim. Adoption A(t)=8.0%, Cumul. Sim. Diffusion N(t)=8.8%, "
        "simulated churn c=8%, Baseline trajectory 47.0% heuristic weight."
    )
    kept_tel, tel_note = reframe_ungrounded_numbers(telemetry_line, NVIDIA_SIGNAL)
    assert kept_tel == telemetry_line
    assert tel_note is None

    # Horizon mismatch guard blocks 7-day & transient short-term trading items in medium-term slots
    assert enforce_horizon_alignment("Short-term (7 days) - 10-15% drop in GPU orders", "medium_term_months")[1] is False
    assert (
        enforce_horizon_alignment(
            "Temporary reduction in short-term speculative trading volume on NVIDIA stock",
            "medium_term_months",
        )[1]
        is False
    )
    assert (
        enforce_horizon_alignment(
            "Accelerated adoption of verified zero-trust compliance reviews across enterprise buyers over 3-6 months.",
            "medium_term_months",
        )[1]
        is True
    )

    # Prompt-anchor comma leak sanitization preserves domain-specific stakeholders
    ctx = classify_event_domain(NVIDIA_SIGNAL)
    cleaned_focus = sanitize_prompt_anchor_leaks("Open-source ecosystems, enterprise CIOs, investors, educators", ctx)
    assert "open-source ecosystems" not in cleaned_focus.lower()
    assert "educators" not in cleaned_focus.lower()
    assert "enterprise CIOs" in cleaned_focus


def test_negation_aware_trigger_classification():
    # "no criminal activity or data breach" -> stabilization/mitigation, NEVER escalation
    assert classify_trigger_and_condition(
        "Release of preliminary board findings confirming no criminal activity or data breach."
    ) == ("mitigation", "stabilization")

    assert classify_trigger_and_condition(
        "Investigation concludes with no findings related to security breaches or foreign intelligence involvement"
    ) == ("mitigation", "stabilization")

    # Confirmed harm / removal / contradiction -> escalation
    assert classify_trigger_and_condition(
        "Formal announcement by the board confirming executive removal or severe operational disruption."
    ) == ("escalation", "escalation")

    assert classify_trigger_and_condition(
        "Publication of specific classified documents or code compromises."
    ) == ("escalation", "escalation")

    assert classify_trigger_and_condition(
        "Leaked documents contradicting the no-breach narrative trigger regulatory subpoenas."
    ) == ("escalation", "escalation")

    assert classify_trigger_and_condition(
        "Board releases interim findings indicating potential security protocol gaps."
    ) == ("escalation", "escalation")

    # Debunked / fabricated -> reversal / falsification
    cond, trig5 = classify_trigger_and_condition(
        "Allegations are definitively debunked or proven to be fabricated by independent evidence."
    )
    assert cond in ("reversal", "falsification")
    assert trig5 in ("reversal", "falsification")


def test_isolated_re_evaluation_of_sim_a650771db6_and_pdf_and_api(isolated_storage_dirs):
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "data" / "sessions" / "sim_a650771db6_1794385571.json"
    if not src_path.is_file():
        pytest.skip("Reference archive sim_a650771db6_1794385571.json not present in workspace")

    orig_bytes = src_path.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()

    # Copy into isolated pytest tmp_path; NEVER touch production data/sessions
    iso_session_path = isolated_storage_dirs["sessions_dir"] / "sim_a650771db6_1794385571.json"
    iso_session_path.write_bytes(orig_bytes)

    raw_data = json.loads(orig_bytes.decode("utf-8"))
    upgraded = ensure_forecast_fields(raw_data)

    fv = upgraded["final_verdict"]
    trace = fv["traceability_refs"]
    assert trace["event_domain"] == "corporate_governance"
    assert "NVIDIA" in " ".join(trace["event_entities"])

    # Meso event-specific stakeholder mapping exists and is NVIDIA/governance grounded
    last_meso = upgraded["iterations"][-1]["meso_state"]
    st_map = last_meso.get("event_stakeholder_mapping") or []
    assert len(st_map) >= 5
    st_blob = " ".join(m["stakeholder"] for m in st_map).lower()
    assert any(k in st_blob for k in ("board", "jensen", "nvidia", "investor", "security", "semiconductor"))

    # Forecast sections must have ZERO contamination, ZERO ungrounded %, ZERO premise repetition,
    # and ZERO reassuring rationalization inside Q3 / Negative / Extreme / Pathway D
    forecast_only_blob = json.dumps({
        "causal_pathways": upgraded["causal_pathways"],
        "order_effects_summary": upgraded["order_effects_summary"],
        "stakeholder_reactions": upgraded["stakeholder_reactions"],
        "scenarios": upgraded["scenarios"],
        "time_horizon_forecast": upgraded["time_horizon_forecast"],
        "epistemic_breakdown": upgraded["epistemic_breakdown"],
        "final_verdict": upgraded["final_verdict"],
    }).lower()

    for bad in BANNED_CONTAMINATION:
        assert bad not in forecast_only_blob, f"Banned contamination '{bad}' found in upgraded forecast JSON"

    # Reassuring outcome must NEVER appear in Q3, Pathway D, Negative, or Extreme
    pw_d = next(pw for pw in upgraded["causal_pathways"] if pw["pathway_id"] == "pathway_tail_cascade")
    neg_sc = next(sc for sc in upgraded["scenarios"] if sc["scenario_type"] == "negative")
    ext_sc = next(sc for sc in upgraded["scenarios"] if sc["scenario_type"] == "extreme")
    adverse_blob = json.dumps({
        "q3": fv["most_dangerous_downside_risk"],
        "pathway_d": [s["effect"] for s in pw_d["steps"]],
        "negative": neg_sc["short_term_consequences"] + neg_sc["medium_term_consequences"],
        "extreme": ext_sc["short_term_consequences"] + ext_sc["medium_term_consequences"],
    }).lower()
    assert "rapid market rationalization" not in adverse_blob
    assert "confirm no operational security breach" not in adverse_blob

    # Strict 3rd-order / medium_term horizon alignment (no 1st-order / 7-day / speculative blip leakage)
    for pw in upgraded["causal_pathways"]:
        for st in pw["steps"]:
            if st["step_id"].endswith("_s3"):
                eff_low = st["effect"].lower()
                assert "short-term (7 days)" not in eff_low
                assert "short-term speculative trading" not in eff_low
    for med_item in upgraded["time_horizon_forecast"]["medium_term_months"]:
        assert "short-term (7 days)" not in med_item.lower()
        assert "short-term speculative trading" not in med_item.lower()

    # Non-repeating 1-to-1 monitoring matrix
    mm = fv.get("monitoring_matrix") or []
    assert len(mm) >= 6
    assert len({r["indicator"].lower() for r in mm}) == len(mm)
    assert len({r["trigger_threshold"].lower() for r in mm}) == len(mm)
    actions_core = [r["analyst_action"].split(" (Threshold:")[0].lower() for r in mm]
    assert len(set(actions_core)) == len(mm), f"Expected unique monitoring actions, got: {actions_core}"

    # Verify API read endpoint returns upgraded fields & stakeholder mapping without mutating disk bytes
    client = TestClient(app)
    resp = client.get("/api/simulations/sim_a650771db6_1794385571")
    assert resp.status_code == 200
    api_body = resp.json()
    assert api_body["final_verdict"]["traceability_refs"]["event_domain"] == "corporate_governance"
    assert len(api_body["iterations"][-1]["meso_state"]["event_stakeholder_mapping"]) >= 5
    assert hashlib.sha256(iso_session_path.read_bytes()).hexdigest() == orig_sha

    # Generate PDF into isolated tmp_reports_dir & audit headings + zero contamination in forecast sections
    pdf_meta = generate_simulation_pdf_report(
        raw_data,
        session_path=str(iso_session_path),
        reports_dir=str(isolated_storage_dirs["reports_dir"]),
        force_regenerate=True,
    )
    with fitz.open(pdf_meta["pdf_path"]) as doc:
        full_pdf_text = "\n".join(doc.load_page(i).get_text() for i in range(doc.page_count))
    low_pdf = full_pdf_text.lower()
    assert "5.2 event-specific stakeholder channel mapping" in low_pdf
    for bad_logistics in (
        "transit lead-time",
        "customs exemption",
        "freight/input",
        "multi-port",
        "retaliatory",
        "procedural waiver",
        "smooth routing",
        "competing benchmark reverses",
        "board initiates formal inquiry",
    ):
        assert bad_logistics not in low_pdf, f"Contaminated text '{bad_logistics}' found in PDF"

    # Confirm Section 14 renders clean 1-to-1 monitoring matrix and telemetry remains preserved
    sec14_m = re.search(r"14\. Key Indicators.*?(?=15\. Final Verdict)", full_pdf_text, re.S)
    assert sec14_m
    assert "53.7%" in full_pdf_text or "8.0%" in full_pdf_text

    # Re-verify production archive untouched
    assert hashlib.sha256(src_path.read_bytes()).hexdigest() == orig_sha


def test_fresh_simulation_end_to_end_populates_domain_grounding_and_meso_mapping(isolated_storage_dirs):
    engine = SimulationEngine(
        client_a=LLMClient(provider="mock", model_name="mock-a"),
        client_b=LLMClient(provider="mock", model_name="mock-b"),
    )
    cfg = SimulationConfig(
        event_title="NVIDIA Board Inquiry Test",
        signal=NVIDIA_SIGNAL,
        max_iterations=3,
        provider_a="mock",
        model_a="mock-a",
        provider_b="mock",
        model_b="mock-b",
    )
    res = engine.run_simulation(cfg)
    dumped = res.model_dump(mode="json")

    assert dumped["final_verdict"]["traceability_refs"]["event_domain"] == "corporate_governance"
    st_map = dumped["iterations"][-1]["meso_state"]["event_stakeholder_mapping"]
    assert len(st_map) >= 5
    assert all(0.0 <= s["relevance_weight"] <= 1.0 for s in st_map)
    assert all(0.0 <= s["meso_node_activation"] <= 1.0 for s in st_map)

    blob = json.dumps(dumped).lower()
    for bad in ("transit lead-time", "customs exemption", "freight/input", "multi-port", "procedural waiver"):
        assert bad not in blob

