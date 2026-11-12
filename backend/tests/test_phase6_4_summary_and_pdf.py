import hashlib
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.core.forecast_tracking as forecast_tracking_mod
import src.core.report_generator as report_generator_mod
import src.core.simulation_engine as simulation_engine_mod
from src.api.main import app
from src.core.event_grounding import classify_event_domain
from src.core.forecast_summarizer import (
    build_extractive_summary,
    compress_summary_for_pdf,
    extract_event_date,
    synthesize_forecast_summary_and_chain,
    validate_summary_text,
)
from src.core.forecasting_engine import ensure_forecast_fields
from src.core.llm_client import LLMClient
from src.core.report_generator import generate_simulation_pdf_report
from src.core.schemas import SimulationConfig
from src.core.simulation_engine import SimulationEngine


REPO_ROOT = Path(__file__).resolve().parents[2]
VERBATIM_DISCLAIMER = (
    "This forecast is generated from multi-model structured behavioral interpretations, network propagation proxies, "
    "and simulated diffusion dynamics. Scenario weights represent heuristic exploratory indices rather than "
    "empirically calibrated statistical probabilities. Confirmed facts are separated from unverified claims and "
    "hypothetical future developments."
)


@pytest.fixture
def isolated_storage_dirs(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    forecasts_dir = tmp_path / "forecasts"
    reports_dir = tmp_path / "reports"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(simulation_engine_mod, "SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setattr(forecast_tracking_mod, "FORECASTS_DIR", str(forecasts_dir))
    monkeypatch.setattr(report_generator_mod, "REPORTS_DIR", str(reports_dir))
    return {
        "sessions_dir": sessions_dir,
        "forecasts_dir": forecasts_dir,
        "reports_dir": reports_dir,
    }


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _assert_valid_single_paragraph_summary(summary: str, expected_weight_int: int | None = None) -> None:
    assert summary and isinstance(summary, str)
    assert "\n" not in summary.strip(), "Summary must be a single cohesive paragraph"
    words = summary.split()
    assert 120 <= len(words) <= 180, f"Expected 120-180 words, got {len(words)}: {summary}"
    low = summary.lower()
    for banned in (
        "a(t)=",
        "n(t)=",
        "verification norms",
        "customs exemption",
        "transit lead-time",
        "raw raw",
        "[unverified",
        "[inferred",
        "hypothesis:",
    ):
        assert banned not in low, f"Found banned boilerplate/metric dump {banned!r} in summary: {summary}"
    assert "heuristic weight" in low
    assert "non-empirical" in low or "not an empirical probability" in low
    assert "limited long-term impact" in low
    if expected_weight_int is not None:
        assert f"{expected_weight_int}%" in summary, f"Expected {expected_weight_int}% in summary: {summary}"


def test_event_date_extraction():
    assert extract_event_date("On January 20, 2025, DeepSeek released R1.") == "January 20, 2025"
    assert extract_event_date("On November 30, 2022, OpenAI released ChatGPT.") == "November 30, 2022"
    assert extract_event_date("The controversy surrounding Jensen Huang triggered an inquiry.") is None


def test_summary_ollama_polisher_acceptance_and_logged_fallback():
    sig = "On January 20, 2025, DeepSeek released its R1 reasoning model at lower training costs."
    ctx = classify_event_domain(sig)
    scenarios = [
        {
            "scenario_type": "baseline",
            "title": "Baseline Trajectory — Contained Spillover & Gradual Adaptation",
            "summary": "Cloud providers and AI developers evaluate R1 claims without abrupt disruption.",
            "conditional_weight": 0.48,
            "short_term_consequences": ["Developers run benchmark comparisons"],
            "medium_term_consequences": ["Public cloud GPU utilization recalibrates around private deployment"],
            "long_term_consequences": ["Gradual equilibrium across open and proprietary model tiers"],
        },
        {"scenario_type": "positive", "title": "Positive", "conditional_weight": 0.22},
        {"scenario_type": "negative", "title": "Negative", "conditional_weight": 0.25},
        {"scenario_type": "extreme", "title": "Extreme", "conditional_weight": 0.05},
    ]
    common_kwargs = dict(
        event_title="DeepSeek R1 Launch",
        signal_text=sig,
        confirmed_facts=["On January 20, 2025, DeepSeek released its R1 reasoning model at lower training costs"],
        unverified_allegations=["Reported training cost reductions remain independently unverified"],
        q1_immediate="Developers and cloud buyers initiate benchmark audits.",
        q2_second="Public cloud GPU utilization recalibrates around private model hosting.",
        q2_third="Long-term hardware competition settles around inference efficiency.",
        q3_downside="Divergence between headline claims and audited accuracy",
        q4_stabilizing="Independent audits validate baseline safety and performance disclosures",
        stakeholder_reactions=[
            {"stakeholder": "AI Cloud Providers"},
            {"stakeholder": "Enterprise Developers"},
            {"stakeholder": "Hardware Vendors"},
        ],
        order_effects={
            "first_order": ["Developers and cloud buyers initiate benchmark audits"],
            "second_order": ["Public cloud GPU utilization recalibrates around private model hosting"],
            "third_order_and_systemic": ["Long-term hardware competition settles around inference efficiency"],
            "prevention_and_reversal_conditions": ["Independent audits confirm benchmark parity and safety compliance"],
        },
        scenarios=scenarios,
        leading_scenario=scenarios[0],
        gaps_classified={"missing_source_evidence": ["Absence of audited training compute logs for DeepSeek R1"]},
        triggers_classified={
            "stabilization": ["Independent audits confirm benchmark parity"],
            "escalation": ["Audited enterprise benchmarks expose reliability failures"],
        },
        macro_state={"attention_indicator": 0.535, "active_adoption": 0.113},
        feedback_state={"systemic_state": {"institutional_posture_dampening": 0.40, "escalation_risk": 0.30}},
        domain_ctx=ctx,
        model_names=("qwen3.5:4b", "gemma3:4b"),
        provider_names=("ollama", "ollama"),
        valid_slots=30,
        expected_slots=30,
        use_llm=True,
    )

    good_prose = (
        "Available reporting confirms DeepSeek released its R1 reasoning model on January 20, 2025, "
        "while cost-reduction claims remain unverified; immediately, developers and cloud buyers initiate benchmark audits. "
        "Key stakeholders including AI Cloud Providers, Enterprise Developers, and Hardware Vendors adjust procurement "
        "as media attention outpaces confirmed adoption shifts. Over subsequent months, public cloud GPU utilization "
        "recalibrates around private deployment while longer-horizon hardware competition settles toward inference efficiency. "
        "The Baseline scenario leads with a 48% heuristic weight (not an empirical probability), pointing toward measured "
        "containment without systemic disruption. Limited long-term impact remains plausible if independent audits confirm "
        "benchmark parity, whereas audited reliability failures could widen disruption across cloud infrastructure budgets. "
        "The decisive remaining uncertainty is the absence of independently audited training compute logs verifying reported cost efficiency."
    )
    mock_ok = MagicMock(status_code=200)
    mock_ok.json.return_value = {"response": good_prose}
    with patch("src.core.forecast_summarizer.requests.post", return_value=mock_ok):
        res_ok = synthesize_forecast_summary_and_chain(**common_kwargs)
    assert res_ok["forecast_summary_meta"]["method"].startswith("ollama_grounded_polish:")
    assert res_ok["forecast_summary_meta"]["fallback_used"] is False
    _assert_valid_single_paragraph_summary(res_ok["forecast_summary"], expected_weight_int=48)

    # Off-domain / hallucinated percentage from Ollama must trigger logged extractive fallback (NEVER Mock)
    bad_prose = "Customs exemptions and 42.5% transit lead-time drops happen next week. " * 10
    mock_bad = MagicMock(status_code=200)
    mock_bad.json.return_value = {"response": bad_prose}
    with patch("src.core.forecast_summarizer.requests.post", return_value=mock_bad):
        res_fb = synthesize_forecast_summary_and_chain(**common_kwargs)
    assert res_fb["forecast_summary_meta"]["method"] == "deterministic_grounded_extractive"
    assert res_fb["forecast_summary_meta"]["fallback_used"] is True
    assert "ollama_output_rejected" in str(res_fb["forecast_summary_meta"]["fallback_reason"])
    _assert_valid_single_paragraph_summary(res_fb["forecast_summary"], expected_weight_int=48)


@pytest.mark.parametrize(
    "rel_path,expected_date",
    [
        ("data/sessions/sim_a650771db6_1794385571.json", None),
        ("data/sessions/sim_9289d0daa5_1794381826.json", "January 20, 2025"),
        ("data/backups/sessions_backup_20261111_054005/sim_1319d438b8_1793727240.json", "November 30, 2022"),
    ],
)
def test_archived_real_simulations_upgrade_summary_and_page1_pdf(isolated_storage_dirs, rel_path, expected_date):
    src_path = REPO_ROOT / rel_path
    if not src_path.is_file():
        pytest.skip(f"Archive not present: {src_path}")

    orig_bytes = src_path.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()
    raw = json.loads(orig_bytes.decode("utf-8"))

    iso_path = isolated_storage_dirs["sessions_dir"] / f"{raw['simulation_id']}.json"
    iso_path.write_bytes(orig_bytes)

    # Generating forecast & PDF must NEVER call LLM inference
    with patch("src.core.llm_client.LLMClient.generate_json") as mock_llm, \
         patch("src.core.forecast_summarizer.requests.post") as mock_sum_http:
        upgraded = ensure_forecast_fields(raw)
        fv = upgraded["final_verdict"]
        _assert_valid_single_paragraph_summary(fv["forecast_summary"])
        assert fv.get("initial_event_date") == expected_date
        assert len(fv.get("causal_chain_compact") or {}) == 5

        pdf_meta = generate_simulation_pdf_report(
            raw,
            session_path=str(iso_path),
            reports_dir=str(isolated_storage_dirs["reports_dir"]),
            force_regenerate=True,
        )
        assert mock_llm.call_count == 0
        assert mock_sum_http.call_count == 0

    with fitz.open(pdf_meta["pdf_path"]) as doc:
        assert doc.page_count >= 12
        p1_norm = _norm_ws(doc.load_page(0).get_text())
        p2_norm = _norm_ws(doc.load_page(1).get_text())

        # Standalone Page 1 Executive Forecast assertions
        for req_p1 in (
            "THE BUTTERFLY ALGORITHM",
            "Future Impact Forecast & Final Verdict",
            "1. Event Information",
            "2. Executive Forecast Summary",
            "3. Most Plausible Future Trajectory",
            "4. Key Causal Chain",
            "5. Four Scenario Weights",
            "6. Final Verdict & Uncertainty",
            "7. Methodological Disclaimer",
            VERBATIM_DISCLAIMER,
        ):
            assert req_p1 in p1_norm, f"Missing '{req_p1}' on PDF Page 1 for {raw['simulation_id']}"

        # Zero Page 2 bleed into Page 1, and Page 2 cleanly opens Detailed Technical Report
        assert "1. Executive Verdict" not in p1_norm
        assert "2. Initial Event & Evidence Baseline" not in p1_norm
        assert "Detailed Technical Report" in p2_norm
        assert "1. Executive Verdict" in p2_norm

        # Verify Page 1 Section 5 percentages sum to 100.0%
        s5_m = re.search(r"5\. Four Scenario Weights.*?6\. Final Verdict", p1_norm)
        assert s5_m
        pct_vals = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", s5_m.group(0))]
        assert len(pct_vals) == 4
        assert abs(sum(pct_vals) - 100.0) <= 0.15

        # Verify all 5 numbered Figure captions across technical report + Page N of N footer
        full_norm = "\n".join(_norm_ws(doc.load_page(i).get_text()) for i in range(doc.page_count))
        for fig_lbl in ("Figure 1:", "Figure 2:", "Figure 3:", "Figure 4:", "Figure 5:"):
            assert fig_lbl in full_norm
        assert f"Page 1 of {doc.page_count}" in p1_norm
        assert f"Page {doc.page_count} of {doc.page_count}" in _norm_ws(doc.load_page(doc.page_count - 1).get_text())

    # Verify FastAPI endpoints serve forecast_summary without mutating disk bytes or calling LLMs
    client = TestClient(app)
    r_sim = client.get(f"/api/simulations/{raw['simulation_id']}")
    assert r_sim.status_code == 200
    assert r_sim.json()["final_verdict"]["forecast_summary"] == fv["forecast_summary"]

    r_fc = client.get(f"/api/simulations/{raw['simulation_id']}/forecast")
    assert r_fc.status_code == 200
    assert r_fc.json()["final_verdict"]["forecast_summary"] == fv["forecast_summary"]

    assert hashlib.sha256(src_path.read_bytes()).hexdigest() == orig_sha


def test_fresh_non_ai_simulation_and_failed_run_page1_pdf(isolated_storage_dirs):
    engine = SimulationEngine(
        client_a=LLMClient(provider="mock", model_name="mock-a"),
        client_b=LLMClient(provider="mock", model_name="mock-b"),
    )
    cfg = SimulationConfig(
        event_title="Regional Grid Transformer Shortage",
        signal=(
            "On September 12, 2024, utilities across the Midwest reported extended delivery backlogs "
            "for high-voltage substation transformers while grid operators opened emergency reserve auctions."
        ),
        max_iterations=3,
        provider_a="mock",
        model_a="mock-a",
        provider_b="mock",
        model_b="mock-b",
    )
    res = engine.run_simulation(cfg)
    dumped = res.model_dump(mode="json")
    fv = dumped["final_verdict"]
    _assert_valid_single_paragraph_summary(fv["forecast_summary"])
    assert fv["initial_event_date"] == "September 12, 2024"

    pdf_meta = generate_simulation_pdf_report(
        dumped,
        reports_dir=str(isolated_storage_dirs["reports_dir"]),
        force_regenerate=True,
    )
    with fitz.open(pdf_meta["pdf_path"]) as doc:
        p1_norm = _norm_ws(doc.load_page(0).get_text())
        assert "Regional Grid Transformer Shortage" in p1_norm
        assert "September 12, 2024" in p1_norm
        assert VERBATIM_DISCLAIMER in p1_norm
        assert "1. Executive Verdict" not in p1_norm
        assert "1. Executive Verdict" in _norm_ws(doc.load_page(1).get_text())

    # Compress helper must preserve complete sentences and never end dangling
    comp = compress_summary_for_pdf(fv["forecast_summary"], max_words=135)
    assert len(comp.split()) <= 135
    assert comp.endswith(".")
    assert "raw raw" not in comp.lower()


def test_failed_execution_handles_summary_and_pdf_honestly(isolated_storage_dirs):
    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"response": "", "done": True, "done_reason": "stop"}

    engine = SimulationEngine(
        client_a=LLMClient(provider="ollama", model_name="qwen3.5:4b"),
        client_b=LLMClient(provider="ollama", model_name="gemma3:4b"),
    )
    cfg = SimulationConfig(
        event_title="Failed Connectivity Benchmark",
        signal="Central bank announces emergency overnight liquidity facility adjustment.",
        max_iterations=3,
        provider_a="ollama",
        model_a="qwen3.5:4b",
        provider_b="ollama",
        model_b="gemma3:4b",
    )
    with patch.dict(os.environ, {"OLLAMA_MAX_RETRIES": "0"}), \
         patch("src.core.simulation_engine.check_ollama_health", return_value={"running": True}), \
         patch("src.core.llm_client.requests.post", return_value=empty_resp):
        res = engine.run_simulation(cfg)

    dumped = res.model_dump(mode="json")
    fv = dumped["final_verdict"]
    assert dumped["status"] == "failed"
    assert fv["forecast_summary"]
    assert "failed" in fv["forecast_summary"].lower()

    pdf_meta = generate_simulation_pdf_report(
        dumped,
        reports_dir=str(isolated_storage_dirs["reports_dir"]),
        force_regenerate=True,
    )
    with fitz.open(pdf_meta["pdf_path"]) as doc:
        p1_norm = _norm_ws(doc.load_page(0).get_text())
        assert "FAILED" in p1_norm
        assert VERBATIM_DISCLAIMER in p1_norm
        assert "1. Executive Verdict" not in p1_norm
