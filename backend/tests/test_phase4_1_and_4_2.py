"""
ba/tests/test_phase4_1_and_4_2.py

Comprehensive regression suite covering Phase 4.1 (Critical Model Corrections & Cross-Layer Validation)
and Phase 4.2 (Simulation Isolation, Past Simulations Management & Ollama JSON Reliability).

All persistence and API tests run in isolated pytest `tmp_path` directories via monkeypatching
`src.core.simulation_engine.SESSIONS_DIR` so existing user simulation history in `data/sessions/`
is never modified or deleted.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.core.simulation_engine as simulation_engine_mod
from src.api.main import app, _JOBS
from src.core.evaluation import TrajectoryEvaluator
from src.core.llm_client import (
    LLMClient,
    LLMInferenceError,
    parse_and_validate_json,
    sanitize_error_text,
)
from src.core.meso_layer import MesoInfluenceGraph, run_meso_layer
from src.core.micro_layer import (
    compute_evidence_and_transmission_scores,
    run_archetype_interpretation,
    run_micro_layer,
)
from src.core.schemas import (
    Archetype,
    EvidenceStatus,
    FeedbackState,
    InterpretationOutput,
    IterationResult,
    MacroState,
    MesoGraphState,
    MicroInterpretation,
    ModelExecutionStatus,
    SimulationConfig,
    SimulationStatus,
)
from src.core.simulation_engine import (
    SimulationEngine,
    compute_model_agreement_details,
    compute_narrative_convergence_details,
)


def _valid_interpretation_payload(**overrides: Any) -> Dict[str, Any]:
    base = {
        "stance": 0.45,
        "virality_estimate": 0.65,
        "perceived_credibility": 0.72,
        "emotional_charge": 0.40,
        "primary_narrative": "Initial verified facts indicate measurable market repricing risk.",
        "key_arguments": ["Primary event confirmed by source disclosure"],
        "narrative_tags": ["market_risk", "disclosure", "verification"],
        "evidence_status": "supported",
        "supporting_evidence": ["Official release quoted directly in prompt"],
        "uncertainty": 0.22,
        "claims": ["Market repricing risk identified in official release"],
    }
    base.update(overrides)
    return base


def _make_valid_micro(
    archetype: Archetype,
    model_slot: str,
    model_name: str,
    stance: float = 0.4,
    virality: float = 0.6,
    credibility: float = 0.75,
    emotional_charge: float = 0.4,
    evidence_status: EvidenceStatus = EvidenceStatus.SUPPORTED,
    uncertainty: float = 0.2,
    tags: List[str] | None = None,
    iteration: int = 1,
) -> MicroInterpretation:
    llm_cred, eff_cred, ev_g, ev_p, t_w = compute_evidence_and_transmission_scores(
        stance=stance,
        virality=virality,
        emotional_charge=emotional_charge,
        model_estimated_credibility=credibility,
        evidence_status=evidence_status.value,
        uncertainty=uncertainty,
        supporting_evidence=["Source anchor"] if evidence_status != EvidenceStatus.UNSUPPORTED else [],
    )
    return MicroInterpretation(
        archetype=archetype,
        archetype_name=archetype.value.replace("_", " ").title(),
        model_name=model_name,
        provider="ollama",
        model_slot=model_slot,
        execution_status=ModelExecutionStatus.SUCCESS.value,
        stance=stance,
        virality=virality,
        credibility=eff_cred,
        emotional_charge=emotional_charge,
        model_estimated_credibility=llm_cred,
        evidence_grounding_score=ev_g,
        evidence_penalty_applied=ev_p,
        transmission_weight=t_w,
        primary_narrative=f"{archetype.value} interpretation by {model_name}",
        key_arguments=["Grounded claim"],
        narrative_tags=tags or ["analysis", archetype.value],
        iteration=iteration,
        evidence_status=evidence_status,
        supporting_evidence=["Source anchor"],
        uncertainty=uncertainty,
        claims=["Structured claim"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# PART A — OLLAMA JSON RELIABILITY & DIAGNOSTIC TELEMETRY TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_secret_redaction_never_leaks_keys():
    raw = "Error calling https://api.example.com?key=AIzaSySecret12345 with Bearer sk-secret-token-999 and gsk_abcd123456"
    clean = sanitize_error_text(raw)
    assert "AIzaSySecret12345" not in clean
    assert "sk-secret-token-999" not in clean
    assert "gsk_abcd123456" not in clean
    assert "[REDACTED]" in clean


def test_parse_and_validate_controlled_recovery():
    schema_cls = InterpretationOutput
    raw_fenced = f"Here is the output:\n```json\n{json.dumps(_valid_interpretation_payload())}\n```"
    parsed, recovered = parse_and_validate_json(raw_fenced, model_name="qwen3.5:4b", response_model=schema_cls)
    assert recovered is True
    assert parsed["stance"] == pytest.approx(0.45)

    # Pure valid JSON needs no recovery
    raw_clean = json.dumps(_valid_interpretation_payload())
    parsed2, recovered2 = parse_and_validate_json(raw_clean, model_name="qwen3.5:4b", response_model=schema_cls)
    assert recovered2 is False
    assert parsed2["primary_narrative"].startswith("Initial verified")


def test_ollama_payload_includes_think_false_and_schema():
    client = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    captured_payloads: List[Dict[str, Any]] = []

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": json.dumps(_valid_interpretation_payload()),
        "done": True,
        "done_reason": "stop",
    }

    def fake_post(url, json=None, timeout=None, headers=None):
        captured_payloads.append(json)
        return mock_resp

    with patch("src.core.llm_client.requests.post", side_effect=fake_post):
        out = client.generate_structured(
            prompt="Analyze test signal.",
            json_schema=InterpretationOutput.strict_json_schema(),
            response_model=InterpretationOutput,
            max_retries=1,
        )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["format"] == InterpretationOutput.strict_json_schema()
    assert payload["options"]["num_predict"] >= 256
    assert out["stance"] == pytest.approx(0.45)
    assert client.last_call_metadata["execution_status"] == ModelExecutionStatus.SUCCESS.value
    assert client.last_call_metadata["retry_count"] == 0


def test_ollama_think_400_fallback_compatibility():
    """Models that reject `think: false` with HTTP 400 fall back automatically once without `think`."""
    client = LLMClient(provider="ollama", model_name="legacy-small-model")
    payloads: List[Dict[str, Any]] = []

    resp_400 = MagicMock()
    resp_400.status_code = 400
    resp_400.text = '{"error":"model does not support think parameter"}'

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "response": json.dumps(_valid_interpretation_payload()),
        "done": True,
        "done_reason": "stop",
    }

    def fake_post(url, json=None, timeout=None, headers=None):
        payloads.append(dict(json))
        return resp_400 if len(payloads) == 1 else resp_200

    with patch("src.core.llm_client.requests.post", side_effect=fake_post):
        out = client.generate_structured(
            prompt="Analyze signal.",
            json_schema=InterpretationOutput.strict_json_schema(),
            response_model=InterpretationOutput,
            max_retries=1,
        )

    assert len(payloads) == 2
    assert "think" in payloads[0]
    assert "think" not in payloads[1]
    assert out["primary_narrative"]


def test_ollama_empty_response_retries_and_recovers():
    """Simulates qwen3.5:4b thinking-exhaustion empty response recovering on retry."""
    client = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    prompts_seen: List[str] = []

    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {
        "response": "",
        "thinking": "Long internal chain of thought that hit token budget...",
        "done": True,
        "done_reason": "length",
    }

    good_resp = MagicMock()
    good_resp.status_code = 200
    good_resp.json.return_value = {
        "response": json.dumps(_valid_interpretation_payload()),
        "done": True,
        "done_reason": "stop",
    }

    def fake_post(url, json=None, timeout=None, headers=None):
        prompts_seen.append(json["prompt"])
        return empty_resp if len(prompts_seen) == 1 else good_resp

    with patch("src.core.llm_client.requests.post", side_effect=fake_post):
        out = client.generate_structured(
            prompt="Analyze test signal.",
            json_schema=InterpretationOutput.strict_json_schema(),
            response_model=InterpretationOutput,
            max_retries=2,
        )

    assert len(prompts_seen) == 2
    assert "CORRECTION REQUIRED" in prompts_seen[1]
    assert client.last_call_metadata["retry_count"] == 1
    assert out["evidence_status"] == "supported"


def test_ollama_exhausted_empty_raises_structured_failure_no_mock():
    """When retries exhaust on empty output, raises LLMInferenceError explicitly without Mock fallback."""
    client = LLMClient(provider="ollama", model_name="qwen3.5:4b")

    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {
        "response": "",
        "thinking": "Reasoning only...",
        "done": True,
        "done_reason": "length",
    }

    with patch("src.core.llm_client.requests.post", return_value=empty_resp):
        with pytest.raises(LLMInferenceError) as exc_info:
            client.generate_structured(
                prompt="Test prompt",
                json_schema=InterpretationOutput.strict_json_schema(),
                response_model=InterpretationOutput,
                max_retries=1,
            )

    err = exc_info.value
    assert err.error_category in ("empty_output", "truncated_output")
    assert err.execution_status == ModelExecutionStatus.INVALID_JSON.value
    assert err.retry_count == 1
    assert err.thinking_length > 0
    assert err.response_length == 0


def test_ollama_malformed_json_then_schema_failure_retries():
    client = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    responses = iter([
        {"response": "{not valid json", "done": True, "done_reason": "stop"},
        {"response": json.dumps(_valid_interpretation_payload(primary_narrative="   ")), "done": True, "done_reason": "stop"},
        {"response": json.dumps(_valid_interpretation_payload()), "done": True, "done_reason": "stop"},
    ])

    def fake_post(url, json=None, timeout=None, headers=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = next(responses)
        return resp

    with patch("src.core.llm_client.requests.post", side_effect=fake_post):
        out = client.generate_structured(
            prompt="Analyze signal",
            json_schema=InterpretationOutput.strict_json_schema(),
            response_model=InterpretationOutput,
            max_retries=2,
        )

    assert client.last_call_metadata["retry_count"] == 2
    assert out["primary_narrative"]


def test_ollama_timeout_maps_to_explicit_timeout_status():
    client = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    with patch("src.core.llm_client.requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        with pytest.raises(LLMInferenceError) as exc_info:
            client.generate_structured(
                prompt="Test prompt",
                json_schema=InterpretationOutput.strict_json_schema(),
                response_model=InterpretationOutput,
                max_retries=1,
            )
    assert exc_info.value.execution_status == ModelExecutionStatus.TIMEOUT.value
    assert exc_info.value.error_category == "timeout"


# ──────────────────────────────────────────────────────────────────────────────
# PART B — MICRO SLOT ISOLATION & NO SILENT SUBSTITUTION
# ──────────────────────────────────────────────────────────────────────────────


def test_model_a_failure_is_explicit_and_never_substituted_with_model_b(monkeypatch):
    """
    Reproduces the scenario where Model A (qwen3.5:4b) fails on JSON/timeout while
    Model B (gemma3:4b) succeeds. Verifies Model A rows remain explicitly failed
    with empty primary_narrative and Model B rows succeed cleanly without cross-fill.
    """
    monkeypatch.setenv("OLLAMA_MAX_RETRIES", "0")
    client_a = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    client_b = LLMClient(provider="mock", model_name="gemma3:4b")

    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"response": "", "done": True, "done_reason": "stop"}

    diagnostics: List[Dict[str, Any]] = []
    with patch("src.core.llm_client.requests.post", return_value=empty_resp), \
         patch("src.core.micro_layer.unload_model_from_vram"):
        interps = run_micro_layer(
            signal="Regional bank deposit outflow rumor reported online.",
            client_a=client_a,
            client_b=client_b,
            iteration=1,
            diagnostics_out=diagnostics,
        )

    assert len(interps) == 10
    slot_a = [m for m in interps if m.model_slot == "model_a"]
    slot_b = [m for m in interps if m.model_slot == "model_b"]
    assert len(slot_a) == 5
    assert len(slot_b) == 5

    for a in slot_a:
        assert a.execution_status == ModelExecutionStatus.INVALID_JSON.value
        assert a.primary_narrative == ""
        assert a.error_category in ("empty_output", "truncated_output")
        assert a.model_name == "qwen3.5:4b"

    for b in slot_b:
        assert b.execution_status == ModelExecutionStatus.SUCCESS.value
        assert len(b.primary_narrative.strip()) > 0
        assert b.model_name == "gemma3:4b"

    assert len(diagnostics) == 10
    assert sum(1 for d in diagnostics if d["execution_status"] == "success") == 5


def test_identical_model_names_still_preserve_distinct_model_slots():
    client_a = LLMClient(provider="mock", model_name="qwen3.5:4b")
    client_b = LLMClient(provider="mock", model_name="qwen3.5:4b")

    interps = run_micro_layer(
        signal="Tech firm announces battery efficiency jump.",
        client_a=client_a,
        client_b=client_b,
        iteration=1,
    )
    assert len(interps) == 10
    assert sum(1 for m in interps if m.model_slot == "model_a") == 5
    assert sum(1 for m in interps if m.model_slot == "model_b") == 5


# ──────────────────────────────────────────────────────────────────────────────
# PART C — MESO LAYER [0, 1] BOUNDS, SYMMETRY, EXTERNAL MEDIA & RAW INFLOWS
# ──────────────────────────────────────────────────────────────────────────────


def test_meso_bounds_under_extreme_saturation_and_raw_inflow_preserved():
    """
    Even if every archetype fires at 1.0 virality, credibility, emotional_charge, and |stance|=1.0,
    normalized current_activation MUST remain in [0.0, 1.0], while raw_influence (>1.0) is
    preserved separately without distorting normalized activation.
    """
    extreme_interps = []
    for arch in Archetype:
        extreme_interps.append(
            _make_valid_micro(
                archetype=arch,
                model_slot="model_a",
                model_name="qwen3.5:4b",
                stance=1.0,
                virality=1.0,
                credibility=1.0,
                emotional_charge=1.0,
                evidence_status=EvidenceStatus.SUPPORTED,
                uncertainty=0.0,
            )
        )

    meso = run_meso_layer(extreme_interps, external_weight=0.4)
    assert meso.activation_semantics == "instantaneous_per_iteration"
    assert meso.meso_metrics_explanation is not None

    for node in meso.nodes:
        assert 0.0 <= node.current_activation <= 1.0, f"{node.id} violated [0,1]: {node.current_activation}"
        assert node.activation_mode == "instantaneous_per_iteration"
        assert node.raw_influence is not None

    # Public nodes have inbound edge weight sums > 1.0, so raw_influence exceeds 1.0 cleanly
    mainstream = next(n for n in meso.nodes if n.id == "mainstream_public")
    subcultures = next(n for n in meso.nodes if n.id == "engaged_subcultures")
    assert mainstream.raw_influence > 1.0
    assert subcultures.raw_influence > 1.0
    assert mainstream.current_activation <= 1.0
    assert subcultures.current_activation <= 1.0

    # External media broadcast must be non-zero when Amplifier/Institutional outputs are active
    ext_node = next(n for n in meso.nodes if n.id == "external_media_broadcast")
    assert ext_node.current_activation > 0.0


def test_meso_stance_symmetry_negative_equals_positive_conviction():
    """Negative conviction (-0.8) must have identical transmission weight to positive conviction (+0.8)."""
    neg_interps = [
        _make_valid_micro(
            archetype=arch,
            model_slot="model_a",
            model_name="m",
            stance=-0.8,
            virality=0.7,
            credibility=0.8,
            emotional_charge=0.5,
            evidence_status=EvidenceStatus.SUPPORTED,
            uncertainty=0.2,
        )
        for arch in Archetype
    ]
    pos_interps = [
        _make_valid_micro(
            archetype=arch,
            model_slot="model_a",
            model_name="m",
            stance=0.8,
            virality=0.7,
            credibility=0.8,
            emotional_charge=0.5,
            evidence_status=EvidenceStatus.SUPPORTED,
            uncertainty=0.2,
        )
        for arch in Archetype
    ]

    meso_neg = run_meso_layer(neg_interps, external_weight=0.4)
    meso_pos = run_meso_layer(pos_interps, external_weight=0.4)

    for nid in meso_neg.node_activations:
        assert meso_neg.node_activations[nid] == pytest.approx(meso_pos.node_activations[nid], abs=1e-6)


def test_meso_repeated_iterations_do_not_accumulate_unbounded():
    """Running Meso layer repeatedly does not compound previous activations above 1.0."""
    graph = MesoInfluenceGraph(external_weight=0.4)
    interps = [
        _make_valid_micro(arch, "model_a", "m", stance=1.0, virality=0.95, credibility=0.95, emotional_charge=0.95)
        for arch in Archetype
    ]
    runs = [graph.compute_propagation(interps) for _ in range(5)]
    for state in runs:
        for node in state.nodes:
            assert 0.0 <= node.current_activation <= 1.0
    assert runs[0].node_activations == runs[-1].node_activations


def test_meso_excludes_failed_interpretations_and_penalizes_unsupported():
    _, eff_supp, g_supp, p_supp, t_supp = compute_evidence_and_transmission_scores(
        stance=0.7,
        virality=0.8,
        emotional_charge=0.6,
        model_estimated_credibility=0.85,
        evidence_status="supported",
        uncertainty=0.15,
        supporting_evidence=["Verified filing"],
    )
    _, eff_unsupp, g_unsupp, p_unsupp, t_unsupp = compute_evidence_and_transmission_scores(
        stance=0.7,
        virality=0.8,
        emotional_charge=0.6,
        model_estimated_credibility=0.85,
        evidence_status="unsupported",
        uncertainty=0.15,
        supporting_evidence=[],
    )
    assert eff_unsupp <= 0.40
    assert p_supp == pytest.approx(1.0)
    assert p_unsupp == pytest.approx(0.40)
    assert g_unsupp < g_supp
    assert t_unsupp < t_supp * 0.6

    # Failed interpretations must contribute 0 transmission
    failed_only = [
        MicroInterpretation(
            archetype=arch,
            archetype_name=arch.value,
            model_name="qwen3.5:4b",
            provider="ollama",
            model_slot="model_a",
            execution_status=ModelExecutionStatus.INVALID_JSON.value,
            stance=1.0,
            virality=1.0,
            credibility=1.0,
            emotional_charge=1.0,
            primary_narrative="",
            iteration=1,
        )
        for arch in Archetype
    ]
    meso_failed = run_meso_layer(failed_only)
    assert all(val == 0.0 for val in meso_failed.node_activations.values())
    assert meso_failed.effective_spread_potential == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# PART D — AGREEMENT, CONVERGENCE, EXECUTION STATES & EVALUATION KEYS
# ──────────────────────────────────────────────────────────────────────────────


def test_agreement_returns_none_when_single_model_or_zero_pairs():
    """Never return fake 1.0 agreement when Model A failed and only Model B succeeded."""
    only_b = [
        _make_valid_micro(arch, "model_b", "gemma3:4b", stance=0.4)
        for arch in Archetype
    ]
    failed_a = [
        MicroInterpretation(
            archetype=arch,
            archetype_name=arch.value,
            model_name="qwen3.5:4b",
            provider="ollama",
            model_slot="model_a",
            execution_status=ModelExecutionStatus.INVALID_JSON.value,
            stance=0.0,
            virality=0.0,
            credibility=0.0,
            emotional_charge=0.0,
            primary_narrative="",
            iteration=1,
        )
        for arch in Archetype
    ]
    details = compute_model_agreement_details(failed_a + only_b)
    assert details["available"] is False
    assert details["score"] is None
    assert details["valid_pairs"] == 0
    assert "N/A" in details["reason"]


def test_agreement_metrics_separate_stance_from_semantic_similarity():
    pair_agree_tags_diff = [
        _make_valid_micro(Archetype.INSTITUTIONAL_ANALYST, "model_a", "qwen3.5:4b", stance=0.6, tags=["bonds", "liquidity"]),
        _make_valid_micro(Archetype.INSTITUTIONAL_ANALYST, "model_b", "gemma3:4b", stance=0.6, tags=["solar", "panels"]),
        _make_valid_micro(Archetype.SKEPTIC_DEBUNKER, "model_a", "qwen3.5:4b", stance=-0.5, tags=["audit", "claim"]),
        _make_valid_micro(Archetype.SKEPTIC_DEBUNKER, "model_b", "gemma3:4b", stance=-0.5, tags=["football", "stadium"]),
    ]
    details = compute_model_agreement_details(pair_agree_tags_diff)
    assert details["available"] is True
    assert details["stance_agreement"] == pytest.approx(1.0)
    assert details["stance_correlation"] == pytest.approx(1.0, abs=1e-4)
    assert details["tag_semantic_similarity"] < 0.5


def test_narrative_convergence_detects_polarization_expansion():
    def _dummy_iter(it_num: int, stances: List[float]) -> IterationResult:
        micros = [
            _make_valid_micro(arch, "model_a", "qwen3.5:4b", stance=s, iteration=it_num)
            for arch, s in zip(Archetype, stances)
        ]
        meso = run_meso_layer(micros)
        macro = MacroState(
            iteration=it_num,
            p_external=0.03,
            q_internal=0.25,
            M_potential=1.0,
            c_churn=0.08,
            cumulative_adoption=0.1,
            active_adoption=0.09,
            adoption_velocity=0.02,
            attention_indicator=0.2,
            attention_vs_adoption_ratio=2.0,
        )
        fb = FeedbackState(
            iteration=it_num,
            macro_summary="s",
            dominant_framing="d",
            polarization_index=0.2,
            prompt_context_for_next_round="c",
        )
        return IterationResult(
            iteration=it_num,
            micro_interpretations=micros,
            meso_state=meso,
            macro_state=macro,
            feedback_state=fb,
        )

    # Expanding variance: round 1 near 0, round 3 split [-0.9, +0.9] -> 0.0 (diverging_polarizing)
    div_details = compute_narrative_convergence_details([
        _dummy_iter(1, [0.1, 0.1, 0.0, 0.0, 0.1]),
        _dummy_iter(2, [-0.5, 0.5, -0.4, 0.4, 0.6]),
        _dummy_iter(3, [-0.9, 0.9, -0.8, 0.8, 0.9]),
    ])
    assert div_details["score"] == 0.0
    assert div_details["direction"] == "diverging_polarizing"

    # Contracting variance: round 1 split, round 3 aligned -> positive convergence
    conv_details = compute_narrative_convergence_details([
        _dummy_iter(1, [-0.9, 0.9, -0.8, 0.8, 0.9]),
        _dummy_iter(2, [-0.4, 0.4, -0.3, 0.3, 0.4]),
        _dummy_iter(3, [0.1, 0.2, 0.1, 0.15, 0.1]),
    ])
    assert conv_details["score"] > 0.7
    assert conv_details["direction"] == "converging"


def test_simulation_engine_statuses_partial_and_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation_engine_mod, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_MAX_RETRIES", "0")

    # Case 1: Model A fails via Ollama error while Model B succeeds -> partially_completed
    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"response": "", "done": True, "done_reason": "stop"}

    client_a = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    client_b = LLMClient(provider="mock", model_name="gemma3:4b")

    engine_partial = SimulationEngine(client_a=client_a, client_b=client_b)
    cfg = SimulationConfig(
        signal="Central bank tests offline retail token.",
        event_title="CBDC Retail Trial",
        max_iterations=3,
        model_a="qwen3.5:4b",
        model_b="gemma3:4b",
    )

    with patch("src.core.simulation_engine.check_ollama_health", return_value={"running": True}), \
         patch("src.core.simulation_engine.is_model_installed", return_value=True), \
         patch("src.core.llm_client.requests.post", return_value=empty_resp), \
         patch("src.core.micro_layer.unload_model_from_vram"):
        res_partial = engine_partial.run_simulation(cfg)

    assert res_partial.status == SimulationStatus.PARTIALLY_COMPLETED.value
    assert res_partial.successful_interpretations_count == 15
    assert res_partial.expected_interpretations_count == 30
    assert res_partial.model_agreement_score is None
    assert len(res_partial.failed_models) == 1
    assert res_partial.failed_models[0]["model_slot"] == "model_a"
    assert res_partial.failed_models[0]["failure_count"] == 15
    assert len(res_partial.diagnostics["provider_errors"]) == 15
    assert res_partial.event_title == "CBDC Retail Trial"

    # Case 2: Both models fail -> status == failed
    client_a2 = LLMClient(provider="ollama", model_name="qwen3.5:4b")
    client_b2 = LLMClient(provider="ollama", model_name="gemma3:4b")
    engine_fail = SimulationEngine(client_a=client_a2, client_b=client_b2)
    with patch("src.core.simulation_engine.check_ollama_health", return_value={"running": True}), \
         patch("src.core.simulation_engine.is_model_installed", return_value=True), \
         patch("src.core.llm_client.requests.post", return_value=empty_resp), \
         patch("src.core.micro_layer.unload_model_from_vram"):
        res_fail = engine_fail.run_simulation(cfg)

    assert res_fail.status == SimulationStatus.FAILED.value
    assert res_fail.successful_interpretations_count == 0
    assert res_fail.model_agreement_score is None

    # Verify evaluation returns UI-compatible numeric keys
    ev = TrajectoryEvaluator.evaluate(res_partial, observed_adoption=[0.05, 0.12, 0.20])
    assert "directional_accuracy" in ev
    assert isinstance(ev["directional_accuracy"], float)
    assert "peak_timing_error" in ev


# ──────────────────────────────────────────────────────────────────────────────
# PART E — STRICT API ISOLATION, PAST SIMULATIONS METADATA & SAFE DELETION
# ──────────────────────────────────────────────────────────────────────────────


def test_api_async_isolation_past_simulations_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation_engine_mod, "SESSIONS_DIR", str(tmp_path))
    _JOBS.clear()
    client = TestClient(app)

    # Start Run 1 (Async)
    payload_1 = {
        "signal": "Synthetic Battery Discovery at 400 Wh/kg reported by Lab One.",
        "event_title": "Run One Battery",
        "max_iterations": 3,
        "provider_a": "mock",
        "model_a": "mock-a",
        "provider_b": "mock",
        "model_b": "mock-b",
    }
    r1 = client.post("/api/simulations/run/async", json=payload_1)
    assert r1.status_code == 200
    d1 = r1.json()
    assert "job_id" in d1 and "simulation_id" in d1
    sim1_id = d1["simulation_id"]
    job1_id = d1["job_id"]

    # Poll until Run 1 completes (FastAPI TestClient executes BackgroundTasks synchronously before returning or shortly after)
    j1 = client.get(f"/jobs/{job1_id}").json()
    assert j1["simulation_id"] == sim1_id
    assert j1["status"] == "completed"

    # Start Run 2 (Async) with completely different prompt
    payload_2 = {
        "signal": "Major shipping canal closure triggers logistics rerouting alerts.",
        "event_title": "Run Two Canal",
        "max_iterations": 3,
        "provider_a": "mock",
        "model_a": "mock-a",
        "provider_b": "mock",
        "model_b": "mock-b",
    }
    r2 = client.post("/api/simulations/run/async", json=payload_2)
    assert r2.status_code == 200
    d2 = r2.json()
    sim2_id = d2["simulation_id"]
    assert sim2_id != sim1_id

    # Verify Past Simulations listing contains rich metadata for BOTH independent runs
    sess_resp = client.get("/sessions")
    assert sess_resp.status_code == 200
    sessions = sess_resp.json()["sessions"]
    by_id = {s["simulation_id"]: s for s in sessions}
    assert sim1_id in by_id
    assert sim2_id in by_id
    assert by_id[sim1_id]["event_title"] == "Run One Battery"
    assert by_id[sim2_id]["event_title"] == "Run Two Canal"
    assert by_id[sim1_id]["status"] == "completed"
    assert by_id[sim1_id]["successful_interpretations_count"] == 30
    assert by_id[sim1_id]["expected_interpretations_count"] == 30

    # Strict retrieval isolation
    get1 = client.get(f"/api/simulations/{sim1_id}")
    get2 = client.get(f"/api/simulations/{sim2_id}")
    assert get1.status_code == 200 and get2.status_code == 200
    assert "Battery Discovery" in get1.json()["signal_text"]
    assert "shipping canal closure" in get2.json()["signal_text"]

    # Path traversal protection
    bad = client.get("/api/simulations/..secret")
    assert bad.status_code == 400

    # Safe deletion removes ONLY sim1 and leaves sim2 intact
    del_resp = client.delete(f"/api/simulations/{sim1_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    assert client.get(f"/api/simulations/{sim1_id}").status_code == 404
    assert client.get(f"/api/simulations/{sim2_id}").status_code == 200
    remaining = client.get("/api/simulations").json()["sessions"]
    assert all(s["simulation_id"] != sim1_id for s in remaining)
    assert any(s["simulation_id"] == sim2_id for s in remaining)
