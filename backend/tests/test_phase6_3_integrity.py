"""
ba/tests/test_phase6_3_integrity.py

Phase 6.3 Verification Suite — Verdict Quality (Q1-Q8), Archetype Differentiation & Consensus vs Collapse,
Causal Pathway Quality & Missing Links, Macro Metric Semantics, Scenario & Counter-Evidence Integrity,
Non-Destructive Upgrade of Legacy Sessions, and Synchronized 15-Section PDF Verification.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from typing import Any, Dict, List

import fitz  # PyMuPDF
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.forecasting_engine import (
    _is_benign_structural_shift,
    _is_unexplained_fragmentation,
    build_forecast_bundle,
    ensure_forecast_fields,
)
from src.core.llm_client import LLMClient
from src.core.meso_layer import run_meso_layer
from src.core.report_generator import generate_simulation_pdf_report
from src.core.schemas import (
    Archetype,
    EvidenceStatus,
    FeedbackState,
    IterationResult,
    MacroState,
    MicroInterpretation,
    ModelExecutionStatus,
    SimulationConfig,
)
from src.core.simulation_engine import (
    SimulationEngine,
    compute_narrative_convergence_details,
)


DEEPSEEK_SIGNAL = (
    "On January 20, 2025, DeepSeek released its R1 reasoning model, attracting significant global "
    "attention because of its reported reasoning capabilities and comparatively low training and inference costs. "
    "The release challenged prevailing assumptions about computing resources and financial investment, raising "
    "questions about AI economics, open-source adoption, GPU demand, and startups entering the market."
)


def _make_rich_micro(
    archetype: Archetype,
    model_slot: str,
    model_name: str,
    *,
    stance: float,
    risk_severity: float | None = None,
    analytical_question: str = "",
    stakeholder_focus: str = "",
    strongest_supporting: str = "",
    strongest_opposing: str = "",
    narrative: str = "",
    mechanism: str = "",
    actions: List[str] | None = None,
    behaviours: List[str] | None = None,
    consequences: List[str] | None = None,
    triggers: List[str] | None = None,
    invalidations: List[str] | None = None,
    supporting_evidence: List[str] | None = None,
    claims: List[str] | None = None,
    tags: List[str] | None = None,
    iteration: int = 1,
) -> MicroInterpretation:
    return MicroInterpretation(
        archetype=archetype,
        archetype_name=archetype.value.replace("_", " ").title(),
        model_name=model_name,
        provider="ollama",
        model_slot=model_slot,
        execution_status=ModelExecutionStatus.SUCCESS.value,
        stance=stance,
        virality=0.64,
        credibility=0.76,
        emotional_charge=0.44,
        model_estimated_credibility=0.76,
        evidence_grounding_score=0.74,
        evidence_penalty_applied=1.0,
        transmission_weight=0.58,
        primary_narrative=narrative or f"{archetype.value} distinct analytical lens.",
        key_arguments=[f"{archetype.value} core causal point"],
        narrative_tags=tags or [archetype.value, "deepseek"],
        analytical_question=analytical_question or f"Core question tailored to {archetype.value}",
        stakeholder_focus=stakeholder_focus or f"Primary cohort for {archetype.value}",
        strongest_supporting_consideration=strongest_supporting or f"Evidence favoring {archetype.value} view",
        strongest_opposing_consideration=strongest_opposing or f"Strongest counterfactual against {archetype.value}",
        risk_severity=risk_severity,
        iteration=iteration,
        evidence_status=EvidenceStatus.INFERRED,
        supporting_evidence=supporting_evidence or [DEEPSEEK_SIGNAL],
        uncertainty=0.35,
        claims=claims or [f"{archetype.value} forward projection"],
        causal_mechanism=mechanism or f"{archetype.value} distinct causal transition",
        expected_stakeholder_actions=actions or [f"{archetype.value} target actor responds"],
        behavioural_changes=behaviours or [f"{archetype.value} cohort changes tooling"],
        emergent_consequences=consequences or [f"{archetype.value} downstream market outcome"],
        time_horizon="short_term_weeks",
        trigger_conditions=triggers or [f"{archetype.value} observable trigger"],
        invalidation_conditions=invalidations or [f"{archetype.value} falsification condition"],
    )


def test_part1_verdict_q1_to_q8_integrity_and_guardrails():
    # Craft an adversarial ensemble mixing:
    # - uncorroborated export-control/ban shock claim
    # - benign startup competition & lower inference cost claims inside Doomer
    # - unexplained fragmentation claim inside Institutional Analyst
    # - genuine harm mechanisms + high decoupled risk_severity inside Doomer
    micros = [
        _make_rich_micro(
            Archetype.INSTITUTIONAL_ANALYST,
            "model_a",
            "qwen3.5:4b",
            stance=0.32,
            risk_severity=0.40,
            analytical_question="How will enterprises and universities adapt governance to R1?",
            stakeholder_focus="Enterprise CIOs & AI governance boards",
            narrative="Enterprises test R1 weights under phased governance audits while avoiding premature fragmentation.",
            mechanism="Institutional procurement gates shift spending toward benchmark verification before workflow rollout.",
            actions=["Enterprise risk committees mandate reproducible evaluation audits"],
            consequences=["Enterprise governance frameworks formalize model provenance tracking"],
            triggers=[
                "Fragmentation accelerates without shared standards",  # Should be filtered as unexplained fragmentation
                "Published independent benchmark reproducibility standards clarify model reliability",
            ],
            invalidations=["Primary evaluation benchmarks are retracted or fail replication"],
        ),
        _make_rich_micro(
            Archetype.SKEPTIC_DEBUNKER,
            "model_a",
            "qwen3.5:4b",
            stance=-0.28,
            risk_severity=0.25,
            analytical_question="Are DeepSeek R1 training-cost claims fully substantiated?",
            stakeholder_focus="Hardware auditors & ML systems engineers",
            narrative="Reported training costs exclude prior pretraining infrastructure and cluster amortization.",
            mechanism="Independent hardware cost audits separate marginal post-training spend from foundational capex.",
            actions=["Analysts audit total compute cluster amortization schedules"],
            consequences=["Infrastructure buyers recalibrate cost-saving assumptions once amortized compute is priced"],
            triggers=["Independent compute accounting reveals hidden pretraining expenditure"],
            invalidations=["Audited third-party reproduction confirms full end-to-end training under headline figure"],
        ),
        _make_rich_micro(
            Archetype.PATTERN_SEEKER,
            "model_a",
            "qwen3.5:4b",
            stance=0.38,
            risk_severity=0.45,
            analytical_question="What historical SaaS/hardware disruption pattern mirrors cheaper reasoning models?",
            stakeholder_focus="Cloud providers, startups, and GPU infrastructure planners",
            narrative="Cheaper inference historically lowers unit margins while expanding aggregate API utilization (Jevons paradox).",
            mechanism="Lower per-token inference cost induces developers to embed reasoning chains across high-volume workflows.",
            actions=["Cloud platforms package distilled reasoning endpoints for application developers"],
            consequences=[
                "Application-layer API inference volume expands into enterprise ticketing and code review",
                "Hyperscaler capital allocation shifts from speculative pretraining toward elastic inference serving",
            ],
            triggers=["Startups commercialize lower-cost reasoning APIs rapidly"],
            invalidations=["Developer retention drops sharply due to reasoning latency overhead"],
        ),
        _make_rich_micro(
            Archetype.AMPLIFIER_INFLUENCER,
            "model_a",
            "qwen3.5:4b",
            stance=0.68,
            risk_severity=0.55,
            analytical_question="How does viral social attention distort immediate market pricing?",
            stakeholder_focus="Retail investors & developer social feeds",
            narrative="Viral benchmarks spark short-term GPU valuation swings well ahead of verified deployment metrics.",
            mechanism="Social virality amplifies cost-disruption headlines before institutional procurement cycles complete.",
            actions=["High-velocity commentary drives elevated equity volatility"],
            consequences=["Short-lived sentiment overshoot widens gap between attention spikes and enterprise revenue"],
            triggers=["Immediate semiconductor valuation pullback exceeds verified contract cancellations"],
            invalidations=["Quarterly hardware order backlogs confirm sustained hyperscaler demand"],
        ),
        _make_rich_micro(
            Archetype.DOOMER,
            "model_a",
            "qwen3.5:4b",
            stance=-0.45,
            risk_severity=0.78,
            analytical_question="What concrete systemic harms arise if unvetted reasoning weights deploy at scale?",
            stakeholder_focus="Critical infrastructure operators & safety compliance teams",
            narrative="Unvetted open weights deployed without safety red-teaming raise security vulnerability propagation.",
            mechanism="Startups adopt cheaper weights without safety regression testing, increasing systemic exploit exposure.",
            actions=["Operators disable unverified automated code execution after exploit incidents"],
            consequences=[
                "Lower cost allows smaller startups to enter market rapidly",  # Benign! Must be excluded from Q3/Doomer harms
                "Unvetted autonomous workflow errors cause production security breaches and institutional liability",
            ],
            triggers=[
                "Emergency bilateral government bans and export-control crackdowns hit immediately",  # Uncorroborated shock! Must be filtered
                "Small startups gain accessible reasoning tools",  # Benign! Must be excluded from Negative/Extreme triggers
                "Confirmed safety bypass incidents trigger coordinated production lockouts",
            ],
            invalidations=["Verified red-teaming suites block known exploit classes before deployment"],
        ),
    ]

    sim_payload = {
        "simulation_id": "sim_phase63_q1_q8_test",
        "event_title": "DeepSeek R1 Release",
        "signal_text": DEEPSEEK_SIGNAL,
        "status": "completed",
        "model_agreement_score": 0.62,
        "model_agreement_details": {"available": True, "causal_semantic_similarity": 0.44},
        "narrative_convergence_score": 0.48,
        "narrative_convergence_details": {
            "available": True,
            "archetype_collapse_warning": False,
            "legitimate_consensus_detected": False,
            "reasoning_differentiation_score": 0.72,
        },
        "iterations": [
            {
                "iteration": 1,
                "micro_interpretations": [m.model_dump(mode="json") for m in micros],
                "meso_state": run_meso_layer(micros).model_dump(mode="json"),
                "macro_state": {
                    "iteration": 1,
                    "p_external": 0.032,
                    "q_internal": 0.310,
                    "M_potential": 1.0,
                    "c_churn": 0.08,
                    "cumulative_adoption": 0.145,
                    "active_adoption": 0.128,
                    "adoption_velocity": 0.092,
                    "attention_indicator": 0.610,
                    "attention_vs_adoption_ratio": 4.77,
                },
                "feedback_state": {
                    "iteration": 1,
                    "macro_summary": "Attention outpaces verified adoption",
                    "dominant_framing": "Cost disruption",
                    "polarization_index": 0.42,
                    "prompt_context_for_next_round": "Monitor evidence quality",
                    "claim_verification_audit": {"supported": 2, "inferred": 3, "unsupported": 0, "total_interpretations": 5},
                    "systemic_state": {
                        "institutional_dampening_index": 0.48,
                        "public_pressure": 0.56,
                        "economic_market_stress": 0.44,
                        "escalation_risk": 0.39,
                        "epistemic_fragility": 0.41,
                    },
                },
            }
        ],
    }

    bundle = build_forecast_bundle(sim_payload)
    fv = bundle["final_verdict"]

    # Q1: explicit mechanism, no auto-invented bans/export-controls, includes insufficiency disclosure when applicable
    q1 = fv["most_likely_immediate_outcome"].lower()
    assert "causal mechanism:" in q1
    assert "export-control crackdown" not in q1
    assert "insufficiency disclosure" in q1

    # Q2: strict order separation with causal arrows
    q2 = fv["most_plausible_secondary_cascade"]
    assert "1st-order:" in q2 and "2nd-order:" in q2 and "3rd-order:" in q2
    assert "--(via " in q2

    # Q3: benign competition / startup entry excluded; explicit harm included + Doomer risk severity surfaced
    q3 = fv["most_dangerous_downside_risk"].lower()
    assert "lower cost allows smaller startups" not in q3
    assert ("breach" in q3 or "exploit" in q3 or "vulnerability" in q3 or "liability" in q3)
    assert "risk severity" in q3

    # Q4: categorized positive outcome, clean punctuation (no '..'), unexplained fragmentation filtered out
    q4 = fv["most_plausible_stabilizing_outcome"]
    assert any(q4.startswith(tag) for tag in ("[Positive Development]", "[Mitigation]", "[Recovery]", "[Uncertainty Reduction]"))
    assert ".." not in q4.replace("...", "")
    assert "fragmentation accelerates" not in q4.lower()

    # Q5: 5-way trigger classification
    t_cls = fv["triggers_classified"]
    assert set(t_cls.keys()) == {"acceleration", "escalation", "stabilization", "reversal", "falsification"}
    assert all(isinstance(v, list) and len(v) >= 1 for v in t_cls.values())
    assert not any("fragmentation accelerates" in item.lower() for item in t_cls["stabilization"])
    assert not any("small startups gain" in item.lower() for item in t_cls["escalation"])

    # Q6: 3-way epistemic gap classification + attributed hypothesis boundary
    g_cls = fv["evidence_gaps_classified"]
    assert set(g_cls.keys()) == {"missing_source_evidence", "model_disagreement", "unsupported_causal_links"}
    assert all(isinstance(v, list) and len(v) >= 1 for v in g_cls.values())

    # Q7: cohesive monitoring_matrix with measurable_variable + supports/contradicts + distinct scenario triggers
    mm = fv["monitoring_matrix"]
    assert len(mm) >= 4
    for row in mm:
        assert row.get("measurable_variable")
        assert row.get("supports_scenario_movement")
        assert row.get("contradicts_scenario_movement")
        assert row.get("scenario_type") in {"baseline", "positive", "negative", "extreme"}
        assert row.get("trigger_threshold")
        assert row.get("analyst_action")

    # Baseline vs Positive and Negative vs Extreme primary triggers should not collide verbatim
    scen_map = {s["scenario_type"]: s for s in bundle["scenarios"]}
    assert scen_map["baseline"]["trigger_conditions"][0] != scen_map["positive"]["trigger_conditions"][0]
    assert scen_map["negative"]["trigger_conditions"][0] != scen_map["extreme"]["trigger_conditions"][0]
    assert not any("export-control crackdown" in t.lower() for t in scen_map["extreme"]["trigger_conditions"])

    # Q8: heuristic weights sum to 1.0000 and is_empirical_calibration is False
    probs = fv["scenario_probabilities"]
    assert probs["is_empirical_calibration"] is False
    wt_sum = sum(probs[k]["conditional_weight"] for k in ("baseline", "positive", "negative", "extreme"))
    assert abs(wt_sum - 1.0) <= 0.002


def test_part2_archetype_fields_decoupled_risk_and_legitimate_consensus_vs_collapse():
    # Mock engine emits all 5 new Phase 6.3 fields cleanly
    engine = SimulationEngine(
        client_a=LLMClient(provider="mock", model_name="mock-a"),
        client_b=LLMClient(provider="mock", model_name="mock-b"),
    )
    cfg = SimulationConfig(
        event_title="DeepSeek R1 Release",
        signal=DEEPSEEK_SIGNAL,
        max_iterations=2,
        provider_a="mock",
        model_a="mock-a",
        provider_b="mock",
        model_b="mock-b",
    )
    res = engine.run_simulation(cfg)
    for m in res.iterations[0].micro_interpretations:
        assert m.analytical_question
        assert m.stakeholder_focus
        assert m.strongest_supporting_consideration
        assert m.strongest_opposing_consideration
        assert m.risk_severity is not None and 0.0 <= m.risk_severity <= 1.0

    # Test legitimate directional consensus: tight stances (+0.41..+0.45) WITH distinct archetype reasoning
    consensus_round = [
        _make_rich_micro(
            Archetype.INSTITUTIONAL_ANALYST, "model_a", "qwen3.5:4b",
            stance=0.42, risk_severity=0.45,
            stakeholder_focus="Hospitality & Municipal regulators",
            narrative="City permitting authorities update zoning guidelines methodically.",
            mechanism="Municipal safety audits convert emergency decrees into transparent compliance codes.",
            actions=["Regulators issue phased compliance bulletins"],
            triggers=["Public permitting portals go live"],
            tags=["governance", "permitting"],
        ),
        _make_rich_micro(
            Archetype.SKEPTIC_DEBUNKER, "model_a", "qwen3.5:4b",
            stance=0.41, risk_severity=0.22,
            stakeholder_focus="Empirical energy & grid auditors",
            narrative="Metered load logs support gradual grid efficiency claims with caveats.",
            mechanism="Smart-meter baselines isolate real peak-shaving from weather fluctuations.",
            actions=["Independent auditors publish baseline methodology"],
            triggers=["Audited telemetry reproduces peak reduction"],
            tags=["telemetry", "audits"],
        ),
        _make_rich_micro(
            Archetype.PATTERN_SEEKER, "model_a", "qwen3.5:4b",
            stance=0.44, risk_severity=0.38,
            stakeholder_focus="Cloud ISVs & mid-market builders",
            narrative="Historical middleware deflation unlocks downstream SMB software tooling.",
            mechanism="API margin compression transfers pricing power to vertical integration vendors.",
            actions=["SaaS builders bundle vertical agent templates"],
            triggers=["Developer SDK downloads cross inflection"],
            tags=["middleware", "saas"],
        ),
        _make_rich_micro(
            Archetype.AMPLIFIER_INFLUENCER, "model_a", "qwen3.5:4b",
            stance=0.45, risk_severity=0.52,
            stakeholder_focus="Tech newsletters & creator channels",
            narrative="Developer tutorials trend rapidly across video & forum feeds.",
            mechanism="Creator demo clips compress educational onboarding cycles worldwide.",
            actions=["Educators publish starter walkthroughs"],
            triggers=["Search volume spikes globally"],
            tags=["viral", "tutorials"],
        ),
        _make_rich_micro(
            Archetype.DOOMER, "model_a", "qwen3.5:4b",
            stance=0.43, risk_severity=0.81,  # High decoupled risk even with positive directional stance!
            stakeholder_focus="Identity security & fraud monitoring teams",
            narrative="Rapid automation adoption simultaneously widens synthetic fraud surfaces.",
            mechanism="Unmonitored automated agents execute social engineering at lower adversary cost.",
            actions=["Security desks mandate strict authentication telemetry"],
            triggers=["Fraud complaint rates rise meaningfully"],
            tags=["fraud", "security"],
        ),
    ]
    iter_stub = lambda n, ms: IterationResult(
        iteration=n,
        micro_interpretations=ms,
        meso_state=run_meso_layer(ms),
        macro_state=MacroState(iteration=n, p_external=0.03, q_internal=0.25, M_potential=1.0, c_churn=0.08, cumulative_adoption=0.2, active_adoption=0.18, adoption_velocity=0.04, attention_indicator=0.5, attention_vs_adoption_ratio=2.5),
        feedback_state=FeedbackState(iteration=n, macro_summary="s", dominant_framing="d", polarization_index=0.1, prompt_context_for_next_round="c"),
    )
    conv_consensus = compute_narrative_convergence_details([
        iter_stub(1, consensus_round),
        iter_stub(3, consensus_round),
    ])
    assert conv_consensus["archetype_stance_std_final"] < 0.08
    assert conv_consensus["reasoning_differentiation_score"] >= 0.45
    assert conv_consensus["legitimate_consensus_detected"] is True
    assert conv_consensus["archetype_collapse_warning"] is False

    # Contrast: true uniform collapse (same narrative, mechanism, stakeholder focus, stance)
    collapsed_round = [
        _make_rich_micro(
            arch, "model_a", "qwen3.5:4b",
            stance=0.42, risk_severity=0.40,
            stakeholder_focus="General tech ecosystem actors",
            narrative="Everyone adopts identical automation software smoothly without friction.",
            mechanism="All stakeholders follow identical rollout patterns synchronously.",
            actions=["Stakeholders deploy automation tools together"],
            triggers=["Adoption rises steadily"],
            tags=["uniform", "automation"],
            claims=["Uniform adoption grows"],
            iteration=3,
        )
        for arch in Archetype
    ]
    conv_collapse = compute_narrative_convergence_details([
        iter_stub(1, collapsed_round),
        iter_stub(3, collapsed_round),
    ])
    assert conv_collapse["archetype_collapse_warning"] is True
    assert conv_collapse["legitimate_consensus_detected"] is False


def test_part3_and_part4_zero_neutral_focus_missing_links_and_simulated_macro_labels(isolated_storage_dirs):
    # Sparse 1st-order-only inputs should surface explicit [MISSING LINK] & causal step fields without generic boilerplate
    sparse_micros = [
        {
            "archetype": arch.value,
            "archetype_name": arch.value.replace("_", " ").title(),
            "model_name": "qwen3.5:4b",
            "execution_status": "success",
            "stance": 0.25 if arch != Archetype.SKEPTIC_DEBUNKER else -0.35,
            "virality": 0.55,
            "credibility": 0.70,
            "uncertainty": 0.45,
            "evidence_status": "inferred",
            "primary_narrative": f"{arch.value} initial observation.",
            "supporting_evidence": [DEEPSEEK_SIGNAL],
            "claims": [],
            "causal_mechanism": "",
            "expected_stakeholder_actions": [],
            "behavioural_changes": [],
            "emergent_consequences": [],
            "trigger_conditions": [],
            "invalidation_conditions": [],
        }
        for arch in Archetype
    ]
    sim_sparse = {
        "simulation_id": "sim_sparse_missing_links",
        "event_title": "DeepSeek R1 Release",
        "signal_text": DEEPSEEK_SIGNAL,
        "status": "completed",
        "iterations": [
            {
                "iteration": 1,
                "micro_interpretations": sparse_micros,
                "meso_state": {"external_influence_score": 0.45, "internal_contagion_score": 0.40, "effective_spread_potential": 0.42},
                "macro_state": {"cumulative_adoption": 0.14, "active_adoption": 0.12, "attention_indicator": 0.58, "c_churn": 0.08},
                "feedback_state": {"polarization_index": 0.30, "claim_verification_audit": {"supported": 0, "inferred": 5, "unsupported": 0, "total_interpretations": 5}, "systemic_state": {}},
            }
        ],
    }

    bundle = build_forecast_bundle(sim_sparse)
    dumped_str = json.dumps(bundle).lower()

    # Part 3: Zero generic boilerplate & zero "actual adoption" claims
    assert "neutral_focus" not in dumped_str
    assert "stakeholders across institutional and public tiers evaluate" not in dumped_str
    assert "actual adoption" not in dumped_str
    assert "actual persistent adoption" not in dumped_str

    all_steps = [st for pw in bundle["causal_pathways"] for st in pw["steps"]]
    missing_steps = [st for st in all_steps if st.get("is_missing_link")]
    assert len(missing_steps) >= 2, "Sparse evidence must explicitly flag missing intermediate causal links"
    for st in all_steps:
        assert st.get("affected_actor")
        assert st.get("behavioral_or_institutional_response")
        assert st.get("invalidation_condition")
        if st.get("is_missing_link"):
            assert st.get("missing_link_reason")

    # Part 5 helpers direct checks
    assert _is_benign_structural_shift("Lower inference cost makes advanced tools more accessible to startups") is True
    assert _is_benign_structural_shift("Startups ship unvetted models causing severe security breaches") is False
    assert _is_unexplained_fragmentation("Deep ecosystem fragmentation spreads across vendors") is True
    assert _is_unexplained_fragmentation("Vendor diversification lowers single-point systemic risk and improves resilience") is False

    # Part 6: Non-destructive ensure_forecast_fields + PDF sync audit
    legacy_sim = {
        "simulation_id": "sim_legacy_upgrade",
        "event_title": "DeepSeek R1 Release",
        "signal_text": DEEPSEEK_SIGNAL,
        "status": "completed",
        "config": {"model_a": "mock-a", "model_b": "mock-b", "provider_a": "mock", "provider_b": "mock"},
        "iterations": copy.deepcopy(sim_sparse["iterations"]),
        "final_verdict": {"headline": "Old v6.2 Headline"},  # Lacks Phase 6.3 classified fields
        "scenarios": [],
    }
    upgraded = ensure_forecast_fields(legacy_sim)
    assert upgraded is not legacy_sim
    assert legacy_sim["final_verdict"] == {"headline": "Old v6.2 Headline"}  # Original dict untouched
    assert upgraded["final_verdict"]["triggers_classified"]
    assert upgraded["final_verdict"]["evidence_gaps_classified"]
    assert len(upgraded["final_verdict"]["monitoring_matrix"]) >= 4

    session_file = isolated_storage_dirs["sessions_dir"] / "sim_legacy_upgrade.json"
    session_file.write_text(json.dumps(upgraded, indent=2), encoding="utf-8")
    pdf_meta = generate_simulation_pdf_report(
        upgraded,
        session_path=str(session_file),
        reports_dir=str(isolated_storage_dirs["reports_dir"]),
        force_regenerate=True,
    )
    with fitz.open(pdf_meta["pdf_path"]) as doc:
        full_text = "\n".join(doc.load_page(i).get_text() for i in range(doc.page_count))
    low_pdf = full_text.lower()
    assert "actual adoption" not in low_pdf
    assert "actual persistent adoption" not in low_pdf
    assert "simulated" in low_pdf
    assert "[missing link]" in low_pdf or "missing link" in low_pdf
    assert "verdict question (q1-q8)" in low_pdf
    assert "1. what happens first? (immediate)" in low_pdf
    assert "5. key triggers & turning points" in low_pdf
    assert "6. evidence & reasoning gaps" in low_pdf
    assert "7. priority monitoring variables" in low_pdf
    assert "8. simulated metrics & epistemic note" in low_pdf


def test_part3b_q1_supported_vs_inferred_and_pathway_d_actor_invalidation_distinctness():
    # Supported Institutional slot with corroborated action omits Q1 insufficiency disclosure;
    # Cloud provider harm step infers Cloud Infrastructure actor (not Education) and distinct invalidation conditions across steps.
    inst_sup = _make_rich_micro(
        Archetype.INSTITUTIONAL_ANALYST,
        "model_a",
        "qwen3.5:4b",
        stance=0.30,
        risk_severity=0.35,
        actions=["Universities and enterprise CIOs publish formal acceptable-use review circulars"],
        consequences=["Institutions codify reproducible evaluation gates for reasoning APIs"],
        triggers=["Formal evaluation circulars take effect across procurement teams"],
        invalidations=[
            "Institutions rescind guidance circular within first week",
            "Procurement boards abandon third-party verification mandates",
        ],
    )
    inst_sup.evidence_status = EvidenceStatus.SUPPORTED

    skep_m = _make_rich_micro(
        Archetype.SKEPTIC_DEBUNKER,
        "model_a",
        "qwen3.5:4b",
        stance=-0.20,
        risk_severity=0.25,
        consequences=["Cost auditors isolate post-training tuning spend from full cluster capex"],
        invalidations=["Vendor financial filings validate end-to-end cluster bill"],
    )
    patt_m = _make_rich_micro(
        Archetype.PATTERN_SEEKER,
        "model_a",
        "qwen3.5:4b",
        stance=0.35,
        risk_severity=0.40,
        consequences=["Application teams integrate lower-cost APIs into batch classification pipelines"],
        invalidations=["Latency bottlenecks block high-volume adoption", "Switching contracts lock buyers to incumbents"],
    )
    ampl_m = _make_rich_micro(
        Archetype.AMPLIFIER_INFLUENCER,
        "model_a",
        "qwen3.5:4b",
        stance=0.55,
        risk_severity=0.50,
        consequences=["Retail commentary amplifies disruption headlines ahead of quarterly earnings"],
        invalidations=["Attention rotates rapidly to competing launches", "Measured API revenue calms volatility"],
    )
    doom_m = _make_rich_micro(
        Archetype.DOOMER,
        "model_a",
        "qwen3.5:4b",
        stance=-0.45,
        risk_severity=0.76,
        actions=["Security responders restrict autonomous agent tool-use permissions"],
        consequences=[
            "Cloud providers face margin compression and unplanned GPU provisioning losses under rapid workload churn",
            "Unverified open-weight deployments experience critical security exploit outages across enterprise tenants",
        ],
        triggers=["Coordinated cloud tenant exploit outages disrupt hosted reasoning APIs"],
        invalidations=[
            "Independent red-teaming suites close exploit vectors before broad rollout",
            "Elastic capacity hedging stabilizes hosting provider margins",
            "Tenant isolation controls prevent lateral exploit propagation",
        ],
    )

    sim_supported = {
        "simulation_id": "sim_phase63_supported_q1_and_pw_d",
        "event_title": "DeepSeek R1 Release",
        "signal_text": DEEPSEEK_SIGNAL,
        "status": "completed",
        "iterations": [
            {
                "iteration": 1,
                "micro_interpretations": [m.model_dump(mode="json") for m in (inst_sup, skep_m, patt_m, ampl_m, doom_m)],
                "meso_state": run_meso_layer([inst_sup, skep_m, patt_m, ampl_m, doom_m]).model_dump(mode="json"),
                "macro_state": {"cumulative_adoption": 0.18, "active_adoption": 0.15, "attention_indicator": 0.62, "c_churn": 0.08},
                "feedback_state": {"polarization_index": 0.34, "claim_verification_audit": {"supported": 1, "inferred": 4, "unsupported": 0, "total_interpretations": 5}, "systemic_state": {}},
            }
        ],
    }

    bundle = build_forecast_bundle(sim_supported)
    q1 = bundle["final_verdict"]["most_likely_immediate_outcome"].lower()
    assert "insufficiency disclosure" not in q1

    pw_d = next(pw for pw in bundle["causal_pathways"] if pw["pathway_id"] == "pathway_tail_cascade")
    p4_s2 = next(st for st in pw_d["steps"] if st["step_id"] == "p4_s2")
    assert p4_s2["affected_actor"] == "Cloud Infrastructure & Hosting Providers"
    invs = [st["invalidation_condition"] for st in pw_d["steps"] if st.get("invalidation_condition")]
    assert len(invs) == 4
    assert len(set(invs)) == 4, f"Expected distinct step invalidations in Pathway D, got {invs}"
