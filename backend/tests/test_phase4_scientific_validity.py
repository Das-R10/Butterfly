"""
ba/tests/test_phase4_scientific_validity.py

Comprehensive regression test suite for Phase 4:
Scientific Validity, Evidence Grounding, and Model Calibration.
Tests:
1. Micro layer evidence grounding fields & uncertainty enforcement.
2. Meso network activation consistency, bounds [0, 1], and non-zero external broadcast.
3. Traceability of external vs internal influence contributions.
4. Bass diffusion active adoption strictly <= cumulative adoption <= M ceiling.
5. Macro trajectory schema compatibility (steps, iterations, churn_volume, churn_loss).
6. Feedback loop separation of verified facts from speculative hypotheses.
7. Narrative amplification warning detection.
8. Iteration drift tracking metrics.
"""

import pytest
import numpy as np
from src.core.schemas import (
    Archetype,
    EvidenceStatus,
    MicroInterpretation,
    InterpretationOutput,
    SimulationConfig,
)
from src.core.llm_client import LLMClient
from src.core.micro_layer import (
    run_archetype_interpretation,
    run_micro_layer,
)
from src.core.meso_layer import (
    MesoInfluenceGraph,
    run_meso_layer,
)
from src.core.macro_layer import (
    compute_macro_dynamics,
)
from src.core.feedback_layer import (
    compute_feedback_state,
)
from src.core.simulation_engine import SimulationEngine


def test_micro_evidence_grounding_fields():
    """Verify that micro interpretations include structured evidence fields."""
    client_a = LLMClient(provider="mock", model_name="mock-model-a")
    signal = "NASA announced the detection of atmospheric water vapor on an exoplanet."
    source_url = "https://nasa.gov/news/exoplanet-water-vapor"

    interp = run_archetype_interpretation(
        archetype=Archetype.INSTITUTIONAL_ANALYST,
        signal=signal,
        client=client_a,
        source_url=source_url,
    )

    assert hasattr(interp, "evidence_status")
    assert interp.evidence_status in (EvidenceStatus.SUPPORTED, EvidenceStatus.INFERRED, EvidenceStatus.UNSUPPORTED)
    assert hasattr(interp, "supporting_evidence")
    assert isinstance(interp.supporting_evidence, list)
    assert hasattr(interp, "uncertainty")
    assert 0.0 <= interp.uncertainty <= 1.0
    assert hasattr(interp, "claims")
    assert isinstance(interp.claims, list)
    assert interp.source_url == source_url


def test_unsupported_claim_credibility_and_uncertainty_penalty():
    """Verify that unsupported claims have uncertainty >= 0.50 and credibility capped <= 0.40."""
    client = LLMClient(provider="mock", model_name="mock-model-a")
    
    # Run Pattern Seeker in mock mode which tests speculative hypothesis handling
    interp = run_archetype_interpretation(
        archetype=Archetype.PATTERN_SEEKER,
        signal="A minor power outage occurred in downtown Seattle.",
        client=client,
    )

    if interp.evidence_status == EvidenceStatus.UNSUPPORTED:
        assert interp.uncertainty >= 0.50, "Unsupported claim must carry high uncertainty"
        assert interp.credibility <= 0.40, "Unsupported claim credibility must be penalized"


def test_meso_network_activation_consistency_and_external_media_nonzero():
    """
    Verify that:
    1. external_media_broadcast activation is non-zero when broadcast transmission exists.
    2. All 8 nodes remain strictly within [0.0, 1.0].
    3. Neither mainstream_public nor engaged_subcultures exceeds 100%.
    """
    client = LLMClient(provider="mock", model_name="mock-model-a")
    interps = run_micro_layer(
        signal="A prominent regulatory agency launched a formal inquiry into algorithmic transparency.",
        client_a=client,
        client_b=client,
        iteration=1,
    )

    meso_state = run_meso_layer(interpretations=interps, external_weight=0.4)

    # Check external_media_broadcast is non-zero
    ext_node = next((n for n in meso_state.nodes if n.id == "external_media_broadcast"), None)
    assert ext_node is not None, "external_media_broadcast node must exist in Meso graph"
    assert ext_node.current_activation > 0.0, "External media broadcast activation must be > 0%"
    assert meso_state.node_activations["external_media_broadcast"] > 0.0

    # Check bounds [0.0, 1.0] across all nodes
    for node in meso_state.nodes:
        assert 0.0 <= node.current_activation <= 1.0, f"Node {node.id} activation {node.current_activation} out of bounds [0, 1]"

    # Verify public segments do not exceed 1.0
    pub_node = next(n for n in meso_state.nodes if n.id == "mainstream_public")
    sub_node = next(n for n in meso_state.nodes if n.id == "engaged_subcultures")
    assert pub_node.current_activation <= 1.0
    assert sub_node.current_activation <= 1.0


def test_meso_traceability_breakdown():
    """Verify that MesoGraphState exposes traceable node influence breakdowns."""
    client = LLMClient(provider="mock", model_name="mock-model-a")
    interps = run_micro_layer(
        signal="Autonomous delivery drones tested in suburban neighborhoods.",
        client_a=client,
        client_b=client,
        iteration=1,
    )
    meso_state = run_meso_layer(interpretations=interps, external_weight=0.4)

    assert meso_state.node_influence_breakdown is not None
    assert "mainstream_public" in meso_state.node_influence_breakdown
    
    pub_breakdown = meso_state.node_influence_breakdown["mainstream_public"]
    assert "external_contribution" in pub_breakdown
    assert "internal_contribution" in pub_breakdown
    assert "calculation_basis" in pub_breakdown
    assert "incoming_sources" in pub_breakdown


def test_macro_bass_dynamics_bounds():
    """
    Verify that active adoption A(t) <= cumulative adoption N(t) <= M ceiling
    across discrete simulation iterations.
    """
    client = LLMClient(provider="mock", model_name="mock-model-a")
    interps = run_micro_layer(
        signal="Breakthrough in solid state battery technology announced.",
        client_a=client,
        client_b=client,
        iteration=1,
    )
    meso_state = run_meso_layer(interpretations=interps, external_weight=0.4)

    prior_macro = None
    for round_num in range(1, 4):
        macro_state = compute_macro_dynamics(
            meso_state=meso_state,
            micro_interpretations=interps,
            iteration=round_num,
            prior_macro_state=prior_macro,
            population_ceiling=1.0,
            churn_rate=0.08,
        )

        assert 0.0 <= macro_state.active_adoption <= macro_state.cumulative_adoption <= macro_state.M_potential
        assert macro_state.dt == 1.0
        assert macro_state.formula_derivation is not None
        assert "p=" in macro_state.formula_derivation
        assert "q=" in macro_state.formula_derivation

        prior_macro = macro_state


def test_macro_trajectory_data_keys():
    """Verify that trajectory dictionary provides steps, iterations, churn_volume, and churn_loss."""
    engine = SimulationEngine(
        client_a=LLMClient(provider="mock", model_name="mock-a"),
        client_b=LLMClient(provider="mock", model_name="mock-b"),
    )
    config = SimulationConfig(
        signal="Central bank evaluates digital currency infrastructure.",
        max_iterations=3,
        model_a="mock-a",
        model_b="mock-b",
    )
    result = engine.run_simulation(config)

    assert "iterations" in result.trajectory
    assert "steps" in result.trajectory
    assert len(result.trajectory["iterations"]) == 3
    assert len(result.trajectory["steps"]) == 3
    assert "churn_loss" in result.trajectory
    assert "churn_volume" in result.trajectory
    assert result.trajectory["iterations"] == [1, 2, 3]


def test_feedback_loop_separates_facts_from_hypotheses():
    """Verify that Feedback Layer marks unverified hypotheses and provides claim audits."""
    # Create synthetic interpretations with an unverified hypothesis
    interp_unsupported = MicroInterpretation(
        archetype=Archetype.PATTERN_SEEKER,
        archetype_name="Pattern Seeker",
        model_name="mock-a",
        provider="mock",
        stance=0.8,
        virality=0.9,
        credibility=0.3,
        emotional_charge=0.8,
        primary_narrative="Shadow agency deploying secret tracking network.",
        key_arguments=["Unverified covert network"],
        narrative_tags=["conspiracy"],
        iteration=1,
        evidence_status=EvidenceStatus.UNSUPPORTED,
        uncertainty=0.75,
    )
    interp_supported = MicroInterpretation(
        archetype=Archetype.INSTITUTIONAL_ANALYST,
        archetype_name="Institutional Analyst",
        model_name="mock-b",
        provider="mock",
        stance=0.1,
        virality=0.3,
        credibility=0.9,
        emotional_charge=0.1,
        primary_narrative="Municipal agency published annual infrastructure audit.",
        key_arguments=["Official report published"],
        narrative_tags=["infrastructure"],
        iteration=1,
        evidence_status=EvidenceStatus.SUPPORTED,
        uncertainty=0.1,
    )

    interps = [interp_unsupported, interp_supported]
    meso_state = run_meso_layer(interps)
    macro_state = compute_macro_dynamics(meso_state, interps, iteration=1)

    feedback = compute_feedback_state(
        micro_interpretations=interps,
        meso_state=meso_state,
        macro_state=macro_state,
        iteration=1,
        original_signal="Municipal agency published annual infrastructure audit.",
    )

    assert feedback.claim_verification_audit is not None
    assert feedback.claim_verification_audit["unsupported"] == 1
    assert feedback.claim_verification_audit["supported"] == 1
    # Check that dominant framing is marked if an unsupported item dominates
    if "Shadow agency" in feedback.dominant_framing:
        assert "[UNVERIFIED HYPOTHESIS" in feedback.dominant_framing
    # Prompt context must include verified fact block
    assert "VERIFIED SOURCE EVIDENCE" in feedback.prompt_context_for_next_round


def test_narrative_amplification_warning():
    """Verify that amplification warning triggers when unsupported claims have high virality."""
    interp_viral_unsupported = MicroInterpretation(
        archetype=Archetype.AMPLIFIER_INFLUENCER,
        archetype_name="Amplifier Influencer",
        model_name="mock-a",
        provider="mock",
        stance=0.9,
        virality=0.95,  # High virality
        credibility=0.2,
        emotional_charge=0.9,
        primary_narrative="Global collapse triggered by covert AI test.",
        key_arguments=["Viral warning"],
        narrative_tags=["viral"],
        iteration=1,
        evidence_status=EvidenceStatus.UNSUPPORTED,
        uncertainty=0.8,
    )
    interps = [interp_viral_unsupported]
    meso_state = run_meso_layer(interps)
    macro_state = compute_macro_dynamics(meso_state, interps, iteration=1)

    feedback = compute_feedback_state(
        micro_interpretations=interps,
        meso_state=meso_state,
        macro_state=macro_state,
        iteration=1,
    )

    assert feedback.amplification_warning is True
    assert feedback.warning_details is not None
    assert "Amplification Warning" in feedback.warning_details


def test_drift_metrics_tracking():
    """Verify that feedback layer calculates iteration-over-iteration drift metrics."""
    interp_r1 = MicroInterpretation(
        archetype=Archetype.INSTITUTIONAL_ANALYST,
        archetype_name="Institutional Analyst",
        model_name="mock-a",
        provider="mock",
        stance=0.2,
        virality=0.4,
        credibility=0.8,
        emotional_charge=0.2,
        primary_narrative="Initial assessment.",
        key_arguments=["Fact"],
        narrative_tags=["tag"],
        iteration=1,
        evidence_status=EvidenceStatus.SUPPORTED,
        uncertainty=0.2,
    )
    interp_r2 = MicroInterpretation(
        archetype=Archetype.INSTITUTIONAL_ANALYST,
        archetype_name="Institutional Analyst",
        model_name="mock-a",
        provider="mock",
        stance=0.6,  # Stance drifted +0.4
        virality=0.6,
        credibility=0.7,  # Credibility drifted -0.1
        emotional_charge=0.3,
        primary_narrative="Updated assessment.",
        key_arguments=["Fact 2"],
        narrative_tags=["tag"],
        iteration=2,
        evidence_status=EvidenceStatus.SUPPORTED,
        uncertainty=0.3,  # Uncertainty drifted +0.1
    )

    meso_r1 = run_meso_layer([interp_r1])
    meso_r2 = run_meso_layer([interp_r2])
    macro_r1 = compute_macro_dynamics(meso_r1, [interp_r1], iteration=1)
    macro_r2 = compute_macro_dynamics(meso_r2, [interp_r2], iteration=2, prior_macro_state=macro_r1)

    feedback = compute_feedback_state(
        micro_interpretations=[interp_r2],
        meso_state=meso_r2,
        macro_state=macro_r2,
        iteration=2,
        prior_interpretations=[interp_r1],
        prior_meso_state=meso_r1,
        prior_macro_state=macro_r1,
    )

    assert feedback.drift_metrics is not None
    assert feedback.drift_metrics["stance_drift"] == pytest.approx(0.4, abs=0.01)
    assert feedback.drift_metrics["credibility_drift"] == pytest.approx(-0.1, abs=0.01)
    assert feedback.drift_metrics["uncertainty_drift"] == pytest.approx(0.1, abs=0.01)
    assert "adoption_drift" in feedback.drift_metrics
    assert "activation_drift" in feedback.drift_metrics
