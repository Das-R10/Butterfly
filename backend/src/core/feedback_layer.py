"""
src/core/feedback_layer.py

Layer 4: FEEDBACK — Macro Influences Subsequent Micro-Level Behavior
Extracts macro dynamics and network consensus from iteration t and packages
a concise feedback context block injected into archetype prompts for iteration t+1.
Enforces a hard limit of 3 to 5 iterations to prevent unlimited recursive execution.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Any
from loguru import logger

from src.core.schemas import (
    EvidenceStatus,
    MicroInterpretation,
    MesoGraphState,
    MacroState,
    FeedbackState,
)


_CONTAINMENT_CUES = (
    "contain", "stabiliz", "de-escalat", "exempt", "buffer", "moderate",
    "fizzle", "dampen", "postpon", "retract", "corrective", "slowdown",
    "rollback", "revers", "bounded", "absorbed", "no evidence of systemic",
)
_MOMENTUM_OR_DISRUPTION_CUES = (
    "widespread", "accelerat", "surge", "disrupt", "transform", "panic",
    "escalat", "viral", "rapid adoption", "systemic", "dislocat", "reshap",
)


def compute_institutional_dampening_index(
    *,
    valid_micro: List[MicroInterpretation],
    macro_state: MacroState,
    public_pressure: float,
    drift_metrics: Optional[Dict[str, Any]] = None,
    has_prior_macro: bool = False,
) -> float:
    """
    Computes an institutional dampening / de-escalation index in [0.0, 1.0].
    Strictly decoupled from raw factual credibility so verified disruptive events
    cannot artificially inflate de-escalation probability.
    """
    if not valid_micro:
        return 0.0

    inst_items = [
        m for m in valid_micro
        if getattr(m.archetype, "value", str(m.archetype)) in ("institutional_analyst", "skeptic_debunker")
    ]
    amp_items = [
        m for m in valid_micro
        if getattr(m.archetype, "value", str(m.archetype)) in ("amplifier_influencer", "pattern_seeker", "doomer")
    ]

    if inst_items:
        # Stance below +0.25 increasingly points toward dampening/containment; high positive stance means escalation/structural adoption
        damp_stance_score = float(
            np.mean([np.clip((0.25 - float(m.stance)) / 1.25, 0.0, 1.0) for m in inst_items])
        )
        containment_scores: List[float] = []
        for m in inst_items:
            blob = " ".join([
                m.primary_narrative or "",
                m.causal_mechanism or "",
                " ".join(m.expected_stakeholder_actions or []),
                " ".join(m.behavioural_changes or []),
                " ".join(m.key_arguments or []),
            ]).lower()
            c_hits = sum(1 for k in _CONTAINMENT_CUES if k in blob)
            a_hits = sum(1 for k in _MOMENTUM_OR_DISRUPTION_CUES if k in blob)
            if c_hits + a_hits == 0:
                containment_scores.append(0.35 if float(m.stance) <= 0.1 else 0.15)
            else:
                containment_scores.append(c_hits / (c_hits + a_hits))
        containment_direction = float(np.mean(containment_scores))
        inst_cred_mean = float(np.mean([float(m.credibility) for m in inst_items]))
    else:
        damp_stance_score = 0.25
        containment_direction = 0.25
        inst_cred_mean = 0.50

    amp_pressure = (
        float(np.mean([max(0.0, float(m.stance)) for m in amp_items]))
        if amp_items else 0.0
    )

    drift_adj = 0.0
    if has_prior_macro and drift_metrics:
        s_drift = float(drift_metrics.get("stance_drift") or 0.0)
        a_drift = float(drift_metrics.get("attention_drift") or 0.0)
        if s_drift < -0.04:
            drift_adj += 0.07
        elif s_drift > 0.04:
            drift_adj -= 0.06
        if a_drift < -0.03:
            drift_adj += 0.05
        elif a_drift > 0.04:
            drift_adj -= 0.04

    cred_quality_factor = 0.55 + 0.45 * inst_cred_mean
    raw_damp = (
        (
            0.45 * damp_stance_score
            + 0.35 * containment_direction
            + 0.20 * max(0.0, 1.0 - amp_pressure)
        )
        * cred_quality_factor
        + drift_adj
        - 0.18 * float(np.clip(public_pressure, 0.0, 1.0))
        - 0.10 * float(np.clip(macro_state.active_adoption, 0.0, 1.0))
    )
    return round(float(np.clip(raw_damp, 0.0, 1.0)), 4)


def compute_feedback_state(
    micro_interpretations: List[MicroInterpretation],
    meso_state: MesoGraphState,
    macro_state: MacroState,
    iteration: int,
    max_iterations: int = 3,
    prior_interpretations: Optional[List[MicroInterpretation]] = None,
    prior_meso_state: Optional[MesoGraphState] = None,
    prior_macro_state: Optional[MacroState] = None,
    original_signal: Optional[str] = None,
) -> FeedbackState:
    """
    Synthesizes the iteration state, audits claim verification status,
    detects drift and narrative amplification, and constructs the prompt context for round t+1.
    Strictly separates verified source facts from speculative archetype hypotheses.
    Enforces a hard bound: 3 <= max_iterations <= 5.
    """
    # Guard max iterations between 3 and 5
    safe_max_iterations = int(np.clip(max_iterations, 3, 5))

    # Filter strictly to valid interpretations (exclude failed slots from consensus & audits)
    valid_micro = [
        m for m in micro_interpretations
        if getattr(m, "execution_status", "success") == "success"
    ]
    failed_count = len(micro_interpretations) - len(valid_micro)

    # Calculate polarization index (standard deviation of archetype stances)
    stances = [m.stance for m in valid_micro]
    polarization_index = float(np.std(stances)) if stances else 0.0

    # ── Claim Verification Audit ──────────────────────────────────────────────
    supported_count = sum(
        1 for m in valid_micro
        if getattr(m, 'evidence_status', None) in ('supported', EvidenceStatus.SUPPORTED)
    )
    inferred_count = sum(
        1 for m in valid_micro
        if getattr(m, 'evidence_status', None) in ('inferred', EvidenceStatus.INFERRED)
    )
    unsupported_count = sum(
        1 for m in valid_micro
        if getattr(m, 'evidence_status', None) in ('unsupported', EvidenceStatus.UNSUPPORTED)
    )

    claim_audit = {
        "supported": supported_count,
        "inferred": inferred_count,
        "unsupported": unsupported_count,
        "total_interpretations": len(valid_micro),
        "failed_interpretations": failed_count,
    }

    # Identify dominant framing (from highest grounding-weighted virality/credibility valid interpretation)
    best_narrative_item = max(
        valid_micro,
        key=lambda m: (
            m.virality * 0.50
            + (getattr(m, "evidence_grounding_score", None) or m.credibility) * 0.50
        ),
        default=None,
    )

    # Label unverified hypotheses explicitly so they don't silently become established facts
    if best_narrative_item:
        status = getattr(best_narrative_item, 'evidence_status', None)
        if status in ('unsupported', EvidenceStatus.UNSUPPORTED):
            dominant_framing = f"[UNVERIFIED HYPOTHESIS - REQUIRES EVIDENCE] {best_narrative_item.primary_narrative}"
        elif status in ('inferred', EvidenceStatus.INFERRED):
            dominant_framing = f"[INFERRED SPECULATION] {best_narrative_item.primary_narrative}"
        else:
            dominant_framing = best_narrative_item.primary_narrative
    else:
        dominant_framing = "No valid narrative produced in this iteration (all model inferences failed)."

    # ── Narrative Drift Tracking ──────────────────────────────────────────────
    drift_metrics: Dict[str, Any] = {
        "stance_drift": 0.0,
        "credibility_drift": 0.0,
        "uncertainty_drift": 0.0,
        "activation_drift": 0.0,
        "adoption_drift": 0.0,
        "attention_drift": 0.0,
    }

    valid_prior = [
        m for m in (prior_interpretations or [])
        if getattr(m, "execution_status", "success") == "success"
    ]

    if valid_prior and valid_micro:
        prior_stances = [m.stance for m in valid_prior]
        drift_metrics["stance_drift"] = round(float(np.mean(stances) - np.mean(prior_stances)), 4)

        curr_creds = [m.credibility for m in valid_micro]
        prior_creds = [m.credibility for m in valid_prior]
        drift_metrics["credibility_drift"] = round(float(np.mean(curr_creds) - np.mean(prior_creds)), 4)

        curr_uncerts = [getattr(m, 'uncertainty', 0.3) for m in valid_micro]
        prior_uncerts = [getattr(m, 'uncertainty', 0.3) for m in valid_prior]
        drift_metrics["uncertainty_drift"] = round(float(np.mean(curr_uncerts) - np.mean(prior_uncerts)), 4)

    if prior_macro_state:
        drift_metrics["adoption_drift"] = round(
            float(macro_state.active_adoption - prior_macro_state.active_adoption), 4
        )
        drift_metrics["attention_drift"] = round(
            float(macro_state.attention_indicator - prior_macro_state.attention_indicator), 4
        )

    if prior_meso_state:
        curr_acts = list(meso_state.node_activations.values())
        prior_acts = list(prior_meso_state.node_activations.values())
        if curr_acts and prior_acts:
            drift_metrics["activation_drift"] = round(float(np.mean(curr_acts) - np.mean(prior_acts)), 4)

    if iteration == 1 or not prior_macro_state:
        iteration_change_summary = (
            f"Baseline Iteration 1: Evaluated {len(valid_micro)} valid interpretations directly from raw source signal "
            f"(active adoption {macro_state.active_adoption:.1%}, attention {macro_state.attention_indicator:.1%}, polarization {polarization_index:.2f})."
        )
    else:
        iteration_change_summary = (
            f"Iteration {iteration} vs Iteration {iteration - 1}: "
            f"Mean stance shifted {drift_metrics['stance_drift']:+.3f}, "
            f"credibility {drift_metrics['credibility_drift']:+.3f}, "
            f"uncertainty {drift_metrics['uncertainty_drift']:+.3f}, "
            f"active adoption {drift_metrics['adoption_drift']:+.1%}, "
            f"attention {drift_metrics['attention_drift']:+.1%}. "
            f"Change driven by injected Round {iteration - 1} macro consensus, claim audit, and Bass imitation/churn integration."
        )

    # ── Narrative Amplification Detection ─────────────────────────────────────
    unsupported_items = [
        m for m in valid_micro
        if getattr(m, 'evidence_status', None) in ('unsupported', EvidenceStatus.UNSUPPORTED)
    ]
    high_virality_unsupported = [m for m in unsupported_items if m.virality >= 0.60]

    amplification_warning = False
    warning_details: Optional[str] = None

    if high_virality_unsupported:
        amplification_warning = True
        archetypes_amplifying = ", ".join(
            set(m.archetype.value if hasattr(m.archetype, 'value') else str(m.archetype) for m in high_virality_unsupported)
        )
        warning_details = (
            f"Amplification Warning: {len(high_virality_unsupported)} unsupported narrative(s) "
            f"exhibit high virality (>=0.60) generated by [{archetypes_amplifying}]. "
            f"Speculative claims risk compounding into network belief without empirical evidence."
        )
    elif unsupported_count > 0 and macro_state.attention_indicator > 0.65:
        amplification_warning = True
        warning_details = (
            f"Attention Warning: Conversational attention is high ({macro_state.attention_indicator:.1%}) "
            f"while {unsupported_count} unsupported hypothesis(es) circulate in the network."
        )

    # Narrative state description
    if macro_state.attention_vs_adoption_ratio > 2.0:
        hype_label = "High Simulated Attention Divergence (Media salience exceeds simulated behavioral diffusion)"
    elif macro_state.attention_vs_adoption_ratio < 0.9:
        hype_label = "Deep Simulated Diffusion (Simulated uptake outpaces media salience)"
    else:
        hype_label = "Balanced Diffusion (Simulated attention and adoption moving proportionally)"

    # Velocity description
    vel = macro_state.adoption_velocity
    velocity_desc = "accelerating rapidly" if vel > 0.08 else "expanding steadily" if vel > 0.02 else "slowing/plateauing"

    macro_summary = (
        f"Active simulated adoption proxy reached {macro_state.active_adoption:.1%} of ceiling ({velocity_desc}, "
        f"net velocity: {vel:+.3f}/cycle, behavioral churn: {macro_state.c_churn:.1%}). "
        f"Simulated media attention is {macro_state.attention_indicator:.1%} ({hype_label}). "
        f"Polarization index is {polarization_index:.2f}. "
        f"Evidence audit: {supported_count} supported, {inferred_count} inferred, {unsupported_count} unverified."
    )

    # ── Simulated Systemic State Evolution ────────────────────────────────────
    if valid_micro:
        mean_uncert = float(np.mean([float(getattr(m, "uncertainty", 0.35)) for m in valid_micro]))
        mean_stance = float(np.mean([float(m.stance) for m in valid_micro]))

        node_acts = meso_state.node_activations or {}
        mainstream_act = float(
            node_acts.get("mainstream_public", node_acts.get("public_mainstream", macro_state.active_adoption))
        )
        public_pressure = round(
            float(np.clip(0.55 * macro_state.attention_indicator + 0.45 * mainstream_act, 0.0, 1.0)),
            4,
        )
        institutional_dampening_index = compute_institutional_dampening_index(
            valid_micro=valid_micro,
            macro_state=macro_state,
            public_pressure=public_pressure,
            drift_metrics=drift_metrics,
            has_prior_macro=prior_macro_state is not None,
        )
        market_stress_raw = (
            0.40 * macro_state.active_adoption
            + 0.35 * max(0.0, mean_stance)
            + 0.25 * float(meso_state.effective_spread_potential)
        )
        economic_market_stress = round(float(np.clip(market_stress_raw, 0.0, 1.0)), 4)

        unsupported_ratio = unsupported_count / len(valid_micro)
        escalation_raw = (
            0.35 * min(1.0, polarization_index)
            + 0.25 * max(0.0, mean_stance)
            + 0.20 * unsupported_ratio
            + 0.20 * float(macro_state.attention_indicator)
        )
        escalation_risk = round(float(np.clip(escalation_raw, 0.0, 1.0)), 4)
        epistemic_fragility = round(
            float(np.clip(0.50 * unsupported_ratio + 0.50 * mean_uncert, 0.0, 1.0)),
            4,
        )
        if institutional_dampening_index >= 0.58 and escalation_risk < 0.50:
            institutional_posture = "Stabilizing / Procedural Containment"
        elif escalation_risk >= 0.62:
            institutional_posture = "Reactive / Defensive Crisis Management"
        elif mean_stance >= 0.30 and institutional_dampening_index < 0.35:
            institutional_posture = "Adaptive Governance Under Structural Pressure"
        else:
            institutional_posture = "Cautious Monitoring / Split Guidance"
    else:
        institutional_dampening_index = 0.0
        public_pressure = 0.0
        economic_market_stress = 0.0
        escalation_risk = 0.0
        epistemic_fragility = 1.0
        institutional_posture = "Unavailable (Zero Valid Model Outputs)"

    systemic_state: Dict[str, Any] = {
        "institutional_posture": institutional_posture,
        "institutional_dampening_index": institutional_dampening_index,
        "public_pressure": public_pressure,
        "economic_market_stress": economic_market_stress,
        "escalation_risk": escalation_risk,
        "epistemic_fragility": epistemic_fragility,
        "narrative_polarization": round(polarization_index, 4),
        "measurement_status": "simulated_systemic_proxy",
    }

    what_changed: List[str] = []
    if iteration == 1 or not prior_macro_state:
        what_changed.append(
            f"Initial shock baseline: Active simulated adoption proxy {macro_state.active_adoption:.1%}, "
            f"simulated attention {macro_state.attention_indicator:.1%}, escalation risk proxy {escalation_risk:.1%}."
        )
        if valid_micro:
            what_changed.append(
                f"Institutional posture classified as '{institutional_posture}' with {supported_count} supported vs {unsupported_count} unverified forward claims."
            )
        else:
            what_changed.append("All micro interpretations failed in Round 1; systemic propagation remains zeroed/unmeasured.")
        why_changed = (
            "Round 1 reflects direct micro reactions from the raw event signal prior to recursive network feedback."
        )
    else:
        what_changed.append(
            f"Active simulated adoption proxy shifted {drift_metrics['adoption_drift']:+.1%} to {macro_state.active_adoption:.1%} (behavioral churn deducted {macro_state.c_churn:.1%})."
        )
        what_changed.append(
            f"Simulated media attention shifted {drift_metrics['attention_drift']:+.1%} to {macro_state.attention_indicator:.1%}; stance drifted {drift_metrics['stance_drift']:+.3f}."
        )
        what_changed.append(
            f"Escalation risk proxy registered {escalation_risk:.1%} alongside institutional dampening proxy {institutional_dampening_index:.1%} and epistemic fragility {epistemic_fragility:.1%}."
        )
        driver = (
            "external media/broadcast amplification"
            if meso_state.external_influence_score >= meso_state.internal_contagion_score
            else "peer-to-peer network contagion"
        )
        why_changed = (
            f"Driven by {driver} (ext={meso_state.external_influence_score:.2f}, int={meso_state.internal_contagion_score:.2f}), "
            f"Round {iteration - 1} consensus carryover, and {macro_state.c_churn:.1%} behavioral churn."
        )

    # Deduplicated simulated consequences & preventive triggers surfaced this round
    seen_conseq: set[str] = set()
    new_consequences_emerged: List[str] = []
    for m in valid_micro:
        for item in (getattr(m, "emergent_consequences", None) or []):
            clean = str(item).strip()
            if clean and clean.lower() not in seen_conseq:
                seen_conseq.add(clean.lower())
                new_consequences_emerged.append(clean)
            if len(new_consequences_emerged) >= 5:
                break
        if len(new_consequences_emerged) >= 5:
            break
    if not new_consequences_emerged and valid_micro:
        new_consequences_emerged = [
            f"Behavioral framing converged around: {dominant_framing[:120]}"
        ]

    _PREMISE_FALSIF_CUES = (
        "never occurred", "did not occur", "was fabricated", "fake event",
        "no release", "hoax", "not released", "never released",
    )
    seen_prev: set[str] = set()
    preventive_conditions: List[str] = []
    for m in valid_micro:
        for item in (getattr(m, "invalidation_conditions", None) or []):
            clean = str(item).strip()
            if not clean:
                continue
            low = clean.lower()
            if any(cue in low for cue in _PREMISE_FALSIF_CUES):
                continue
            if low not in seen_prev:
                seen_prev.add(low)
                preventive_conditions.append(clean)
            if len(preventive_conditions) >= 4:
                break
        if len(preventive_conditions) >= 4:
            break
    if not preventive_conditions and valid_micro:
        preventive_conditions = [
            "Verified empirical clarification refuting speculative second-order escalation claims",
            "Credible governance guardrails and operational capacity preventing destabilizing spillovers",
        ]

    # Context block injected into iteration t+1 archetype prompt
    if iteration < safe_max_iterations:
        fact_base = (
            f"• VERIFIED SOURCE EVIDENCE (Initial Event): \"{original_signal[:220]}...\"\n"
            if original_signal else ""
        )
        warning_text = f"• ALERT: {warning_details}\n" if amplification_warning else ""

        prompt_context = (
            f"--- MACRO FEEDBACK FROM ROUND {iteration} ---\n"
            f"{fact_base}"
            f"• PREVIOUS ROUND LEADING HYPOTHESIS (Unverified Projection): \"{dominant_framing}\"\n"
            f"• [SIMULATED PRIOR SYSTEMIC STATE — HYPOTHESIS ONLY, NOT VERIFIED FACT]:\n"
            f"  - Institutional Posture: {institutional_posture} (Dampening Proxy={institutional_dampening_index:.1%})\n"
            f"  - Public Pressure Proxy: {public_pressure:.1%} | Systemic Activity Stress Proxy: {economic_market_stress:.1%}\n"
            f"  - Escalation Risk Proxy: {escalation_risk:.1%} | Epistemic Fragility: {epistemic_fragility:.1%}\n"
            f"• Forward Hypothesis Audit: {supported_count} supported directly by signal | {inferred_count} logically inferred | {unsupported_count} speculative.\n"
            f"• Simulated Active Adoption Proxy (Bass Model): {macro_state.active_adoption:.1%} ({velocity_desc}).\n"
            f"• Simulated Media Attention Index: {macro_state.attention_indicator:.1%} — {hype_label}.\n"
            f"• Network Polarization: {polarization_index:.2f} (dispersion across archetype stances).\n"
            f"{warning_text}"
            f"GUARDRAILS FOR ROUND {iteration + 1}:\n"
            f"- Ground all analyses strictly in verified source facts. Do NOT treat prior round projections or simulated metrics as real-world events.\n"
            f"- Rate `evidence_status` STRICTLY by proof for your FORWARD causal claims (mark projections 'inferred' or 'unsupported', NEVER 'supported' merely because the initial event occurred).\n"
            f"- Keep your assigned archetype `analytical_question`, `stakeholder_focus`, and `causal_mechanism` distinct on the uniform [-1.0, +1.0] stance axis; consensus in stance sign is fine ONLY if your reasoning perspective remains authentically differentiated.\n"
            f"- Factor simulated adoption momentum ({macro_state.active_adoption:.1%}) and behavioral churn ({macro_state.c_churn:.1%}) into your updated causal forecast."
        )
    else:
        prompt_context = "SIMULATION_FINAL_CYCLE"

    feedback_state = FeedbackState(
        iteration=iteration,
        macro_summary=macro_summary,
        dominant_framing=dominant_framing,
        polarization_index=round(polarization_index, 4),
        prompt_context_for_next_round=prompt_context,
        claim_verification_audit=claim_audit,
        drift_metrics=drift_metrics,
        iteration_change_summary=iteration_change_summary,
        amplification_warning=amplification_warning,
        warning_details=warning_details,
        systemic_state=systemic_state,
        what_changed=what_changed,
        why_changed=why_changed,
        new_consequences_emerged=new_consequences_emerged,
        preventive_conditions=preventive_conditions,
    )

    logger.info(
        f"[Feedback Layer] Iteration {iteration}/{safe_max_iterations} | "
        f"Polarization: {feedback_state.polarization_index:.3f} | "
        f"Dominant Framing: '{feedback_state.dominant_framing[:60]}...' | "
        f"Claims: {claim_audit} | Amplification Warning: {amplification_warning}"
    )
    return feedback_state

