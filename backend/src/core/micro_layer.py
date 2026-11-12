"""
src/core/micro_layer.py

Layer 1: MICRO — Individual Behavioral Interpretations
Executes the five behavioral archetypes across two independent LLMs (Model A & Model B).
Total: 10 structured slots per iteration (5 archetypes × 2 models), with explicit
per-model execution status tracking and zero silent substitution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from src.core.schemas import (
    Archetype,
    CORE_RESEARCH_QUESTION,
    InterpretationOutput,
    MicroInterpretation,
    ModelExecutionDiagnostic,
    ModelExecutionStatus,
)
from src.core.llm_client import LLMClient, LLMInferenceError, sanitize_error_text
from src.core.ollama_service import unload_model_from_vram

ARCHETYPE_PROMPTS = {
    Archetype.INSTITUTIONAL_ANALYST: (
        "You are an Institutional Analyst specializing in governance, regulatory policy, enterprise compliance, and organizational incentives. "
        "ANALYTICAL QUESTION: How could formal institutions, boards, regulators, enterprise buyers, and oversight bodies modify policies, disclosures, procurement, or standards in response? "
        "STAKEHOLDER FOCUS: Derive the exact institutional stakeholders directly implied by the SOURCE MATERIAL (e.g., boards, oversight committees, regulators, enterprise customers, or ministries — never insert unrelated sectors like classrooms unless stated). "
        "SCIENTIFIC RULE: Distinguish verified event facts from projected post-trigger institutional actions (do NOT predict an already-initiated inquiry/event from SOURCE MATERIAL as a new future action), "
        "name concrete mechanisms before predicting regulatory curbs, never invent uncorroborated export bans or emergency crackdowns, and articulate both supporting drivers and organizational inertia."
    ),
    Archetype.SKEPTIC_DEBUNKER: (
        "You are a Skeptic Debunker and empirical investigator trained in historical base rates, evidentiary standards, technical/governance verification bottlenecks, and dampening dynamics. "
        "ANALYTICAL QUESTION: What missing primary evidence, factual substantiation gaps, operational constraints, or dampening frictions could slow, bound, or refute exaggerated headlines? "
        "STAKEHOLDER FOCUS: Derive the exact auditing, risk-review, and analytical skeptics implied by the SOURCE MATERIAL (e.g., independent counsel/auditors, CFOs, technical reviewers, or skeptical institutional analysts). "
        "SCIENTIFIC RULE: Challenge unsupported escalation leaps and surface concrete falsifying conditions without denying explicit facts documented in the SOURCE MATERIAL. Never invent ungrounded numerical % stock/sales forecasts."
    ),
    Archetype.PATTERN_SEEKER: (
        "You are a Pattern Seeker and cross-domain systems analyst tracking indirect 2nd- and 3rd-order spillovers across customers, competitors, partners, incentives, and structural ecosystems. "
        "ANALYTICAL QUESTION: What non-obvious 2nd-order competitive repositioning, customer hedging, partnership sensitivity, or 3rd-order structural shifts follow changes in actor incentives? "
        "STAKEHOLDER FOCUS: Identify the specific ecosystem cohorts directly connected to the SOURCE MATERIAL (e.g., enterprise/cloud customers, competing vendors, commercial partners, employees, or capital allocators — never insert unrelated open-source/education boilerplate). "
        "SCIENTIFIC RULE: Trace distinct multi-order causal transitions (1st -> 2nd -> 3rd order) with explicit intermediate causal links and matching time horizons; NEVER fabricate conspiracies or fake dates."
    ),
    Archetype.DOOMER: (
        "You are a Downside Risk & Fragility Analyst (Doomer) focused strictly on concrete adverse harm mechanisms, misuse vectors, governance/security failures, operational disruption, and systemic fragility. "
        "ANALYTICAL QUESTION: Through what specific adverse failure mechanisms (e.g., confirmed security/IP compromise, forced executive/leadership disruption, customer defection, or compliance failure) could this event cause identifiable harm? "
        "STAKEHOLDER FOCUS: Risk officers, internal security/compliance teams, exposed employees/investors, and dependent customers/counterparties directly implied by the SOURCE MATERIAL. "
        "CRITICAL FILTER: NEVER treat normal market competition, legitimate product entry, lower user costs, or exonerating 'no wrongdoing found' outcomes as adverse threats unless paired with an explicit harm mechanism. "
        "SCIENTIFIC RULE: Analyze downside risks proportionally without assuming automatic catastrophe; express your view on the SAME uniform stance axis as every other analyst "
        "(-1.0 = negligible/contained impact, +1.0 = high accelerating/disruptive impact — use `risk_severity` [0.0..1.0] separately for adverse harm severity, NEVER invert stance sign merely to mark something bad)."
    ),
    Archetype.AMPLIFIER_INFLUENCER: (
        "You are an Attention & Social Contagion Analyst (Amplifier Influencer) studying media framing, viral narrative salience, market/public sentiment divergence, and crowd herding. "
        "ANALYTICAL QUESTION: How do headline salience, social/financial media amplification, and sentiment divergence shape near-term reaction vs premature overreaction? "
        "STAKEHOLDER FOCUS: Media commentators, retail/public observers, investor feeds, and reactive stakeholders tied to the SOURCE MATERIAL. "
        "SCIENTIFIC RULE: Separate simulated viral buzz/attention from sustained institutional or fundamental impact; specify concrete narrative fatigue and debunking reversal triggers; never fabricate quotes or fake numeric market predictions."
    ),
}

INTERPRETATION_INSTRUCTION = """
RESEARCH QUESTION:
{research_question}

Produce a concise, deeply differentiated assessment strictly from your assigned archetype lens covering:
1) Your specific `analytical_question` & event-grounded `stakeholder_focus`,
2) `causal_mechanism` + discrete post-trigger stakeholder actions & behavioural shifts,
3) Temporally ordered `emergent_consequences` matching order & horizon:
   - `"1st-order: ..."` for immediate days (`immediate_days`),
   - `"2nd-order: ..."` for weeks (`short_term_weeks`),
   - `"3rd-order: ..."` for months (`medium_term_months`), without putting short-term 7-day drops inside 3rd-order,
4) Balanced `strongest_supporting_consideration`, `strongest_opposing_consideration`, triggers & `invalidation_conditions`.

SOURCE MATERIAL:
{signal}

SOURCE REFERENCE URL:
{source_url}

{feedback_section}

CALIBRATION & EPISTEMIC RULES (READ CAREFULLY):
1. UNIFORM STANCE AXIS [-1.0 to +1.0]:
   - Negative (-1.0 to -0.1): predicts dampening, containment, skepticism, reversal, or minor systemic impact.
   - Zero (0.0): balanced / ambiguous countervailing forces.
   - Positive (+0.1 to +1.0): predicts accelerating momentum, strong diffusion, structural disruption, or high escalation impact.
   (Even when analyzing downside risks or institutional friction, use THIS exact single axis consistently. Put harm intensity in `risk_severity` [0.0..1.0].)
2. FORWARD-LOOKING `evidence_status`: Rate your FORWARD CAUSAL HYPOTHESIS & PREDICTED CLAIMS — NOT whether the initial event occurred:
   - "supported": forward causal claim is directly documented inside the source material with minimal extrapolation.
   - "inferred": logical forward causal deduction extending from documented source facts (typical for forecasts; uncertainty ~0.25-0.55).
   - "unsupported": speculative multi-hop or tail-risk conjecture lacking direct corroboration (uncertainty ~0.55-0.90).
3. DISTINCT REASONING (NO ROLE COLLAPSE): Your `analytical_question`, `stakeholder_focus`, `causal_mechanism`, and `strongest_opposing_consideration` must reflect YOUR unique archetype lens, not echo peer roles. If shared source facts point multiple archetypes toward similar directional `stance`, keep authentic stance values (never invent fake divergence) while keeping reasoning content strictly distinct.
4. EVENT GROUNDING & NO UNSUPPORTED NUMBERS:
   - Stay strictly anchored to the entities, actors, and mechanism in SOURCE MATERIAL. Do NOT import unrelated domains (such as tariffs, customs, ports, shipping, or classrooms) unless explicitly bridged step-by-step from the event.
   - Do NOT repeat an already-occurred premise action from SOURCE MATERIAL (e.g., if SOURCE MATERIAL says an inquiry was triggered, predict POST-trigger findings, disclosures, governance continuity, or market consequences — not initiating the inquiry).
   - NEVER invent ungrounded numerical forecasts (such as "5-10% stock decline" or "10-15% sales drop") unless documented in SOURCE MATERIAL; state the qualitative mechanism and what empirical evidence would be required to measure magnitude. Keep every field crisp (1-2 sentences, <=3 array items) so JSON completes cleanly.

Return ONLY a strictly valid JSON object matching this exact schema:
{{
  "evidence_status": "<supported | inferred | unsupported (rate your forward causal hypothesis, NOT just whether initial event happened)>",
  "supporting_evidence": ["<specific fact or detail grounded in the source signal>"],
  "uncertainty": <float 0.0 (nearly certain forecast) to 1.0 (highly speculative)>,
  "claims": ["<discrete analytical or forward causal assertion 1>", "<assertion 2>"],
  "stance": <float -1.0 (dampening/skeptical) to +1.0 (accelerating/impactful)>,
  "virality_estimate": <float 0.0 (niche) to 1.0 (viral)>,
  "perceived_credibility": <float 0.0 (weak grounding) to 1.0 (strong grounding)>,
  "emotional_charge": <float 0.0 (analytical/calm) to 1.0 (urgent/charged)>,
  "risk_severity": <float 0.0 (benign/contained) to 1.0 (severe adverse harm)>,
  "analytical_question": "<1 crisp sentence stating this archetype's specific analytical question>",
  "stakeholder_focus": "<specific stakeholder groups/subsystems examined by this archetype>",
  "primary_narrative": "<1-2 sentence immediate interpretation & forward implication from your unique archetype angle>",
  "key_arguments": ["<key rationale or main uncertainty 1>", "<rationale 2>"],
  "narrative_tags": ["<tag1>", "<tag2>"],
  "causal_mechanism": "<concise how-and-why causal link from the initial event to stakeholder action & effect>",
  "strongest_supporting_consideration": "<strongest source-grounded observation or structural argument supporting this trajectory>",
  "strongest_opposing_consideration": "<strongest friction, counter-evidence, or dampening dynamic opposing this trajectory>",
  "expected_stakeholder_actions": ["<concrete action by your target stakeholders>"],
  "behavioural_changes": ["<observable shift in user, institutional, or market behaviour>"],
  "emergent_consequences": ["1st-order: <immediate direct consequence>", "2nd-order: <mediated downstream consequence>"],
  "time_horizon": "<immediate_days | short_term_weeks | medium_term_months | long_term_quarters>",
  "trigger_conditions": ["<observable condition that accelerates/activates this branch>"],
  "invalidation_conditions": ["<observable evidence that would falsify or reverse your causal hypothesis>"]
}}

Return ONLY the single valid JSON object. No markdown fences or prose outside the JSON.
"""

EVIDENCE_STATUS_BASE = {
    "supported": 1.0,
    "inferred": 0.72,
    "unsupported": 0.30,
}

EVIDENCE_PENALTY_MULTIPLIER = {
    "supported": 1.0,
    "inferred": 0.85,
    "unsupported": 0.40,
}


def compute_evidence_and_transmission_scores(
    *,
    stance: float,
    virality: float,
    emotional_charge: float,
    model_estimated_credibility: float,
    evidence_status: str,
    uncertainty: float,
    supporting_evidence: List[str],
) -> Tuple[float, float, float, float, float]:
    """
    Deterministically computes separated scoring layers without stance-sign bias:
      - effective_uncertainty
      - penalized_credibility (capped <= 0.40 if unsupported)
      - evidence_grounding_score [0, 1]
      - evidence_penalty_applied [0.40..1.0]
      - transmission_weight [0, 1] (symmetric wrt |stance|)
    """
    status_clean = evidence_status if evidence_status in EVIDENCE_STATUS_BASE else "inferred"
    effective_uncertainty = float(max(0.55, uncertainty) if status_clean == "unsupported" else uncertainty)
    effective_uncertainty = max(0.0, min(1.0, effective_uncertainty))

    # Evidence grounding reflects factual status, cited evidence presence, and epistemic uncertainty
    base_ev = EVIDENCE_STATUS_BASE[status_clean]
    has_evidence = any(str(e).strip() for e in (supporting_evidence or []))
    evidence_bonus = 0.08 if has_evidence else 0.0
    grounding_raw = min(1.0, base_ev + evidence_bonus) * (1.0 - 0.50 * effective_uncertainty)
    evidence_grounding_score = round(max(0.0, min(1.0, grounding_raw)), 4)

    penalty_mult = EVIDENCE_PENALTY_MULTIPLIER[status_clean]
    penalized_credibility = (
        min(model_estimated_credibility, 0.40)
        if status_clean == "unsupported"
        else model_estimated_credibility
    )
    penalized_credibility = round(max(0.0, min(1.0, penalized_credibility)), 4)

    # Symmetric conviction factor: strongly negative and strongly positive stances have equal rhetorical intensity
    conviction_factor = 0.20 + 0.80 * abs(float(stance))
    reach_intensity = 0.50 * float(virality) + 0.50 * float(emotional_charge)
    epistemic_quality = 0.50 * evidence_grounding_score + 0.50 * float(model_estimated_credibility)

    transmission_raw = conviction_factor * (0.50 * reach_intensity + 0.50 * epistemic_quality) * penalty_mult
    transmission_weight = round(max(0.0, min(1.0, transmission_raw)), 4)

    return (
        effective_uncertainty,
        penalized_credibility,
        evidence_grounding_score,
        penalty_mult,
        transmission_weight,
    )


def format_micro_prompt(
    archetype: Archetype,
    signal: str,
    source_url: str = "",
    feedback_context: str = "",
) -> tuple[str, str]:
    """Generates the system prompt and user prompt for a specific archetype."""
    system_prompt = ARCHETYPE_PROMPTS[archetype]
    feedback_section = ""
    if feedback_context:
        feedback_section = (
            f"\nPRIOR ITERATION RECURRENT AUDIT CONTEXT:\n"
            f"{feedback_context}\n"
            f"Ground your updated forecast in the SOURCE MATERIAL above. Maintain your distinct archetype question & stakeholder focus, "
            f"use the uniform [-1, +1] stance axis (never force artificial stance divergence when evidence aligns), and rate evidence_status based on proof for your forward causal claims.\n"
        )
    user_prompt = INTERPRETATION_INSTRUCTION.format(
        research_question=CORE_RESEARCH_QUESTION,
        signal=signal,
        source_url=source_url or "Not provided (Local benchmark signal)",
        feedback_section=feedback_section,
    )
    return system_prompt, user_prompt


def run_archetype_interpretation(
    client: LLMClient,
    archetype: Archetype,
    signal: str,
    source_url: str = "",
    feedback_context: str = "",
    iteration: int = 1,
    model_slot: str = "model_a",
    diagnostics_out: Optional[List[Dict[str, Any]]] = None,
) -> MicroInterpretation:
    """
    Executes a single archetype interpretation through the given LLM client.
    Captures explicit success/failure status and diagnostics without silent mock/peer substitution.
    """
    system_prompt, user_prompt = format_micro_prompt(archetype, signal, source_url, feedback_context)

    try:
        raw_data = client.generate_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=InterpretationOutput.strict_json_schema(),
            response_model=InterpretationOutput,
        )
        parsed = InterpretationOutput(**raw_data)
        meta = dict(getattr(client, "last_call_metadata", {}) or {})

        raw_cred = float(parsed.credibility if parsed.credibility is not None else parsed.perceived_credibility)
        raw_vir = float(parsed.virality if parsed.virality is not None else parsed.virality_estimate)
        raw_emo = float(parsed.emotion if parsed.emotion is not None else parsed.emotional_charge)

        (
            eff_uncertainty,
            cal_credibility,
            ev_grounding,
            ev_penalty,
            trans_weight,
        ) = compute_evidence_and_transmission_scores(
            stance=float(parsed.stance),
            virality=raw_vir,
            emotional_charge=raw_emo,
            model_estimated_credibility=raw_cred,
            evidence_status=parsed.evidence_status,
            uncertainty=float(parsed.uncertainty),
            supporting_evidence=parsed.supporting_evidence,
        )

        if diagnostics_out is not None:
            diagnostics_out.append(
                ModelExecutionDiagnostic(
                    iteration=iteration,
                    archetype=archetype.value,
                    model_slot=model_slot,
                    model_name=client.model_name,
                    provider=client.provider,
                    execution_status=ModelExecutionStatus.SUCCESS.value,
                    retry_count=int(meta.get("retry_count", 0)),
                    finish_reason=meta.get("finish_reason"),
                    response_length=meta.get("response_length"),
                    thinking_length=meta.get("thinking_length"),
                    duration_sec=float(meta.get("duration_sec", client.last_duration_sec or 0.0)),
                    recovered=bool(meta.get("recovered", False)),
                ).model_dump()
            )

        return MicroInterpretation(
            archetype=archetype,
            archetype_name=archetype.display_name,
            model_name=client.model_name,
            provider=client.provider,
            model_slot=model_slot,
            execution_status=ModelExecutionStatus.SUCCESS.value,
            duration_sec=float(meta.get("duration_sec", client.last_duration_sec or 0.0)),
            retry_count=int(meta.get("retry_count", 0)),
            finish_reason=meta.get("finish_reason"),
            response_length=meta.get("response_length"),
            stance=float(parsed.stance),
            virality=raw_vir,
            credibility=cal_credibility,
            emotional_charge=raw_emo,
            model_estimated_credibility=round(raw_cred, 4),
            evidence_grounding_score=ev_grounding,
            evidence_penalty_applied=ev_penalty,
            transmission_weight=trans_weight,
            measurement_source="synthetic_llm_estimate",
            evidence_status=parsed.evidence_status,
            supporting_evidence=parsed.supporting_evidence,
            uncertainty=round(eff_uncertainty, 4),
            claims=parsed.claims or [parsed.primary_narrative],
            primary_narrative=parsed.primary_narrative,
            key_arguments=parsed.key_arguments,
            narrative_tags=parsed.narrative_tags,
            causal_mechanism=parsed.causal_mechanism,
            analytical_question=parsed.analytical_question,
            stakeholder_focus=parsed.stakeholder_focus,
            strongest_supporting_consideration=parsed.strongest_supporting_consideration,
            strongest_opposing_consideration=parsed.strongest_opposing_consideration,
            risk_severity=parsed.risk_severity,
            expected_stakeholder_actions=parsed.expected_stakeholder_actions,
            behavioural_changes=parsed.behavioural_changes,
            emergent_consequences=parsed.emergent_consequences,
            time_horizon=parsed.time_horizon,
            trigger_conditions=parsed.trigger_conditions,
            invalidation_conditions=parsed.invalidation_conditions,
            iteration=iteration,
            source_url=source_url or None,
            raw_response=None,
        )

    except LLMInferenceError as e:
        logger.error(
            f"[Micro Layer] {model_slot} ({client.provider}/{client.model_name}) failed on "
            f"{archetype.value} [iter={iteration}]: status={e.execution_status} "
            f"category={e.error_category} retries={e.retry_count} err={e}"
        )
        if diagnostics_out is not None:
            diagnostics_out.append(
                ModelExecutionDiagnostic(
                    iteration=iteration,
                    archetype=archetype.value,
                    model_slot=model_slot,
                    model_name=client.model_name,
                    provider=client.provider,
                    execution_status=e.execution_status,
                    error_category=e.error_category,
                    error_message=str(e),
                    retry_count=e.retry_count,
                    finish_reason=e.finish_reason,
                    response_length=e.response_length,
                    thinking_length=e.thinking_length,
                    duration_sec=e.duration_sec,
                    recovered=False,
                ).model_dump()
            )

        return MicroInterpretation(
            archetype=archetype,
            archetype_name=archetype.display_name,
            model_name=client.model_name,
            provider=client.provider,
            model_slot=model_slot,
            execution_status=e.execution_status,
            error_category=e.error_category,
            error_message=str(e),
            duration_sec=e.duration_sec,
            retry_count=e.retry_count,
            finish_reason=e.finish_reason,
            response_length=e.response_length,
            stance=0.0,
            virality=0.0,
            credibility=0.0,
            emotional_charge=0.0,
            model_estimated_credibility=None,
            evidence_grounding_score=0.0,
            evidence_penalty_applied=0.0,
            transmission_weight=0.0,
            measurement_source="execution_failed",
            evidence_status="unsupported",
            supporting_evidence=[],
            uncertainty=1.0,
            claims=[],
            primary_narrative="",
            key_arguments=[],
            narrative_tags=[],
            iteration=iteration,
            source_url=source_url or None,
            raw_response=None,
        )

    except Exception as e:
        clean_err = sanitize_error_text(str(e))
        logger.error(
            f"[Micro Layer] Unexpected validation error on {model_slot} "
            f"({client.model_name}) {archetype.value}: {clean_err}"
        )
        if diagnostics_out is not None:
            diagnostics_out.append(
                ModelExecutionDiagnostic(
                    iteration=iteration,
                    archetype=archetype.value,
                    model_slot=model_slot,
                    model_name=client.model_name,
                    provider=client.provider,
                    execution_status=ModelExecutionStatus.VALIDATION_FAILURE.value,
                    error_category="schema_validation_failure",
                    error_message=clean_err,
                    duration_sec=client.last_duration_sec or 0.0,
                ).model_dump()
            )

        return MicroInterpretation(
            archetype=archetype,
            archetype_name=archetype.display_name,
            model_name=client.model_name,
            provider=client.provider,
            model_slot=model_slot,
            execution_status=ModelExecutionStatus.VALIDATION_FAILURE.value,
            error_category="schema_validation_failure",
            error_message=clean_err,
            duration_sec=client.last_duration_sec or 0.0,
            retry_count=0,
            stance=0.0,
            virality=0.0,
            credibility=0.0,
            emotional_charge=0.0,
            model_estimated_credibility=None,
            evidence_grounding_score=0.0,
            evidence_penalty_applied=0.0,
            transmission_weight=0.0,
            measurement_source="execution_failed",
            evidence_status="unsupported",
            supporting_evidence=[],
            uncertainty=1.0,
            claims=[],
            primary_narrative="",
            key_arguments=[],
            narrative_tags=[],
            iteration=iteration,
            source_url=source_url or None,
            raw_response=None,
        )


def run_micro_layer(
    signal: str,
    client_a: LLMClient,
    client_b: LLMClient,
    source_url: str = "",
    feedback_context: str = "",
    iteration: int = 1,
    diagnostics_out: Optional[List[Dict[str, Any]]] = None,
) -> List[MicroInterpretation]:
    """
    Executes the complete Micro Layer for iteration t:
      5 archetypes × 2 models = 10 interpretation slots.
    Failed calls remain explicitly tagged with error diagnostics; they are never
    replaced by the peer model or by deterministic Mock responses.
    """
    archetypes = [
        Archetype.INSTITUTIONAL_ANALYST,
        Archetype.SKEPTIC_DEBUNKER,
        Archetype.PATTERN_SEEKER,
        Archetype.DOOMER,
        Archetype.AMPLIFIER_INFLUENCER,
    ]

    interpretations: List[MicroInterpretation] = []

    # Model A passes (Sequential execution for 8GB VRAM safety)
    logger.info(f"[Micro Layer] Executing Model A ({client_a.provider} / {client_a.model_name}) across 5 archetypes...")
    for arch in archetypes:
        interp_a = run_archetype_interpretation(
            client=client_a,
            archetype=arch,
            signal=signal,
            source_url=source_url,
            feedback_context=feedback_context,
            iteration=iteration,
            model_slot="model_a",
            diagnostics_out=diagnostics_out,
        )
        interpretations.append(interp_a)

    # If both models use Ollama and are distinct, unload Model A to free up VRAM for Model B
    if (
        client_a.provider == "ollama"
        and client_b.provider == "ollama"
        and client_a.model_name.lower().strip() != client_b.model_name.lower().strip()
    ):
        logger.info(f"[Micro Layer] Sequential VRAM guard: unloading Model A '{client_a.model_name}' before loading Model B.")
        unload_model_from_vram(client_a.model_name)

    # Model B passes (Sequential execution)
    logger.info(f"[Micro Layer] Executing Model B ({client_b.provider} / {client_b.model_name}) across 5 archetypes...")
    for arch in archetypes:
        interp_b = run_archetype_interpretation(
            client=client_b,
            archetype=arch,
            signal=signal,
            source_url=source_url,
            feedback_context=feedback_context,
            iteration=iteration,
            model_slot="model_b",
            diagnostics_out=diagnostics_out,
        )
        interpretations.append(interp_b)

    success_count = sum(1 for m in interpretations if m.execution_status == ModelExecutionStatus.SUCCESS.value)
    logger.info(
        f"[Micro Layer] Completed iteration {iteration}: {success_count}/{len(interpretations)} valid interpretations."
    )
    return interpretations
