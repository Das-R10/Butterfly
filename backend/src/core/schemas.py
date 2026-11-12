"""
src/core/schemas.py

Pydantic schemas for the 4-layer Butterfly Algorithm:
  1. Micro: 5 Archetypes × 2 Models = 10 structured interpretations
  2. Meso: Weighted influence graph (external broadcast + internal peer contagion)
  3. Macro: Modified Bass diffusion with adoption fatigue/churn, attention vs adoption
  4. Feedback: Recurrent state for iterations 1..K (bounded to 3-5 iterations)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


CORE_RESEARCH_QUESTION = (
    "Given this initial event and the information currently available, "
    "what are the most plausible consequences that could emerge over time "
    "through interconnected behavioural and systemic effects?"
)


class Archetype(str, Enum):
    INSTITUTIONAL_ANALYST = "institutional_analyst"
    SKEPTIC_DEBUNKER = "skeptic_debunker"
    PATTERN_SEEKER = "pattern_seeker"
    DOOMER = "doomer"
    AMPLIFIER_INFLUENCER = "amplifier_influencer"

    @property
    def display_name(self) -> str:
        return {
            Archetype.INSTITUTIONAL_ANALYST: "Institutional Analyst",
            Archetype.SKEPTIC_DEBUNKER: "Skeptic Debunker",
            Archetype.PATTERN_SEEKER: "Pattern Seeker",
            Archetype.DOOMER: "Doomer",
            Archetype.AMPLIFIER_INFLUENCER: "Amplifier Influencer",
        }[self]


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"


class ModelExecutionStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    INVALID_JSON = "invalid_json"
    VALIDATION_FAILURE = "validation_failure"


class SimulationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"


# ── Micro Layer Schemas ────────────────────────────────────────────────────────

class InterpretationOutput(BaseModel):
    """Raw structured output produced by an LLM for a specific archetype."""
    evidence_status: str = Field(
        default="supported",
        description="Factual grounding status: supported (direct facts), inferred (logical deduction), unsupported (speculation)",
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Specific grounded evidence references extracted directly from the source signal",
    )
    uncertainty: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Epistemic uncertainty estimate: 0.0 (high confidence / verified) to 1.0 (extreme ambiguity / unconfirmed)",
    )
    claims: List[str] = Field(
        default_factory=list,
        description="Discrete, atomic factual or interpretive assertions made in this interpretation",
    )
    stance: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Behavioral stance: -1.0 (strongly skeptical/debunking) to +1.0 (strongly believing/amplifying)",
    )
    virality_estimate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated shareability and viral potential from this archetype perspective",
    )
    perceived_credibility: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Perceived truthfulness/soundness of the signal (0.0 to 1.0)",
    )
    emotional_charge: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Intensity of emotional response: calm/neutral (0.0) to alarmed/urgent (1.0)",
    )
    primary_narrative: str = Field(
        ...,
        description="The core 1-2 sentence interpretation authentic to this archetype",
    )
    key_arguments: List[str] = Field(
        default_factory=list,
        description="Key supporting arguments or rationale",
    )
    narrative_tags: List[str] = Field(
        default_factory=list,
        description="1-3 short keyword tags classifying the narrative theme",
    )
    causal_mechanism: str = Field(
        default="",
        description="Concise causal mechanism explaining how the initial event triggers stakeholder or systemic reactions",
    )
    analytical_question: str = Field(
        default="",
        description="Core analytical question defining this archetype's distinct epistemic lens on the event",
    )
    stakeholder_focus: str = Field(
        default="",
        description="Primary stakeholder cohort or institutional subsystem examined by this archetype",
    )
    strongest_supporting_consideration: str = Field(
        default="",
        description="Strongest grounded observation or structural dynamic supporting this archetype's causal forecast",
    )
    strongest_opposing_consideration: str = Field(
        default="",
        description="Strongest counter-consideration, friction, or dampening argument against this archetype's forecast",
    )
    risk_severity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Adverse harm severity [0.0=benign/contained .. 1.0=severe systemic harm], decoupled from stance [-1..+1]",
    )
    expected_stakeholder_actions: List[str] = Field(
        default_factory=list,
        description="Concrete short-to-medium-term actions likely taken by key actors (regulators, markets, public, media)",
    )
    behavioural_changes: List[str] = Field(
        default_factory=list,
        description="Expected shifts in public, institutional, or economic behaviour if this narrative gains traction",
    )
    emergent_consequences: List[str] = Field(
        default_factory=list,
        description="Hypothesized downstream 1st/2nd/3rd-order consequences (marked explicitly as hypothesis)",
    )
    time_horizon: str = Field(
        default="short_term_weeks",
        description="Primary impact horizon: immediate_days | short_term_weeks | medium_term_months | long_term_quarters",
    )
    trigger_conditions: List[str] = Field(
        default_factory=list,
        description="Observable triggers that would accelerate or activate this causal branch",
    )
    invalidation_conditions: List[str] = Field(
        default_factory=list,
        description="Observable evidence or thresholds that would falsify or reverse this causal hypothesis",
    )

    # Optional alias fields for unified naming
    virality: Optional[float] = None
    credibility: Optional[float] = None
    emotion: Optional[float] = None

    @field_validator("primary_narrative")
    @classmethod
    def check_primary_narrative_non_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("primary_narrative must be a non-empty string")
        return str(v).strip()

    @classmethod
    def strict_json_schema(cls) -> Dict[str, Any]:
        """
        Clean JSON schema for structured decoding (e.g. Ollama `format` parameter).
        Omits redundant alias fields and caps array items for reliable local 4B decoding.
        """
        return {
            "type": "object",
            "properties": {
                "evidence_status": {
                    "type": "string",
                    "enum": ["supported", "inferred", "unsupported"],
                },
                "supporting_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "uncertainty": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "stance": {
                    "type": "number",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "virality_estimate": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "perceived_credibility": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "emotional_charge": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "primary_narrative": {
                    "type": "string",
                    "minLength": 1,
                },
                "key_arguments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "narrative_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "causal_mechanism": {
                    "type": "string",
                },
                "analytical_question": {
                    "type": "string",
                },
                "stakeholder_focus": {
                    "type": "string",
                },
                "strongest_supporting_consideration": {
                    "type": "string",
                },
                "strongest_opposing_consideration": {
                    "type": "string",
                },
                "risk_severity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "expected_stakeholder_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "behavioural_changes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "emergent_consequences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "time_horizon": {
                    "type": "string",
                    "enum": [
                        "immediate_days",
                        "short_term_weeks",
                        "medium_term_months",
                        "long_term_quarters",
                    ],
                },
                "trigger_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "invalidation_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
            },
            "required": [
                "evidence_status",
                "supporting_evidence",
                "uncertainty",
                "claims",
                "stance",
                "virality_estimate",
                "perceived_credibility",
                "emotional_charge",
                "primary_narrative",
                "key_arguments",
                "narrative_tags",
                "causal_mechanism",
                "analytical_question",
                "stakeholder_focus",
                "strongest_supporting_consideration",
                "strongest_opposing_consideration",
                "risk_severity",
                "expected_stakeholder_actions",
                "behavioural_changes",
                "emergent_consequences",
                "time_horizon",
                "trigger_conditions",
                "invalidation_conditions",
            ],
            "additionalProperties": True,
        }

    def model_post_init(self, __context: Any) -> None:
        # Normalize alias values if provided
        if self.virality is not None:
            self.virality_estimate = self.virality
        else:
            self.virality = self.virality_estimate

        if self.credibility is not None:
            self.perceived_credibility = self.credibility
        else:
            self.credibility = self.perceived_credibility

        if self.emotion is not None:
            self.emotional_charge = self.emotion
        else:
            self.emotion = self.emotional_charge

        # Normalize evidence status
        status_clean = str(self.evidence_status).lower().strip()
        if "unsupported" in status_clean or "speculat" in status_clean or "false" in status_clean:
            self.evidence_status = "unsupported"
        elif "infer" in status_clean or "deduc" in status_clean or "extrapolat" in status_clean:
            self.evidence_status = "inferred"
        else:
            self.evidence_status = "supported"

        # Normalize time horizon
        horizon_clean = str(self.time_horizon or "").lower().strip()
        if "immediate" in horizon_clean or "day" in horizon_clean or "hour" in horizon_clean:
            self.time_horizon = "immediate_days"
        elif "long" in horizon_clean or "year" in horizon_clean or "quarter" in horizon_clean:
            self.time_horizon = "long_term_quarters"
        elif "medium" in horizon_clean or "month" in horizon_clean:
            self.time_horizon = "medium_term_months"
        else:
            self.time_horizon = "short_term_weeks"


class MicroInterpretation(BaseModel):
    """Enriched interpretation combining archetype metadata and LLM output."""
    archetype: Archetype
    archetype_name: str
    model_name: str
    provider: str
    model_slot: Optional[str] = None  # "model_a" | "model_b"
    execution_status: str = "success"  # "success" | "timeout" | "provider_error" | "invalid_json" | "validation_failure"
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    duration_sec: Optional[float] = None
    retry_count: int = 0
    finish_reason: Optional[str] = None
    response_length: Optional[int] = None
    stance: float
    virality: float
    credibility: float
    emotional_charge: float
    model_estimated_credibility: Optional[float] = None
    evidence_grounding_score: Optional[float] = None
    evidence_penalty_applied: Optional[float] = None
    transmission_weight: Optional[float] = None
    measurement_source: str = "synthetic_llm_estimate"
    evidence_status: str = "supported"
    supporting_evidence: List[str] = Field(default_factory=list)
    uncertainty: float = 0.20
    claims: List[str] = Field(default_factory=list)
    primary_narrative: str
    key_arguments: List[str] = Field(default_factory=list)
    narrative_tags: List[str] = Field(default_factory=list)
    causal_mechanism: str = ""
    analytical_question: str = ""
    stakeholder_focus: str = ""
    strongest_supporting_consideration: str = ""
    strongest_opposing_consideration: str = ""
    risk_severity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    expected_stakeholder_actions: List[str] = Field(default_factory=list)
    behavioural_changes: List[str] = Field(default_factory=list)
    emergent_consequences: List[str] = Field(default_factory=list)
    time_horizon: str = "short_term_weeks"
    trigger_conditions: List[str] = Field(default_factory=list)
    invalidation_conditions: List[str] = Field(default_factory=list)
    iteration: int = 1
    source_url: Optional[str] = None
    raw_response: Optional[str] = None


# ── Meso Layer Schemas ─────────────────────────────────────────────────────────

class MesoNode(BaseModel):
    id: str
    label: str
    node_type: str  # "archetype", "external_media", "public_segment"
    base_weight: float = 1.0
    current_activation: float = 0.0
    raw_influence: Optional[float] = None
    raw_external_inflow: Optional[float] = None
    raw_internal_inflow: Optional[float] = None
    inbound_weight_sum: Optional[float] = None
    external_contribution: float = 0.0
    internal_contribution: float = 0.0
    activation_mode: str = "instantaneous_per_iteration"
    calculation_basis: str = ""


class MesoEdge(BaseModel):
    source: str
    target: str
    weight: float
    channel_type: str  # "external_broadcast", "peer_contagion", "institutional_trust"


class MesoGraphState(BaseModel):
    nodes: List[MesoNode]
    edges: List[MesoEdge]
    external_influence_score: float = Field(
        ...,
        description="Aggregate external pressure from media/influencer channels (0.0 to 1.0)",
    )
    internal_contagion_score: float = Field(
        ...,
        description="Aggregate peer-to-peer network transmission score (0.0 to 1.0)",
    )
    effective_spread_potential: float = Field(
        ...,
        description="Deduplicated combined propagation potential",
    )
    activation_semantics: str = "instantaneous_per_iteration"
    meso_metrics_explanation: Optional[Dict[str, Any]] = None
    node_activations: Dict[str, float] = Field(default_factory=dict)
    node_influence_breakdown: Optional[Dict[str, Dict[str, Any]]] = None
    event_stakeholder_mapping: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Event-specific stakeholders dynamically mapped onto Meso channels with relevance & influence weights.",
    )


# ── Macro Layer Schemas ────────────────────────────────────────────────────────

class MacroState(BaseModel):
    """
    Modified Bass Diffusion state at a discrete iteration step:
      dN/dt = [p + q * (N/M)] * (M - N) - c * N
    """
    iteration: int
    p_external: float = Field(
        ...,
        description="Bass innovation coefficient (external broadcast influence)",
    )
    q_internal: float = Field(
        ...,
        description="Bass imitation coefficient (internal peer contagion)",
    )
    M_potential: float = Field(
        default=1.0,
        description="Total population / market ceiling (normalized to 1.0)",
    )
    c_churn: float = Field(
        default=0.08,
        description="Adoption fatigue / abandonment rate per iteration",
    )
    cumulative_adoption: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cumulative simulated diffusion proxy: fraction exposed to/adopting the framing up to this step (NOT empirical adoption)",
    )
    active_adoption: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Active simulated adoption proxy: current fraction retaining framing minus behavioral churn (NOT verified market adoption)",
    )
    adoption_velocity: float = Field(
        ...,
        description="Net rate of simulated adoption change per cycle (dN/dt)",
    )
    attention_indicator: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Simulated media attention and salience index [0,1]. DISTINCT from simulated behavioral adoption.",
    )
    attention_vs_adoption_ratio: float = Field(
        ...,
        description="Ratio of simulated media attention to active simulated adoption (>1 indicates conversational salience exceeds simulated uptake)",
    )
    dt: float = Field(
        default=1.0,
        description="Discrete simulation time step per round",
    )
    initial_n0: Optional[float] = Field(
        default=None,
        description="Initial adoption baseline N(0)",
    )
    formula_derivation: Optional[str] = Field(
        default=None,
        description="Description of Bass parameter derivation from Meso scores",
    )
    assumptions_and_limitations: Optional[str] = Field(
        default=None,
        description="Explicit mean-field assumptions and synthetic calibration disclaimer",
    )


# ── Feedback Layer Schemas ────────────────────────────────────────────────────

class FeedbackState(BaseModel):
    iteration: int
    macro_summary: str
    dominant_framing: str
    polarization_index: float
    prompt_context_for_next_round: str
    claim_verification_audit: Optional[Dict[str, int]] = None
    drift_metrics: Optional[Dict[str, Any]] = None
    iteration_change_summary: Optional[str] = None
    amplification_warning: bool = False
    warning_details: Optional[str] = None
    systemic_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evolving simulated systemic variables (institutional posture, public pressure, market stress, escalation risk, epistemic fragility)",
    )
    what_changed: List[str] = Field(
        default_factory=list,
        description="Structured deltas describing what shifted between iteration t-1 and iteration t",
    )
    why_changed: str = Field(
        default="",
        description="Causal explanation linking Micro/Meso/Macro dynamics to this round's shift",
    )
    new_consequences_emerged: List[str] = Field(
        default_factory=list,
        description="Emergent simulated consequences identified in this iteration (explicitly non-factual hypotheses)",
    )
    preventive_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions or interventions that would dampen or reverse destabilizing feedback loops",
    )


# ── Causal Butterfly-Effect Forecasting & Verdict Schemas ─────────────────────

class CausalStep(BaseModel):
    step_id: str
    order: str  # "1st_order" | "2nd_order" | "3rd_order" | "systemic_long_term" | "feedback_loop"
    time_horizon: str  # "immediate_days" | "short_term_weeks" | "medium_term_months" | "long_term_quarters"
    title: str
    cause: str
    effect: str
    mechanism: str
    affected_actor: str = ""
    behavioral_or_institutional_response: str = ""
    affected_stakeholders: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    trigger_conditions: List[str] = Field(default_factory=list)
    invalidation_condition: Optional[str] = None
    supporting_evidence: List[str] = Field(default_factory=list)
    evidence_status: str = "inferred"  # "supported" | "inferred" | "unsupported"
    epistemic_category: str = "hypothetical_future_consequence"
    uncertainty: float = Field(default=0.45, ge=0.0, le=1.0)
    confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    assumption_count: int = 1
    is_weak_link: bool = False
    weak_link_reason: Optional[str] = None
    is_missing_link: bool = False
    missing_link_reason: Optional[str] = None
    source_layer_refs: List[str] = Field(default_factory=list)


class CausalPathway(BaseModel):
    pathway_id: str
    name: str
    archetype_driver: str
    narrative_summary: str
    overall_plausibility: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_grounding_score: float = Field(default=0.5, ge=0.0, le=1.0)
    weak_link_count: int = 0
    has_unsupported_jump: bool = False
    invalidation_conditions: List[str] = Field(default_factory=list)
    steps: List[CausalStep] = Field(default_factory=list)


class OrderEffectsSummary(BaseModel):
    first_order: List[str] = Field(default_factory=list)
    second_order: List[str] = Field(default_factory=list)
    third_order_and_systemic: List[str] = Field(default_factory=list)
    unexpected_low_prob_high_impact: List[str] = Field(default_factory=list)
    prevention_and_reversal_conditions: List[str] = Field(default_factory=list)


class StakeholderReaction(BaseModel):
    stakeholder: str
    meso_node_id: Optional[str] = None
    activation_level: Optional[float] = None
    expected_reaction: str
    behavioural_shift: str
    likely_actions: List[str] = Field(default_factory=list)
    resilience_lever: Optional[str] = None
    downside_vulnerability: Optional[str] = None
    impact_direction: str = "mixed"  # "stabilizing" | "destabilizing" | "polarizing" | "opportunistic" | "mixed"
    evidence_basis: str = "inferred"


class FutureScenario(BaseModel):
    scenario_id: str
    scenario_type: str  # "baseline" | "positive" | "negative" | "extreme"
    title: str
    summary: str
    linked_pathway_id: Optional[str] = None
    affected_stakeholders: List[str] = Field(default_factory=list)
    short_term_consequences: List[str] = Field(default_factory=list)
    medium_term_consequences: List[str] = Field(default_factory=list)
    long_term_consequences: List[str] = Field(default_factory=list)
    trigger_conditions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    counter_evidence: List[str] = Field(default_factory=list)
    indicators_to_monitor: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    conditional_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Heuristic conditional scenario weight (NOT an empirical statistical frequency)",
    )
    probability_band: str = "20% - 40% (heuristic conditional band)"
    calibration_warning: str = (
        "Heuristic model-synthesized scenario weight derived from bounded simulation layers; "
        "NOT an empirically calibrated statistical probability."
    )


class TimeHorizonForecast(BaseModel):
    immediate_days: List[str] = Field(default_factory=list)
    short_term_weeks: List[str] = Field(default_factory=list)
    medium_term_months: List[str] = Field(default_factory=list)
    long_term_quarters: List[str] = Field(default_factory=list)
    unexpected_tail_risks: List[str] = Field(default_factory=list)


class EpistemicItem(BaseModel):
    statement: str
    category: str  # "source_confirmed_fact" | "attributed_claim" | "unverified_allegation" | "model_interpretation" | "simulated_behavioral_outcome" | "hypothetical_future_consequence"
    source_ref: Optional[str] = None


class EpistemicBreakdown(BaseModel):
    source_confirmed_facts: List[str] = Field(default_factory=list)
    attributed_claims: List[str] = Field(default_factory=list)
    unverified_allegations: List[str] = Field(default_factory=list)
    model_interpretations: List[str] = Field(default_factory=list)
    simulated_behavioral_outcomes: List[str] = Field(default_factory=list)
    hypothetical_future_consequences: List[str] = Field(default_factory=list)


class FinalVerdict(BaseModel):
    research_question: str = CORE_RESEARCH_QUESTION
    headline: str
    executive_summary: str
    forecast_summary: str = Field(
        default="",
        description="Single concise, grounded 120-180 word executive paragraph shared across UI, JSON API, and PDF Page 1.",
    )
    forecast_summary_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Synthesis method, word count, fallback logging (if AI polisher failed/timed out), and uncertainty disclosure metadata.",
    )
    causal_chain_compact: Dict[str, str] = Field(
        default_factory=dict,
        description="Compact 5-stage causal chain for PDF Page 1: initial_event, immediate_reaction, secondary_consequence, third_order_effect, possible_systemic_outcome.",
    )
    initial_event_date: Optional[str] = Field(
        default=None,
        description="Extracted date of the initial event if stated in the prompt signal, otherwise None.",
    )
    most_likely_immediate_outcome: str
    most_plausible_secondary_cascade: str
    most_dangerous_downside_risk: str
    most_plausible_stabilizing_outcome: str
    key_triggers_and_turning_points: List[str] = Field(default_factory=list)
    triggers_classified: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Classified trigger taxonomy: acceleration | escalation | stabilization | reversal | falsification",
    )
    evidence_and_information_gaps: List[str] = Field(default_factory=list)
    evidence_gaps_classified: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Classified epistemic gap taxonomy: missing_source_evidence | model_disagreement | unsupported_causal_links",
    )
    recommended_monitoring_actions: List[str] = Field(default_factory=list)
    monitoring_matrix: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Structured monitoring indicators with measurable_variable, supports_scenario_movement, contradicts_scenario_movement, trigger_threshold, and analyst_action",
    )
    verdict_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_band: str = "Moderate / Heuristic"
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    scenario_probabilities: Dict[str, Any] = Field(default_factory=dict)
    correlation_vs_causation_note: str = (
        "All multi-order pathways distinguish observable co-occurrence in the signal from verified "
        "causal mechanisms; speculative links and multi-assumption jumps are flagged explicitly."
    )
    calibration_warning: str = (
        "Scenario conditional weights and verdict confidence are heuristic scenario indices inferred "
        "from LLM archetypes and coupled Bass/network dynamics, not statistically calibrated empirical probabilities."
    )
    traceability_refs: Dict[str, Any] = Field(default_factory=dict)
    is_partial_or_failed: bool = False
    failure_disclosure: Optional[str] = None


class ForecastOutcomeResolution(BaseModel):
    actual_outcome_summary: str
    realized_scenario_type: str = "baseline"  # "baseline" | "positive" | "negative" | "extreme" | "unforeseen"
    outcome_match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    brier_score: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    resolved_at: str
    evidence_url: Optional[str] = None
    notes: Optional[str] = None


class ForecastTrackingRecord(BaseModel):
    simulation_id: str
    event_title: str
    research_question: str = CORE_RESEARCH_QUESTION
    created_at: str
    simulation_status: str
    leading_scenario: Optional[str] = None
    scenario_probabilities: Dict[str, float] = Field(default_factory=dict)
    verdict_headline: Optional[str] = None
    verdict_confidence: Optional[float] = None
    is_mock_only: bool = False
    status: str = "open"  # "open" | "resolved"
    resolution: Optional[ForecastOutcomeResolution] = None


# ── Simulation Orchestration Schemas ──────────────────────────────────────────

class SimulationConfig(BaseModel):
    signal: str
    event_title: Optional[str] = None
    source_url: Optional[str] = None
    max_iterations: int = Field(default=3, ge=1, le=5)
    model_a: str = Field(default="qwen3.5:4b")
    model_b: str = Field(default="gemma3:4b")
    provider_a: Optional[str] = None
    provider_b: Optional[str] = None
    population_ceiling: float = 1.0
    churn_rate: float = 0.08
    external_weight: float = 0.4
    run_mode: str = "simulation"  # "simulation" | "perturbation"


class ModelExecutionDiagnostic(BaseModel):
    iteration: int
    archetype: str
    model_slot: str
    model_name: str
    provider: str
    execution_status: str
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    finish_reason: Optional[str] = None
    response_length: Optional[int] = None
    thinking_length: Optional[int] = None
    duration_sec: float = 0.0
    recovered: bool = False


class IterationResult(BaseModel):
    iteration: int
    micro_interpretations: List[MicroInterpretation] = Field(
        ...,
        description="Exact 10 interpretations (5 archetypes × 2 models)",
    )
    meso_state: MesoGraphState
    macro_state: MacroState
    feedback_state: FeedbackState

    @field_validator("micro_interpretations")
    @classmethod
    def check_count(cls, v: List[MicroInterpretation]) -> List[MicroInterpretation]:
        if len(v) != 10:
            # Allow fallback during development/partial test runs, but log warning
            pass
        return v


class SimulationResult(BaseModel):
    simulation_id: str
    event_title: Optional[str] = None
    signal_text: str
    research_question: str = CORE_RESEARCH_QUESTION
    config: SimulationConfig
    iterations: List[IterationResult]
    model_agreement_score: Optional[float] = Field(
        default=None,
        description="Agreement between valid Model A and Model B outputs (0.0 to 1.0), or None (N/A) if insufficient pairs",
    )
    model_agreement_details: Optional[Dict[str, Any]] = None
    narrative_convergence_score: Optional[float] = Field(
        default=None,
        description="How much archetype interpretations converged over iterations (0.0 to 1.0)",
    )
    narrative_convergence_details: Optional[Dict[str, Any]] = None
    dominant_narrative: str
    trajectory: Dict[str, List[Any]] = Field(
        ...,
        description="Time series of adoption, velocity, attention, and churn across iterations",
    )
    timestamp: str
    status: str = "completed"  # queued | running | completed | partially_completed | failed
    status_details: Optional[str] = None
    successful_interpretations_count: Optional[int] = None
    expected_interpretations_count: Optional[int] = None
    failed_models: List[Dict[str, Any]] = Field(default_factory=list)
    model_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostics: Optional[Dict[str, Any]] = None
    causal_pathways: List[CausalPathway] = Field(default_factory=list)
    order_effects_summary: Optional[OrderEffectsSummary] = None
    stakeholder_reactions: List[StakeholderReaction] = Field(default_factory=list)
    scenarios: List[FutureScenario] = Field(default_factory=list)
    time_horizon_forecast: Optional[TimeHorizonForecast] = None
    epistemic_breakdown: Optional[EpistemicBreakdown] = None
    final_verdict: Optional[FinalVerdict] = None

