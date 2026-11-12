// frontend/src/types/simulation.ts

export type ArchetypeKey =
  | 'institutional_analyst'
  | 'skeptic_debunker'
  | 'pattern_seeker'
  | 'doomer'
  | 'amplifier_influencer';

export type EvidenceStatus = 'supported' | 'inferred' | 'unsupported';

export type TimeHorizonKey =
  | 'immediate_days'
  | 'short_term_weeks'
  | 'medium_term_months'
  | 'long_term_quarters';

export type ModelExecutionStatus =
  | 'success'
  | 'timeout'
  | 'provider_error'
  | 'invalid_json'
  | 'validation_failure';

export type SimulationStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'partially_completed'
  | 'failed';

export interface InterpretationOutput {
  stance: number; // -1.0 to 1.0
  virality_estimate: number; // 0.0 to 1.0
  perceived_credibility: number; // 0.0 to 1.0
  emotional_charge: number; // 0.0 to 1.0
  primary_narrative: string;
  key_arguments: string[];
  narrative_tags: string[];
  evidence_status?: EvidenceStatus;
  supporting_evidence?: string[];
  uncertainty?: number;
  claims?: string[];
  analytical_question?: string;
  stakeholder_focus?: string;
  strongest_supporting_consideration?: string;
  strongest_opposing_consideration?: string;
  risk_severity?: number | null;
  causal_mechanism?: string;
  expected_stakeholder_actions?: string[];
  behavioural_changes?: string[];
  emergent_consequences?: string[];
  time_horizon?: TimeHorizonKey | string;
  trigger_conditions?: string[];
  invalidation_conditions?: string[];
}

export interface MicroInterpretation {
  archetype: ArchetypeKey;
  archetype_name: string;
  model_name: string;
  provider: string;
  model_slot?: 'model_a' | 'model_b' | string | null;
  execution_status?: ModelExecutionStatus | string;
  error_category?: string | null;
  error_message?: string | null;
  duration_sec?: number | null;
  retry_count?: number;
  finish_reason?: string | null;
  response_length?: number | null;
  stance: number;
  virality: number;
  credibility: number;
  emotional_charge: number;
  model_estimated_credibility?: number | null;
  evidence_grounding_score?: number | null;
  evidence_penalty_applied?: number | null;
  transmission_weight?: number | null;
  measurement_source?: string;
  primary_narrative: string;
  key_arguments: string[];
  narrative_tags: string[];
  analytical_question?: string;
  stakeholder_focus?: string;
  strongest_supporting_consideration?: string;
  strongest_opposing_consideration?: string;
  risk_severity?: number | null;
  causal_mechanism?: string;
  expected_stakeholder_actions?: string[];
  behavioural_changes?: string[];
  emergent_consequences?: string[];
  time_horizon?: TimeHorizonKey | string;
  trigger_conditions?: string[];
  invalidation_conditions?: string[];
  iteration: number;
  raw_response?: string | null;
  evidence_status?: EvidenceStatus;
  supporting_evidence?: string[];
  uncertainty?: number;
  claims?: string[];
  source_url?: string;
}

export interface MesoNode {
  id: string;
  label: string;
  node_type: 'archetype' | 'external_media' | 'public_segment' | string;
  base_weight: number;
  current_activation: number;
  raw_influence?: number | null;
  raw_external_inflow?: number | null;
  raw_internal_inflow?: number | null;
  inbound_weight_sum?: number | null;
  external_contribution?: number;
  internal_contribution?: number;
  activation_mode?: string;
  calculation_basis?: string;
}

export interface MesoEdge {
  source: string;
  target: string;
  weight: number;
  channel_type: 'external_broadcast' | 'peer_contagion' | 'institutional_trust' | string;
}

export interface EventStakeholderMapping {
  stakeholder: string;
  meso_node_id: string;
  relevance_weight: number;
  meso_node_activation: number;
  weighted_influence_score?: number;
  causal_role?: string;
  impact_direction?: string;
}

export interface MesoGraphState {
  nodes: MesoNode[];
  edges: MesoEdge[];
  external_influence_score: number;
  internal_contagion_score: number;
  effective_spread_potential: number;
  activation_semantics?: string;
  meso_metrics_explanation?: Record<string, any> | null;
  node_activations: Record<string, number>;
  node_influence_breakdown?: Record<string, any>;
  event_stakeholder_mapping?: EventStakeholderMapping[];
}

export interface MacroState {
  iteration: number;
  p_external: number;
  q_internal: number;
  M_potential: number;
  c_churn: number;
  cumulative_adoption: number;
  active_adoption: number;
  adoption_velocity: number;
  attention_indicator: number;
  attention_vs_adoption_ratio: number;
  dt?: number;
  initial_n0?: number;
  formula_derivation?: string;
  assumptions_and_limitations?: string | null;
}

export interface SystemicStateSnapshot {
  institutional_posture?: string;
  institutional_dampening_index?: number;
  public_pressure?: number;
  economic_market_stress?: number;
  escalation_risk?: number;
  epistemic_fragility?: number;
  narrative_polarization?: number;
  measurement_status?: string;
}

export interface FeedbackState {
  iteration: number;
  macro_summary: string;
  dominant_framing: string;
  polarization_index: number;
  prompt_context_for_next_round: string;
  claim_verification_audit?: {
    supported: number;
    inferred: number;
    unsupported: number;
    total_interpretations?: number;
    failed_interpretations?: number;
  };
  drift_metrics?: Record<string, any>;
  iteration_change_summary?: string | null;
  amplification_warning?: boolean;
  warning_details?: string | null;
  systemic_state?: SystemicStateSnapshot;
  what_changed?: string[];
  why_changed?: string;
  new_consequences_emerged?: string[];
  preventive_conditions?: string[];
}

export interface CausalStep {
  step_id: string;
  order: '1st_order' | '2nd_order' | '3rd_order' | 'systemic_long_term' | 'feedback_loop' | string;
  time_horizon: TimeHorizonKey | string;
  title: string;
  cause: string;
  effect: string;
  mechanism: string;
  affected_actor?: string;
  behavioral_or_institutional_response?: string;
  affected_stakeholders: string[];
  dependencies: string[];
  trigger_conditions: string[];
  invalidation_condition?: string | null;
  supporting_evidence: string[];
  evidence_status: EvidenceStatus | string;
  epistemic_category: string;
  uncertainty: number;
  confidence: number;
  assumption_count: number;
  is_weak_link: boolean;
  weak_link_reason?: string | null;
  is_missing_link?: boolean;
  missing_link_reason?: string | null;
  source_layer_refs: string[];
}

export interface CausalPathway {
  pathway_id: string;
  name: string;
  archetype_driver: string;
  narrative_summary: string;
  overall_plausibility: number;
  evidence_grounding_score: number;
  weak_link_count: number;
  has_unsupported_jump: boolean;
  invalidation_conditions: string[];
  steps: CausalStep[];
}

export interface OrderEffectsSummary {
  first_order: string[];
  second_order: string[];
  third_order_and_systemic: string[];
  unexpected_low_prob_high_impact: string[];
  prevention_and_reversal_conditions: string[];
}

export interface StakeholderReaction {
  stakeholder: string;
  meso_node_id?: string | null;
  activation_level?: number | null;
  expected_reaction: string;
  behavioural_shift: string;
  likely_actions: string[];
  impact_direction: 'stabilizing' | 'destabilizing' | 'polarizing' | 'opportunistic' | 'mixed' | string;
  evidence_basis: string;
  resilience_lever?: string | null;
  downside_vulnerability?: string | null;
}

export interface FutureScenario {
  scenario_id: string;
  scenario_type: 'baseline' | 'positive' | 'negative' | 'extreme' | string;
  title: string;
  summary: string;
  linked_pathway_id?: string | null;
  affected_stakeholders: string[];
  short_term_consequences: string[];
  medium_term_consequences: string[];
  long_term_consequences: string[];
  trigger_conditions: string[];
  supporting_evidence: string[];
  counter_evidence: string[];
  indicators_to_monitor: string[];
  confidence?: number | null;
  uncertainty: number;
  conditional_weight?: number | null;
  probability_band: string;
  calibration_warning: string;
}

export interface TimeHorizonForecast {
  immediate_days: string[];
  short_term_weeks: string[];
  medium_term_months: string[];
  long_term_quarters: string[];
  unexpected_tail_risks: string[];
}

export interface EpistemicItem {
  statement: string;
  category: string;
  source_ref?: string | null;
}

export interface EpistemicBreakdown {
  source_confirmed_facts: string[];
  attributed_claims: string[];
  unverified_allegations: string[];
  model_interpretations: string[];
  simulated_behavioral_outcomes: string[];
  hypothetical_future_consequences: string[];
}

export interface ScenarioProbabilitySummaryItem {
  title: string;
  conditional_weight: number | null;
  probability_band: string;
}

export interface MonitoringMatrixRow {
  indicator: string;
  measurable_variable?: string;
  scenario_type: string;
  trigger_threshold: string;
  supports_scenario_movement?: string;
  contradicts_scenario_movement?: string;
  analyst_action: string;
}

export interface FinalVerdict {
  research_question: string;
  headline: string;
  executive_summary: string;
  forecast_summary?: string;
  forecast_summary_meta?: Record<string, any>;
  causal_chain_compact?: Record<string, string>;
  initial_event_date?: string | null;
  most_likely_immediate_outcome: string;
  most_plausible_secondary_cascade: string;
  most_dangerous_downside_risk: string;
  most_plausible_stabilizing_outcome: string;
  key_triggers_and_turning_points: string[];
  triggers_classified?: Record<string, string[]>;
  evidence_and_information_gaps: string[];
  evidence_gaps_classified?: Record<string, string[]>;
  recommended_monitoring_actions: string[];
  monitoring_matrix?: MonitoringMatrixRow[];
  verdict_confidence?: number | null;
  confidence_band: string;
  uncertainty: number;
  scenario_probabilities: Record<string, any>;
  correlation_vs_causation_note: string;
  calibration_warning: string;
  traceability_refs: Record<string, any>;
  is_partial_or_failed: boolean;
  failure_disclosure?: string | null;
}

export interface ForecastOutcomeResolution {
  actual_outcome_summary: string;
  realized_scenario_type: 'baseline' | 'positive' | 'negative' | 'extreme' | 'unforeseen' | string;
  outcome_match_score?: number | null;
  brier_score?: number | null;
  resolved_at: string;
  evidence_url?: string | null;
  notes?: string | null;
}

export interface ForecastTrackingRecord {
  simulation_id: string;
  event_title: string;
  research_question: string;
  created_at: string;
  simulation_status: string;
  leading_scenario?: string | null;
  scenario_probabilities: Record<string, number>;
  verdict_headline?: string | null;
  verdict_confidence?: number | null;
  is_mock_only: boolean;
  status: 'open' | 'resolved' | string;
  resolution?: ForecastOutcomeResolution | null;
}

export interface ForecastResolveRequest {
  actual_outcome_summary: string;
  realized_scenario_type: 'baseline' | 'positive' | 'negative' | 'extreme' | 'unforeseen';
  evidence_url?: string;
  notes?: string;
}

export interface SimulationForecastResponse {
  simulation_id: string;
  event_title?: string | null;
  status: SimulationStatus | string;
  research_question?: string | null;
  causal_pathways: CausalPathway[];
  order_effects_summary?: OrderEffectsSummary | null;
  stakeholder_reactions: StakeholderReaction[];
  scenarios: FutureScenario[];
  time_horizon_forecast?: TimeHorizonForecast | null;
  epistemic_breakdown?: EpistemicBreakdown | null;
  final_verdict?: FinalVerdict | null;
  tracking?: ForecastTrackingRecord | null;
}

export interface IterationResult {
  iteration: number;
  micro_interpretations: MicroInterpretation[];
  meso_state: MesoGraphState;
  macro_state: MacroState;
  feedback_state: FeedbackState;
}

export interface SimulationConfig {
  signal: string;
  event_title?: string;
  source_url?: string;
  max_iterations: number;
  model_a: string;
  model_b: string;
  provider_a?: string;
  provider_b?: string;
  population_ceiling?: number;
  churn_rate?: number;
  external_weight?: number;
  run_mode?: 'simulation' | 'perturbation';
}

export interface ModelExecutionDiagnostic {
  iteration: number;
  archetype: string;
  model_slot: string;
  model_name: string;
  provider: string;
  execution_status: ModelExecutionStatus | string;
  error_category?: string | null;
  error_message?: string | null;
  retry_count: number;
  finish_reason?: string | null;
  response_length?: number | null;
  thinking_length?: number | null;
  duration_sec: number;
  recovered?: boolean;
}

export interface SimulationResult {
  simulation_id: string;
  event_title?: string | null;
  signal_text: string;
  research_question?: string | null;
  config: SimulationConfig;
  iterations: IterationResult[];
  model_agreement_score: number | null;
  model_agreement_details?: Record<string, any> | null;
  narrative_convergence_score: number | null;
  narrative_convergence_details?: Record<string, any> | null;
  dominant_narrative: string;
  trajectory: {
    steps?: number[];
    iterations?: number[];
    cumulative_adoption: number[];
    active_adoption: number[];
    adoption_velocity: number[];
    attention_indicator: number[];
    churn_loss?: number[];
    churn_volume?: number[];
  };
  timestamp: string;
  status: SimulationStatus | string;
  status_details?: string | null;
  successful_interpretations_count?: number;
  expected_interpretations_count?: number;
  failed_models?: Array<Record<string, any>>;
  model_diagnostics?: ModelExecutionDiagnostic[];
  diagnostics?: Record<string, any> | null;
  causal_pathways?: CausalPathway[];
  order_effects_summary?: OrderEffectsSummary | null;
  stakeholder_reactions?: StakeholderReaction[];
  scenarios?: FutureScenario[];
  time_horizon_forecast?: TimeHorizonForecast | null;
  epistemic_breakdown?: EpistemicBreakdown | null;
  final_verdict?: FinalVerdict | null;
}

export interface JobStatusResponse {
  job_id: string;
  simulation_id?: string | null;
  status: 'queued' | 'running' | 'completed' | 'complete' | 'partially_completed' | 'failed';
  progress: string[];
  result?: SimulationResult;
  error?: string;
}

export interface SessionSummary {
  session_id: string;
  simulation_id?: string;
  event_title?: string;
  signal: string;
  short_description?: string;
  source_url?: string | null;
  model_a?: string;
  model_b?: string;
  provider_a?: string;
  provider_b?: string;
  iterations: number;
  max_iterations?: number;
  status?: SimulationStatus | string;
  status_details?: string | null;
  successful_interpretations_count?: number;
  expected_interpretations_count?: number;
  failed_models?: Array<Record<string, any>>;
  model_agreement_score?: number | null;
  timestamp: string;
  dominant_narrative?: string;
  verdict_headline?: string | null;
  leading_scenario?: string | null;
  verdict_confidence?: number | null;
  research_question?: string | null;
  has_report?: boolean;
  report_generated_at?: string | null;
  report_filename?: string | null;
}

export interface PDFReportResponse {
  simulation_id: string;
  event_title: string;
  report_exists: boolean;
  cached: boolean;
  filename: string;
  file_size_bytes: number;
  generated_at: string;
  download_url: string;
}

export interface PDFReportStatusResponse {
  simulation_id: string;
  report_exists: boolean;
  filename: string;
  file_size_bytes?: number;
  generated_at?: string | null;
  source_sha256?: string | null;
  stale?: boolean;
  download_url: string;
}

export type PerturbationType =
  | 'debunk_injection'
  | 'amplifier_blast'
  | 'fatigue_spike'
  | 'institutional_freeze';

export interface PerturbRequest {
  signal: string;
  max_iterations?: number;
  shock_type: PerturbationType;
  shock_iteration?: number;
  intensity?: number;
}

export interface PerturbationResult {
  shock_type: string;
  shock_iteration: number;
  intensity: number;
  baseline_simulation_id: string;
  perturbed_simulation_id: string;
  baseline_trajectory: {
    steps: number[];
    cumulative_adoption: number[];
    active_adoption: number[];
    adoption_velocity: number[];
    attention_indicator: number[];
  };
  perturbed_trajectory: {
    steps: number[];
    cumulative_adoption: number[];
    active_adoption: number[];
    adoption_velocity: number[];
    attention_indicator: number[];
  };
  adoption_divergence: number[];
  final_divergence: number;
  effect_summary: string;
}

export interface EvaluateRequest {
  simulation_id: string;
  observed_adoption: number[];
  observed_attention?: number[];
}

export interface EvaluationResult {
  evaluated_steps: number;
  adoption_mse: number;
  adoption_mae: number;
  peak_timing_error: number;
  peak_timing_error_steps?: number;
  directional_accuracy: number;
  directional_accuracy_pct?: string;
  attention_mse?: number;
  attention_mae?: number;
  error?: string;
}

// ── LLM Diagnostics & Hardware Types ─────────────────────────────────────────

export interface GpuDiagnostics {
  available: boolean;
  name: string | null;
  vram_total_mb: number | null;
  vram_used_mb: number | null;
  vram_free_mb: number | null;
  driver_version?: string | null;
  verified: boolean;
  sequential_recommended: boolean;
  details: string;
}

export interface OllamaHealth {
  installed: boolean;
  binary_path?: string | null;
  running: boolean;
  base_url: string;
  version?: string | null;
  error?: string | null;
}

export interface LLMHealthResponse {
  status: string;
  timestamp: string;
  ollama: OllamaHealth;
  gpu: GpuDiagnostics;
  cloud_providers: {
    groq: {
      configured: boolean;
      available: boolean;
    };
    gemini: {
      configured: boolean;
      available: boolean;
    };
  };
  mock: {
    available: boolean;
  };
}

export interface InstalledOllamaModel {
  name: string;
  model: string;
  size_bytes: number;
  size_human: string;
  modified_at?: string;
  digest?: string;
}

export interface RecommendedModel {
  model: string;
  display_name: string;
  role: string;
  parameter_size: string;
  approx_vram_gb: number;
  pull_command: string;
  is_installed: boolean;
}

export interface LLMModelsResponse {
  ollama_connected: boolean;
  base_url: string;
  version?: string | null;
  installed_models: InstalledOllamaModel[];
  recommended_models: RecommendedModel[];
  error?: string | null;
}

export interface LLMProviderInfo {
  id: string;
  name: string;
  category: 'local' | 'cloud' | 'mock';
  description: string;
  requires_api_key: boolean;
  api_key_configured: boolean;
  is_available: boolean;
  status_message: string;
  recommended_models: string[];
}

export interface LLMProvidersResponse {
  providers: LLMProviderInfo[];
  recommended_local_config: {
    model_a: {
      provider: string;
      model: string;
      role: string;
    };
    model_b: {
      provider: string;
      model: string;
      role: string;
    };
  };
}
