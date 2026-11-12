// frontend/src/components/forecast/FinalVerdictPanel.tsx

import React, { useState } from 'react';
import {
  EpistemicBreakdown,
  FinalVerdict,
  ForecastResolveRequest,
  ForecastTrackingRecord,
} from '../../types/simulation';
import { api } from '../../api/client';
import {
  Target,
  AlertTriangle,
  ShieldCheck,
  CheckCircle2,
  Scale,
  Layers,
  Activity,
  FileCheck2,
  Info,
  Sparkles,
} from 'lucide-react';

interface FinalVerdictPanelProps {
  simulationId: string;
  verdict: FinalVerdict;
  epistemicBreakdown?: EpistemicBreakdown | null;
  tracking?: ForecastTrackingRecord | null;
  onTrackingUpdated?: (record: ForecastTrackingRecord) => void;
  showVerdictQuestions?: boolean;
  showEpistemicAndResolution?: boolean;
  isMockOnly?: boolean;
}

export const FinalVerdictPanel: React.FC<FinalVerdictPanelProps> = ({
  simulationId,
  verdict,
  epistemicBreakdown,
  tracking,
  onTrackingUpdated,
  showVerdictQuestions = true,
  showEpistemicAndResolution = false,
  isMockOnly = false,
}) => {
  const [actualSummary, setActualSummary] = useState(
    tracking?.resolution?.actual_outcome_summary || ''
  );
  const [realizedType, setRealizedType] = useState<
    'baseline' | 'positive' | 'negative' | 'extreme' | 'unforeseen'
  >(
    (tracking?.resolution?.realized_scenario_type as
      | 'baseline'
      | 'positive'
      | 'negative'
      | 'extreme'
      | 'unforeseen') || 'baseline'
  );
  const [evidenceUrl, setEvidenceUrl] = useState(tracking?.resolution?.evidence_url || '');
  const [notes, setNotes] = useState(tracking?.resolution?.notes || '');
  const [submitting, setSubmitting] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [resolveSuccess, setResolveSuccess] = useState<string | null>(null);

  const handleResolveSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!actualSummary.trim()) {
      setResolveError('Please summarize the observed real-world outcome.');
      return;
    }
    setSubmitting(true);
    setResolveError(null);
    setResolveSuccess(null);
    try {
      const payload: ForecastResolveRequest = {
        actual_outcome_summary: actualSummary.trim(),
        realized_scenario_type: realizedType,
        evidence_url: evidenceUrl.trim() || undefined,
        notes: notes.trim() || undefined,
      };
      const updated = await api.resolveForecastOutcome(simulationId, payload);
      onTrackingUpdated?.(updated);
      setResolveSuccess(
        `Outcome logged. Match Score: ${
          updated.resolution?.outcome_match_score != null
            ? `${Math.round(updated.resolution.outcome_match_score * 100)}%`
            : 'N/A'
        } • Brier Score: ${
          updated.resolution?.brier_score != null ? updated.resolution.brier_score.toFixed(4) : 'N/A'
        }`
      );
    } catch (err: any) {
      setResolveError(err?.message || 'Failed to record forecast outcome resolution.');
    } finally {
      setSubmitting(false);
    }
  };

  const trace = verdict.traceability_refs || {};
  const summaryMeta = verdict.forecast_summary_meta || {};
  const summaryParagraph =
    (verdict.forecast_summary && verdict.forecast_summary.trim()) ||
    verdict.executive_summary ||
    '';

  const leadingTypeRaw =
    summaryMeta.leading_scenario ||
    (['baseline', 'positive', 'negative', 'extreme']
      .map((k) => ({
        k,
        w: verdict.scenario_probabilities?.[k]?.conditional_weight ?? -1,
      }))
      .sort((a, b) => Number(b.w) - Number(a.w))[0]?.k ??
      'baseline');
  const leadingLabel =
    String(leadingTypeRaw).charAt(0).toUpperCase() + String(leadingTypeRaw).slice(1);
  const leadingWeightRaw =
    summaryMeta.leading_weight ??
    verdict.scenario_probabilities?.[leadingTypeRaw]?.conditional_weight ??
    null;
  const leadingWeightLabel =
    leadingWeightRaw != null
      ? `${(Number(leadingWeightRaw) * 100).toFixed(1)}% Heuristic Weight`
      : 'Weights Unavailable';
  const synthesisMethodLabel = String(summaryMeta.method || '').startsWith('ollama_')
    ? 'AI-Polished Grounded Summary'
    : 'Grounded Extractive Summary';

  const offDomainFiltered =
    trace.relevance_audit?.rejected_off_domain_count ??
    trace.relevance_audit?.rejected_off_domain ??
    0;
  const withheldNumericCount =
    trace.relevance_audit?.withheld_numeric_count ??
    trace.relevance_audit?.withheld_numeric ??
    0;

  return (
    <div className="space-y-6">
      {showVerdictQuestions && (
        <>
          {/* Phase 6.4: Top-of-Section Single-Paragraph Executive Forecast Summary */}
          {summaryParagraph && (
            <section
              aria-label="Butterfly-Effect Forecast Summary"
              className="rounded-2xl bg-gradient-to-br from-cyan-950/35 via-slate-900 to-slate-950 border border-cyan-500/35 p-6 space-y-3.5 shadow-xl"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                  <h2 className="text-sm md:text-base font-bold text-slate-100 tracking-tight">
                    Butterfly-Effect Forecast Summary
                  </h2>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-[10px] font-mono">
                  <span className="px-2.5 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-700/60 text-cyan-300 font-semibold">
                    {leadingLabel} &bull; {leadingWeightLabel}
                  </span>
                  {verdict.initial_event_date && (
                    <span className="px-2 py-0.5 rounded-full bg-slate-900 border border-slate-700 text-slate-300">
                      Event Date: {verdict.initial_event_date}
                    </span>
                  )}
                </div>
              </div>

              <p
                data-testid="forecast-summary-paragraph"
                className="text-sm text-slate-100 leading-relaxed font-normal"
              >
                {summaryParagraph}
              </p>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-2.5 border-t border-slate-800/80 text-[11px] text-slate-400">
                <span className="truncate max-w-2xl">
                  <span className="text-amber-300 font-medium">Key Uncertainty:</span>{' '}
                  {summaryMeta.uncertainty_disclosure ||
                    verdict.evidence_and_information_gaps?.[0] ||
                    'Primary source verification pending.'}
                </span>
                <span className="font-mono text-[10px] text-slate-400">
                  {synthesisMethodLabel}
                  {summaryMeta.fallback_used && summaryMeta.fallback_reason
                    ? ` (fallback: ${summaryMeta.fallback_reason})`
                    : ''}
                  {' • '}Heuristic indices (non-empirical)
                </span>
              </div>
            </section>
          )}

          {/* Core Research Question & Executive Headline Banner */}
          <div className="rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-slate-800 p-6 space-y-4 shadow-xl">
            <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
              <div className="space-y-2 max-w-3xl">
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/60 text-cyan-300 text-[10px] font-mono uppercase">
                  <Target className="w-3.5 h-3.5" />
                  <span>Core Research Question</span>
                </div>
                <p className="text-xs text-slate-300 italic leading-relaxed">
                  &ldquo;{verdict.research_question}&rdquo;
                </p>
                <h2 className="text-xl font-extrabold text-slate-100 leading-snug pt-1">
                  {verdict.headline}
                </h2>
                <p className="text-xs text-slate-300 leading-relaxed">{verdict.executive_summary}</p>
              </div>

              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 min-w-[210px] space-y-1.5 font-mono">
                <div className="text-[10px] uppercase text-slate-400">
                  Verdict Confidence Index
                </div>
                <div className="text-2xl font-bold text-cyan-400">
                  {verdict.verdict_confidence != null
                    ? `${Math.round(verdict.verdict_confidence * 100)}%`
                    : 'N/A'}
                </div>
                <div className="text-[11px] text-slate-300">{verdict.confidence_band}</div>
                <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800">
                  Uncertainty: {(verdict.uncertainty * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            {verdict.is_partial_or_failed && verdict.failure_disclosure && (
              <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/70 text-xs text-rose-200 flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-rose-300">Execution Completeness Disclosure:</span>{' '}
                  {verdict.failure_disclosure}
                </div>
              </div>
            )}

            {/* Layer Traceability Pills */}
            <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-300">
              <span className="text-slate-400 uppercase mr-1 flex items-center gap-1">
                <Layers className="w-3 h-3 text-cyan-400" /> Traceability:
              </span>
              {trace.valid_slots != null && (
                <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800">
                  Slots {trace.valid_slots}/{trace.expected_slots ?? '?'}
                </span>
              )}
              {trace.model_agreement_score !== undefined && (
                <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800">
                  Agreement{' '}
                  {trace.model_agreement_score != null
                    ? `${Math.round(trace.model_agreement_score * 100)}%`
                    : 'N/A'}
                </span>
              )}
              {trace.final_active_adoption != null && (
                <span
                  title="Modified Bass active simulated adoption proxy (NOT verified empirical adoption)"
                  className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800"
                >
                  Active Sim. Adoption {(Number(trace.final_active_adoption) * 100).toFixed(1)}%
                </span>
              )}
              {trace.final_attention_indicator != null && (
                <span
                  title="Simulated conversational/media attention index (NOT empirical adoption)"
                  className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800"
                >
                  Sim. Attention {(Number(trace.final_attention_indicator) * 100).toFixed(1)}%
                </span>
              )}
              {trace.reasoning_differentiation_score != null && (
                <span className="px-2 py-0.5 rounded bg-indigo-950/50 border border-indigo-800/60 text-indigo-300">
                  Reasoning Diff {(Number(trace.reasoning_differentiation_score) * 100).toFixed(0)}%
                </span>
              )}
              {trace.legitimate_consensus_detected && (
                <span className="px-2 py-0.5 rounded bg-emerald-950/50 border border-emerald-800/60 text-emerald-300">
                  Legitimate Consensus (Distinct Reasoning)
                </span>
              )}
              {trace.archetype_collapse_warning && (
                <span className="px-2 py-0.5 rounded bg-rose-950/60 border border-rose-800/70 text-rose-300">
                  Archetype Collapse Warning
                </span>
              )}
              {trace.weak_links_detected != null && (
                <span className="px-2 py-0.5 rounded bg-amber-950/60 border border-amber-800/60 text-amber-300">
                  {trace.weak_links_detected} Weak Causal Link(s)
                </span>
              )}
              {trace.scenario_diversity_score != null && (
                <span className="px-2 py-0.5 rounded bg-cyan-950/50 border border-cyan-800/60 text-cyan-300">
                  Scenario Diversity {(Number(trace.scenario_diversity_score) * 100).toFixed(0)}%
                </span>
              )}
              {trace.event_domain && (
                <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-700 text-slate-200">
                  Domain: {String(trace.event_domain)}
                </span>
              )}
              {trace.relevance_audit &&
                (offDomainFiltered > 0 || withheldNumericCount > 0) && (
                  <span
                    title="Groundedness filter telemetry: rejected off-domain claims and reframed ungrounded numerical projections"
                    className="px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-800/60 text-emerald-300"
                  >
                    Audit Guard: {offDomainFiltered} off-domain / {withheldNumericCount}{' '}
                    ungrounded numbers reframed
                  </span>
                )}
            </div>
          </div>

          {/* 8 Core Final Verdict Questions Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
              <div className="text-[10px] font-mono uppercase text-cyan-400 font-bold">
                Q1 • Most Likely Immediate Outcome (Days)
              </div>
              <p className="text-slate-200 leading-relaxed">
                {verdict.most_likely_immediate_outcome}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
              <div className="text-[10px] font-mono uppercase text-violet-400 font-bold">
                Q2 • Most Plausible Secondary Cascade (Weeks–Months)
              </div>
              <p className="text-slate-200 leading-relaxed">
                {verdict.most_plausible_secondary_cascade}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/50 space-y-1.5">
              <div className="text-[10px] font-mono uppercase text-rose-400 font-bold">
                Q3 • Most Dangerous Downside Risk
              </div>
              <p className="text-slate-200 leading-relaxed">
                {verdict.most_dangerous_downside_risk}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-800/50 space-y-1.5">
              <div className="text-[10px] font-mono uppercase text-emerald-400 font-bold">
                Q4 • Most Plausible Stabilizing / Positive Outcome
              </div>
              <p className="text-slate-200 leading-relaxed">
                {verdict.most_plausible_stabilizing_outcome}
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="text-[10px] font-mono uppercase text-amber-400 font-bold">
                Q5 • Key Triggers &amp; Turning Points (5-Way Classified)
              </div>
              {verdict.triggers_classified &&
              Object.values(verdict.triggers_classified).some((arr) => Array.isArray(arr) && arr.length > 0) ? (
                <div className="space-y-1.5 text-[11px]">
                  {(
                    [
                      ['acceleration', 'Acceleration', 'text-cyan-300'],
                      ['escalation', 'Escalation', 'text-rose-300'],
                      ['stabilization', 'Stabilization', 'text-emerald-300'],
                      ['reversal', 'Reversal', 'text-amber-300'],
                      ['falsification', 'Falsification', 'text-violet-300'],
                    ] as const
                  ).map(([key, label, cls]) => {
                    const items = verdict.triggers_classified?.[key] || [];
                    if (items.length === 0) return null;
                    return (
                      <div key={key}>
                        <span className={`font-mono uppercase text-[10px] font-bold ${cls}`}>
                          {label}:
                        </span>
                        <ul className="list-disc list-inside text-slate-300 space-y-0.5 ml-1">
                          {items.map((item, idx) => (
                            <li key={idx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                  {verdict.key_triggers_and_turning_points.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              )}
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="text-[10px] font-mono uppercase text-rose-300 font-bold flex items-center justify-between gap-2">
                <span>Q6 • Evidence &amp; Information Gaps</span>
                <span className="text-[9px] text-amber-300/90 normal-case font-normal">
                  Hypotheses Separated From Evidence
                </span>
              </div>
              {verdict.evidence_gaps_classified &&
              Object.values(verdict.evidence_gaps_classified).some((arr) => Array.isArray(arr) && arr.length > 0) ? (
                <div className="space-y-1.5 text-[11px]">
                  {(
                    [
                      ['missing_source_evidence', 'Missing Source Evidence', 'text-amber-300'],
                      ['model_disagreement', 'Model Disagreement', 'text-cyan-300'],
                      ['unsupported_causal_links', 'Unsupported Causal Links', 'text-rose-300'],
                    ] as const
                  ).map(([key, label, cls]) => {
                    const items = verdict.evidence_gaps_classified?.[key] || [];
                    if (items.length === 0) return null;
                    return (
                      <div key={key}>
                        <span className={`font-mono uppercase text-[10px] font-bold ${cls}`}>
                          {label}:
                        </span>
                        <ul className="list-disc list-inside text-slate-300 space-y-0.5 ml-1">
                          {items.map((item, idx) => (
                            <li key={idx}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                  {verdict.evidence_and_information_gaps.length > 0 ? (
                    verdict.evidence_and_information_gaps.map((item, idx) => <li key={idx}>{item}</li>)
                  ) : (
                    <li className="text-slate-500">No major structural evidence gaps recorded</li>
                  )}
                </ul>
              )}
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
              <div className="text-[10px] font-mono uppercase text-cyan-300 font-bold">
                Q7 • Recommended Indicators &amp; Actions to Monitor Next
              </div>
              <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                {verdict.recommended_monitoring_actions.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="text-[10px] font-mono uppercase text-indigo-300 font-bold flex items-center justify-between gap-1.5">
                <span className="flex items-center gap-1.5">
                  <Scale className="w-3.5 h-3.5" />
                  <span>Q8 • Heuristic Scenario Weights</span>
                </span>
                <span className="text-[9px] text-amber-300 border border-amber-800/60 bg-amber-950/40 px-1.5 py-0.5 rounded normal-case">
                  {verdict.scenario_probabilities?.is_empirical_calibration === true
                    ? 'Empirically Calibrated'
                    : 'Non-Empirical Heuristic (Sum = 100%)'}
                </span>
              </div>
              <div className="space-y-1 text-[11px] font-mono text-slate-300">
                {(['baseline', 'positive', 'negative', 'extreme'] as const).map((k) => {
                  const row = verdict.scenario_probabilities?.[k];
                  const wt = row && typeof row === 'object' ? row.conditional_weight : null;
                  return (
                    <div key={k} className="flex items-center justify-between">
                      <span className="capitalize text-slate-400">{k}:</span>
                      <span className="text-slate-100">
                        {wt != null ? `${(Number(wt) * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="text-[10px] text-amber-300/90 pt-1 border-t border-slate-800">
                {verdict.calibration_warning}
              </p>
            </div>
          </div>

          {/* Full Q7 Monitoring Matrix */}
          {((verdict.monitoring_matrix && verdict.monitoring_matrix.length > 0) ||
            (Array.isArray(trace.monitoring_matrix) && trace.monitoring_matrix.length > 0)) && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  <span>Q7 • Scenario-Linked Monitoring Matrix (Supports vs Contradicts)</span>
                </h3>
                <span className="text-[10px] font-mono text-slate-400">
                  Strict Scenario-Specific Falsification &amp; Escalation Thresholds
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px]">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-mono text-[10px] uppercase">
                      <th className="py-2 px-3">Measurable Variable / Indicator</th>
                      <th className="py-2 px-3">Scenario</th>
                      <th className="py-2 px-3 text-emerald-300">Supports Movement</th>
                      <th className="py-2 px-3 text-rose-300">Contradicts Movement</th>
                      <th className="py-2 px-3">Threshold &amp; Analyst Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {(verdict.monitoring_matrix && verdict.monitoring_matrix.length > 0
                      ? verdict.monitoring_matrix
                      : (trace.monitoring_matrix as any[])
                    ).map((row, idx) => (
                      <tr key={idx} className="align-top hover:bg-slate-950/50">
                        <td className="py-2.5 px-3 font-medium text-slate-200 max-w-xs">
                          {row.measurable_variable || row.indicator}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[10px] uppercase">
                          <span
                            className={`px-2 py-0.5 rounded border ${
                              row.scenario_type === 'extreme'
                                ? 'bg-rose-950/60 text-rose-300 border-rose-800/70'
                                : row.scenario_type === 'negative'
                                ? 'bg-amber-950/60 text-amber-300 border-amber-800/70'
                                : row.scenario_type === 'positive'
                                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/70'
                                : 'bg-cyan-950/60 text-cyan-300 border-cyan-800/70'
                            }`}
                          >
                            {row.scenario_type}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-emerald-200/90 max-w-xs">
                          {row.supports_scenario_movement || row.trigger_threshold}
                        </td>
                        <td className="py-2.5 px-3 text-rose-200/90 max-w-xs">
                          {row.contradicts_scenario_movement || 'Sustained null evidence / reversal'}
                        </td>
                        <td className="py-2.5 px-3 text-slate-300 max-w-sm space-y-1">
                          <div>
                            <strong className="text-slate-400">Trigger:</strong>{' '}
                            {row.trigger_threshold}
                          </div>
                          <div>
                            <strong className="text-cyan-300">Action:</strong>{' '}
                            {row.analyst_action}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {showEpistemicAndResolution && (
        <div className="space-y-6">
          {/* Correlation vs Causation + Calibration Guardrail */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1.5">
              <div className="text-[10px] font-mono uppercase text-cyan-400 font-bold flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5" />
                <span>Correlation vs. Causation Boundary</span>
              </div>
              <p className="text-[11px] text-slate-300 leading-relaxed">
                {verdict.correlation_vs_causation_note}
              </p>
            </div>
            <div className="p-4 rounded-xl bg-amber-950/25 border border-amber-800/60 space-y-1.5">
              <div className="text-[10px] font-mono uppercase text-amber-300 font-bold flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Probability &amp; Calibration Warning</span>
              </div>
              <p className="text-[11px] text-amber-200/90 leading-relaxed">
                {verdict.calibration_warning}
              </p>
            </div>
          </div>

          {/* 6-Tier Epistemic Separation Inspector */}
          {epistemicBreakdown && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span>6-Tier Epistemic Separation Audit</span>
                </h3>
                <span className="text-[10px] font-mono text-slate-400">
                  Strict Boundary Between Confirmed Evidence &amp; Simulated Hypotheses
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5 text-xs">
                {[
                  {
                    title: '1. Source-Confirmed Facts',
                    items: epistemicBreakdown.source_confirmed_facts,
                    cls: 'border-emerald-800/60 bg-emerald-950/15 text-emerald-300',
                  },
                  {
                    title: '2. Attributed Claims',
                    items: epistemicBreakdown.attributed_claims,
                    cls: 'border-cyan-800/60 bg-cyan-950/15 text-cyan-300',
                  },
                  {
                    title: '3. Unverified Allegations',
                    items: epistemicBreakdown.unverified_allegations,
                    cls: 'border-rose-800/60 bg-rose-950/15 text-rose-300',
                  },
                  {
                    title: '4. Model Interpretations',
                    items: epistemicBreakdown.model_interpretations,
                    cls: 'border-indigo-800/60 bg-indigo-950/15 text-indigo-300',
                  },
                  {
                    title: '5. Simulated Behavioral Outcomes',
                    items: epistemicBreakdown.simulated_behavioral_outcomes,
                    cls: 'border-violet-800/60 bg-violet-950/15 text-violet-300',
                  },
                  {
                    title: '6. Hypothetical Future Consequences',
                    items: epistemicBreakdown.hypothetical_future_consequences,
                    cls: 'border-amber-800/60 bg-amber-950/15 text-amber-300',
                  },
                ].map((tier) => (
                  <div
                    key={tier.title}
                    className={`p-3.5 rounded-xl border ${tier.cls} space-y-1.5`}
                  >
                    <div className="text-[10px] font-mono uppercase font-bold">{tier.title}</div>
                    <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-200">
                      {tier.items.length > 0 ? (
                        tier.items.map((stmt, i) => <li key={i}>{stmt}</li>)
                      ) : (
                        <li className="text-slate-500">None recorded</li>
                      )}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Post-Event Outcome Resolution & Forecast Accuracy Tracking */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <FileCheck2 className="w-4 h-4 text-cyan-400" />
                <span>Post-Event Forecast Tracking &amp; Realized Outcome Resolution</span>
              </h3>
              {tracking && (
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border uppercase ${
                    tracking.status === 'resolved'
                      ? 'bg-emerald-950/70 text-emerald-300 border-emerald-700'
                      : 'bg-slate-950 text-slate-300 border-slate-700'
                  }`}
                >
                  {tracking.status === 'resolved' ? 'Resolved & Scored' : 'Open Forecast'}
                </span>
              )}
            </div>

            {tracking?.resolution && (
              <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-800/60 space-y-2 text-xs">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-bold text-emerald-300 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" />
                    Realized Trajectory: {tracking.resolution.realized_scenario_type.toUpperCase()}
                  </span>
                  <div className="flex items-center gap-3 font-mono text-[11px]">
                    <span>
                      Match:{' '}
                      <strong className="text-emerald-300">
                        {tracking.resolution.outcome_match_score != null
                          ? `${Math.round(tracking.resolution.outcome_match_score * 100)}%`
                          : 'N/A'}
                      </strong>
                    </span>
                    <span>
                      Brier:{' '}
                      <strong className="text-cyan-300">
                        {tracking.resolution.brier_score != null
                          ? tracking.resolution.brier_score.toFixed(4)
                          : 'N/A'}
                      </strong>
                    </span>
                  </div>
                </div>
                <p className="text-[11px] text-slate-200">
                  {tracking.resolution.actual_outcome_summary}
                </p>
                {tracking.resolution.evidence_url && (
                  <div className="text-[10px] font-mono text-cyan-400 truncate">
                    Evidence: {tracking.resolution.evidence_url}
                  </div>
                )}
              </div>
            )}

            {isMockOnly && (
              <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 text-[11px] text-amber-200">
                Note: This run used Deterministic Mock providers; persistence to production{' '}
                <code>data/forecasts/</code> is restricted to real persisted sessions.
              </div>
            )}

            {resolveError && (
              <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-200">
                {resolveError}
              </div>
            )}
            {resolveSuccess && (
              <div className="p-3 rounded-lg bg-emerald-950/40 border border-emerald-800/60 text-xs text-emerald-200">
                {resolveSuccess}
              </div>
            )}

            <form onSubmit={handleResolveSubmit} className="space-y-3 text-xs">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Realized Scenario Class
                  </label>
                  <select
                    value={realizedType}
                    onChange={(e) =>
                      setRealizedType(
                        e.target.value as
                          | 'baseline'
                          | 'positive'
                          | 'negative'
                          | 'extreme'
                          | 'unforeseen'
                      )
                    }
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 px-3 py-2 outline-none focus:border-cyan-500"
                  >
                    <option value="baseline">Baseline (Contained / Expected Path)</option>
                    <option value="positive">Positive (Stabilization / De-escalation)</option>
                    <option value="negative">Negative (Disruption / Polarization)</option>
                    <option value="extreme">Extreme (Tail Escalation)</option>
                    <option value="unforeseen">Unforeseen (Surprise Beyond Modelled Space)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1">
                    Reference Evidence URL (Optional)
                  </label>
                  <input
                    type="url"
                    value={evidenceUrl}
                    onChange={(e) => setEvidenceUrl(e.target.value)}
                    placeholder="https://..."
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 px-3 py-2 outline-none focus:border-cyan-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-[11px] font-medium text-slate-400 mb-1">
                  Actual Observed Outcome Summary
                </label>
                <textarea
                  rows={2}
                  value={actualSummary}
                  onChange={(e) => setActualSummary(e.target.value)}
                  placeholder="Describe what actually unfolded over subsequent days/weeks..."
                  className="w-full rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 p-2.5 outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="block text-[11px] font-medium text-slate-400 mb-1">
                  Evaluation Notes (Optional)
                </label>
                <input
                  type="text"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Where assumptions held or missed..."
                  className="w-full rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 px-3 py-2 outline-none focus:border-cyan-500"
                />
              </div>

              <div className="flex items-center justify-end">
                <button
                  type="submit"
                  disabled={submitting || isMockOnly}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 font-bold text-xs transition"
                >
                  <Activity className="w-3.5 h-3.5" />
                  <span>{submitting ? 'Scoring...' : 'Save Observed Outcome & Compute Brier'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
