// frontend/src/components/forecast/CausalChainDiagram.tsx

import React, { useState } from 'react';
import { CausalPathway, CausalStep } from '../../types/simulation';
import {
  GitBranch,
  AlertTriangle,
  ArrowRight,
  Clock,
  Layers,
  ShieldAlert,
  CheckCircle2,
  HelpCircle,
  Users,
  Zap,
} from 'lucide-react';

interface CausalChainDiagramProps {
  pathways: CausalPathway[];
  eventTitle?: string | null;
  signalText?: string | null;
}

const ORDER_LABELS: Record<string, string> = {
  '1st_order': '1st-Order Immediate',
  '2nd_order': '2nd-Order Spillover',
  '3rd_order': '3rd-Order Cross-Sector',
  systemic_long_term: 'Systemic Long-Term',
  feedback_loop: 'Feedback Loop',
};

const HORIZON_LABELS: Record<string, string> = {
  immediate_days: 'Immediate (Days)',
  short_term_weeks: 'Short-Term (Weeks)',
  medium_term_months: 'Medium-Term (Months)',
  long_term_quarters: 'Long-Term (Quarters)',
};

export const CausalChainDiagram: React.FC<CausalChainDiagramProps> = ({
  pathways,
  eventTitle,
  signalText,
}) => {
  const [selectedPathwayId, setSelectedPathwayId] = useState<string>(
    pathways[0]?.pathway_id || ''
  );
  const [selectedStepId, setSelectedStepId] = useState<string>(
    pathways[0]?.steps?.[0]?.step_id || ''
  );

  if (!pathways || pathways.length === 0) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center space-y-2">
        <GitBranch className="w-8 h-8 text-slate-600 mx-auto" />
        <div className="text-sm font-semibold text-slate-300">
          No Multi-Order Causal Pathways Available
        </div>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          Causal butterfly-effect chains require at least one valid Micro interpretation.
        </p>
      </div>
    );
  }

  const activePathway =
    pathways.find((p) => p.pathway_id === selectedPathwayId) || pathways[0];
  const activeStep: CausalStep | undefined =
    activePathway.steps.find((s) => s.step_id === selectedStepId) || activePathway.steps[0];

  const getEvidenceBadge = (status: string, isWeak: boolean, isMissing?: boolean) => {
    if (isMissing) {
      return {
        label: 'Missing Link Gap',
        cls: 'bg-violet-950/80 text-violet-200 border-violet-600/80',
        cardBorder: 'border-dashed border-violet-500/80 bg-violet-950/20',
      };
    }
    if (status === 'unsupported') {
      return {
        label: 'Unsupported Jump',
        cls: 'bg-rose-950/70 text-rose-300 border-rose-700/80',
        cardBorder: 'border-dashed border-rose-600/80 bg-rose-950/15',
      };
    }
    if (isWeak) {
      return {
        label: 'Weak Link (>=3 Assump.)',
        cls: 'bg-amber-950/70 text-amber-300 border-amber-700/80',
        cardBorder: 'border-dashed border-amber-500/80 bg-amber-950/15',
      };
    }
    if (status === 'supported') {
      return {
        label: 'Source Supported',
        cls: 'bg-emerald-950/70 text-emerald-300 border-emerald-700/80',
        cardBorder: 'border-emerald-700/60 bg-slate-950/80',
      };
    }
    return {
      label: 'Model Inferred',
      cls: 'bg-cyan-950/70 text-cyan-300 border-cyan-800/70',
      cardBorder: 'border-slate-700 bg-slate-950/80',
    };
  };

  return (
    <div className="space-y-5">
      {/* Pathway Selector Bar */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-cyan-400" />
              <span>Competing Multi-Order Causal Butterfly-Effect Pathways</span>
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Select a pathway and click any causal step to inspect underlying mechanisms, triggers,
              dependencies, and epistemic weak/missing links.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono">
            <span className="flex items-center gap-1 text-emerald-400">
              <span className="w-2.5 h-0.5 bg-emerald-400" /> Supported
            </span>
            <span className="flex items-center gap-1 text-amber-400">
              <span className="w-2.5 h-0.5 border-t border-dashed border-amber-400" /> Weak Link
            </span>
            <span className="flex items-center gap-1 text-violet-300">
              <span className="w-2.5 h-0.5 border-t border-dashed border-violet-400" /> Missing Link
            </span>
            <span className="flex items-center gap-1 text-rose-400">
              <span className="w-2.5 h-0.5 border-t border-dashed border-rose-400" /> Unsupported Jump
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {pathways.map((pw) => {
            const isSelected = pw.pathway_id === activePathway.pathway_id;
            return (
              <button
                key={pw.pathway_id}
                type="button"
                onClick={() => {
                  setSelectedPathwayId(pw.pathway_id);
                  setSelectedStepId(pw.steps[0]?.step_id || '');
                }}
                className={`text-left p-3 rounded-xl border transition-all ${
                  isSelected
                    ? 'bg-cyan-950/40 border-cyan-500/70 shadow-md shadow-cyan-950/40'
                    : 'bg-slate-950/70 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`text-xs font-bold truncate ${
                      isSelected ? 'text-cyan-300' : 'text-slate-200'
                    }`}
                    title={pw.name}
                  >
                    {pw.name}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 truncate mt-0.5">
                  Driver: {pw.archetype_driver}
                </div>
                <div className="flex flex-wrap items-center gap-1.5 mt-2 font-mono text-[10px]">
                  <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-cyan-300">
                    Plaus {(pw.overall_plausibility * 100).toFixed(0)}%
                  </span>
                  <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-emerald-300">
                    Ground {(pw.evidence_grounding_score * 100).toFixed(0)}%
                  </span>
                  {pw.weak_link_count > 0 && (
                    <span className="px-1.5 py-0.5 rounded bg-amber-950/70 border border-amber-800/70 text-amber-300">
                      {pw.weak_link_count} Weak
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Interactive Chain Flow Visualization */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 pb-3 border-b border-slate-800">
          <div>
            <div className="text-sm font-bold text-slate-100">{activePathway.name}</div>
            <p className="text-xs text-slate-400 mt-0.5">{activePathway.narrative_summary}</p>
          </div>
          {activePathway.has_unsupported_jump && (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-rose-950/50 border border-rose-800/70 text-[11px] font-mono text-rose-300 self-start">
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
              Contains Unsupported Jump (Hypothesis Only)
            </span>
          )}
        </div>

        {/* Horizontal Scrollable Node Chain */}
        <div className="overflow-x-auto pb-2">
          <div className="flex items-stretch gap-2 min-w-[760px]">
            {/* Initial Event Anchor Card */}
            <div className="w-48 flex-shrink-0 rounded-xl border border-indigo-500/50 bg-indigo-950/30 p-3 flex flex-col justify-between">
              <div className="space-y-1.5">
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                  Initial Event (t=0)
                </span>
                <div className="text-xs font-bold text-slate-100 line-clamp-2">
                  {eventTitle || 'Initial Event Signal'}
                </div>
                <p className="text-[11px] text-slate-300/90 line-clamp-3">
                  {signalText || 'Primary observed signal anchor.'}
                </p>
              </div>
              <div className="text-[10px] font-mono text-indigo-300 pt-2 border-t border-indigo-900/50">
                Exogenous Shock
              </div>
            </div>

            {activePathway.steps.map((step) => {
              const badge = getEvidenceBadge(step.evidence_status, step.is_weak_link, step.is_missing_link);
              const isStepSelected = activeStep?.step_id === step.step_id;

              return (
                <React.Fragment key={step.step_id}>
                  <div className="flex items-center justify-center flex-shrink-0 px-0.5">
                    <ArrowRight
                      className={`w-4 h-4 ${
                        step.is_missing_link
                          ? 'text-violet-400'
                          : step.evidence_status === 'unsupported'
                          ? 'text-rose-400'
                          : step.is_weak_link
                          ? 'text-amber-400'
                          : 'text-cyan-400'
                      }`}
                    />
                  </div>

                  <button
                    type="button"
                    onClick={() => setSelectedStepId(step.step_id)}
                    className={`w-56 flex-shrink-0 text-left rounded-xl border p-3 flex flex-col justify-between transition-all ${
                      badge.cardBorder
                    } ${
                      isStepSelected
                        ? 'ring-2 ring-cyan-400 shadow-lg shadow-cyan-950/50'
                        : 'hover:border-slate-500'
                    }`}
                  >
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-[10px] font-mono uppercase text-slate-400">
                          {ORDER_LABELS[step.order] || step.order}
                        </span>
                        {step.is_weak_link && (
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                        )}
                      </div>

                      <div className="text-xs font-bold text-slate-100 line-clamp-2">
                        {step.title}
                      </div>

                      <p className="text-[11px] text-slate-300 line-clamp-3">{step.effect}</p>
                    </div>

                    <div className="space-y-1.5 pt-2 mt-2 border-t border-slate-800/80">
                      <div className="flex items-center justify-between text-[10px] font-mono">
                        <span className={`px-1.5 py-0.5 rounded border ${badge.cls}`}>
                          {badge.label}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                        <span>Conf: {(step.confidence * 100).toFixed(0)}%</span>
                        <span>Uncert: {(step.uncertainty * 100).toFixed(0)}%</span>
                        <span>{step.assumption_count}x Assump</span>
                      </div>
                    </div>
                  </button>
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Selected Step Detail Inspector */}
        {activeStep && (
          <div className="rounded-xl bg-slate-950/90 border border-slate-800 p-4 space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-800 pb-3">
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2 text-[10px] font-mono">
                  <span className="px-2 py-0.5 rounded bg-cyan-950/70 text-cyan-300 border border-cyan-800">
                    {ORDER_LABELS[activeStep.order] || activeStep.order}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-700 flex items-center gap-1">
                    <Clock className="w-3 h-3 text-cyan-400" />
                    {HORIZON_LABELS[activeStep.time_horizon] || activeStep.time_horizon}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-900 text-amber-300 border border-slate-700">
                    Epistemic: {activeStep.epistemic_category.replace(/_/g, ' ')}
                  </span>
                </div>
                <h4 className="text-sm font-bold text-slate-100">{activeStep.title}</h4>
              </div>

              <div className="text-right font-mono text-xs">
                <div className="text-cyan-400 font-bold">
                  Step Confidence: {(activeStep.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-slate-400">
                  Uncertainty {(activeStep.uncertainty * 100).toFixed(0)}% •{' '}
                  {activeStep.assumption_count} assumption hop(s)
                </div>
              </div>
            </div>

            {activeStep.is_missing_link && (
              <div className="p-3 rounded-lg bg-violet-950/40 border border-violet-700/80 text-xs text-violet-200 flex items-start gap-2">
                <ShieldAlert className="w-4 h-4 text-violet-300 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-violet-200">[MISSING LINK] Empirical Continuity Gap</div>
                  <p className="text-[11px] mt-0.5">
                    {activeStep.missing_link_reason ||
                      'Causal step crosses an empirical evidence gap without verified primary-source confirmation; treat strictly as speculative exploration.'}
                  </p>
                </div>
              </div>
            )}

            {activeStep.is_weak_link && (
              <div className="p-3 rounded-lg bg-amber-950/40 border border-amber-800/70 text-xs text-amber-200 flex items-start gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-amber-300">Flagged Weak Link</div>
                  <p className="text-[11px] mt-0.5">
                    {activeStep.weak_link_reason ||
                      'Multi-assumption or unverified causal transition; treat strictly as conditional hypothesis.'}
                  </p>
                </div>
              </div>
            )}

            {(activeStep.affected_actor ||
              activeStep.behavioral_or_institutional_response ||
              activeStep.invalidation_condition) && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-900/70 border border-slate-800/80 space-y-1">
                  <div className="text-[10px] font-mono uppercase text-violet-300">
                    Affected Actor
                  </div>
                  <p className="text-slate-200 leading-relaxed">
                    {activeStep.affected_actor ||
                      activeStep.affected_stakeholders[0] ||
                      'Target cohort'}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/70 border border-slate-800/80 space-y-1">
                  <div className="text-[10px] font-mono uppercase text-indigo-300">
                    Behavioural / Institutional Response
                  </div>
                  <p className="text-slate-200 leading-relaxed">
                    {activeStep.behavioral_or_institutional_response || activeStep.mechanism}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-800/50 space-y-1">
                  <div className="text-[10px] font-mono uppercase text-emerald-300">
                    Step Invalidation Condition
                  </div>
                  <p className="text-slate-200 leading-relaxed">
                    {activeStep.invalidation_condition ||
                      'Direct empirical falsification of this causal link'}
                  </p>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-900/70 border border-slate-800/80 space-y-1">
                <div className="text-[10px] font-mono uppercase text-slate-400">Cause / Driver</div>
                <p className="text-slate-200 leading-relaxed">{activeStep.cause}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-900/70 border border-slate-800/80 space-y-1">
                <div className="text-[10px] font-mono uppercase text-cyan-400">
                  Causal Mechanism
                </div>
                <p className="text-slate-200 leading-relaxed">{activeStep.mechanism}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-900/70 border border-slate-800/80 space-y-1">
                <div className="text-[10px] font-mono uppercase text-emerald-400">
                  Resulting Effect
                </div>
                <p className="text-slate-200 leading-relaxed">{activeStep.effect}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="space-y-1.5">
                <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1">
                  <Users className="w-3 h-3 text-violet-400" />
                  <span>Affected Stakeholders</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {activeStep.affected_stakeholders.length > 0 ? (
                    activeStep.affected_stakeholders.map((st, idx) => (
                      <span
                        key={idx}
                        className="text-[11px] px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-300"
                      >
                        {st}
                      </span>
                    ))
                  ) : (
                    <span className="text-slate-500 text-[11px]">General ecosystem</span>
                  )}
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1">
                  <Zap className="w-3 h-3 text-amber-400" />
                  <span>Trigger Conditions</span>
                </div>
                <ul className="list-disc list-inside space-y-0.5 text-[11px] text-slate-300">
                  {activeStep.trigger_conditions.length > 0 ? (
                    activeStep.trigger_conditions.map((t, i) => <li key={i}>{t}</li>)
                  ) : (
                    <li className="text-slate-500">Conditional continuation of prior step</li>
                  )}
                </ul>
              </div>

              <div className="space-y-1.5">
                <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                  <span>Evidence &amp; Layer Traceability</span>
                </div>
                <ul className="list-disc list-inside space-y-0.5 text-[11px] text-slate-300">
                  {activeStep.supporting_evidence.length > 0 ? (
                    activeStep.supporting_evidence.map((ev, i) => <li key={i}>{ev}</li>)
                  ) : (
                    <li className="text-amber-300/90">
                      No direct empirical citation — multi-step extrapolation
                    </li>
                  )}
                </ul>
                {activeStep.source_layer_refs.length > 0 && (
                  <div className="flex flex-wrap gap-1 pt-1">
                    {activeStep.source_layer_refs.map((r, i) => (
                      <span
                        key={i}
                        className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/60 text-indigo-300"
                      >
                        {r}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Pathway Invalidation / Breakage Conditions */}
        {activePathway.invalidation_conditions.length > 0 && (
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 flex items-start gap-2.5 text-xs">
            <HelpCircle className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-mono text-[10px] uppercase text-cyan-400 font-semibold">
                Pathway Invalidation / Reversal Conditions
              </div>
              <ul className="list-disc list-inside text-[11px] text-slate-300 space-y-0.5">
                {activePathway.invalidation_conditions.map((cond, idx) => (
                  <li key={idx}>{cond}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400 pt-1">
          <Layers className="w-3.5 h-3.5 text-slate-500" />
          <span>
            Causal pathways represent structured multi-order hypotheses conditioned on 4-layer
            Micro/Meso/Macro/Feedback propagation — NOT guaranteed physical determinism.
          </span>
        </div>
      </div>
    </div>
  );
};
