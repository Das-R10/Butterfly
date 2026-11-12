// frontend/src/components/forecast/ScenarioExplorer.tsx

import React, { useState } from 'react';
import {
  FutureScenario,
  OrderEffectsSummary,
  StakeholderReaction,
  TimeHorizonForecast,
} from '../../types/simulation';
import {
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Activity,
  Clock,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  Flame,
  Compass,
  Users,
} from 'lucide-react';

interface ScenarioExplorerProps {
  scenarios: FutureScenario[];
  orderEffects?: OrderEffectsSummary | null;
  stakeholders?: StakeholderReaction[];
  timeHorizon?: TimeHorizonForecast | null;
  section?: 'scenarios' | 'stakeholders_horizons' | 'all';
}

const SCENARIO_STYLE: Record<
  string,
  {
    badge: string;
    border: string;
    bar: string;
    label: string;
    icon: React.FC<{ className?: string }>;
  }
> = {
  baseline: {
    badge: 'bg-cyan-950/70 text-cyan-300 border-cyan-700/70',
    border: 'border-cyan-500/50',
    bar: 'bg-cyan-400',
    label: 'Baseline',
    icon: Compass,
  },
  positive: {
    badge: 'bg-emerald-950/70 text-emerald-300 border-emerald-700/70',
    border: 'border-emerald-500/50',
    bar: 'bg-emerald-400',
    label: 'Positive / Stabilizing',
    icon: TrendingUp,
  },
  negative: {
    badge: 'bg-amber-950/70 text-amber-300 border-amber-700/70',
    border: 'border-amber-500/50',
    bar: 'bg-amber-400',
    label: 'Negative / Disruptive',
    icon: TrendingDown,
  },
  extreme: {
    badge: 'bg-rose-950/70 text-rose-300 border-rose-700/70',
    border: 'border-rose-500/50',
    bar: 'bg-rose-400',
    label: 'Extreme Tail Risk',
    icon: Flame,
  },
};

export const ScenarioExplorer: React.FC<ScenarioExplorerProps> = ({
  scenarios,
  orderEffects,
  stakeholders = [],
  timeHorizon,
  section = 'scenarios',
}) => {
  const [selectedType, setSelectedType] = useState<string>(
    scenarios[0]?.scenario_type || 'baseline'
  );

  const activeScenario =
    scenarios.find((s) => s.scenario_type === selectedType) || scenarios[0] || null;

  const hasValidWeights = scenarios.some((s) => s.conditional_weight != null);
  const weightSum = scenarios.reduce((acc, s) => acc + (s.conditional_weight ?? 0), 0);

  return (
    <div className="space-y-6">
      {(section === 'scenarios' || section === 'all') && (
        <div className="space-y-5">
          {/* Heuristic Calibration Warning Banner */}
          <div className="p-3.5 rounded-xl bg-amber-950/35 border border-amber-800/60 text-xs text-amber-200 flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-bold text-amber-300">
                Heuristic Conditional Weights — Not Empirical Probabilities:
              </span>{' '}
              {scenarios[0]?.calibration_warning ||
                'Scenario weights are bounded heuristic allocations inferred from LLM interpretations, Meso spread, and Bass diffusion dynamics.'}
            </div>
          </div>

          {/* Comparative Scenario Strip */}
          {scenarios.length > 0 && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                  Four Future Scenarios — Heuristic Weight Comparison
                </h3>
                <span className="text-[11px] font-mono text-slate-400">
                  {hasValidWeights
                    ? `Conditional Weights Sum: ${(weightSum * 100).toFixed(1)}%`
                    : 'Weights Unavailable (Failed Execution)'}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {scenarios.map((sc) => {
                  const style = SCENARIO_STYLE[sc.scenario_type] || SCENARIO_STYLE.baseline;
                  const Icon = style.icon;
                  const isSelected = activeScenario?.scenario_id === sc.scenario_id;
                  const pct =
                    sc.conditional_weight != null ? Math.round(sc.conditional_weight * 1000) / 10 : null;

                  return (
                    <button
                      key={sc.scenario_id}
                      type="button"
                      onClick={() => setSelectedType(sc.scenario_type)}
                      className={`text-left p-4 rounded-xl border transition-all space-y-2.5 ${
                        isSelected
                          ? `bg-slate-950 ${style.border} ring-1 ring-cyan-400/40 shadow-lg`
                          : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${style.badge}`}>
                          {style.label}
                        </span>
                        <Icon className="w-4 h-4 text-slate-400 flex-shrink-0" />
                      </div>
                      <div className="text-xs font-bold text-slate-100 line-clamp-2">{sc.title}</div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-[11px] font-mono">
                          <span className="text-slate-400">Weight</span>
                          <span className="text-slate-100 font-bold">
                            {pct != null ? `${pct.toFixed(1)}%` : 'N/A'}
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${style.bar}`}
                            style={{ width: `${pct != null ? Math.min(pct, 100) : 0}%` }}
                          />
                        </div>
                      </div>
                      <div className="text-[10px] font-mono text-slate-400 truncate">
                        {sc.probability_band}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Detailed Inspector for Selected Scenario */}
          {activeScenario && (
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-5">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded border ${
                        (SCENARIO_STYLE[activeScenario.scenario_type] || SCENARIO_STYLE.baseline).badge
                      }`}
                    >
                      {activeScenario.scenario_type} scenario
                    </span>
                    {activeScenario.linked_pathway_id && (
                      <span className="text-[10px] font-mono text-slate-400">
                        Pathway: {activeScenario.linked_pathway_id}
                      </span>
                    )}
                  </div>
                  <h4 className="text-base font-bold text-slate-100">{activeScenario.title}</h4>
                  <p className="text-xs text-slate-300 leading-relaxed">{activeScenario.summary}</p>
                </div>

                <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-xs flex-shrink-0 space-y-1">
                  <div className="text-slate-400 text-[10px] uppercase">Heuristic Weight</div>
                  <div className="text-lg font-bold text-cyan-400">
                    {activeScenario.conditional_weight != null
                      ? `${(activeScenario.conditional_weight * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    Conf:{' '}
                    {activeScenario.confidence != null
                      ? `${(activeScenario.confidence * 100).toFixed(0)}%`
                      : 'N/A'}{' '}
                    | Uncert: {(activeScenario.uncertainty * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              {/* Short / Medium / Long-Term Consequences */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="text-[10px] font-mono uppercase text-cyan-400 font-semibold">
                    Short-Term Consequences (Days–Weeks)
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                    {activeScenario.short_term_consequences.length > 0 ? (
                      activeScenario.short_term_consequences.map((c, i) => <li key={i}>{c}</li>)
                    ) : (
                      <li className="text-slate-500">None isolated</li>
                    )}
                  </ul>
                </div>

                <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="text-[10px] font-mono uppercase text-violet-400 font-semibold">
                    Medium-Term Consequences (Months)
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                    {activeScenario.medium_term_consequences.length > 0 ? (
                      activeScenario.medium_term_consequences.map((c, i) => <li key={i}>{c}</li>)
                    ) : (
                      <li className="text-slate-500">None isolated</li>
                    )}
                  </ul>
                </div>

                <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="text-[10px] font-mono uppercase text-amber-400 font-semibold">
                    Long-Term Systemic Implications (Quarters+)
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                    {activeScenario.long_term_consequences.length > 0 ? (
                      activeScenario.long_term_consequences.map((c, i) => <li key={i}>{c}</li>)
                    ) : (
                      <li className="text-slate-500">None isolated</li>
                    )}
                  </ul>
                </div>
              </div>

              {/* Supporting vs Counter-Evidence + Triggers + Indicators */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-800/50 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-emerald-400 flex items-center justify-between gap-1.5">
                    <span className="flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>Model Hypotheses &amp; Supporting Basis</span>
                    </span>
                    <span className="text-[9px] text-amber-300/90 normal-case">
                      Attributed Hypotheses ≠ Empirical Proof
                    </span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                    {activeScenario.supporting_evidence.length > 0 ? (
                      activeScenario.supporting_evidence.map((s, i) => <li key={i}>{s}</li>)
                    ) : (
                      <li className="text-slate-500">No direct supporting citations recorded</li>
                    )}
                  </ul>
                </div>

                <div className="p-3.5 rounded-lg bg-rose-950/20 border border-rose-800/50 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-rose-400 flex items-center gap-1.5">
                    <XCircle className="w-3.5 h-3.5" />
                    <span>Counter-Evidence &amp; Falsifiers</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                    {activeScenario.counter_evidence.length > 0 ? (
                      activeScenario.counter_evidence.map((c, i) => <li key={i}>{c}</li>)
                    ) : (
                      <li className="text-slate-500">None flagged directly</li>
                    )}
                  </ul>
                </div>

                <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-amber-400">
                    Activation Triggers
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                    {activeScenario.trigger_conditions.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </div>

                <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-cyan-400 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5" />
                    <span>Indicators to Monitor Next</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                    {activeScenario.indicators_to_monitor.map((ind, i) => (
                      <li key={i}>{ind}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {(section === 'stakeholders_horizons' || section === 'all') && (
        <div className="space-y-6">
          {/* Time-Horizon Forecast Grid */}
          {timeHorizon && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <span>Multi-Horizon Impact Timeline</span>
                </h3>
                <span className="text-[10px] font-mono text-slate-400">
                  Days -&gt; Weeks -&gt; Months -&gt; Quarters+
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                {[
                  {
                    title: 'Immediate (Days)',
                    items: timeHorizon.immediate_days,
                    cls: 'border-cyan-800/60 text-cyan-300',
                  },
                  {
                    title: 'Short-Term (Weeks)',
                    items: timeHorizon.short_term_weeks,
                    cls: 'border-emerald-800/60 text-emerald-300',
                  },
                  {
                    title: 'Medium-Term (Months)',
                    items: timeHorizon.medium_term_months,
                    cls: 'border-violet-800/60 text-violet-300',
                  },
                  {
                    title: 'Long-Term (Quarters)',
                    items: timeHorizon.long_term_quarters,
                    cls: 'border-amber-800/60 text-amber-300',
                  },
                ].map((col) => (
                  <div
                    key={col.title}
                    className={`p-3.5 rounded-xl bg-slate-950/75 border ${col.cls} space-y-2`}
                  >
                    <div className="text-[11px] font-mono font-bold uppercase">{col.title}</div>
                    <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                      {col.items.length > 0 ? (
                        col.items.map((item, idx) => <li key={idx}>{item}</li>)
                      ) : (
                        <li className="text-slate-500">No items captured</li>
                      )}
                    </ul>
                  </div>
                ))}
              </div>

              {timeHorizon.unexpected_tail_risks.length > 0 && (
                <div className="p-3.5 rounded-xl bg-rose-950/25 border border-rose-800/60 space-y-1.5 text-xs">
                  <div className="text-[10px] font-mono uppercase text-rose-300 font-bold flex items-center gap-1.5">
                    <Flame className="w-3.5 h-3.5 text-rose-400" />
                    <span>Unexpected / Low-Probability High-Impact Tail Risks</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-rose-100/90">
                    {timeHorizon.unexpected_tail_risks.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Stakeholder Reaction & Behavioural Impact Matrix */}
          {stakeholders.length > 0 && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Users className="w-4 h-4 text-violet-400" />
                  <span>Stakeholder Reaction &amp; Meso-Network Alignment Matrix</span>
                </h3>
                <span className="text-[10px] font-mono text-slate-400">
                  {stakeholders.length} Tracked Actor Cohorts
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {stakeholders.map((sr, idx) => {
                  const dirBadge =
                    sr.impact_direction === 'stabilizing'
                      ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/70'
                      : sr.impact_direction === 'destabilizing'
                      ? 'bg-rose-950/60 text-rose-300 border-rose-800/70'
                      : sr.impact_direction === 'polarizing'
                      ? 'bg-amber-950/60 text-amber-300 border-amber-800/70'
                      : 'bg-cyan-950/60 text-cyan-300 border-cyan-800/70';

                  return (
                    <div
                      key={`${sr.stakeholder}-${idx}`}
                      className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2.5 text-xs"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="font-bold text-slate-100">{sr.stakeholder}</div>
                        <div className="flex items-center gap-1.5 font-mono text-[10px]">
                          <span className={`px-2 py-0.5 rounded border uppercase ${dirBadge}`}>
                            {sr.impact_direction}
                          </span>
                          {sr.activation_level != null && (
                            <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-cyan-300">
                              Meso {(sr.activation_level * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="text-[11px] text-slate-300 leading-relaxed">
                        {sr.expected_reaction}
                      </p>
                      <div className="text-[11px] text-slate-400 bg-slate-900/60 p-2 rounded border border-slate-800/70">
                        <strong className="text-slate-300">Behavioural Shift:</strong>{' '}
                        {sr.behavioural_shift}
                      </div>
                      {(sr.resilience_lever || sr.downside_vulnerability) && (
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[10px]">
                          {sr.resilience_lever && (
                            <div className="p-2 rounded bg-emerald-950/20 border border-emerald-800/40 text-emerald-200">
                              <strong className="text-emerald-400">Resilience Lever:</strong>{' '}
                              {sr.resilience_lever}
                            </div>
                          )}
                          {sr.downside_vulnerability && (
                            <div className="p-2 rounded bg-rose-950/20 border border-rose-800/40 text-rose-200">
                              <strong className="text-rose-400">Downside Risk:</strong>{' '}
                              {sr.downside_vulnerability}
                            </div>
                          )}
                        </div>
                      )}
                      {sr.likely_actions.length > 0 && (
                        <ul className="list-disc list-inside text-[11px] text-slate-300 space-y-0.5">
                          {sr.likely_actions.slice(0, 3).map((act, aIdx) => (
                            <li key={aIdx}>{act}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Order-of-Effects Summary & Prevention / Reversal Conditions */}
          {orderEffects && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Orders of Effects &amp; Reversal / Prevention Conditions</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="p-3.5 rounded-lg bg-slate-950/75 border border-slate-800 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-cyan-400">
                    1st-Order Direct Effects
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                    {orderEffects.first_order.map((x, i) => (
                      <li key={i}>{x}</li>
                    ))}
                  </ul>
                </div>
                <div className="p-3.5 rounded-lg bg-slate-950/75 border border-slate-800 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-violet-400">
                    2nd-Order Indirect Effects
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                    {orderEffects.second_order.map((x, i) => (
                      <li key={i}>{x}</li>
                    ))}
                  </ul>
                </div>
                <div className="p-3.5 rounded-lg bg-slate-950/75 border border-slate-800 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-amber-400">
                    3rd-Order &amp; Systemic Effects
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-300">
                    {orderEffects.third_order_and_systemic.map((x, i) => (
                      <li key={i}>{x}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {orderEffects.prevention_and_reversal_conditions.length > 0 && (
                <div className="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-800/60 space-y-1.5 text-xs">
                  <div className="text-[10px] font-mono uppercase text-emerald-300 font-bold">
                    Conditions That Would Prevent, Dampen, or Reverse Negative Consequences
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-200">
                    {orderEffects.prevention_and_reversal_conditions.map((p, i) => (
                      <li key={i}>{p}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

