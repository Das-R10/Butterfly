// frontend/src/components/common/CrossLayerExplanationPanel.tsx

import React, { useState } from 'react';
import { SimulationResult } from '../../types/simulation';
import { isValidInterpretation } from '../../utils/modelSlots';
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  Cpu,
  Share2,
  TrendingUp,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';

interface CrossLayerExplanationPanelProps {
  simulation: SimulationResult | null;
  selectedIteration: number;
  defaultExpanded?: boolean;
  highlightLayer?: 'micro' | 'meso' | 'macro' | 'feedback' | 'all';
}

export const CrossLayerExplanationPanel: React.FC<CrossLayerExplanationPanelProps> = ({
  simulation,
  selectedIteration,
  defaultExpanded = false,
  highlightLayer = 'all',
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [activeFocus, setActiveFocus] = useState<'all' | 'micro' | 'meso' | 'macro' | 'feedback'>(
    highlightLayer
  );

  if (!simulation) return null;

  const iterations = simulation.iterations || [];
  const iterData =
    iterations.find((it) => it.iteration === selectedIteration) || iterations[iterations.length - 1];

  if (!iterData) return null;

  const validMicro = (iterData.micro_interpretations || []).filter((m) => isValidInterpretation(m));
  const failedMicro = (iterData.micro_interpretations || []).length - validMicro.length;
  const meanTransmission =
    validMicro.length > 0
      ? validMicro.reduce((acc, m) => acc + (m.transmission_weight ?? m.virality * m.credibility), 0) /
        validMicro.length
      : 0;
  const meanCred =
    validMicro.length > 0
      ? validMicro.reduce((acc, m) => acc + (m.model_estimated_credibility ?? m.credibility), 0) /
        validMicro.length
      : 0;
  const meanGrounding =
    validMicro.length > 0
      ? validMicro.reduce((acc, m) => acc + (m.evidence_grounding_score ?? m.credibility), 0) /
        validMicro.length
      : 0;

  const meso = iterData.meso_state;
  const macro = iterData.macro_state;
  const feedback = iterData.feedback_state;

  const showLayer = (layer: 'micro' | 'meso' | 'macro' | 'feedback') =>
    activeFocus === 'all' || activeFocus === layer;

  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl overflow-hidden shadow-lg">
      {/* Header Bar */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-4 flex items-center justify-between text-left hover:bg-slate-900/90 transition"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
            <BookOpen className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-slate-100 flex items-center gap-2">
              <span>Cross-Layer Pipeline Mechanics & Traceability</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                Round {iterData.iteration}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Micro → Meso → Macro → Feedback inputs, exact formulas, active numerical outputs, and
              scientific limitations.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
          <span>{expanded ? 'Hide Details' : 'Inspect Pipeline Math'}</span>
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {expanded && (
        <div className="p-5 pt-2 border-t border-slate-800 space-y-5 animate-fade-in">
          {/* Layer Focus Pills */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono">
              {(['all', 'micro', 'meso', 'macro', 'feedback'] as const).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveFocus(tab)}
                  className={`px-2.5 py-1 rounded-md border transition ${
                    activeFocus === tab
                      ? 'bg-cyan-950/70 border-cyan-700 text-cyan-300 font-semibold'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {tab === 'all'
                    ? 'All 4 Layers'
                    : tab === 'micro'
                    ? 'L1: Micro'
                    : tab === 'meso'
                    ? 'L2: Meso'
                    : tab === 'macro'
                    ? 'L3: Macro'
                    : 'L4: Feedback'}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2 text-[10px] font-mono">
              <span className="px-2 py-0.5 rounded bg-violet-950/60 border border-violet-800/60 text-violet-300">
                Micro Credibility: Synthetic LLM Estimate
              </span>
              <span className="px-2 py-0.5 rounded bg-amber-950/60 border border-amber-800/60 text-amber-300">
                Fact Verification: Not Empirically Measured
              </span>
            </div>
          </div>

          {/* Flow Cards Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* L1 MICRO */}
            {showLayer('micro') && (
              <div className="p-4 rounded-xl bg-slate-950/80 border border-cyan-900/40 space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-cyan-400">
                    <Cpu className="w-4 h-4" />
                    <span>L1: Micro Layer (Dual-Model Cognitive Parsing)</span>
                  </div>
                  <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/50 px-2 py-0.5 rounded">
                    {validMicro.length} valid / {failedMicro} failed
                  </span>
                </div>
                <div className="text-[11px] text-slate-300 space-y-1.5 leading-relaxed">
                  <div>
                    <span className="text-slate-400 font-semibold">Inputs:</span> Raw event signal
                    {simulation.config.source_url ? ' + source reference URL' : ''}, 5 archetype
                    persona prompts, and Round {Math.max(1, iterData.iteration - 1)} feedback
                    guardrails.
                  </div>
                  <div className="font-mono text-[10px] bg-black/40 p-2 rounded border border-slate-800 text-cyan-300">
                    Conviction = 0.20 + 0.80·|stance| (stance-sign unbiased)
                    <br />
                    Transmission = Conv · (0.45·V + 0.35·E_ground + 0.20·Emo) · penalty
                  </div>
                  <div>
                    <span className="text-slate-400 font-semibold">Round {iterData.iteration} Outputs:</span>{' '}
                    Mean LLM Credibility <span className="text-cyan-300 font-mono">{(meanCred * 100).toFixed(1)}%</span>,
                    Evidence Grounding <span className="text-emerald-300 font-mono">{(meanGrounding * 100).toFixed(1)}%</span>,
                    Mean Transmission <span className="text-amber-300 font-mono">{meanTransmission.toFixed(3)}</span>.
                  </div>
                  <div className="text-[10px] text-amber-300/90 flex items-start gap-1.5 pt-1 border-t border-slate-800/80">
                    <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    <span>
                      <strong>Limitation:</strong> Archetype credibility and stances are synthetic LLM
                      assessments, not verified empirical fact checks. Unsupported claims trigger a{' '}
                      <code>0.40x</code> transmission penalty and uncertainty floor{' '}
                      <code>≥0.55</code>.
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* L2 MESO */}
            {showLayer('meso') && (
              <div className="p-4 rounded-xl bg-slate-950/80 border border-violet-900/40 space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-violet-400">
                    <Share2 className="w-4 h-4" />
                    <span>L2: Meso Layer (Capacity-Normalized Influence Graph)</span>
                  </div>
                  <span className="text-[10px] font-mono text-violet-300 bg-violet-950/50 px-2 py-0.5 rounded">
                    {meso?.activation_semantics || 'instantaneous_per_iteration'}
                  </span>
                </div>
                <div className="text-[11px] text-slate-300 space-y-1.5 leading-relaxed">
                  <div>
                    <span className="text-slate-400 font-semibold">Inputs:</span> Valid Micro
                    transmission weights only (failed slots excluded), external weight α=
                    {(simulation.config.external_weight ?? 0.4).toFixed(2)}, 8-node directed graph.
                  </div>
                  <div className="font-mono text-[10px] bg-black/40 p-2 rounded border border-slate-800 text-violet-300">
                    NormAct ∈ [0,1] = α·NormExt + (1-α)·NormPeer (instantaneous, non-accumulating)
                    <br />
                    ExtMedia = 0.5·T(Amp) + 0.5·T(Inst) | ExtScore = 0.5·ExtMedia + 0.5·T(Amp)
                  </div>
                  <div>
                    <span className="text-slate-400 font-semibold">Round {iterData.iteration} Outputs:</span>{' '}
                    ExtInfluence <span className="text-cyan-300 font-mono">{(meso?.external_influence_score ?? 0).toFixed(3)}</span>,
                    InternalContagion <span className="text-violet-300 font-mono">{(meso?.internal_contagion_score ?? 0).toFixed(3)}</span>,
                    EffectiveSpread <span className="text-emerald-300 font-mono">{(meso?.effective_spread_potential ?? 0).toFixed(3)}</span>.
                  </div>
                  <div className="text-[10px] text-slate-400 flex items-start gap-1.5 pt-1 border-t border-slate-800/80">
                    <ArrowRight className="w-3.5 h-3.5 text-violet-400 flex-shrink-0 mt-0.5" />
                    <span>
                      <strong>Interpretation:</strong> Edge weights represent structural channels; raw
                      inflows can exceed 1.0 when multiple peers fire simultaneously, while normalized
                      node activation measures relative capacity saturation in [0, 1].
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* L3 MACRO */}
            {showLayer('macro') && (
              <div className="p-4 rounded-xl bg-slate-950/80 border border-emerald-900/40 space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-emerald-400">
                    <TrendingUp className="w-4 h-4" />
                    <span>L3: Macro Layer (Modified Bass Diffusion + Churn)</span>
                  </div>
                  <span className="text-[10px] font-mono text-emerald-300 bg-emerald-950/50 px-2 py-0.5 rounded">
                    Ratio: {(macro?.attention_vs_adoption_ratio ?? 0).toFixed(2)}x
                  </span>
                </div>
                <div className="text-[11px] text-slate-300 space-y-1.5 leading-relaxed">
                  <div>
                    <span className="text-slate-400 font-semibold">Inputs:</span> Meso External (→p=
                    {(macro?.p_external ?? 0).toFixed(4)}) &amp; Internal (→q=
                    {(macro?.q_internal ?? 0).toFixed(4)}) scores, churn c=
                    {(macro?.c_churn ?? 0.08).toFixed(2)}, ceiling M=
                    {(macro?.M_potential ?? 1).toFixed(2)}, valid Micro virality/emotion.
                  </div>
                  <div className="font-mono text-[10px] bg-black/40 p-2 rounded border border-slate-800 text-emerald-300">
                    dN/dt = [p + q·(N/M)]·(M - N) - c·N (Active Sim. Adoption ≤ Cumulative Sim. Diffusion ≤ M)
                  </div>
                  <div>
                    <span className="text-slate-400 font-semibold">Round {iterData.iteration} Simulated Proxies:</span>{' '}
                    Active Sim. <span className="text-emerald-300 font-mono">{((macro?.active_adoption ?? 0) * 100).toFixed(1)}%</span>,
                    Cumul. Sim. <span className="text-cyan-300 font-mono">{((macro?.cumulative_adoption ?? 0) * 100).toFixed(1)}%</span>,
                    Sim. Attention <span className="text-amber-300 font-mono">{((macro?.attention_indicator ?? 0) * 100).toFixed(1)}%</span>,
                    dN/dt <span className="text-slate-200 font-mono">{(macro?.adoption_velocity ?? 0).toFixed(4)}</span>.
                  </div>
                  <div className="text-[10px] text-slate-400 flex items-start gap-1.5 pt-1 border-t border-slate-800/80">
                    <AlertTriangle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                    <span>
                      <strong>Limitation:</strong> {macro?.assumptions_and_limitations || 'Mean-field deterministic Bass approximation assuming homogeneous population mixing per step Δt=1; conversational buzz S(t) is decoupled from persistent adoption N(t).'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* L4 FEEDBACK */}
            {showLayer('feedback') && (
              <div className="p-4 rounded-xl bg-slate-950/80 border border-amber-900/40 space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-amber-400">
                    <RefreshCw className="w-4 h-4" />
                    <span>L4: Feedback Layer (Polarization & Drift Guardrail)</span>
                  </div>
                  <span className="text-[10px] font-mono text-amber-300 bg-amber-950/50 px-2 py-0.5 rounded">
                    Polarization: {((feedback?.polarization_index ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-[11px] text-slate-300 space-y-1.5 leading-relaxed">
                  <div>
                    <span className="text-slate-400 font-semibold">Inputs:</span> Valid Micro stance
                    std dev, evidence verification counts (
                    {feedback?.claim_verification_audit?.supported ?? 0} supported /{' '}
                    {feedback?.claim_verification_audit?.inferred ?? 0} inferred /{' '}
                    {feedback?.claim_verification_audit?.unsupported ?? 0} unsupported), and Macro
                    adoption/attention state.
                  </div>
                  {feedback?.iteration_change_summary && (
                    <div className="text-[10px] bg-black/40 p-2 rounded border border-slate-800 text-amber-200 leading-snug">
                      {feedback.iteration_change_summary}
                    </div>
                  )}
                  <div>
                    <span className="text-slate-400 font-semibold">Why Round {iterData.iteration} Shifted:</span>{' '}
                    Stance Δ={Number(feedback?.drift_metrics?.stance_drift ?? 0).toFixed(3)} |
                    Cred Δ={Number(feedback?.drift_metrics?.credibility_drift ?? 0).toFixed(3)} |
                    Adopt Δ={(Number(feedback?.drift_metrics?.adoption_drift ?? 0) * 100).toFixed(1)}%.
                  </div>
                  <div className="text-[10px] text-slate-400 flex items-start gap-1.5 pt-1 border-t border-slate-800/80">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                    <span>
                      <strong>Epistemic Safeguard:</strong> Unverified hypotheses from Round t are
                      tagged <code>[UNVERIFIED HYPOTHESIS]</code> when injected into Round t+1 so
                      speculation cannot silently harden into verified baseline facts.
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

