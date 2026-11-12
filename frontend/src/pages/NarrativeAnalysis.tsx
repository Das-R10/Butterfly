// frontend/src/pages/NarrativeAnalysis.tsx

import React, { useState } from 'react';
import { useSimulation } from '../context/SimulationContext';
import { ArchetypeBadge } from '../components/common/ArchetypeBadge';
import { MetricCard } from '../components/common/MetricCard';
import { CrossLayerExplanationPanel } from '../components/common/CrossLayerExplanationPanel';
import { ReportDownloadButton } from '../components/common/ReportDownloadButton';
import {
  groupByArchetypeAndSlot,
  isValidInterpretation,
  getExecutionStatusBadge,
} from '../utils/modelSlots';
import { MicroInterpretation } from '../types/simulation';
import {
  MessageSquareText,
  Layers,
  Scale,
  Sparkles,
  Tag,
  Cpu,
  ShieldAlert,
  Clock,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  X,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

const ModelInterpretationBox: React.FC<{
  slotLabel: 'Model A' | 'Model B';
  configuredModel?: string;
  accent: 'cyan' | 'violet';
  item?: MicroInterpretation;
}> = ({ slotLabel, configuredModel, accent, item }) => {
  const valid = isValidInterpretation(item);
  const badge = item ? getExecutionStatusBadge(item.execution_status) : null;
  const headingColor = accent === 'cyan' ? 'text-cyan-400' : 'text-violet-400';

  if (!item) {
    return (
      <div className="p-4 rounded-lg bg-rose-950/20 border border-rose-900/50 space-y-2 text-xs">
        <div className="flex items-center justify-between">
          <span className={`font-bold ${headingColor}`}>
            {slotLabel} ({configuredModel || 'Configured Engine'})
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800/60">
            Missing Output
          </span>
        </div>
        <p className="text-rose-200/80 text-[11px]">
          No interpretation record produced for this model slot in this iteration (not substituted
          with another model).
        </p>
      </div>
    );
  }

  if (!valid) {
    return (
      <div className="p-4 rounded-lg bg-rose-950/25 border border-rose-800/70 space-y-2.5 text-xs">
        <div className="flex items-center justify-between gap-2">
          <span className={`font-bold ${headingColor}`}>
            {slotLabel} ({item.model_name || configuredModel})
          </span>
          {badge && (
            <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${badge.className}`}>
              {badge.label}
            </span>
          )}
        </div>
        <div className="p-3 rounded bg-black/40 border border-rose-900/60 space-y-1.5">
          <div className="flex items-center gap-1.5 text-rose-300 font-semibold">
            <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>
              Inference Failed: {item.error_category || item.execution_status}
            </span>
          </div>
          <p className="text-[11px] font-mono text-rose-200/90 break-words">
            {item.error_message ||
              'Provider returned invalid or empty JSON; failure explicitly recorded without mock substitution.'}
          </p>
          <div className="flex flex-wrap gap-3 text-[10px] font-mono text-rose-300/80 pt-1">
            {item.duration_sec != null && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" /> {item.duration_sec.toFixed(1)}s
              </span>
            )}
            {item.retry_count != null && <span>Retries: {item.retry_count}</span>}
            {item.finish_reason && <span>Finish: {item.finish_reason}</span>}
            {item.response_length != null && <span>Length: {item.response_length} chars</span>}
          </div>
        </div>
      </div>
    );
  }

  const llmCred = item.model_estimated_credibility ?? item.credibility;
  const evGround = item.evidence_grounding_score;

  return (
    <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className={`font-bold ${headingColor}`}>
            {slotLabel} ({item.model_name})
          </span>
          {badge && (
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${badge.className}`}>
              {badge.label}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3 font-mono text-slate-300">
          <span>
            Stance:{' '}
            <strong className={item.stance >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
              {item.stance > 0 ? `+${item.stance.toFixed(2)}` : item.stance.toFixed(2)}
            </strong>
          </span>
          {item.risk_severity != null && (
            <span
              title="Decoupled risk severity assessment [0,1]"
              className="px-1.5 py-0.5 rounded bg-rose-950/50 border border-rose-800/60 text-rose-300 text-[10px]"
            >
              Risk Sev: {Math.round(item.risk_severity * 100)}%
            </span>
          )}
        </div>
      </div>

      <p className="text-xs text-slate-200 leading-relaxed italic">"{item.primary_narrative}"</p>

      {/* Evidence Grounding & Synthetic vs Empirical Distinction */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono bg-slate-900/70 px-2.5 py-1.5 rounded border border-slate-800/80">
        <div className="flex items-center gap-1.5">
          <span
            className={`px-1.5 py-0.5 rounded uppercase border ${
              item.evidence_status === 'supported'
                ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60'
                : item.evidence_status === 'unsupported'
                ? 'bg-rose-950/60 text-rose-400 border-rose-800/60'
                : 'bg-amber-950/60 text-amber-400 border-amber-800/60'
            }`}
          >
            {item.evidence_status || 'inferred'}
          </span>
          {item.evidence_penalty_applied != null && item.evidence_penalty_applied < 1.0 && (
            <span className="text-rose-300">Penalty {item.evidence_penalty_applied.toFixed(2)}x</span>
          )}
        </div>
        <span className="text-slate-400">
          Empirical Truth: <strong className="text-amber-300">Not Empirically Measured</strong>
        </span>
      </div>

      {/* Supporting Evidence & Claims */}
      {item.supporting_evidence && item.supporting_evidence.length > 0 && (
        <div className="text-[11px] text-slate-400 space-y-1">
          <div className="font-mono text-[10px] uppercase text-slate-400">Supporting Evidence</div>
          <ul className="list-disc list-inside space-y-0.5 text-slate-300">
            {item.supporting_evidence.slice(0, 3).map((ev, idx) => (
              <li key={idx} className="truncate" title={ev}>
                {ev}
              </li>
            ))}
          </ul>
        </div>
      )}

      {item.source_url && (
        <div className="text-[10px] font-mono text-cyan-400/90 flex items-center gap-1 truncate">
          <ExternalLink className="w-3 h-3 flex-shrink-0" />
          <span className="truncate">{item.source_url}</span>
        </div>
      )}

      {/* Archetype Causal & Behavioural Hypotheses */}
      {(item.analytical_question ||
        item.stakeholder_focus ||
        item.causal_mechanism ||
        item.strongest_supporting_consideration ||
        item.strongest_opposing_consideration ||
        (item.expected_stakeholder_actions && item.expected_stakeholder_actions.length > 0) ||
        (item.emergent_consequences && item.emergent_consequences.length > 0) ||
        (item.invalidation_conditions && item.invalidation_conditions.length > 0)) && (
        <div className="p-2.5 rounded-lg bg-slate-900/70 border border-slate-800/80 space-y-1.5 text-[11px]">
          <div className="flex items-center justify-between text-[10px] font-mono uppercase text-cyan-400">
            <span>Archetype Differentiation &amp; Causal Hypothesis</span>
            {item.time_horizon && (
              <span className="text-slate-400">{item.time_horizon.replace(/_/g, ' ')}</span>
            )}
          </div>
          {item.analytical_question && (
            <p className="text-cyan-200/95 leading-snug">
              <strong className="text-cyan-400">Question:</strong> {item.analytical_question}
            </p>
          )}
          {item.stakeholder_focus && (
            <p className="text-violet-200/95 leading-snug">
              <strong className="text-violet-400">Stakeholder Focus:</strong>{' '}
              {item.stakeholder_focus}
            </p>
          )}
          {item.causal_mechanism && (
            <p className="text-slate-200 leading-snug">
              <strong className="text-slate-400">Mechanism:</strong> {item.causal_mechanism}
            </p>
          )}
          {(item.strongest_supporting_consideration || item.strongest_opposing_consideration) && (
            <div className="grid grid-cols-1 gap-1 pt-0.5">
              {item.strongest_supporting_consideration && (
                <div className="text-emerald-200/90">
                  <strong className="text-emerald-400">Strongest Support:</strong>{' '}
                  {item.strongest_supporting_consideration}
                </div>
              )}
              {item.strongest_opposing_consideration && (
                <div className="text-rose-200/90">
                  <strong className="text-rose-400">Strongest Opposing:</strong>{' '}
                  {item.strongest_opposing_consideration}
                </div>
              )}
            </div>
          )}
          {item.expected_stakeholder_actions && item.expected_stakeholder_actions.length > 0 && (
            <div className="text-slate-300">
              <strong className="text-slate-400">Likely Actions:</strong>{' '}
              {item.expected_stakeholder_actions.slice(0, 2).join('; ')}
            </div>
          )}
          {item.emergent_consequences && item.emergent_consequences.length > 0 && (
            <div className="text-amber-200/90">
              <strong className="text-amber-400">Emergent Consequence (Hyp.):</strong>{' '}
              {item.emergent_consequences.slice(0, 2).join('; ')}
            </div>
          )}
          {item.invalidation_conditions && item.invalidation_conditions.length > 0 && (
            <div className="text-emerald-200/90">
              <strong className="text-emerald-400">Invalidation Criteria:</strong>{' '}
              {item.invalidation_conditions.slice(0, 2).join('; ')}
            </div>
          )}
        </div>
      )}

      {/* Separated Metrics Footer */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 text-[10px] font-mono text-slate-400 pt-2 border-t border-slate-800">
        <span title="Synthetic model self-estimate (NOT verified ground truth)">
          LLM Cred (Synth): <strong className="text-cyan-300">{Math.round(llmCred * 100)}%</strong>
        </span>
        <span>
          Grounding:{' '}
          <strong className="text-emerald-300">
            {evGround != null ? `${Math.round(evGround * 100)}%` : 'N/A'}
          </strong>
        </span>
        <span>
          Trans. Wt:{' '}
          <strong className="text-indigo-300">
            {item.transmission_weight != null ? item.transmission_weight.toFixed(2) : 'N/A'}
          </strong>
        </span>
        <span>
          Virality: <strong className="text-amber-300">{Math.round(item.virality * 100)}%</strong>
        </span>
        <span>
          Emotion:{' '}
          <strong className="text-purple-300">{Math.round(item.emotional_charge * 100)}%</strong>
        </span>
        <span>
          Uncertainty:{' '}
          <strong className="text-slate-200">{Math.round((item.uncertainty ?? 0.3) * 100)}%</strong>
        </span>
      </div>
    </div>
  );
};

export const NarrativeAnalysis: React.FC = () => {
  const { currentSimulation, selectedIteration, setSelectedIteration, recentSessions } =
    useSimulation();
  const [selectedArchetype, setSelectedArchetype] = useState<string>('all');
  const [reportError, setReportError] = useState<string | null>(null);
  const [reportNotice, setReportNotice] = useState<string | null>(null);

  if (!currentSimulation) {
    return (
      <div className="py-16 text-center space-y-4">
        <MessageSquareText className="w-12 h-12 text-slate-600 mx-auto" />
        <h2 className="text-lg font-bold text-slate-200">No Simulation Active</h2>
        <p className="text-xs text-slate-400 max-w-sm mx-auto">
          Launch a new simulation or open a past run to explore multi-model narrative
          interpretations.
        </p>
      </div>
    );
  }

  const iterations = currentSimulation.iterations || [];
  const currentIterData =
    iterations.find((it) => it.iteration === selectedIteration) || iterations[iterations.length - 1];

  const interps = currentIterData?.micro_interpretations || [];
  const groupedEntries = groupByArchetypeAndSlot(interps, currentSimulation);

  const filteredEntries =
    selectedArchetype === 'all'
      ? groupedEntries
      : groupedEntries.filter((entry) => entry.archetypeKey === selectedArchetype);

  // Stance comparison bar chart strictly over valid outputs
  const comparisonChartData = groupedEntries.map((entry) => ({
    name: entry.name,
    stanceA: isValidInterpretation(entry.modelA)
      ? Number((entry.modelA?.stance ?? 0).toFixed(2))
      : null,
    stanceB: isValidInterpretation(entry.modelB)
      ? Number((entry.modelB?.stance ?? 0).toFixed(2))
      : null,
  }));

  const validTagsOnly = interps.filter((i) => isValidInterpretation(i));
  const allTags = Array.from(new Set(validTagsOnly.flatMap((i) => i.narrative_tags || [])));

  const agrScore = currentSimulation.model_agreement_score;
  const agrDetails = currentSimulation.model_agreement_details;
  const convScore = currentSimulation.narrative_convergence_score;
  const convDetails = currentSimulation.narrative_convergence_details;

  const isPersistedRecord = Boolean(
    currentSimulation.simulation_id &&
      recentSessions.some(
        (s) =>
          s.simulation_id === currentSimulation.simulation_id ||
          s.session_id === currentSimulation.simulation_id
      )
  );
  const isUnpersistedMock = Boolean(
    currentSimulation.diagnostics?.is_mock_only && !isPersistedRecord
  );

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
            <MessageSquareText className="w-6 h-6 text-cyan-400" />
            <span>Multi-Model Narrative Analysis</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Side-by-side comparison of{' '}
            <code className="text-cyan-300">{currentSimulation.config?.model_a || 'Model A'}</code>{' '}
            vs{' '}
            <code className="text-violet-300">{currentSimulation.config?.model_b || 'Model B'}</code>{' '}
            across 5 cognitive archetypes.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 self-start md:self-auto">
          {currentSimulation.simulation_id && iterations.length > 0 && (
            <ReportDownloadButton
              simulationId={currentSimulation.simulation_id}
              disabled={isUnpersistedMock}
              disabledReason="Deterministic mock test runs are excluded from database storage; run a persisted simulation to generate PDF reports."
              onError={(msg) => {
                setReportNotice(null);
                setReportError(msg);
              }}
              onSuccess={({ filename, cached }) => {
                setReportError(null);
                setReportNotice(
                  cached
                    ? `Downloaded PDF research report: ${filename}`
                    : `Generated & downloaded PDF research report: ${filename}`
                );
              }}
            />
          )}

          {/* Iteration Selector */}
          {iterations.length > 0 && (
            <div className="flex items-center gap-1.5 bg-slate-900 p-1.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 px-2">Iteration:</span>
              {iterations.map((it) => (
                <button
                  key={it.iteration}
                  onClick={() => setSelectedIteration(it.iteration)}
                  className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition ${
                    selectedIteration === it.iteration
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Round {it.iteration}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {reportError && (
        <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/70 text-xs text-rose-200 flex items-center justify-between gap-3">
          <span>{reportError}</span>
          <button
            onClick={() => setReportError(null)}
            className="text-rose-400 hover:text-rose-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {reportNotice && (
        <div className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-800/70 text-xs text-emerald-200 flex items-center justify-between gap-3">
          <span>{reportNotice}</span>
          <button
            onClick={() => setReportNotice(null)}
            className="text-emerald-400 hover:text-emerald-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Agreement Unavailable / Partial / Stance-Without-Reasoning / Collapse Warnings */}
      {agrScore === null && agrDetails?.reason && (
        <div className="p-3.5 rounded-xl bg-amber-950/35 border border-amber-800/60 text-xs text-amber-200 flex items-center gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span>{agrDetails.reason}</span>
        </div>
      )}

      {(agrDetails?.stance_without_reasoning_warning || convDetails?.archetype_collapse_warning) && (
        <div className="p-3.5 rounded-xl bg-amber-950/30 border border-amber-800/60 text-xs text-amber-200 space-y-1">
          {agrDetails?.stance_without_reasoning_warning && (
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <span>
                <strong>Reasoning Divergence Diagnostic:</strong> Models show high stance
                agreement despite low overlap in causal mechanisms and narrative themes.
              </span>
            </div>
          )}
          {convDetails?.archetype_collapse_warning && (
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <span>
                <strong>Archetype Collapse Warning:</strong> Distinct archetypes converged onto
                near-identical stances AND uniform causal/stakeholder reasoning in the final iteration.
              </span>
            </div>
          )}
        </div>
      )}

      {convDetails?.legitimate_consensus_detected && !convDetails?.archetype_collapse_warning && (
        <div className="p-3.5 rounded-xl bg-emerald-950/25 border border-emerald-800/60 text-xs text-emerald-200 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span>
            <strong>Legitimate Cross-Archetype Consensus:</strong> Directional stances aligned across archetypes while causal mechanisms and stakeholder perspectives remained distinctly differentiated (
            {convDetails?.reasoning_differentiation_score != null
              ? `Diff=${Math.round(convDetails.reasoning_differentiation_score * 100)}%`
              : 'differentiated'}
            ).
          </span>
        </div>
      )}

      {/* Narrative Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard
          label="Model Agreement Score"
          value={agrScore !== null && agrScore !== undefined ? `${Math.round(agrScore * 100)}%` : 'N/A'}
          badge={
            agrDetails?.valid_pairs != null
              ? `${agrDetails.valid_pairs}/${agrDetails.total_archetypes || 5} Pairs`
              : undefined
          }
          subtext={
            agrDetails?.available
              ? `Stance: ${Math.round((agrDetails.stance_agreement ?? 0) * 100)}% • r=${
                  agrDetails.stance_correlation ?? 'N/A'
                } • CausalSim: ${Math.round(
                  (agrDetails.causal_semantic_similarity ?? agrDetails.tag_semantic_similarity ?? 0) * 100
                )}%`
              : agrDetails?.reason || 'Requires matched valid (Model A, Model B) pairs'
          }
          icon={Scale}
          accent={agrScore !== null ? 'cyan' : 'amber'}
        />
        <MetricCard
          label="Narrative Convergence"
          value={
            convScore !== null && convScore !== undefined ? `${Math.round(convScore * 100)}%` : 'N/A'
          }
          badge={convDetails?.direction ? String(convDetails.direction).replace(/_/g, ' ') : undefined}
          subtext={
            convDetails?.available
              ? `Var: ${convDetails.var_initial ?? '?'} → ${convDetails.var_final ?? '?'}${
                  convDetails.reasoning_differentiation_score != null
                    ? ` • Diff: ${Math.round(convDetails.reasoning_differentiation_score * 100)}%`
                    : ''
                }`
              : convDetails?.reason || 'Requires >=2 valid iterations'
          }
          icon={Layers}
          accent="violet"
        />
        <MetricCard
          label="Dominant Narrative"
          value={
            (currentSimulation.dominant_narrative || 'No valid consensus').length > 42
              ? `${currentSimulation.dominant_narrative.slice(0, 42)}...`
              : currentSimulation.dominant_narrative || 'No valid consensus'
          }
          tooltip={currentSimulation.dominant_narrative}
          subtext="Highest grounding-weighted framing (valid outputs only)"
          icon={Sparkles}
          accent="amber"
        />
      </div>

      {/* Model A vs Model B Stance Comparison Chart */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Archetype Stance Comparison: Model A vs Model B
            </h3>
            <p className="text-[11px] text-slate-400">
              Stance range: -1.0 (strongly skeptical/debunking) to +1.0 (strongly believing/amplifying). Failed slots are omitted.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-cyan-400">
              <span className="w-2.5 h-2.5 rounded bg-cyan-400" />{' '}
              Model A ({currentSimulation.config?.model_a || 'Primary'})
            </span>
            <span className="flex items-center gap-1.5 text-violet-400">
              <span className="w-2.5 h-2.5 rounded bg-violet-400" />{' '}
              Model B ({currentSimulation.config?.model_b || 'Reviewer'})
            </span>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="name" stroke="#64748B" fontSize={11} />
              <YAxis stroke="#64748B" fontSize={11} domain={[-1, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0F172A',
                  borderColor: '#1E293B',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Bar dataKey="stanceA" name="Model A Stance" fill="#38BDF8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="stanceB" name="Model B Stance" fill="#818CF8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Narrative Tags Cloud */}
      {allTags.length > 0 && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Tag className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Identified Narrative Tags & Conceptual Themes (Valid Interpretations)
            </h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {allTags.map((tag, idx) => (
              <span
                key={idx}
                className="px-3 py-1 rounded-full text-xs font-mono bg-slate-950 border border-slate-800 text-cyan-300 hover:border-cyan-500/40 transition"
              >
                #{tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Side-by-Side Comparison for Each Archetype */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>Independent Interpretations by Cognitive Archetype</span>
          </h3>

          <select
            value={selectedArchetype}
            onChange={(e) => setSelectedArchetype(e.target.value)}
            className="rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 px-3 py-1.5 outline-none"
          >
            <option value="all">All Archetypes ({groupedEntries.length})</option>
            {groupedEntries.map((g) => (
              <option key={g.archetypeKey} value={g.archetypeKey}>
                {g.name}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-4">
          {filteredEntries.map((entry) => {
            const bothValid =
              isValidInterpretation(entry.modelA) && isValidInterpretation(entry.modelB);
            const delta = bothValid
              ? Math.abs((entry.modelA?.stance ?? 0) - (entry.modelB?.stance ?? 0))
              : null;

            return (
              <div
                key={entry.archetypeKey}
                className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4"
              >
                <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                  <ArchetypeBadge archetype={entry.archetypeKey} size="lg" />
                  <div className="text-xs font-mono text-slate-400">
                    Stance Delta:{' '}
                    {delta !== null ? (
                      <span
                        className={
                          delta < 0.25 ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'
                        }
                      >
                        {delta.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-rose-400 font-semibold">N/A (Incomplete pair)</span>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ModelInterpretationBox
                    slotLabel="Model A"
                    configuredModel={currentSimulation.config?.model_a}
                    accent="cyan"
                    item={entry.modelA}
                  />
                  <ModelInterpretationBox
                    slotLabel="Model B"
                    configuredModel={currentSimulation.config?.model_b}
                    accent="violet"
                    item={entry.modelB}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <CrossLayerExplanationPanel
        simulation={currentSimulation}
        selectedIteration={selectedIteration}
        highlightLayer="micro"
      />
    </div>
  );
};
