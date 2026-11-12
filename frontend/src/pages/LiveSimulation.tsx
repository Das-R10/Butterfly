// frontend/src/pages/LiveSimulation.tsx

import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSimulation } from '../context/SimulationContext';
import { ArchetypeBadge } from '../components/common/ArchetypeBadge';
import { MetricCard } from '../components/common/MetricCard';
import { AttentionAdoptionNotice } from '../components/common/AttentionAdoptionNotice';
import { CrossLayerExplanationPanel } from '../components/common/CrossLayerExplanationPanel';
import { ReportDownloadButton } from '../components/common/ReportDownloadButton';
import { FinalVerdictPanel } from '../components/forecast/FinalVerdictPanel';
import {
  partitionInterpretationsBySlot,
  isValidInterpretation,
  getStatusLabel,
  getExecutionStatusBadge,
} from '../utils/modelSlots';
import {
  Activity,
  AlertCircle,
  Terminal,
  UserCheck,
  Share2,
  TrendingUp,
  RefreshCw,
  Gauge,
  Archive,
  Play,
  AlertTriangle,
  ShieldAlert,
  Clock,
  X,
  Target,
  ArrowRight,
} from 'lucide-react';
import {
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  AreaChart,
  Area,
} from 'recharts';

export const LiveSimulation: React.FC = () => {
  const navigate = useNavigate();
  const {
    currentSimulation,
    activeSimulationId,
    activeJobId,
    isSimulating,
    isReadOnlyArchive,
    executionLogs,
    simulationError,
    selectedIteration,
    setSelectedIteration,
    recentSessions,
    resetForNewSimulation,
  } = useSimulation();

  const [activeLayerTab, setActiveLayerTab] = useState<
    'micro' | 'meso' | 'macro' | 'feedback' | 'verdict'
  >('micro');
  const [selectedModel, setSelectedModel] = useState<'all' | 'model_a' | 'model_b'>('all');
  const [reportNotice, setReportNotice] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const logContainerRef = useRef<HTMLDivElement>(null);

  // Strict simulation identity check: never render stale outputs when a new simulation is running
  const displaySim =
    !isSimulating || !activeSimulationId || currentSimulation?.simulation_id === activeSimulationId
      ? currentSimulation
      : null;

  // Auto-scroll execution logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [executionLogs]);

  // Empty initial / new workspace view
  if (!displaySim && !isSimulating) {
    return (
      <div className="py-16 text-center space-y-5 animate-fade-in">
        <Activity className="w-12 h-12 text-slate-600 mx-auto" />
        <div className="space-y-2">
          <h2 className="text-lg font-bold text-slate-200">No Active Simulation Selected</h2>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Start a fresh isolated simulation or inspect a completed record from Past Simulations.
          </p>
        </div>
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => {
              resetForNewSimulation();
              navigate('/new');
            }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold transition"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Configure New Simulation</span>
          </button>
          <button
            onClick={() => navigate('/history')}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition"
          >
            <Archive className="w-3.5 h-3.5" />
            <span>Browse Past Simulations</span>
          </button>
        </div>
      </div>
    );
  }

  // Derive active iteration data
  const iterations = displaySim?.iterations || [];
  const currentIterIndex = Math.min(
    Math.max(0, selectedIteration - 1),
    Math.max(0, iterations.length - 1)
  );
  const iterData = iterations[currentIterIndex];

  const interps = iterData?.micro_interpretations || [];
  const { modelA, modelB, slotByItem } = partitionInterpretationsBySlot(interps, displaySim);
  const filteredInterps =
    selectedModel === 'model_a'
      ? modelA
      : selectedModel === 'model_b'
      ? modelB
      : interps;

  // Build trajectory points cleanly supporting both `.iterations` (v4.1+) and `.steps` (legacy)
  const traj = displaySim?.trajectory;
  const trajSteps =
    traj?.iterations && traj.iterations.length > 0
      ? traj.iterations
      : traj?.steps && traj.steps.length > 0
      ? traj.steps
      : iterations.map((it) => it.iteration);

  const trajectoryChartData = trajSteps.map((step, idx) => ({
    iteration: `Round ${step}`,
    cumulativeAdoption: traj?.cumulative_adoption?.[idx] ?? iterations[idx]?.macro_state?.cumulative_adoption ?? 0,
    activeAdoption: traj?.active_adoption?.[idx] ?? iterations[idx]?.macro_state?.active_adoption ?? 0,
    attentionIndicator: traj?.attention_indicator?.[idx] ?? iterations[idx]?.macro_state?.attention_indicator ?? 0,
    adoptionVelocity: traj?.adoption_velocity?.[idx] ?? iterations[idx]?.macro_state?.adoption_velocity ?? 0,
  }));

  const simStatus = isSimulating ? 'running' : displaySim?.status || 'completed';
  const statusDotClass =
    simStatus === 'running' || simStatus === 'queued'
      ? 'bg-cyan-400 animate-pulse'
      : simStatus === 'failed'
      ? 'bg-rose-400'
      : simStatus === 'partially_completed'
      ? 'bg-amber-400'
      : 'bg-emerald-400';

  const hasLegacyOverflow =
    iterData?.meso_state?.nodes?.some((n) => (n.current_activation ?? 0) > 1.0001) ?? false;

  const isPersistedRecord = Boolean(
    displaySim?.simulation_id &&
      recentSessions.some(
        (s) =>
          s.simulation_id === displaySim.simulation_id || s.session_id === displaySim.simulation_id
      )
  );
  const isUnpersistedMock = Boolean(
    displaySim?.diagnostics?.is_mock_only && !isPersistedRecord
  );

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Read-Only Archived Banner */}
      {isReadOnlyArchive && displaySim && (
        <div className="rounded-xl border border-indigo-800/60 bg-indigo-950/30 px-4 py-3 text-xs text-indigo-200 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <Archive className="w-4 h-4 text-indigo-400 flex-shrink-0" />
            <span>
              <strong>Read-Only Archived Results:</strong> Inspecting persisted simulation{' '}
              <code className="font-mono text-indigo-300">{displaySim.simulation_id}</code>. No
              model inference is running.
            </span>
          </div>
          <button
            onClick={() => navigate('/history')}
            className="text-[11px] font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-700 hover:border-indigo-500 text-indigo-300 transition flex-shrink-0"
          >
            Past Simulations
          </button>
        </div>
      )}

      {/* Simulation Header / Status Banner */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1.5 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${statusDotClass}`} />
            <span className="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold">
              {isSimulating
                ? 'Live Simulation Executing'
                : `Simulation ${getStatusLabel(simStatus)}`}
            </span>
            {(displaySim?.simulation_id || activeSimulationId) && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                ID: {displaySim?.simulation_id || activeSimulationId}
              </span>
            )}
            {activeJobId && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-950 text-slate-400 border border-slate-800">
                Job: {activeJobId}
              </span>
            )}
            {displaySim?.expected_interpretations_count != null && (
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-950 text-emerald-400 border border-slate-800">
                {displaySim.successful_interpretations_count ?? 0}/
                {displaySim.expected_interpretations_count} Valid Interps
              </span>
            )}
          </div>
          <h1 className="text-xl font-bold text-slate-100 truncate">
            {displaySim
              ? displaySim.event_title || displaySim.signal_text
              : `Initializing isolated simulation ${activeSimulationId || ''}...`}
          </h1>
          {displaySim?.event_title && (
            <p className="text-xs text-slate-400 line-clamp-2">{displaySim.signal_text}</p>
          )}
        </div>

        {/* Actions & Iteration Selector Pill Buttons */}
        <div className="flex flex-wrap items-center gap-2.5 self-start md:self-auto flex-shrink-0">
          {!isSimulating && displaySim?.simulation_id && iterations.length > 0 && (
            <>
              <button
                type="button"
                onClick={() => navigate('/forecast')}
                className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-cyan-500/15 hover:bg-cyan-500 hover:text-slate-950 text-cyan-300 border border-cyan-500/40 text-xs font-semibold transition"
              >
                <Target className="w-3.5 h-3.5" />
                <span>Forecast &amp; Verdict</span>
              </button>
              <ReportDownloadButton
              simulationId={displaySim.simulation_id}
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
            </>
          )}

          {iterations.length > 0 && (
            <div className="flex items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 px-2">Cycle:</span>
              {iterations.map((it) => (
                <button
                  key={it.iteration}
                  onClick={() => setSelectedIteration(it.iteration)}
                  className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition ${
                    selectedIteration === it.iteration
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                  }`}
                >
                  Step {it.iteration}
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

      {/* Explicit Error / Partial Completion Warning */}
      {(simulationError || (displaySim?.failed_models && displaySim.failed_models.length > 0)) && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/35 text-rose-200 text-xs space-y-2">
          <div className="flex items-center gap-2 font-semibold text-rose-300">
            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>
              {simulationError ||
                displaySim?.status_details ||
                'Some model inferences encountered execution errors.'}
            </span>
          </div>
          {displaySim?.failed_models && displaySim.failed_models.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
              {displaySim.failed_models.slice(0, 4).map((f, idx) => (
                <div
                  key={idx}
                  className="p-2 rounded bg-black/30 border border-rose-900/50 font-mono text-[10px] text-rose-300"
                >
                  Iter {f.iteration} • {String(f.model_slot || '').toUpperCase()} ({f.model_name}) •{' '}
                  {f.archetype}: <strong>{f.error_category || f.execution_status}</strong>
                  {f.error_message ? ` — ${String(f.error_message).slice(0, 120)}` : ''}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Live Terminal Progress Log (stays populated while active or on error) */}
      {(isSimulating || executionLogs.length > 0) && !isReadOnlyArchive && (
        <div className="bg-slate-950 border border-slate-800/90 rounded-xl overflow-hidden shadow-lg">
          <div className="bg-slate-900 px-4 py-2 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
              <Terminal className="w-4 h-4" />
              <span>Real-Time Pipeline Execution Log</span>
            </div>
            <div className="flex items-center gap-1.5">
              {isSimulating ? (
                <>
                  <div className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                  <span className="text-[10px] font-mono text-cyan-300">Streaming Active</span>
                </>
              ) : (
                <span className="text-[10px] font-mono text-slate-400">Execution Log Complete</span>
              )}
            </div>
          </div>
          <div
            ref={logContainerRef}
            className="p-4 max-h-44 overflow-y-auto font-mono text-[11px] text-slate-300 space-y-1 bg-black/40"
          >
            {executionLogs.length === 0 ? (
              <div className="text-slate-500">Connecting to simulation stream...</div>
            ) : (
              executionLogs.map((log, idx) => (
                <div key={idx} className="leading-relaxed">
                  <span
                    className={
                      log.includes('[COMPLETED]') || log.includes('[COMPLETE]')
                        ? 'text-emerald-400 font-bold'
                        : log.includes('[ERROR]') || log.includes('[FAILED]')
                        ? 'text-rose-400 font-semibold'
                        : log.includes('[PARTIAL')
                        ? 'text-amber-300 font-semibold'
                        : log.includes('[STREAM]')
                        ? 'text-cyan-300'
                        : 'text-slate-300'
                    }
                  >
                    {log}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* While actively simulating and Round 1 not finished yet, pause Layer tabs cleanly */}
      {isSimulating && !displaySim ? (
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 text-center space-y-3">
          <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto" />
          <div className="text-sm font-semibold text-slate-200">
            Executing Isolated Simulation Pipeline ({activeSimulationId})
          </div>
          <p className="text-xs text-slate-400 max-w-lg mx-auto">
            Sequential dual-model local inference is evaluating all 5 archetypes across iterations.
            Results will appear automatically once persisted by the backend.
          </p>
        </div>
      ) : (
        <>
          <AttentionAdoptionNotice />

          {/* Layer Navigation Tabs */}
          <div className="flex items-center gap-2 border-b border-slate-800 pb-2 overflow-x-auto">
            <button
              onClick={() => setActiveLayerTab('micro')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeLayerTab === 'micro'
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <UserCheck className="w-4 h-4" />
              <span>Micro Layer ({interps.length} Slots)</span>
            </button>

            <button
              onClick={() => setActiveLayerTab('meso')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeLayerTab === 'meso'
                  ? 'bg-violet-500/10 text-violet-400 border border-violet-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Share2 className="w-4 h-4" />
              <span>Meso Layer (Influence Network)</span>
            </button>

            <button
              onClick={() => setActiveLayerTab('macro')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeLayerTab === 'macro'
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span>Macro Layer (Bass Diffusion Trajectory)</span>
            </button>

            <button
              onClick={() => setActiveLayerTab('feedback')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeLayerTab === 'feedback'
                  ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <RefreshCw className="w-4 h-4" />
              <span>Feedback Layer (Recurrent State)</span>
            </button>

            <button
              onClick={() => setActiveLayerTab('verdict')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition ${
                activeLayerTab === 'verdict'
                  ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Target className="w-4 h-4" />
              <span>Forecast &amp; Verdict</span>
            </button>
          </div>

          {/* LAYER 1: MICRO LAYER VIEW */}
          {activeLayerTab === 'micro' && (
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-slate-900/60 p-3.5 rounded-xl border border-slate-800">
                <div>
                  <div className="text-xs font-bold text-slate-200">
                    Iteration {selectedIteration} Archetype Cognitive Matrix
                  </div>
                  <div className="text-[11px] text-slate-400">
                    5 behavioral archetypes evaluated independently by Model A (
                    <code className="text-cyan-300">{displaySim?.config?.model_a || 'Primary'}</code>) and Model B (
                    <code className="text-violet-300">{displaySim?.config?.model_b || 'Reviewer'}</code>).
                  </div>
                </div>

                {/* Model Filter */}
                <div className="flex items-center gap-1.5 self-start sm:self-auto">
                  <span className="text-[10px] font-mono text-slate-400">View:</span>
                  <button
                    onClick={() => setSelectedModel('all')}
                    className={`px-2.5 py-1 rounded text-xs font-mono transition ${
                      selectedModel === 'all'
                        ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    All ({interps.length})
                  </button>
                  <button
                    onClick={() => setSelectedModel('model_a')}
                    className={`px-2.5 py-1 rounded text-xs font-mono transition ${
                      selectedModel === 'model_a'
                        ? 'bg-cyan-950/60 text-cyan-400 border border-cyan-800'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Model A ({modelA.length})
                  </button>
                  <button
                    onClick={() => setSelectedModel('model_b')}
                    className={`px-2.5 py-1 rounded text-xs font-mono transition ${
                      selectedModel === 'model_b'
                        ? 'bg-violet-950/60 text-violet-400 border border-violet-800'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Model B ({modelB.length})
                  </button>
                </div>
              </div>

              {filteredInterps.length === 0 ? (
                <div className="py-12 text-center text-slate-500 text-xs">
                  No micro interpretations available for this filter.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {filteredInterps.map((interp, idx) => {
                    const slot = slotByItem.get(interp) || 'model_a';
                    const valid = isValidInterpretation(interp);
                    const statusBadge = getExecutionStatusBadge(interp.execution_status);
                    const llmCred = interp.model_estimated_credibility ?? interp.credibility;
                    const evGround = interp.evidence_grounding_score;

                    return (
                      <div
                        key={`${interp.archetype}-${slot}-${idx}`}
                        className={`bg-slate-900/70 border rounded-xl p-4 space-y-3 transition ${
                          valid
                            ? 'border-slate-800 hover:border-slate-700'
                            : 'border-rose-800/70 bg-rose-950/15'
                        }`}
                      >
                        {/* Header: Archetype + Slot + Status + Model Badge */}
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <ArchetypeBadge archetype={interp.archetype} />
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span
                              className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                                slot === 'model_a'
                                  ? 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60'
                                  : 'bg-violet-950/60 text-violet-300 border-violet-800/60'
                              }`}
                            >
                              {slot === 'model_a' ? 'Model A' : 'Model B'}: {interp.model_name}
                            </span>
                            <span
                              className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${statusBadge.className}`}
                            >
                              {statusBadge.label}
                            </span>
                          </div>
                        </div>

                        {!valid ? (
                          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 space-y-1.5 text-xs">
                            <div className="font-semibold text-rose-300 flex items-center gap-1.5">
                              <ShieldAlert className="w-4 h-4 text-rose-400" />
                              <span>
                                Model execution failed ({interp.error_category || interp.execution_status})
                              </span>
                            </div>
                            <p className="text-[11px] text-rose-200/90 font-mono break-words">
                              {interp.error_message ||
                                'No valid JSON response returned. Not substituted with mock or fallback output.'}
                            </p>
                            <div className="flex flex-wrap gap-3 text-[10px] font-mono text-rose-300/80 pt-1">
                              {interp.duration_sec != null && (
                                <span className="flex items-center gap-1">
                                  <Clock className="w-3 h-3" /> {interp.duration_sec.toFixed(1)}s
                                </span>
                              )}
                              {interp.retry_count != null && <span>Retries: {interp.retry_count}</span>}
                              {interp.finish_reason && <span>Finish: {interp.finish_reason}</span>}
                              {interp.response_length != null && (
                                <span>Len: {interp.response_length} chars</span>
                              )}
                            </div>
                          </div>
                        ) : (
                          <>
                            {/* Primary Narrative Quote */}
                            <p className="text-xs text-slate-200 leading-relaxed italic bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
                              "{interp.primary_narrative}"
                            </p>

                            {(interp.analytical_question || interp.stakeholder_focus) && (
                              <div className="text-[11px] bg-slate-950/50 border border-slate-800/70 rounded-lg p-2.5 space-y-1">
                                {interp.analytical_question && (
                                  <div className="text-cyan-200/90">
                                    <strong className="text-cyan-400">Focus Q:</strong>{' '}
                                    {interp.analytical_question}
                                  </div>
                                )}
                                {interp.stakeholder_focus && (
                                  <div className="text-violet-200/90">
                                    <strong className="text-violet-400">Stakeholders:</strong>{' '}
                                    {interp.stakeholder_focus}
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Epistemic Evidence Status Row */}
                            <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono">
                              <div className="flex items-center gap-1.5">
                                <span
                                  className={`px-2 py-0.5 rounded border uppercase ${
                                    interp.evidence_status === 'supported'
                                      ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60'
                                      : interp.evidence_status === 'unsupported'
                                      ? 'bg-rose-950/60 text-rose-400 border-rose-800/60'
                                      : 'bg-amber-950/60 text-amber-400 border-amber-800/60'
                                  }`}
                                >
                                  Evidence: {interp.evidence_status || 'inferred'}
                                </span>
                                {interp.evidence_penalty_applied != null &&
                                  interp.evidence_penalty_applied < 1.0 && (
                                    <span className="px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/60">
                                      Penalty {interp.evidence_penalty_applied.toFixed(2)}x
                                    </span>
                                  )}
                                {interp.risk_severity != null && (
                                  <span className="px-1.5 py-0.5 rounded bg-rose-950/50 text-rose-300 border border-rose-800/60">
                                    Risk Sev {(interp.risk_severity * 100).toFixed(0)}%
                                  </span>
                                )}
                              </div>
                              <div className="text-slate-400 flex items-center gap-2">
                                {evGround != null && (
                                  <span>
                                    Grounding:{' '}
                                    <strong className="text-emerald-400">
                                      {(evGround * 100).toFixed(0)}%
                                    </strong>
                                  </span>
                                )}
                                {interp.transmission_weight != null && (
                                  <span>
                                    T:{' '}
                                    <strong className="text-cyan-300">
                                      {interp.transmission_weight.toFixed(2)}
                                    </strong>
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Behavioral Dimension Meters */}
                            <div className="grid grid-cols-2 gap-3 pt-1">
                              <div className="space-y-1">
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-slate-400">Stance</span>
                                  <span
                                    className={
                                      interp.stance >= 0
                                        ? 'text-emerald-400 font-bold'
                                        : 'text-rose-400 font-bold'
                                    }
                                  >
                                    {interp.stance > 0
                                      ? `+${interp.stance.toFixed(2)}`
                                      : interp.stance.toFixed(2)}
                                  </span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden flex">
                                  <div
                                    className={`h-full transition-all ${
                                      interp.stance >= 0 ? 'bg-emerald-400 ml-auto' : 'bg-rose-400'
                                    }`}
                                    style={{ width: `${Math.min(Math.abs(interp.stance) * 50, 100)}%` }}
                                  />
                                </div>
                              </div>

                              <div className="space-y-1">
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-slate-400">Credibility (LLM Est.)</span>
                                  <span className="text-cyan-400 font-bold">
                                    {(llmCred * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-cyan-400 rounded-full transition-all"
                                    style={{ width: `${Math.min(llmCred * 100, 100)}%` }}
                                  />
                                </div>
                              </div>

                              <div className="space-y-1">
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-slate-400">Virality</span>
                                  <span className="text-amber-400 font-bold">
                                    {(interp.virality * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-amber-400 rounded-full transition-all"
                                    style={{ width: `${Math.min(interp.virality * 100, 100)}%` }}
                                  />
                                </div>
                              </div>

                              <div className="space-y-1">
                                <div className="flex justify-between text-[10px] font-mono">
                                  <span className="text-slate-400">Emotion / Uncert</span>
                                  <span className="text-purple-400 font-bold">
                                    {(interp.emotional_charge * 100).toFixed(0)}% /{' '}
                                    {((interp.uncertainty ?? 0.3) * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-purple-400 rounded-full transition-all"
                                    style={{
                                      width: `${Math.min(interp.emotional_charge * 100, 100)}%`,
                                    }}
                                  />
                                </div>
                              </div>
                            </div>

                            {interp.narrative_tags && interp.narrative_tags.length > 0 && (
                              <div className="flex flex-wrap gap-1.5 pt-1">
                                {interp.narrative_tags.map((tag, tIdx) => (
                                  <span
                                    key={tIdx}
                                    className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800/80 text-slate-300 border border-slate-700/60"
                                  >
                                    #{tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* LAYER 2: MESO LAYER VIEW */}
          {activeLayerTab === 'meso' && (
            <div className="space-y-6">
              {hasLegacyOverflow && (
                <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-800/60 text-xs text-amber-200 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                  <span>
                    <strong>Legacy Pre-v4.1 Archived Session:</strong> Some raw node activations in
                    this archived JSON exceed 100% because capacity normalization was added in v4.1.
                    Progress bars below are visually capped at 100%.
                  </span>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                  label="External Influence Score"
                  value={(iterData?.meso_state?.external_influence_score ?? 0).toFixed(3)}
                  subtext="Aggregate broadcast media pressure"
                  icon={Share2}
                  accent="cyan"
                />
                <MetricCard
                  label="Internal Contagion Score"
                  value={(iterData?.meso_state?.internal_contagion_score ?? 0).toFixed(3)}
                  subtext="Peer-to-peer viral transmission"
                  icon={Activity}
                  accent="violet"
                />
                <MetricCard
                  label="Effective Spread Potential"
                  value={(iterData?.meso_state?.effective_spread_potential ?? 0).toFixed(3)}
                  subtext="Deduplicated propagation capacity"
                  icon={Gauge}
                  accent="emerald"
                />
              </div>

              {/* Node Activations Summary Table */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    Network Nodes & Activation Levels (Iteration {selectedIteration})
                  </h3>
                  <span className="text-[10px] font-mono text-slate-400">
                    Semantics: {iterData?.meso_state?.activation_semantics || 'instantaneous_per_iteration'} • Bounded [0.0, 1.0]
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {iterData?.meso_state?.nodes?.map((node) => {
                    const normalizedPct = Math.min(Math.max(node.current_activation * 100, 0), 100);
                    const isLegacyOverflow = node.current_activation > 1.0001;

                    return (
                      <div
                        key={node.id}
                        className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 flex items-center justify-between gap-3"
                      >
                        <div className="space-y-0.5 min-w-0">
                          <div className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                            <span className="truncate">{node.label}</span>
                            {isLegacyOverflow && (
                              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800">
                                Legacy Raw
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] font-mono text-slate-400">
                            Type: {node.node_type} | Base: {node.base_weight.toFixed(2)}
                            {node.raw_influence != null && (
                              <span className="text-violet-400 ml-2">
                                | Raw Inflow: {node.raw_influence.toFixed(3)}
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="text-right flex-shrink-0">
                          <div className="text-xs font-bold font-mono text-cyan-400">
                            {(node.current_activation * 100).toFixed(1)}%
                          </div>
                          <div className="w-20 h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1">
                            <div
                              className="h-full bg-cyan-400 rounded-full"
                              style={{ width: `${normalizedPct}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* LAYER 3: MACRO LAYER VIEW */}
          {activeLayerTab === 'macro' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <MetricCard
                  label="Cumulative Sim. Diffusion"
                  value={`${((iterData?.macro_state?.cumulative_adoption ?? 0) * 100).toFixed(1)}%`}
                  subtext="Bass cumulative proxy"
                  accent="cyan"
                />
                <MetricCard
                  label="Active Sim. Adoption"
                  value={`${((iterData?.macro_state?.active_adoption ?? 0) * 100).toFixed(1)}%`}
                  subtext="Simulated net of churn c"
                  accent="emerald"
                />
                <MetricCard
                  label="Sim. Attention Index"
                  value={`${((iterData?.macro_state?.attention_indicator ?? 0) * 100).toFixed(1)}%`}
                  subtext="Simulated media salience"
                  accent="amber"
                />
                <MetricCard
                  label="Velocity (dN/dt)"
                  value={`${(iterData?.macro_state?.adoption_velocity ?? 0).toFixed(4)}`}
                  subtext="Net Bass flow proxy"
                  accent="violet"
                />
                <MetricCard
                  label="Sim. Attention / Adoption"
                  value={`${(iterData?.macro_state?.attention_vs_adoption_ratio ?? 0).toFixed(2)}x`}
                  subtext="Salience-to-adoption ratio"
                  accent="rose"
                />
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                      Macro Simulated Trajectory Dynamics Across Iterations
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      Modified Bass Proxy Model: dN/dt = [p + q(N/M)](M-N) - cN (Simulated Proxies Only)
                    </p>
                  </div>
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="flex items-center gap-1.5 text-emerald-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Active Sim. Adoption
                    </span>
                    <span className="flex items-center gap-1.5 text-amber-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-amber-400" /> Sim. Attention Index
                    </span>
                  </div>
                </div>

                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trajectoryChartData}>
                      <defs>
                        <linearGradient id="colorAdopt" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10B981" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#10B981" stopOpacity={0.0} />
                        </linearGradient>
                        <linearGradient id="colorAttn" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#F59E0B" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="iteration" stroke="#64748B" fontSize={11} />
                      <YAxis stroke="#64748B" fontSize={11} domain={[0, 1]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0F172A',
                          borderColor: '#1E293B',
                          borderRadius: '8px',
                          fontSize: '12px',
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="activeAdoption"
                        name="Active Sim. Adoption (Bass Proxy)"
                        stroke="#10B981"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorAdopt)"
                      />
                      <Area
                        type="monotone"
                        dataKey="attentionIndicator"
                        name="Sim. Attention Index (Buzz)"
                        stroke="#F59E0B"
                        strokeWidth={2}
                        strokeDasharray="4 4"
                        fillOpacity={1}
                        fill="url(#colorAttn)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {iterData?.macro_state?.assumptions_and_limitations && (
                  <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 text-[11px] text-slate-400">
                    <strong className="text-slate-300">Model Boundary Note:</strong>{' '}
                    {iterData.macro_state.assumptions_and_limitations}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* LAYER 4: FEEDBACK LAYER VIEW */}
          {activeLayerTab === 'feedback' && (
            <div className="space-y-4">
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    Iteration {selectedIteration} Feedback State & Epistemic Audit
                  </h3>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950/60 text-amber-400 border border-amber-800">
                    Polarization: {((iterData?.feedback_state?.polarization_index ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>

                {iterData?.feedback_state?.amplification_warning && (
                  <div className="p-3.5 rounded-lg bg-rose-950/40 border border-rose-800/70 text-xs text-rose-200 flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-bold text-rose-300">Unverified Amplification Warning</div>
                      <div className="text-[11px] mt-0.5">
                        {iterData.feedback_state.warning_details}
                      </div>
                    </div>
                  </div>
                )}

                {iterData?.feedback_state?.iteration_change_summary && (
                  <div className="p-4 rounded-lg bg-indigo-950/25 border border-indigo-800/50 space-y-1.5">
                    <div className="text-[11px] font-mono text-indigo-300 uppercase">
                      What Changed Between Iterations
                    </div>
                    <p className="text-xs text-slate-200 leading-relaxed">
                      {iterData.feedback_state.iteration_change_summary}
                    </p>
                  </div>
                )}

                {iterData?.feedback_state?.claim_verification_audit && (
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center font-mono text-xs">
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Supported</div>
                      <div className="text-base font-bold text-emerald-400">
                        {iterData.feedback_state.claim_verification_audit.supported}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Inferred</div>
                      <div className="text-base font-bold text-amber-400">
                        {iterData.feedback_state.claim_verification_audit.inferred}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Unsupported</div>
                      <div className="text-base font-bold text-rose-400">
                        {iterData.feedback_state.claim_verification_audit.unsupported}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Valid Slots</div>
                      <div className="text-base font-bold text-cyan-400">
                        {iterData.feedback_state.claim_verification_audit.total_interpretations ??
                          interps.filter((m) => isValidInterpretation(m)).length}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Failed Slots</div>
                      <div className="text-base font-bold text-slate-300">
                        {iterData.feedback_state.claim_verification_audit.failed_interpretations ??
                          interps.filter((m) => !isValidInterpretation(m)).length}
                      </div>
                    </div>
                  </div>
                )}

                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="text-[11px] font-mono text-cyan-400 uppercase">
                    Macro Summary State
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed">
                    {iterData?.feedback_state?.macro_summary || 'No feedback state available.'}
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="text-[11px] font-mono text-violet-400 uppercase">
                    Dominant Social Framing
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed">
                    {iterData?.feedback_state?.dominant_framing || 'Consensus forming.'}
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="text-[11px] font-mono text-amber-400 uppercase">
                    Context Injected Into Next Cycle
                  </div>
                  <pre className="text-[11px] font-mono text-slate-300 bg-black/40 p-3 rounded overflow-x-auto whitespace-pre-wrap">
                    {iterData?.feedback_state?.prompt_context_for_next_round ||
                      'Final cycle completed.'}
                  </pre>
                </div>

                {/* Systemic State Evolution (v5+) */}
                {iterData?.feedback_state?.systemic_state &&
                  Object.keys(iterData.feedback_state.systemic_state).length > 0 && (
                    <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="text-[11px] font-mono text-cyan-400 uppercase">
                          Simulated Systemic State Proxies (Hypothesis Only)
                        </div>
                        <span className="text-[10px] font-mono text-slate-300 px-2 py-0.5 rounded bg-slate-900 border border-slate-700">
                          Posture: {iterData.feedback_state.systemic_state.institutional_posture || 'N/A'}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center font-mono text-xs">
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <div className="text-[10px] text-slate-400">Dampening Proxy</div>
                          <div className="font-bold text-emerald-400">
                            {((iterData.feedback_state.systemic_state.institutional_dampening_index ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <div className="text-[10px] text-slate-400">Public Press. Proxy</div>
                          <div className="font-bold text-amber-400">
                            {((iterData.feedback_state.systemic_state.public_pressure ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <div className="text-[10px] text-slate-400">Market Stress Proxy</div>
                          <div className="font-bold text-cyan-400">
                            {((iterData.feedback_state.systemic_state.economic_market_stress ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <div className="text-[10px] text-slate-400">Esc. Risk Proxy</div>
                          <div className="font-bold text-rose-400">
                            {((iterData.feedback_state.systemic_state.escalation_risk ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <div className="text-[10px] text-slate-400">Epistemic Fragility</div>
                          <div className="font-bold text-violet-400">
                            {((iterData.feedback_state.systemic_state.epistemic_fragility ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>

                      {iterData.feedback_state.why_changed && (
                        <div className="text-xs text-slate-300">
                          <strong className="text-cyan-300">Why Conditions Shifted:</strong>{' '}
                          {iterData.feedback_state.why_changed}
                        </div>
                      )}

                      {iterData.feedback_state.what_changed &&
                        iterData.feedback_state.what_changed.length > 0 && (
                          <div className="text-xs text-slate-300 space-y-1">
                            <div className="text-[10px] font-mono uppercase text-slate-400">
                              What Changed This Cycle
                            </div>
                            <ul className="list-disc list-inside space-y-0.5 text-[11px]">
                              {iterData.feedback_state.what_changed.map((w, idx) => (
                                <li key={idx}>{w}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        {iterData.feedback_state.new_consequences_emerged &&
                          iterData.feedback_state.new_consequences_emerged.length > 0 && (
                            <div className="p-3 rounded bg-slate-900/70 border border-slate-800 space-y-1">
                              <div className="text-[10px] font-mono uppercase text-amber-300">
                                Emergent Simulated Consequences
                              </div>
                              <ul className="list-disc list-inside space-y-0.5 text-[11px] text-slate-300">
                                {iterData.feedback_state.new_consequences_emerged.map((c, idx) => (
                                  <li key={idx}>{c}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        {iterData.feedback_state.preventive_conditions &&
                          iterData.feedback_state.preventive_conditions.length > 0 && (
                            <div className="p-3 rounded bg-slate-900/70 border border-slate-800 space-y-1">
                              <div className="text-[10px] font-mono uppercase text-emerald-300">
                                Preventive / Dampening Conditions
                              </div>
                              <ul className="list-disc list-inside space-y-0.5 text-[11px] text-slate-300">
                                {iterData.feedback_state.preventive_conditions.map((p, idx) => (
                                  <li key={idx}>{p}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          )}

          {/* FORECAST & FINAL VERDICT TAB */}
          {activeLayerTab === 'verdict' && (
            <div className="space-y-4">
              {displaySim?.final_verdict ? (
                <>
                  <FinalVerdictPanel
                    key={`live-verdict-${displaySim.simulation_id}`}
                    simulationId={displaySim.simulation_id}
                    verdict={displaySim.final_verdict}
                    epistemicBreakdown={displaySim.epistemic_breakdown}
                    showVerdictQuestions={true}
                    showEpistemicAndResolution={false}
                    isMockOnly={isUnpersistedMock}
                  />
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={() => navigate('/forecast')}
                      className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold transition"
                    >
                      <span>Open Full Causal Diagram, 4 Scenarios &amp; Outcome Tracker</span>
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </>
              ) : (
                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 text-center space-y-3">
                  <div className="text-sm font-semibold text-slate-200">
                    Forecast &amp; Verdict Available via Dedicated Synthesis View
                  </div>
                  <p className="text-xs text-slate-400 max-w-lg mx-auto">
                    Inspect multi-order causal butterfly pathways, four future scenarios, stakeholder
                    impact cascades, and the 8-question Final Verdict.
                  </p>
                  <button
                    type="button"
                    onClick={() => navigate('/forecast')}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold transition"
                  >
                    <span>Open Forecast &amp; Verdict Workspace</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Cross-Layer Explanation & Model Limitations Panel */}
          <CrossLayerExplanationPanel
            simulation={displaySim}
            selectedIteration={selectedIteration}
            highlightLayer={activeLayerTab === 'verdict' ? 'feedback' : activeLayerTab}
          />
        </>
      )}
    </div>
  );
};
