// frontend/src/pages/Overview.tsx

import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useSimulation } from '../context/SimulationContext';
import { MetricCard } from '../components/common/MetricCard';
import { AttentionAdoptionNotice } from '../components/common/AttentionAdoptionNotice';
import { getStatusLabel } from '../utils/modelSlots';
import {
  Play,
  Activity,
  Layers,
  ArrowRight,
  Database,
  Cpu,
  Sparkles,
  Target,
} from 'lucide-react';

export const Overview: React.FC = () => {
  const navigate = useNavigate();
  const {
    currentSimulation,
    recentSessions,
    backendConnected,
    backendVersion,
    loadSession,
    resetForNewSimulation,
  } = useSimulation();

  const handleOpenSession = async (sessionId: string) => {
    try {
      await loadSession(sessionId);
      navigate('/live');
    } catch {
      // ignore
    }
  };

  const handleLaunchNew = () => {
    resetForNewSimulation();
    navigate('/new');
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 p-6 md:p-8 shadow-xl">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 rounded-full bg-cyan-500/5 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 -mb-16 w-60 h-60 rounded-full bg-violet-500/5 blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-3 max-w-3xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/50 text-cyan-400 text-xs font-mono">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Butterfly-Effect Consequence Forecasting &amp; Behavioral Diffusion Engine</span>
            </div>
            <h1 className="text-2xl md:text-4xl font-extrabold tracking-tight text-slate-100">
              The Butterfly Algorithm
            </h1>
            <p className="text-sm md:text-base text-slate-300 leading-relaxed">
              Given an initial event, forecast how interconnected behavioural, social, economic,
              political, and institutional feedback loops produce multi-order consequences, four
              bounded scenarios, and a reasoned Final Verdict.
            </p>
            <div className="p-3 rounded-xl bg-slate-950/70 border border-cyan-900/50 text-xs text-cyan-100/90 flex items-start gap-2.5">
              <Target className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-mono text-[10px] uppercase text-cyan-400 block font-bold">
                  Core Research Question
                </span>
                <span className="italic">
                  &ldquo;Given this initial event and the information currently available, what are
                  the most plausible consequences that could emerge over time through interconnected
                  behavioural and systemic effects?&rdquo;
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row md:flex-col items-stretch gap-2.5 flex-shrink-0">
            <button
              onClick={handleLaunchNew}
              className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-sm font-bold shadow-lg shadow-cyan-500/20 transition-all duration-150 transform hover:-translate-y-0.5"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Launch New Simulation</span>
            </button>
            <button
              onClick={() => navigate('/forecast')}
              className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-500/20 hover:bg-indigo-500/30 text-cyan-200 border border-cyan-500/40 text-xs font-semibold transition-all"
            >
              <Target className="w-4 h-4 text-cyan-400" />
              <span>Open Forecast &amp; Verdict</span>
            </button>
            <button
              onClick={() => navigate('/lab')}
              className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition-all"
            >
              <span>Experiment Lab</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Active Verdict Highlight Strip when simulation loaded */}
      {currentSimulation?.final_verdict && (
        <div className="rounded-xl bg-slate-900/80 border border-cyan-800/50 p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1 min-w-0">
            <div className="text-[10px] font-mono uppercase text-cyan-400 font-bold">
              Active Forecast Verdict ({currentSimulation.simulation_id})
            </div>
            <div className="text-sm font-bold text-slate-100 truncate">
              {currentSimulation.final_verdict.headline}
            </div>
            <p className="text-xs text-slate-400 line-clamp-1">
              {currentSimulation.final_verdict.executive_summary}
            </p>
          </div>
          <button
            onClick={() => navigate('/forecast')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500/15 hover:bg-cyan-500 hover:text-slate-950 text-cyan-300 border border-cyan-500/40 text-xs font-bold transition flex-shrink-0"
          >
            <span>Inspect Pathways &amp; Verdict</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Methodological Guidance Banner */}
      <AttentionAdoptionNotice />

      {/* High-Level Research Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Engine Status"
          value={backendConnected ? 'ONLINE' : 'DISCONNECTED'}
          subtext={`FastAPI Backend v${backendVersion || '4.2.0'}`}
          icon={Activity}
          accent={backendConnected ? 'emerald' : 'rose'}
        />
        <MetricCard
          label="Past Simulations"
          value={recentSessions.length}
          subtext="Persisted simulation records"
          icon={Database}
          accent="cyan"
        />
        <MetricCard
          label="Model Agreement"
          value={
            currentSimulation &&
            currentSimulation.model_agreement_score !== null &&
            currentSimulation.model_agreement_score !== undefined
              ? `${Math.round(currentSimulation.model_agreement_score * 100)}%`
              : 'N/A'
          }
          subtext={
            !currentSimulation
              ? 'No simulation active'
              : currentSimulation.model_agreement_score == null
              ? 'Requires >=2 valid models'
              : 'Matched pairs stance & semantic alignment'
          }
          icon={Cpu}
          accent="violet"
        />
        <MetricCard
          label="Narrative Convergence"
          value={
            currentSimulation &&
            currentSimulation.narrative_convergence_score !== null &&
            currentSimulation.narrative_convergence_score !== undefined
              ? `${Math.round(currentSimulation.narrative_convergence_score * 100)}%`
              : 'N/A'
          }
          subtext={
            !currentSimulation
              ? 'No simulation active'
              : currentSimulation.narrative_convergence_score == null
              ? 'Insufficient valid iterations'
              : 'Stance variance contraction across rounds'
          }
          icon={Layers}
          accent="amber"
        />
      </div>

      {/* 4-Layer Architecture Deep Dive Cards */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-slate-200 flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <span>The 4-Layer Behavioral Pipeline</span>
          </h2>
          <span className="text-xs font-mono text-slate-400">Deterministic Math + LLM Interpretation</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Micro */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 hover:border-cyan-500/40 transition flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold font-mono">
                L1
              </div>
              <h3 className="text-sm font-bold text-slate-100">Micro Layer</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                5 distinct cognitive archetypes analyzed independently by 2 LLMs (10 structured
                interpretations per iteration). Captures stance, virality, credibility, and emotional charge.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-cyan-400">
              5 Archetypes × 2 Models
            </div>
          </div>

          {/* Meso */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 hover:border-violet-500/40 transition flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-violet-500/10 border border-violet-500/30 text-violet-400 flex items-center justify-center font-bold font-mono">
                L2
              </div>
              <h3 className="text-sm font-bold text-slate-100">Meso Layer</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                8-node weighted influence network separating external broadcast media pressure from
                internal peer-to-peer contagion. Prevents double-counting influence.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-violet-400">
              Weighted Graph Contagion
            </div>
          </div>

          {/* Macro */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 hover:border-emerald-500/40 transition flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center font-bold font-mono">
                L3
              </div>
              <h3 className="text-sm font-bold text-slate-100">Macro Layer</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Modified Bass diffusion differential model: <br />
                <code className="text-[10px] text-emerald-300 font-mono">
                  dN/dt = [p + q(N/M)](M-N) - cN
                </code>
                <br />
                Features adoption fatigue/churn parameter c and strict attention separation.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-emerald-400">
              Bass Diffusion + Churn (c)
            </div>
          </div>

          {/* Feedback */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 hover:border-amber-500/40 transition flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 flex items-center justify-center font-bold font-mono">
                L4
              </div>
              <h3 className="text-sm font-bold text-slate-100">Feedback Layer</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Aggregated macro state and polarization index inject updated environmental context
                into subsequent iterations. Bounded strictly to 3–5 iterations.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-amber-400">
              Bounded 3–5 Iterations
            </div>
          </div>
        </div>
      </div>

      {/* Recent Simulation History Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-100">Recent Simulation Records</h3>
          </div>
          <button
            onClick={() => navigate('/history')}
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-medium"
          >
            <span>View All Records</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {recentSessions.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs">
            No saved simulations found. Click "Launch New Simulation" to run your first simulation.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase text-[10px]">
                  <th className="py-2.5 px-3">Simulation ID</th>
                  <th className="py-2.5 px-3">Event / Signal</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Iterations</th>
                  <th className="py-2.5 px-3">Dominant Framing</th>
                  <th className="py-2.5 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {recentSessions.slice(0, 5).map((session) => (
                  <tr key={session.session_id} className="hover:bg-slate-800/30 transition">
                    <td className="py-3 px-3 font-mono text-cyan-400 font-medium">
                      {session.simulation_id || session.session_id}
                    </td>
                    <td className="py-3 px-3 text-slate-300 max-w-md truncate">
                      <div className="font-medium text-slate-200 truncate">
                        {session.event_title || session.signal}
                      </div>
                      {session.event_title && (
                        <div className="text-[11px] text-slate-400 truncate">
                          {session.short_description || session.signal}
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-3 font-mono">
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded border ${
                          session.status === 'failed'
                            ? 'bg-rose-950/50 text-rose-400 border-rose-800/60'
                            : session.status === 'partially_completed'
                            ? 'bg-amber-950/50 text-amber-400 border-amber-800/60'
                            : 'bg-emerald-950/50 text-emerald-400 border-emerald-800/60'
                        }`}
                      >
                        {getStatusLabel(session.status || 'completed')}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-400">
                      {session.iterations || 3}
                    </td>
                    <td className="py-3 px-3 text-slate-400 max-w-xs truncate">
                      {session.dominant_narrative || 'No narrative recorded'}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => handleOpenSession(session.simulation_id || session.session_id)}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-cyan-500 hover:text-slate-950 text-slate-300 font-medium transition text-xs"
                      >
                        Open Results
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

