// frontend/src/pages/MacroTrajectory.tsx

import React from 'react';
import { useSimulation } from '../context/SimulationContext';
import { MetricCard } from '../components/common/MetricCard';
import { AttentionAdoptionNotice } from '../components/common/AttentionAdoptionNotice';
import { CrossLayerExplanationPanel } from '../components/common/CrossLayerExplanationPanel';
import {
  TrendingUp,
  Info,
  Sliders,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  AreaChart,
  Area,
} from 'recharts';

export const MacroTrajectory: React.FC = () => {
  const { currentSimulation, selectedIteration } = useSimulation();

  if (!currentSimulation) {
    return (
      <div className="py-16 text-center space-y-4">
        <TrendingUp className="w-12 h-12 text-slate-600 mx-auto" />
        <h2 className="text-lg font-bold text-slate-200">No Simulation Active</h2>
        <p className="text-xs text-slate-400 max-w-sm mx-auto">
          Execute a simulation or open a previous session to examine the Modified Bass diffusion
          trajectory.
        </p>
      </div>
    );
  }

  const traj = currentSimulation.trajectory;
  const iterations = currentSimulation.iterations || [];
  const latestIter = iterations[iterations.length - 1];
  const macroState = latestIter?.macro_state;

  // Build combined chart points with robust fallback across steps / iterations / iterations array
  const stepsList =
    traj?.iterations && traj.iterations.length > 0
      ? traj.iterations
      : traj?.steps && traj.steps.length > 0
      ? traj.steps
      : iterations.map((it) => it.iteration);

  const chartData = stepsList.map((step, idx) => {
    const iterState = iterations[idx]?.macro_state;
    const cumul = traj?.cumulative_adoption?.[idx] ?? iterState?.cumulative_adoption ?? 0;
    const active = traj?.active_adoption?.[idx] ?? iterState?.active_adoption ?? 0;
    const attn = traj?.attention_indicator?.[idx] ?? iterState?.attention_indicator ?? 0;
    const vel = traj?.adoption_velocity?.[idx] ?? iterState?.adoption_velocity ?? 0;
    const churn =
      traj?.churn_volume?.[idx] ??
      traj?.churn_loss?.[idx] ??
      (iterState ? iterState.c_churn * iterState.active_adoption : 0);

    return {
      iteration: `Round ${step}`,
      cumulativeAdoption: Number((cumul * 100).toFixed(2)),
      activeAdoption: Number((active * 100).toFixed(2)),
      attentionIndicator: Number((attn * 100).toFixed(2)),
      adoptionVelocity: Number((vel * 100).toFixed(2)),
      churnVolume: Number((churn * 100).toFixed(2)),
    };
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-emerald-400" />
          <span>Macro Simulated Diffusion Trajectory</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Modified Bass Diffusion proxy with simulated behavioral churn parameter c, rigorously separating
          simulated media attention salience S(t) from simulated active adoption N(t).
        </p>
      </div>

      <AttentionAdoptionNotice />

      {/* Model Parameter Bar */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="Innovation Rate (p)"
          value={(macroState?.p_external ?? 0.03).toFixed(4)}
          subtext="External innovation coefficient"
          accent="cyan"
        />
        <MetricCard
          label="Imitation Rate (q)"
          value={(macroState?.q_internal ?? 0.38).toFixed(4)}
          subtext="Internal social contagion"
          accent="violet"
        />
        <MetricCard
          label="Market Ceiling (M)"
          value={(macroState?.M_potential ?? 1.0).toFixed(2)}
          subtext="Maximum addressable population"
          accent="emerald"
        />
        <MetricCard
          label="Adoption Fatigue (c)"
          value={(macroState?.c_churn ?? 0.08).toFixed(2)}
          subtext="Abandonment / churn rate"
          accent="amber"
        />
        <MetricCard
          label="Peak Attention / Adoption"
          value={`${(macroState?.attention_vs_adoption_ratio ?? 1.2).toFixed(2)}x`}
          subtext="Virality-to-adoption ratio"
          accent="rose"
        />
      </div>

      {/* Chart 1: Active Adoption vs Attention Indicator */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Primary Simulated Trajectory: Active Sim. Adoption vs Media Attention Index
            </h3>
            <p className="text-[11px] text-slate-400">
              Shows divergence between simulated media attention salience S(t) and Modified Bass active simulated adoption N(t) (proxy model, not empirical proof).
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-emerald-400">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Active Sim. Adoption %
            </span>
            <span className="flex items-center gap-1.5 text-amber-400">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400" /> Sim. Attention Index %
            </span>
            <span className="flex items-center gap-1.5 text-cyan-400">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" /> Cumul. Sim. Diffusion %
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="adoptFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="attnFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#F59E0B" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="iteration" stroke="#64748B" fontSize={11} />
              <YAxis stroke="#64748B" fontSize={11} domain={[0, 100]} unit="%" />
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
                dataKey="cumulativeAdoption"
                name="Cumulative Sim. Diffusion %"
                stroke="#38BDF8"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                fill="none"
              />
              <Area
                type="monotone"
                dataKey="activeAdoption"
                name="Active Sim. Adoption % (Bass Proxy)"
                stroke="#10B981"
                strokeWidth={2.5}
                fillOpacity={1}
                fill="url(#adoptFill)"
              />
              <Area
                type="monotone"
                dataKey="attentionIndicator"
                name="Sim. Attention Index % (Buzz)"
                stroke="#F59E0B"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#attnFill)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Adoption Velocity & Churn Decay */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Adoption Velocity */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Adoption Velocity (dN/dt)
            </h3>
            <span className="text-[10px] font-mono text-cyan-400">Net Growth Rate</span>
          </div>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="iteration" stroke="#64748B" fontSize={11} />
                <YAxis stroke="#64748B" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#1E293B',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="adoptionVelocity"
                  name="Velocity (dN/dt)"
                  stroke="#818CF8"
                  strokeWidth={2}
                  dot={{ fill: '#818CF8', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Churn Fatigue Volume */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Adoption Churn & Abandonment Volume
            </h3>
            <span className="text-[10px] font-mono text-amber-400">c · N Decay</span>
          </div>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="iteration" stroke="#64748B" fontSize={11} />
                <YAxis stroke="#64748B" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#1E293B',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="churnVolume" name="Churn Volume %" fill="#F43F5E" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Discrete Iteration Time Series Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Discrete Iteration Trajectory Breakdown (Simulated Values)
          </h3>
          <span className="text-[10px] font-mono text-slate-400">Verified Deterministic Output</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase">
                <th className="py-2.5 px-3">Iteration</th>
                <th className="py-2.5 px-3">p (Ext)</th>
                <th className="py-2.5 px-3">q (Peer)</th>
                <th className="py-2.5 px-3">Cumul. Sim. N(t)</th>
                <th className="py-2.5 px-3 text-emerald-400">Active Sim. A(t)</th>
                <th className="py-2.5 px-3 text-amber-400">Sim. Attention S(t)</th>
                <th className="py-2.5 px-3">Velocity dN/dt</th>
                <th className="py-2.5 px-3">Attn/Adopt</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {iterations.map((it) => (
                <tr key={it.iteration} className="hover:bg-slate-800/30 transition">
                  <td className="py-3 px-3 text-cyan-400 font-bold">Round {it.iteration}</td>
                  <td className="py-3 px-3 text-slate-300">{it.macro_state.p_external.toFixed(4)}</td>
                  <td className="py-3 px-3 text-slate-300">{it.macro_state.q_internal.toFixed(4)}</td>
                  <td className="py-3 px-3 text-slate-300">
                    {(it.macro_state.cumulative_adoption * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-emerald-400 font-bold">
                    {(it.macro_state.active_adoption * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-amber-400 font-bold">
                    {(it.macro_state.attention_indicator * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {it.macro_state.adoption_velocity.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-slate-400">
                    {it.macro_state.attention_vs_adoption_ratio.toFixed(2)}x
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Parameter Traceability Panel */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Bass Diffusion Parameter Derivation & Traceability
            </h3>
          </div>
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 px-2 py-0.5 rounded">
            dN/dt = [p + q(N/M)](M - N) - c·N
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3 space-y-1.5">
            <div className="text-[11px] font-bold text-slate-300">p (Innovation Rate)</div>
            <div className="text-emerald-400 font-mono text-sm">{(macroState?.p_external ?? 0.03).toFixed(4)}</div>
            <p className="text-[10px] text-slate-400">
              Derived from Meso external influence: <code className="text-slate-300">0.01 + 0.05 × ExtScore</code>. Clamped in [0.005, 0.08]. Represents exogenous broadcast impact.
            </p>
          </div>

          <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3 space-y-1.5">
            <div className="text-[11px] font-bold text-slate-300">q (Imitation Rate)</div>
            <div className="text-violet-400 font-mono text-sm">{(macroState?.q_internal ?? 0.38).toFixed(4)}</div>
            <p className="text-[10px] text-slate-400">
              Derived from Meso internal contagion: <code className="text-slate-300">0.15 + 0.35 × IntScore</code>. Clamped in [0.10, 0.55]. Represents peer-to-peer social reinforcement.
            </p>
          </div>

          <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3 space-y-1.5">
            <div className="text-[11px] font-bold text-slate-300">Boundary Constraint</div>
            <div className="text-cyan-400 font-mono text-sm">A(t) ≤ N(t) ≤ M</div>
            <p className="text-[10px] text-slate-400">
              Active adoption A(t) is strictly bounded by cumulative adopters N(t) and ceiling M ({(macroState?.M_potential ?? 1.0).toFixed(2)}). Time step Δt = 1.0 round.
            </p>
          </div>
        </div>

        {macroState?.formula_derivation && (
          <div className="bg-slate-950/80 border border-slate-800/60 rounded p-2.5 text-[11px] font-mono text-slate-400">
            <span className="text-slate-500 font-bold uppercase tracking-wider mr-2">Derivation Log:</span>
            {macroState.formula_derivation}
          </div>
        )}
      </div>

      {/* Trajectory Assumptions & Limitations */}
      <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2 text-slate-200">
          <Info className="w-4 h-4 text-amber-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider">
            Trajectory Assumptions, Interpretation Guide & Model Limitations
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-400 leading-relaxed">
          <div className="space-y-2">
            <h4 className="text-[11px] font-bold text-slate-300">Interpreting Attention vs Adoption Divergence</h4>
            <p>
              <strong>High Buzz / Low Adoption (Ratio &gt; 1.5x):</strong> When conversational attention S(t) heavily exceeds active adoption A(t), the narrative operates as a viral spectacle, sensationalized outrage, or speculative rumor without deep behavioral commitment.
            </p>
            <p>
              <strong>Deep Penetration (Ratio &lt; 0.9x):</strong> Indicates structural adoption where individuals or institutions incorporate the signal into persistent routines with little ongoing media buzz.
            </p>
          </div>
          <div className="space-y-2">
            <h4 className="text-[11px] font-bold text-slate-300">Model Assumptions & Scientific Boundaries</h4>
            <p>
              <strong>Homogeneous Population Mixing:</strong> The Modified Bass differential equation assumes mean-field interaction within defined sub-segments rather than individual agent-level micro-contact graphs.
            </p>
            <p>
              <strong>Empirical Calibration Notice:</strong> Diffusion trajectories reflect mathematical projections derived from calibrated archetype outputs. They are sensitivity analyses, not empirical certainty. External shocks (perturbations) will disrupt smooth trajectories.
            </p>
          </div>
        </div>
        {macroState?.assumptions_and_limitations && (
          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 text-[11px] text-slate-300">
            <strong>Engine Specification:</strong> {macroState.assumptions_and_limitations}
          </div>
        )}
      </div>

      <CrossLayerExplanationPanel
        simulation={currentSimulation}
        selectedIteration={selectedIteration}
        highlightLayer="macro"
      />
    </div>
  );
};
