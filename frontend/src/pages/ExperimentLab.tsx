// frontend/src/pages/ExperimentLab.tsx

import React, { useState } from 'react';
import { useSimulation } from '../context/SimulationContext';
import { api } from '../api/client';
import {
  PerturbationType,
  PerturbationResult,
  EvaluationResult,
} from '../types/simulation';
import { MetricCard } from '../components/common/MetricCard';
import { AttentionAdoptionNotice } from '../components/common/AttentionAdoptionNotice';
import {
  FlaskConical,
  Zap,
  Play,
  Scale,
  BarChart2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

export const ExperimentLab: React.FC = () => {
  const { currentSimulation } = useSimulation();

  // Perturbation Experiment State
  const [signal, setSignal] = useState(
    currentSimulation?.signal_text ||
      'Major research lab announces breakthrough room-temperature superconductor with verified zero resistance at 295K.'
  );
  const [shockType, setShockType] = useState<PerturbationType>('fatigue_spike');
  const [shockIteration, setShockIteration] = useState<number>(2);
  const [intensity, setIntensity] = useState<number>(0.5);
  const [isPerturbing, setIsPerturbing] = useState<boolean>(false);
  const [perturbResult, setPerturbResult] = useState<PerturbationResult | null>(null);
  const [perturbError, setPerturbError] = useState<string | null>(null);

  // Empirical Evaluator State
  const [evalObservedAdoption, setEvalObservedAdoption] = useState('0.05, 0.12, 0.28');
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [evalResult, setEvalResult] = useState<EvaluationResult | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  const handleRunPerturbation = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPerturbing(true);
    setPerturbError(null);

    try {
      const res = await api.runPerturbation({
        signal: signal.trim(),
        max_iterations: 3,
        shock_type: shockType,
        shock_iteration: Number(shockIteration),
        intensity: Number(intensity),
      });
      setPerturbResult(res);
    } catch (err: any) {
      setPerturbError(err.message || 'Failed to execute perturbation experiment.');
    } finally {
      setIsPerturbing(false);
    }
  };

  const handleRunEvaluation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentSimulation) {
      setEvalError('Please load or execute a simulation first.');
      return;
    }

    const points = evalObservedAdoption
      .split(',')
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n));

    if (points.length === 0) {
      setEvalError('Please provide valid comma-separated numeric values.');
      return;
    }

    setIsEvaluating(true);
    setEvalError(null);

    try {
      const res = await api.evaluateSimulation({
        simulation_id: currentSimulation.simulation_id,
        observed_adoption: points,
      });
      setEvalResult(res);
    } catch (err: any) {
      setEvalError(err.message || 'Evaluation failed.');
    } finally {
      setIsEvaluating(false);
    }
  };

  // Comparative Chart Data
  const comparativeChartData = perturbResult
    ? (perturbResult.baseline_trajectory?.steps || []).map((step, idx) => ({
        iteration: `Round ${step}`,
        baselineAdoption: Number(
          ((perturbResult.baseline_trajectory.active_adoption?.[idx] ?? 0) * 100).toFixed(2)
        ),
        perturbedAdoption: Number(
          ((perturbResult.perturbed_trajectory.active_adoption?.[idx] ?? 0) * 100).toFixed(2)
        ),
        divergenceDelta: Number(
          ((perturbResult.adoption_divergence?.[idx] ?? 0) * 100).toFixed(2)
        ),
      }))
    : [];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <FlaskConical className="w-6 h-6 text-amber-400" />
          <span>Experiment Lab: Counterfactual Shocks & Evaluation</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Simulate exogenous behavioral shocks (debunking interventions, influencer blasts, fatigue
          spikes) and empirically evaluate trajectories against ground truth.
        </p>
      </div>

      <AttentionAdoptionNotice />

      {/* MODULE 1: COUNTERFACTUAL PERTURBATION EXPERIMENT */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">
              Counterfactual Shock Experiment
            </h2>
          </div>
          <span className="text-[10px] font-mono text-cyan-400">Exogenous Intervention Test</span>
        </div>

        <form onSubmit={handleRunPerturbation} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Baseline Signal
            </label>
            <textarea
              rows={2}
              required
              value={signal}
              onChange={(e) => setSignal(e.target.value)}
              className="w-full rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-100 p-3 outline-none focus:border-amber-500"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Shock Type */}
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-slate-400">Intervention Type</label>
              <select
                value={shockType}
                onChange={(e) => setShockType(e.target.value as PerturbationType)}
                className="w-full rounded bg-slate-950 border border-slate-800 text-xs text-slate-200 p-2.5 outline-none"
              >
                <option value="debunk_injection">Debunking Evidence Injection</option>
                <option value="amplifier_blast">Viral Influencer Blast</option>
                <option value="fatigue_spike">Adoption Fatigue / Churn Spike</option>
                <option value="institutional_freeze">Institutional Regulatory Freeze</option>
              </select>
            </div>

            {/* Shock Iteration */}
            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-slate-400">
                Target Injection Iteration
              </label>
              <select
                value={shockIteration}
                onChange={(e) => setShockIteration(Number(e.target.value))}
                className="w-full rounded bg-slate-950 border border-slate-800 text-xs text-slate-200 p-2.5 outline-none"
              >
                <option value={1}>Iteration 1 (Immediate shock)</option>
                <option value={2}>Iteration 2 (Mid-flight intervention)</option>
                <option value={3}>Iteration 3 (Late-stage shock)</option>
              </select>
            </div>

            {/* Intensity Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Shock Intensity</span>
                <span className="font-mono text-amber-400 font-bold">
                  {(intensity * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.1}
                value={intensity}
                onChange={(e) => setIntensity(Number(e.target.value))}
                className="w-full h-2 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-400 mt-2"
              />
            </div>
          </div>

          {perturbError && (
            <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-xs text-rose-300">
              {perturbError}
            </div>
          )}

          <div className="flex justify-end pt-2">
            <button
              type="submit"
              disabled={isPerturbing}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 text-xs font-bold transition shadow-md shadow-amber-500/20"
            >
              {isPerturbing ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Computing Comparative Shocks...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Run Counterfactual Shock</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Perturbation Results Display */}
        {perturbResult && (
          <div className="space-y-6 pt-4 border-t border-slate-800 animate-fade-in">
            {/* Effect Summary Card */}
            <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/60 flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-amber-400">Intervention Outcome Summary</div>
                <div className="text-sm font-semibold text-slate-100 mt-0.5">
                  {perturbResult.effect_summary}
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] font-mono text-slate-400 uppercase">Final Net Divergence</div>
                <div
                  className={`text-xl font-bold font-mono ${
                    perturbResult.final_divergence >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {perturbResult.final_divergence > 0 ? `+${(perturbResult.final_divergence * 100).toFixed(1)}%` : `${(perturbResult.final_divergence * 100).toFixed(1)}%`}
                </div>
              </div>
            </div>

            {/* Comparative Adoption Trajectory Chart */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                  Trajectory Divergence: Baseline vs Perturbed Curve
                </h3>
                <div className="flex items-center gap-4 text-xs font-mono">
                  <span className="flex items-center gap-1.5 text-slate-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-400" /> Baseline Active
                  </span>
                  <span className="flex items-center gap-1.5 text-amber-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400" /> Perturbed Active
                  </span>
                </div>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={comparativeChartData}>
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
                    <Line
                      type="monotone"
                      dataKey="baselineAdoption"
                      name="Baseline Active Adoption %"
                      stroke="#94A3B8"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={{ fill: '#94A3B8', r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="perturbedAdoption"
                      name="Perturbed Active Adoption %"
                      stroke="#F59E0B"
                      strokeWidth={3}
                      dot={{ fill: '#F59E0B', r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* MODULE 2: EMPIRICAL BENCHMARK EVALUATOR */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Scale className="w-5 h-5 text-cyan-400" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">
              Empirical Trajectory Evaluator
            </h2>
          </div>
          <span className="text-[10px] font-mono text-cyan-400">Validation Against Observations</span>
        </div>

        <form onSubmit={handleRunEvaluation} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Held-Out Empirical Adoption Observations (Comma-separated decimals 0.0 - 1.0)
            </label>
            <input
              type="text"
              value={evalObservedAdoption}
              onChange={(e) => setEvalObservedAdoption(e.target.value)}
              placeholder="0.05, 0.12, 0.28"
              className="w-full rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-100 p-3 font-mono outline-none focus:border-cyan-500"
            />
            <p className="text-[11px] text-slate-500 mt-1">
              Compares simulated active adoption proxy trajectory against empirical observation array.
            </p>
          </div>

          {evalError && (
            <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-xs text-rose-300">
              {evalError}
            </div>
          )}

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={isEvaluating}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 text-xs font-bold transition shadow-md shadow-cyan-500/20"
            >
              {isEvaluating ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Computing Statistical Metrics...</span>
                </>
              ) : (
                <>
                  <BarChart2 className="w-4 h-4" />
                  <span>Evaluate Accuracy</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Evaluation Metrics Cards */}
        {evalResult && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 border-t border-slate-800 animate-fade-in">
            <MetricCard
              label="Mean Squared Error (MSE)"
              value={evalResult.adoption_mse.toFixed(6)}
              subtext="Adoption variance"
              accent="cyan"
            />
            <MetricCard
              label="Mean Absolute Error (MAE)"
              value={evalResult.adoption_mae.toFixed(4)}
              subtext="Mean trajectory deviation"
              accent="violet"
            />
            <MetricCard
              label="Directional Accuracy"
              value={`${Math.round((evalResult.directional_accuracy ?? 0) * 100)}%`}
              subtext="Trend concordance"
              accent="emerald"
            />
            <MetricCard
              label="Peak Timing Error"
              value={`${evalResult.peak_timing_error ?? evalResult.peak_timing_error_steps ?? 0} Steps`}
              subtext="Iteration peak offset"
              accent="amber"
            />
          </div>
        )}
      </div>
    </div>
  );
};

