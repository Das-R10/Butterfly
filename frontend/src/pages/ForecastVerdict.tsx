// frontend/src/pages/ForecastVerdict.tsx

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSimulation } from '../context/SimulationContext';
import { api } from '../api/client';
import {
  SimulationForecastResponse,
  ForecastTrackingRecord,
} from '../types/simulation';
import { CausalChainDiagram } from '../components/forecast/CausalChainDiagram';
import { ScenarioExplorer } from '../components/forecast/ScenarioExplorer';
import { FinalVerdictPanel } from '../components/forecast/FinalVerdictPanel';
import { ReportDownloadButton } from '../components/common/ReportDownloadButton';
import { getStatusLabel } from '../utils/modelSlots';
import {
  Target,
  GitBranch,
  Users,
  ShieldCheck,
  Compass,
  Play,
  Archive,
  Activity,
  AlertCircle,
  X,
  Loader2,
} from 'lucide-react';

export const ForecastVerdict: React.FC = () => {
  const navigate = useNavigate();
  const { currentSimulation, recentSessions, resetForNewSimulation } = useSimulation();

  const [activeTab, setActiveTab] = useState<'verdict' | 'pathways' | 'stakeholders' | 'epistemic'>(
    'verdict'
  );
  const [fetchedForSim, setFetchedForSim] = useState<{
    simId: string;
    data: SimulationForecastResponse;
  } | null>(null);
  const [trackingOverride, setTrackingOverride] = useState<{
    simId: string;
    record: ForecastTrackingRecord | null;
  } | null>(null);
  const [loadingSimId, setLoadingSimId] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [reportNotice, setReportNotice] = useState<string | null>(null);

  const simId = currentSimulation?.simulation_id || '';

  useEffect(() => {
    if (!simId) return;

    let cancelled = false;
    const controller = new AbortController();

    api
      .getForecast(simId, controller.signal)
      .then((res) => {
        if (cancelled) return;
        setFetchedForSim({ simId, data: res });
        setTrackingOverride({ simId, record: res.tracking || null });
        setLoadingSimId(null);
      })
      .catch(() => {
        if (cancelled) return;
        if (currentSimulation) {
          setFetchedForSim({
            simId,
            data: {
              simulation_id: currentSimulation.simulation_id,
              event_title: currentSimulation.event_title,
              status: currentSimulation.status,
              research_question: currentSimulation.research_question,
              causal_pathways: currentSimulation.causal_pathways || [],
              order_effects_summary: currentSimulation.order_effects_summary || null,
              stakeholder_reactions: currentSimulation.stakeholder_reactions || [],
              scenarios: currentSimulation.scenarios || [],
              time_horizon_forecast: currentSimulation.time_horizon_forecast || null,
              epistemic_breakdown: currentSimulation.epistemic_breakdown || null,
              final_verdict: currentSimulation.final_verdict || null,
              tracking: null,
            },
          });
        }
        setLoadingSimId(null);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [simId, currentSimulation]);

  const forecastData = fetchedForSim?.simId === simId ? fetchedForSim.data : null;
  const tracking = trackingOverride?.simId === simId ? trackingOverride.record : null;
  const loading = Boolean(simId && !forecastData && loadingSimId !== simId);

  if (!currentSimulation) {
    return (
      <div className="py-16 text-center space-y-5 animate-fade-in">
        <Target className="w-12 h-12 text-slate-600 mx-auto" />
        <div className="space-y-2">
          <h2 className="text-lg font-bold text-slate-200">No Active Forecast Available</h2>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Launch a new consequence forecast simulation or open a completed run from Past
            Simulations to inspect causal chains, scenarios, and the Final Verdict.
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
            <span>Launch New Forecast</span>
          </button>
          <button
            onClick={() => navigate('/history')}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition"
          >
            <Archive className="w-3.5 h-3.5" />
            <span>Past Simulations</span>
          </button>
        </div>
      </div>
    );
  }

  const effectiveForecast: SimulationForecastResponse = forecastData || {
    simulation_id: currentSimulation.simulation_id,
    event_title: currentSimulation.event_title,
    status: currentSimulation.status,
    research_question: currentSimulation.research_question,
    causal_pathways: currentSimulation.causal_pathways || [],
    order_effects_summary: currentSimulation.order_effects_summary || null,
    stakeholder_reactions: currentSimulation.stakeholder_reactions || [],
    scenarios: currentSimulation.scenarios || [],
    time_horizon_forecast: currentSimulation.time_horizon_forecast || null,
    epistemic_breakdown: currentSimulation.epistemic_breakdown || null,
    final_verdict: currentSimulation.final_verdict || null,
    tracking,
  };

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
      {/* Page Header */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1.5 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
              Butterfly-Effect Forecast &amp; Verdict
            </span>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              ID: {currentSimulation.simulation_id}
            </span>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                currentSimulation.status === 'failed'
                  ? 'bg-rose-950/60 text-rose-400 border-rose-800/70'
                  : currentSimulation.status === 'partially_completed'
                  ? 'bg-amber-950/60 text-amber-400 border-amber-800/70'
                  : 'bg-emerald-950/60 text-emerald-400 border-emerald-800/70'
              }`}
            >
              {getStatusLabel(currentSimulation.status)}
            </span>
          </div>
          <h1 className="text-xl font-bold text-slate-100 truncate">
            {currentSimulation.event_title || currentSimulation.signal_text}
          </h1>
          {currentSimulation.event_title && (
            <p className="text-xs text-slate-400 line-clamp-2">{currentSimulation.signal_text}</p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2.5 flex-shrink-0">
          <button
            type="button"
            onClick={() => navigate('/live')}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition"
          >
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            <span>4-Layer Pipeline</span>
          </button>

          <ReportDownloadButton
            simulationId={currentSimulation.simulation_id}
            disabled={isUnpersistedMock}
            disabledReason="Deterministic mock runs are excluded from database storage; run a persisted simulation to generate the 15-section PDF report."
            onError={(msg) => {
              setReportNotice(null);
              setReportError(msg);
            }}
            onSuccess={({ filename, cached }) => {
              setReportError(null);
              setReportNotice(
                cached
                  ? `Downloaded 15-section PDF report: ${filename}`
                  : `Generated & downloaded 15-section PDF report: ${filename}`
              );
            }}
          />
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

      {/* Sub-Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2 overflow-x-auto">
        <button
          type="button"
          onClick={() => setActiveTab('verdict')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition whitespace-nowrap ${
            activeTab === 'verdict'
              ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Compass className="w-4 h-4" />
          <span>Verdict &amp; Scenarios</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('pathways')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition whitespace-nowrap ${
            activeTab === 'pathways'
              ? 'bg-violet-500/10 text-violet-400 border border-violet-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <GitBranch className="w-4 h-4" />
          <span>Causal Pathways Diagram ({effectiveForecast.causal_pathways.length})</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('stakeholders')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition whitespace-nowrap ${
            activeTab === 'stakeholders'
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Users className="w-4 h-4" />
          <span>Stakeholders &amp; Horizons</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('epistemic')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition whitespace-nowrap ${
            activeTab === 'epistemic'
              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <ShieldCheck className="w-4 h-4" />
          <span>Epistemic Audit &amp; Resolution</span>
        </button>
      </div>

      {loading && !effectiveForecast.final_verdict ? (
        <div className="py-16 text-center space-y-3">
          <Loader2 className="w-8 h-8 animate-spin text-cyan-400 mx-auto" />
          <div className="text-xs text-slate-400">Synthesizing causal pathways &amp; verdict...</div>
        </div>
      ) : !effectiveForecast.final_verdict ? (
        <div className="p-6 rounded-xl bg-amber-950/30 border border-amber-800/60 text-xs text-amber-200 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0" />
          <span>
            Forecast verdict data is unavailable for this simulation record. Run a complete
            simulation to generate scenarios and multi-order causal chains.
          </span>
        </div>
      ) : (
        <>
          {activeTab === 'verdict' && (
            <div className="space-y-6">
              <FinalVerdictPanel
                key={`verdict-${effectiveForecast.simulation_id}-${tracking?.status || 'open'}`}
                simulationId={effectiveForecast.simulation_id}
                verdict={effectiveForecast.final_verdict}
                epistemicBreakdown={effectiveForecast.epistemic_breakdown}
                tracking={tracking}
                onTrackingUpdated={(rec) => setTrackingOverride({ simId, record: rec })}
                showVerdictQuestions={true}
                showEpistemicAndResolution={false}
                isMockOnly={isUnpersistedMock}
              />
              <ScenarioExplorer
                key={`scen-${effectiveForecast.simulation_id}`}
                scenarios={effectiveForecast.scenarios}
                orderEffects={effectiveForecast.order_effects_summary}
                section="scenarios"
              />
            </div>
          )}

          {activeTab === 'pathways' && (
            <CausalChainDiagram
              key={`pathways-${effectiveForecast.simulation_id}`}
              pathways={effectiveForecast.causal_pathways}
              eventTitle={effectiveForecast.event_title}
              signalText={currentSimulation.signal_text}
            />
          )}

          {activeTab === 'stakeholders' && (
            <ScenarioExplorer
              key={`stakeholders-${effectiveForecast.simulation_id}`}
              scenarios={effectiveForecast.scenarios}
              orderEffects={effectiveForecast.order_effects_summary}
              stakeholders={effectiveForecast.stakeholder_reactions}
              timeHorizon={effectiveForecast.time_horizon_forecast}
              section="stakeholders_horizons"
            />
          )}

          {activeTab === 'epistemic' && (
            <FinalVerdictPanel
              key={`epistemic-${effectiveForecast.simulation_id}-${tracking?.status || 'open'}`}
              simulationId={effectiveForecast.simulation_id}
              verdict={effectiveForecast.final_verdict}
              epistemicBreakdown={effectiveForecast.epistemic_breakdown}
              tracking={tracking}
              onTrackingUpdated={(rec) => setTrackingOverride({ simId, record: rec })}
              showVerdictQuestions={false}
              showEpistemicAndResolution={true}
              isMockOnly={isUnpersistedMock}
            />
          )}
        </>
      )}
    </div>
  );
};
