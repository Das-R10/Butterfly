// frontend/src/pages/SimulationHistory.tsx

import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSimulation } from '../context/SimulationContext';
import { api } from '../api/client';
import { SessionSummary, SimulationConfig, SimulationResult } from '../types/simulation';
import { getStatusLabel } from '../utils/modelSlots';
import { ReportDownloadButton } from '../components/common/ReportDownloadButton';
import {
  History,
  Search,
  RefreshCw,
  Database,
  Eye,
  Copy,
  Trash2,
  FolderOpen,
  X,
  AlertTriangle,
  ExternalLink,
  ArrowUpDown,
  Filter,
} from 'lucide-react';

type SortOption = 'date_desc' | 'date_asc' | 'title_asc' | 'status' | 'iterations_desc';

export const SimulationHistory: React.FC = () => {
  const navigate = useNavigate();
  const { recentSessions, refreshSessions, loadSession, deleteSimulation, resetForNewSimulation } =
    useSimulation();

  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<SortOption>('date_desc');
  const [loading, setLoading] = useState(false);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [reportNotice, setReportNotice] = useState<string | null>(null);

  // View Details Modal state (isolated fetch without altering active workspace unless Open Results clicked)
  const [detailsModalSim, setDetailsModalSim] = useState<SimulationResult | null>(null);
  const [detailsSummary, setDetailsSummary] = useState<SessionSummary | null>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // Delete Confirmation Modal state
  const [deleteTarget, setDeleteTarget] = useState<SessionSummary | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  const handleRefresh = async () => {
    setLoading(true);
    setActionError(null);
    await refreshSessions();
    setLoading(false);
  };

  const handleOpenResults = async (simId: string) => {
    setLoadingSessionId(simId);
    setActionError(null);
    try {
      await loadSession(simId);
      navigate('/live');
    } catch (err: any) {
      setActionError(err?.message || `Failed to open simulation ${simId}`);
    } finally {
      setLoadingSessionId(null);
    }
  };

  const handleViewDetails = async (session: SessionSummary) => {
    const simId = session.simulation_id || session.session_id;
    setDetailsSummary(session);
    setDetailsModalSim(null);
    setLoadingDetails(true);
    try {
      const fullSim = await api.getSimulation(simId);
      setDetailsModalSim(fullSim);
    } catch (err: any) {
      setActionError(err?.message || `Failed to fetch details for ${simId}`);
      setDetailsSummary(null);
    } finally {
      setLoadingDetails(false);
    }
  };

  const handleDuplicateConfig = async (session: SessionSummary) => {
    const simId = session.simulation_id || session.session_id;
    setActionError(null);
    try {
      // Fetch exact saved config cleanly if needed
      const fullSim = await api.getSimulation(simId);
      const duplicateConfig: Partial<SimulationConfig> = {
        event_title: fullSim.event_title || session.event_title || undefined,
        signal: fullSim.signal_text || session.signal,
        source_url: fullSim.config?.source_url || session.source_url || undefined,
        max_iterations: fullSim.config?.max_iterations || session.max_iterations || 3,
        provider_a: fullSim.config?.provider_a || session.provider_a || 'ollama',
        model_a: fullSim.config?.model_a || session.model_a || 'qwen3.5:4b',
        provider_b: fullSim.config?.provider_b || session.provider_b || 'ollama',
        model_b: fullSim.config?.model_b || session.model_b || 'gemma3:4b',
        population_ceiling: fullSim.config?.population_ceiling ?? 1.0,
        churn_rate: fullSim.config?.churn_rate ?? 0.08,
        external_weight: fullSim.config?.external_weight ?? 0.4,
      };
      resetForNewSimulation();
      navigate('/new', { state: { duplicateConfig } });
    } catch {
      // Fallback using summary row metadata
      const duplicateConfig: Partial<SimulationConfig> = {
        event_title: session.event_title || undefined,
        signal: session.signal,
        source_url: session.source_url || undefined,
        max_iterations: session.max_iterations || session.iterations || 3,
        provider_a: session.provider_a || 'ollama',
        model_a: session.model_a || 'qwen3.5:4b',
        provider_b: session.provider_b || 'ollama',
        model_b: session.model_b || 'gemma3:4b',
      };
      resetForNewSimulation();
      navigate('/new', { state: { duplicateConfig } });
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    const simId = deleteTarget.simulation_id || deleteTarget.session_id;
    setIsDeleting(true);
    setActionError(null);
    try {
      await deleteSimulation(simId);
      setDeleteTarget(null);
    } catch (err: any) {
      setActionError(err?.message || `Failed to delete simulation ${simId}`);
    } finally {
      setIsDeleting(false);
    }
  };

  const filteredAndSorted = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    const filtered = recentSessions.filter((session) => {
      const status = (session.status || 'completed').toLowerCase();
      if (statusFilter !== 'all' && status !== statusFilter) return false;
      if (!q) return true;
      return (
        (session.event_title || '').toLowerCase().includes(q) ||
        (session.signal || '').toLowerCase().includes(q) ||
        (session.short_description || '').toLowerCase().includes(q) ||
        (session.simulation_id || session.session_id || '').toLowerCase().includes(q) ||
        (session.model_a || '').toLowerCase().includes(q) ||
        (session.model_b || '').toLowerCase().includes(q) ||
        (session.dominant_narrative || '').toLowerCase().includes(q)
      );
    });

    return [...filtered].sort((a, b) => {
      if (sortBy === 'date_desc') {
        return new Date(b.timestamp || 0).getTime() - new Date(a.timestamp || 0).getTime();
      }
      if (sortBy === 'date_asc') {
        return new Date(a.timestamp || 0).getTime() - new Date(b.timestamp || 0).getTime();
      }
      if (sortBy === 'title_asc') {
        return (a.event_title || a.signal || '').localeCompare(b.event_title || b.signal || '');
      }
      if (sortBy === 'status') {
        return (a.status || 'completed').localeCompare(b.status || 'completed');
      }
      if (sortBy === 'iterations_desc') {
        return (b.iterations || 0) - (a.iterations || 0);
      }
      return 0;
    });
  }, [recentSessions, searchTerm, statusFilter, sortBy]);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
            <History className="w-6 h-6 text-cyan-400" />
            <span>Past Simulations</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Isolated archival records persisted in{' '}
            <code className="font-mono text-cyan-300">data/sessions/</code>. Opening results is
            strictly read-only and triggers no new model inference.
          </p>
        </div>

        <button
          onClick={handleRefresh}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-xs text-slate-300 font-medium transition self-start md:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh List</span>
        </button>
      </div>

      {actionError && (
        <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/70 text-xs text-rose-200 flex items-center justify-between gap-3">
          <span>{actionError}</span>
          <button
            onClick={() => setActionError(null)}
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

      {/* Search, Filter & Sort Controls */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col lg:flex-row items-stretch lg:items-center gap-3">
        <div className="flex items-center gap-2.5 flex-1 bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-2">
          <Search className="w-4 h-4 text-slate-500 flex-shrink-0" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by title, description, simulation ID, or model..."
            className="w-full bg-transparent text-xs text-slate-200 placeholder-slate-500 outline-none"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="text-xs text-slate-500 hover:text-slate-300 font-mono"
            >
              Clear
            </button>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-transparent text-slate-200 outline-none text-xs"
            >
              <option value="all" className="bg-slate-900">
                All Statuses
              </option>
              <option value="completed" className="bg-slate-900">
                Completed
              </option>
              <option value="partially_completed" className="bg-slate-900">
                Partially Completed
              </option>
              <option value="failed" className="bg-slate-900">
                Failed
              </option>
            </select>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5">
            <ArrowUpDown className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortOption)}
              className="bg-transparent text-slate-200 outline-none text-xs"
            >
              <option value="date_desc" className="bg-slate-900">
                Newest First
              </option>
              <option value="date_asc" className="bg-slate-900">
                Oldest First
              </option>
              <option value="title_asc" className="bg-slate-900">
                Title (A–Z)
              </option>
              <option value="status" className="bg-slate-900">
                Status
              </option>
              <option value="iterations_desc" className="bg-slate-900">
                Most Iterations
              </option>
            </select>
          </div>
        </div>
      </div>

      {/* Past Simulations Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
        {filteredAndSorted.length === 0 ? (
          <div className="py-16 text-center space-y-3">
            <Database className="w-10 h-10 text-slate-600 mx-auto" />
            <div className="text-sm font-semibold text-slate-300">No Simulation Records Found</div>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              {searchTerm || statusFilter !== 'all'
                ? 'No saved simulations match your current search or filter.'
                : 'Execute a simulation to persist your first record.'}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase text-[10px] bg-slate-950/50">
                  <th className="py-3 px-4">ID &amp; Date/Time</th>
                  <th className="py-3 px-4">Event Title &amp; Description</th>
                  <th className="py-3 px-4">Model A / Model B</th>
                  <th className="py-3 px-4">Iters</th>
                  <th className="py-3 px-4">Status &amp; Coverage</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredAndSorted.map((session) => {
                  const simId = session.simulation_id || session.session_id;
                  const status = session.status || 'completed';
                  const validCount = session.successful_interpretations_count;
                  const expCount = session.expected_interpretations_count;

                  return (
                    <tr key={simId} className="hover:bg-slate-800/30 transition group">
                      <td className="py-3.5 px-4 align-top whitespace-nowrap">
                        <div className="font-mono text-cyan-400 font-semibold text-[11px]">
                          {simId}
                        </div>
                        <div className="text-[10px] font-mono text-slate-500 mt-0.5">
                          {session.timestamp
                            ? new Date(session.timestamp).toLocaleString()
                            : 'Archived'}
                        </div>
                      </td>

                      <td className="py-3.5 px-4 align-top max-w-sm">
                        <div className="font-semibold text-slate-100 truncate">
                          {session.event_title || session.signal}
                        </div>
                        <div className="text-[11px] text-slate-400 line-clamp-2 mt-0.5">
                          {session.short_description || session.signal}
                        </div>
                        {session.source_url && (
                          <div className="text-[10px] font-mono text-cyan-400/80 flex items-center gap-1 mt-1 truncate">
                            <ExternalLink className="w-3 h-3 flex-shrink-0" />
                            <span className="truncate">{session.source_url}</span>
                          </div>
                        )}
                      </td>

                      <td className="py-3.5 px-4 align-top font-mono text-[11px] whitespace-nowrap">
                        <div className="text-cyan-300">
                          A: {session.model_a || 'unknown'}
                          {session.provider_a && (
                            <span className="text-slate-500 ml-1">({session.provider_a})</span>
                          )}
                        </div>
                        <div className="text-violet-300 mt-0.5">
                          B: {session.model_b || 'unknown'}
                          {session.provider_b && (
                            <span className="text-slate-500 ml-1">({session.provider_b})</span>
                          )}
                        </div>
                      </td>

                      <td className="py-3.5 px-4 align-top font-mono text-slate-300 whitespace-nowrap">
                        <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                          {session.iterations ?? session.max_iterations ?? 3}
                        </span>
                      </td>

                      <td className="py-3.5 px-4 align-top whitespace-nowrap">
                        <span
                          className={`inline-block text-[10px] font-mono px-2 py-0.5 rounded border ${
                            status === 'failed'
                              ? 'bg-rose-950/60 text-rose-400 border-rose-800/70'
                              : status === 'partially_completed'
                              ? 'bg-amber-950/60 text-amber-400 border-amber-800/70'
                              : 'bg-emerald-950/60 text-emerald-400 border-emerald-800/70'
                          }`}
                        >
                          {getStatusLabel(status)}
                        </span>
                        {validCount != null && expCount != null && (
                          <div className="text-[10px] font-mono text-slate-400 mt-1">
                            {validCount}/{expCount} valid interps
                          </div>
                        )}
                      </td>

                      <td className="py-3.5 px-4 align-top text-right whitespace-nowrap">
                        <div className="inline-flex items-center gap-1.5">
                          <button
                            onClick={() => handleOpenResults(simId)}
                            disabled={loadingSessionId === simId}
                            title="Open Read-Only Simulation Results"
                            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-cyan-500/15 hover:bg-cyan-500 hover:text-slate-950 text-cyan-300 border border-cyan-500/30 font-medium transition text-[11px]"
                          >
                            {loadingSessionId === simId ? (
                              <div className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                            ) : (
                              <>
                                <FolderOpen className="w-3.5 h-3.5" />
                                <span>Open Results</span>
                              </>
                            )}
                          </button>

                          <ReportDownloadButton
                            compact
                            simulationId={simId}
                            initialHasReport={Boolean(session.has_report)}
                            initialFilename={session.report_filename}
                            onError={(msg) => {
                              setReportNotice(null);
                              setActionError(msg);
                            }}
                            onSuccess={({ filename, cached }) => {
                              setActionError(null);
                              setReportNotice(
                                cached
                                  ? `Downloaded PDF report: ${filename}`
                                  : `Generated & downloaded PDF report: ${filename}`
                              );
                            }}
                          />

                          <button
                            onClick={() => handleViewDetails(session)}
                            title="View Metadata & Model Diagnostics"
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>

                          <button
                            onClick={() => handleDuplicateConfig(session)}
                            title="Duplicate Configuration into New Simulation"
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 text-slate-300 hover:text-white border border-slate-700 transition"
                          >
                            <Copy className="w-3.5 h-3.5" />
                          </button>

                          <button
                            onClick={() => setDeleteTarget(session)}
                            title="Delete Simulation Record"
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-rose-600 text-slate-300 hover:text-white border border-slate-700 transition"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* View Details Modal */}
      {detailsSummary && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 space-y-4 shadow-2xl">
            <div className="flex items-start justify-between gap-3 border-b border-slate-800 pb-3">
              <div>
                <div className="text-[11px] font-mono text-cyan-400">
                  {detailsSummary.simulation_id || detailsSummary.session_id}
                </div>
                <h2 className="text-base font-bold text-slate-100 mt-0.5">
                  {detailsModalSim?.event_title ||
                    detailsSummary.event_title ||
                    'Simulation Record Details'}
                </h2>
              </div>
              <button
                onClick={() => {
                  setDetailsSummary(null);
                  setDetailsModalSim(null);
                }}
                className="text-slate-400 hover:text-slate-200 p-1"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {loadingDetails ? (
              <div className="py-12 text-center text-xs text-slate-400">
                Loading full simulation diagnostics...
              </div>
            ) : detailsModalSim ? (
              <div className="space-y-4 text-xs">
                <div className="p-3.5 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1.5">
                  <div className="text-[10px] font-mono uppercase text-slate-400">Signal Prompt</div>
                  <p className="text-slate-200 leading-relaxed">{detailsModalSim.signal_text}</p>
                  {detailsModalSim.config?.source_url && (
                    <div className="text-[11px] font-mono text-cyan-400 truncate">
                      Source: {detailsModalSim.config.source_url}
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Status</div>
                    <div className="font-bold text-slate-100">
                      {getStatusLabel(detailsModalSim.status)}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Valid Interps</div>
                    <div className="font-bold text-emerald-400">
                      {detailsModalSim.successful_interpretations_count ?? '?'}/
                      {detailsModalSim.expected_interpretations_count ?? '?'}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Agreement</div>
                    <div className="font-bold text-cyan-400">
                      {detailsModalSim.model_agreement_score != null
                        ? `${Math.round(detailsModalSim.model_agreement_score * 100)}%`
                        : 'N/A'}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <div className="text-[10px] text-slate-400">Convergence</div>
                    <div className="font-bold text-violet-400">
                      {detailsModalSim.narrative_convergence_score != null
                        ? `${Math.round(detailsModalSim.narrative_convergence_score * 100)}%`
                        : 'N/A'}
                    </div>
                  </div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1 font-mono text-[11px]">
                  <div className="text-[10px] uppercase text-slate-400 mb-1">
                    Configuration Parameters
                  </div>
                  <div>
                    Model A: <span className="text-cyan-300">{detailsModalSim.config?.model_a}</span> (
                    {detailsModalSim.config?.provider_a || 'auto'})
                  </div>
                  <div>
                    Model B: <span className="text-violet-300">{detailsModalSim.config?.model_b}</span>{' '}
                    ({detailsModalSim.config?.provider_b || 'auto'})
                  </div>
                  <div className="text-slate-400">
                    Iterations: {detailsModalSim.config?.max_iterations} | Ceiling M:{' '}
                    {detailsModalSim.config?.population_ceiling ?? 1.0} | Churn c:{' '}
                    {detailsModalSim.config?.churn_rate ?? 0.08} | Alpha:{' '}
                    {detailsModalSim.config?.external_weight ?? 0.4}
                  </div>
                </div>

                {detailsModalSim.status_details && (
                  <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 text-amber-200 text-[11px]">
                    <strong>Status Note:</strong> {detailsModalSim.status_details}
                  </div>
                )}

                {detailsModalSim.model_diagnostics &&
                  detailsModalSim.model_diagnostics.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-[10px] font-mono uppercase text-slate-400">
                        Execution Diagnostics ({detailsModalSim.model_diagnostics.length} calls)
                      </div>
                      <div className="max-h-48 overflow-y-auto border border-slate-800 rounded-lg divide-y divide-slate-800/60 font-mono text-[10px]">
                        {detailsModalSim.model_diagnostics.map((d, i) => (
                          <div
                            key={i}
                            className="px-3 py-1.5 flex items-center justify-between gap-2 bg-slate-900/40"
                          >
                            <span className="text-slate-300 truncate">
                              R{d.iteration} • {d.model_slot} ({d.model_name}) • {d.archetype}
                            </span>
                            <span
                              className={
                                d.execution_status === 'success'
                                  ? 'text-emerald-400'
                                  : 'text-rose-400'
                              }
                            >
                              {d.execution_status} ({d.duration_sec.toFixed(1)}s, r={d.retry_count}
                              {d.finish_reason ? `, ${d.finish_reason}` : ''})
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                <div className="flex flex-wrap items-center justify-end gap-2 pt-2 border-t border-slate-800">
                  <ReportDownloadButton
                    simulationId={detailsModalSim.simulation_id}
                    initialHasReport={Boolean(detailsSummary.has_report)}
                    initialFilename={detailsSummary.report_filename}
                    onError={(msg) => {
                      setReportNotice(null);
                      setActionError(msg);
                    }}
                    onSuccess={({ filename, cached }) => {
                      setActionError(null);
                      setReportNotice(
                        cached
                          ? `Downloaded PDF report: ${filename}`
                          : `Generated & downloaded PDF report: ${filename}`
                      );
                    }}
                  />
                  <button
                    onClick={() => handleDuplicateConfig(detailsSummary)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 text-slate-200 text-xs font-medium transition"
                  >
                    <Copy className="w-3.5 h-3.5" />
                    <span>Duplicate Config</span>
                  </button>
                  <button
                    onClick={() => {
                      const id = detailsModalSim.simulation_id;
                      setDetailsSummary(null);
                      setDetailsModalSim(null);
                      void handleOpenResults(id);
                    }}
                    className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold transition"
                  >
                    <FolderOpen className="w-3.5 h-3.5" />
                    <span>Open Full Results</span>
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-rose-800/70 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-rose-950 border border-rose-800 text-rose-400">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-bold text-slate-100">Delete Past Simulation?</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Are you sure you want to permanently delete simulation{' '}
                  <code className="font-mono text-rose-300">
                    {deleteTarget.simulation_id || deleteTarget.session_id}
                  </code>{' '}
                  (<strong className="text-slate-200">{deleteTarget.event_title || deleteTarget.signal}</strong>)?
                  Only this simulation file will be removed.
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => setDeleteTarget(null)}
                className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleConfirmDelete}
                className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-white text-xs font-bold transition"
              >
                {isDeleting ? 'Deleting...' : 'Delete Permanently'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
