// frontend/src/context/SimulationContext.tsx

import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { SimulationResult, SessionSummary } from '../types/simulation';
import { api, connectSimulationWebSocket } from '../api/client';

const WORKSPACE_STORAGE_KEY = 'butterfly_workspace_state_v2';
const LEGACY_STORAGE_KEY = 'butterfly_active_simulation';

interface StoredWorkspace {
  activeSimulationId: string;
  simulation: SimulationResult;
  isReadOnlyArchive?: boolean;
}

function loadInitialWorkspace(): {
  activeSimulationId: string | null;
  simulationsById: Record<string, SimulationResult>;
  isReadOnlyArchive: boolean;
} {
  try {
    const raw = localStorage.getItem(WORKSPACE_STORAGE_KEY);
    if (raw) {
      const parsed: StoredWorkspace = JSON.parse(raw);
      if (parsed?.activeSimulationId && parsed?.simulation?.simulation_id === parsed.activeSimulationId) {
        return {
          activeSimulationId: parsed.activeSimulationId,
          simulationsById: { [parsed.activeSimulationId]: parsed.simulation },
          isReadOnlyArchive: Boolean(parsed.isReadOnlyArchive),
        };
      }
    }
    // Migrate or clean legacy key if valid
    const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (legacy) {
      const sim: SimulationResult = JSON.parse(legacy);
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      if (sim?.simulation_id) {
        return {
          activeSimulationId: sim.simulation_id,
          simulationsById: { [sim.simulation_id]: sim },
          isReadOnlyArchive: false,
        };
      }
    }
  } catch {
    localStorage.removeItem(WORKSPACE_STORAGE_KEY);
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  }
  return { activeSimulationId: null, simulationsById: {}, isReadOnlyArchive: false };
}

interface SimulationContextType {
  simulationsById: Record<string, SimulationResult>;
  activeSimulationId: string | null;
  setActiveSimulationId: (id: string | null) => void;
  currentSimulation: SimulationResult | null;
  setCurrentSimulation: (sim: SimulationResult | null) => void;
  selectedIteration: number;
  setSelectedIteration: (iter: number) => void;
  activeJobId: string | null;
  setActiveJobId: (jobId: string | null) => void;
  isSimulating: boolean;
  setIsSimulating: (isSim: boolean) => void;
  isReadOnlyArchive: boolean;
  simulationError: string | null;
  setSimulationError: (err: string | null) => void;
  executionLogs: string[];
  appendExecutionLog: (msg: string, expectedJobId?: string | null) => void;
  recentSessions: SessionSummary[];
  refreshSessions: () => Promise<void>;
  loadSession: (sessionId: string) => Promise<void>;
  deleteSimulation: (sessionId: string) => Promise<void>;
  markReportAvailable: (simId: string, filename: string, generatedAt: string) => void;
  resetForNewSimulation: () => void;
  startAsyncSimulationWorkspace: (jobId: string, simulationId: string) => void;
  completeJobSimulation: (
    sim: SimulationResult,
    jobId: string,
    simulationId?: string | null
  ) => void;
  failJobSimulation: (
    err: string,
    jobId: string,
    simulationId?: string | null,
    partialResult?: SimulationResult | null
  ) => void;
  backendConnected: boolean;
  backendVersion: string | null;
}

const SimulationContext = createContext<SimulationContextType | undefined>(undefined);

export const SimulationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const initial = loadInitialWorkspace();

  const [simulationsById, setSimulationsById] = useState<Record<string, SimulationResult>>(
    initial.simulationsById
  );
  const [activeSimulationId, setActiveSimulationIdState] = useState<string | null>(
    initial.activeSimulationId
  );
  const [selectedIteration, setSelectedIteration] = useState<number>(() => {
    const active = initial.activeSimulationId
      ? initial.simulationsById[initial.activeSimulationId]
      : null;
    return Math.max(1, active?.iterations?.length || 1);
  });
  const [activeJobId, setActiveJobIdState] = useState<string | null>(null);
  const [isSimulating, setIsSimulatingState] = useState<boolean>(false);
  const [isReadOnlyArchive, setIsReadOnlyArchive] = useState<boolean>(initial.isReadOnlyArchive);
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const [executionLogs, setExecutionLogs] = useState<string[]>([]);
  const [recentSessions, setRecentSessions] = useState<SessionSummary[]>([]);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [backendVersion, setBackendVersion] = useState<string | null>(null);

  // Refs for strict async identity guards against stale responses / late WebSocket frames
  const activeJobIdRef = useRef<string | null>(activeJobId);
  const activeSimulationIdRef = useRef<string | null>(activeSimulationId);
  const isSimulatingRef = useRef<boolean>(isSimulating);
  const isReadOnlyArchiveRef = useRef<boolean>(initial.isReadOnlyArchive);
  const simulationsByIdRef = useRef<Record<string, SimulationResult>>(initial.simulationsById);
  const restoredFromCacheRef = useRef<Set<string>>(
    new Set(initial.activeSimulationId ? [initial.activeSimulationId] : [])
  );
  const latestLoadTargetRef = useRef<string | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    simulationsByIdRef.current = simulationsById;
  }, [simulationsById]);

  useEffect(() => {
    isReadOnlyArchiveRef.current = isReadOnlyArchive;
  }, [isReadOnlyArchive]);

  const setActiveJobId = useCallback((jobId: string | null) => {
    activeJobIdRef.current = jobId;
    setActiveJobIdState(jobId);
  }, []);

  const setActiveSimulationId = useCallback((simId: string | null) => {
    activeSimulationIdRef.current = simId;
    setActiveSimulationIdState(simId);
  }, []);

  const setIsSimulating = useCallback((simming: boolean) => {
    isSimulatingRef.current = simming;
    setIsSimulatingState(simming);
  }, []);

  const persistWorkspace = useCallback(
    (sim: SimulationResult | null, readOnlyFlag: boolean) => {
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      // Do not persist pure deterministic mock runs into localStorage across reloads
      if (sim && sim.simulation_id && !sim.diagnostics?.is_mock_only) {
        const payload: StoredWorkspace = {
          activeSimulationId: sim.simulation_id,
          simulation: sim,
          isReadOnlyArchive: readOnlyFlag,
        };
        localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(payload));
      } else {
        localStorage.removeItem(WORKSPACE_STORAGE_KEY);
      }
    },
    []
  );

  const setCurrentSimulation = useCallback(
    (sim: SimulationResult | null) => {
      if (sim && sim.simulation_id) {
        setSimulationsById((prev) => ({ ...prev, [sim.simulation_id]: sim }));
        setActiveSimulationId(sim.simulation_id);
        setSelectedIteration(Math.max(1, sim.iterations?.length || 1));
        setIsReadOnlyArchive(false);
        persistWorkspace(sim, false);
      } else {
        setActiveSimulationId(null);
        setSelectedIteration(1);
        setIsReadOnlyArchive(false);
        persistWorkspace(null, false);
      }
    },
    [persistWorkspace, setActiveSimulationId]
  );

  const resetForNewSimulation = useCallback(() => {
    // Cancel any in-flight Past Simulation fetch
    if (loadAbortRef.current) {
      loadAbortRef.current.abort();
      loadAbortRef.current = null;
    }
    latestLoadTargetRef.current = null;

    setActiveJobId(null);
    setActiveSimulationId(null);
    setIsSimulating(false);
    setIsReadOnlyArchive(false);
    setSimulationError(null);
    setExecutionLogs([]);
    setSelectedIteration(1);
    persistWorkspace(null, false);
  }, [persistWorkspace, setActiveJobId, setActiveSimulationId, setIsSimulating]);

  const startAsyncSimulationWorkspace = useCallback(
    (jobId: string, simulationId: string) => {
      if (loadAbortRef.current) {
        loadAbortRef.current.abort();
        loadAbortRef.current = null;
      }
      latestLoadTargetRef.current = null;

      setActiveJobId(jobId);
      setActiveSimulationId(simulationId);
      setIsSimulating(true);
      setIsReadOnlyArchive(false);
      setSimulationError(null);
      setExecutionLogs([
        `[INIT] Queued isolated simulation ${simulationId} (Job ID: ${jobId})`,
      ]);
      setSelectedIteration(1);
      persistWorkspace(null, false);
    },
    [persistWorkspace, setActiveJobId, setActiveSimulationId, setIsSimulating]
  );

  const appendExecutionLog = useCallback((msg: string, expectedJobId?: string | null) => {
    if (expectedJobId && expectedJobId !== activeJobIdRef.current) {
      return; // Drop stale log from superseded job
    }
    setExecutionLogs((prev) => [...prev, msg]);
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const data = await api.listSessions();
      const sessions = data.sessions || [];
      setRecentSessions(sessions);

      // Reconcile stale localStorage/archived workspace cache against authoritative backend list
      const activeId = activeSimulationIdRef.current;
      if (!isSimulatingRef.current && activeId) {
        const existsInBackend = sessions.some(
          (s) => s.simulation_id === activeId || s.session_id === activeId
        );
        if (!existsInBackend) {
          const cachedSim = simulationsByIdRef.current[activeId];
          const isMockLive = Boolean(cachedSim?.diagnostics?.is_mock_only);
          const wasRestoredFromStorage = restoredFromCacheRef.current.has(activeId);
          if (isReadOnlyArchiveRef.current || wasRestoredFromStorage || !isMockLive) {
            restoredFromCacheRef.current.delete(activeId);
            setSimulationsById((prev) => {
              const next = { ...prev };
              delete next[activeId];
              return next;
            });
            setActiveSimulationId(null);
            setIsReadOnlyArchive(false);
            setSelectedIteration(1);
            setSimulationError(null);
            persistWorkspace(null, false);
          }
        }
      }
    } catch (err) {
      console.error('Failed to fetch sessions:', err);
    }
  }, [persistWorkspace, setActiveSimulationId]);

  const markReportAvailable = useCallback(
    (simId: string, filename: string, generatedAt: string) => {
      setRecentSessions((prev) =>
        prev.map((s) =>
          s.simulation_id === simId || s.session_id === simId
            ? {
                ...s,
                has_report: true,
                report_filename: filename,
                report_generated_at: generatedAt,
              }
            : s
        )
      );
    },
    []
  );

  const completeJobSimulation = useCallback(
    (sim: SimulationResult, jobId: string, simulationId?: string | null) => {
      if (jobId !== activeJobIdRef.current) {
        return; // Guard against stale late WebSocket/poll update
      }
      if (
        simulationId &&
        activeSimulationIdRef.current &&
        simulationId !== activeSimulationIdRef.current
      ) {
        return;
      }
      setSimulationsById((prev) => ({ ...prev, [sim.simulation_id]: sim }));
      setActiveSimulationId(sim.simulation_id);
      setIsSimulating(false);
      setIsReadOnlyArchive(false);
      setSelectedIteration(Math.max(1, sim.iterations?.length || 1));
      if (sim.status === 'failed' || sim.status === 'partially_completed') {
        setSimulationError(sim.status_details || `Simulation finished with status: ${sim.status}`);
      } else {
        setSimulationError(null);
      }
      setExecutionLogs((prev) => [
        ...prev,
        `[${String(sim.status || 'COMPLETED').toUpperCase()}] Simulation ${sim.simulation_id} finished (${
          sim.successful_interpretations_count ?? '?'
        }/${sim.expected_interpretations_count ?? '?'} valid interpretations).`,
      ]);
      persistWorkspace(sim, false);
      void refreshSessions();
    },
    [persistWorkspace, refreshSessions, setActiveSimulationId, setIsSimulating]
  );

  const failJobSimulation = useCallback(
    (
      err: string,
      jobId: string,
      simulationId?: string | null,
      partialResult?: SimulationResult | null
    ) => {
      if (jobId !== activeJobIdRef.current) {
        return; // Guard against stale error from previous run
      }
      if (
        simulationId &&
        activeSimulationIdRef.current &&
        simulationId !== activeSimulationIdRef.current
      ) {
        return;
      }
      setIsSimulating(false);
      setSimulationError(err);
      setExecutionLogs((prev) => [...prev, `[ERROR] ${err}`]);
      if (partialResult && partialResult.simulation_id) {
        setSimulationsById((prev) => ({
          ...prev,
          [partialResult.simulation_id]: partialResult,
        }));
        setActiveSimulationId(partialResult.simulation_id);
        setSelectedIteration(Math.max(1, partialResult.iterations?.length || 1));
        persistWorkspace(partialResult, false);
      }
      void refreshSessions();
    },
    [persistWorkspace, refreshSessions, setActiveSimulationId, setIsSimulating]
  );

  const loadSession = useCallback(
    async (sessionId: string) => {
      // Cancel previous fetch cleanly so out-of-order responses never leak
      if (loadAbortRef.current) {
        loadAbortRef.current.abort();
      }
      const controller = new AbortController();
      loadAbortRef.current = controller;
      latestLoadTargetRef.current = sessionId;

      // Clear live streaming state immediately
      setActiveJobId(null);
      setIsSimulating(false);
      setSimulationError(null);
      setExecutionLogs([]);

      try {
        const sim = await api.getSimulation(sessionId, controller.signal);
        if (latestLoadTargetRef.current !== sessionId) {
          return; // User requested a different session or started New Simulation meanwhile
        }
        setSimulationsById((prev) => ({ ...prev, [sim.simulation_id]: sim }));
        setActiveSimulationId(sim.simulation_id);
        setSelectedIteration(Math.max(1, sim.iterations?.length || 1));
        setIsReadOnlyArchive(true);
        if (sim.status === 'failed' || sim.status === 'partially_completed') {
          setSimulationError(sim.status_details || `Archived run status: ${sim.status}`);
        }
        persistWorkspace(sim, true);
      } catch (err: any) {
        if (err?.name === 'AbortError') return;
        console.error(`Failed to load session ${sessionId}:`, err);
        throw err;
      }
    },
    [persistWorkspace, setActiveJobId, setActiveSimulationId, setIsSimulating]
  );

  const deleteSimulation = useCallback(
    async (sessionId: string) => {
      await api.deleteSimulation(sessionId);
      setSimulationsById((prev) => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });
      setRecentSessions((prev) =>
        prev.filter((s) => s.simulation_id !== sessionId && s.session_id !== sessionId)
      );
      if (activeSimulationIdRef.current === sessionId) {
        setActiveSimulationId(null);
        setSelectedIteration(1);
        setIsReadOnlyArchive(false);
        setSimulationError(null);
        persistWorkspace(null, false);
      }
    },
    [persistWorkspace, setActiveSimulationId]
  );

  // Derive currentSimulation strictly by activeSimulationId
  const currentSimulation =
    activeSimulationId && simulationsById[activeSimulationId]
      ? simulationsById[activeSimulationId]
      : null;

  // Context-level live WebSocket + polling subscription scoped strictly to activeJobId & activeSimulationId.
  // Keeps streaming active across tab navigation and tears down cleanly on resetForNewSimulation().
  useEffect(() => {
    if (!activeJobId || !isSimulating) return;

    const jobSnapshot = activeJobId;
    const simSnapshot = activeSimulationIdRef.current;

    appendExecutionLog(`[WS] Subscribed to live stream for simulation ${simSnapshot || jobSnapshot}`, jobSnapshot);

    const cleanup = connectSimulationWebSocket(jobSnapshot, simSnapshot, {
      onProgress: (msg, jId) => {
        appendExecutionLog(`[STREAM] ${msg}`, jId);
      },
      onComplete: (result, jId, sId) => {
        completeJobSimulation(result, jId, sId);
      },
      onError: (err, jId, sId, partialResult) => {
        failJobSimulation(err, jId, sId, partialResult);
      },
    });

    return () => {
      cleanup();
    };
  }, [activeJobId, isSimulating, appendExecutionLog, completeJobSimulation, failJobSimulation]);

  // Heartbeat check on mount & periodically
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const health = await api.getHealth();
        setBackendConnected(true);
        setBackendVersion(health.version || '4.2.0');
      } catch {
        setBackendConnected(false);
      }
    };

    checkBackend();
    refreshSessions();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, [refreshSessions]);

  return (
    <SimulationContext.Provider
      value={{
        simulationsById,
        activeSimulationId,
        setActiveSimulationId,
        currentSimulation,
        setCurrentSimulation,
        selectedIteration,
        setSelectedIteration,
        activeJobId,
        setActiveJobId,
        isSimulating,
        setIsSimulating,
        isReadOnlyArchive,
        simulationError,
        setSimulationError,
        executionLogs,
        appendExecutionLog,
        recentSessions,
        refreshSessions,
        loadSession,
        deleteSimulation,
        markReportAvailable,
        resetForNewSimulation,
        startAsyncSimulationWorkspace,
        completeJobSimulation,
        failJobSimulation,
        backendConnected,
        backendVersion,
      }}
    >
      {children}
    </SimulationContext.Provider>
  );
};

export const useSimulation = () => {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
};
