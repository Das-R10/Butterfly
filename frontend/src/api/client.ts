// frontend/src/api/client.ts

import {
  SimulationConfig,
  SimulationResult,
  JobStatusResponse,
  SessionSummary,
  PerturbRequest,
  PerturbationResult,
  EvaluateRequest,
  EvaluationResult,
  LLMHealthResponse,
  LLMModelsResponse,
  LLMProvidersResponse,
  PDFReportResponse,
  PDFReportStatusResponse,
  SimulationForecastResponse,
  ForecastResolveRequest,
  ForecastTrackingRecord,
} from '../types/simulation';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail =
          typeof errorJson.detail === 'string'
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
      }
    } catch {
      // ignore non-json error bodies
    }
    throw new Error(errorDetail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  async getHealth(signal?: AbortSignal): Promise<{ status: string; version: string; layers: string[] }> {
    return fetchJson<{ status: string; version: string; layers: string[] }>('/health', { signal });
  },

  async runSimulationSync(config: SimulationConfig, signal?: AbortSignal): Promise<SimulationResult> {
    return fetchJson<SimulationResult>('/api/simulations/run', {
      method: 'POST',
      body: JSON.stringify(config),
      signal,
    });
  },

  async runSimulationAsync(
    config: SimulationConfig,
    signal?: AbortSignal
  ): Promise<{ job_id: string; simulation_id: string; status: string; message: string }> {
    return fetchJson<{ job_id: string; simulation_id: string; status: string; message: string }>(
      '/api/simulations/run/async',
      {
        method: 'POST',
        body: JSON.stringify(config),
        signal,
      }
    );
  },

  async getJob(jobId: string, signal?: AbortSignal): Promise<JobStatusResponse> {
    return fetchJson<JobStatusResponse>(`/jobs/${jobId}`, { signal });
  },

  async getSimulation(simId: string, signal?: AbortSignal): Promise<SimulationResult> {
    return fetchJson<SimulationResult>(`/api/simulations/${encodeURIComponent(simId)}`, { signal });
  },

  async deleteSimulation(
    simId: string,
    signal?: AbortSignal
  ): Promise<{ deleted: boolean; simulation_id: string }> {
    return fetchJson<{ deleted: boolean; simulation_id: string }>(
      `/api/simulations/${encodeURIComponent(simId)}`,
      {
        method: 'DELETE',
        signal,
      }
    );
  },

  async getTrajectory(
    simId: string,
    signal?: AbortSignal
  ): Promise<{
    simulation_id: string;
    trajectory: SimulationResult['trajectory'];
    model_agreement_score: number | null;
    narrative_convergence_score: number | null;
    dominant_narrative: string;
  }> {
    return fetchJson(`/api/simulations/${encodeURIComponent(simId)}/trajectory`, { signal });
  },

  async listSessions(signal?: AbortSignal): Promise<{ sessions: SessionSummary[] }> {
    return fetchJson<{ sessions: SessionSummary[] }>('/sessions', { signal });
  },

  async getSession(sessionId: string, signal?: AbortSignal): Promise<SimulationResult> {
    return fetchJson<SimulationResult>(`/sessions/${encodeURIComponent(sessionId)}`, { signal });
  },

  async runPerturbation(req: PerturbRequest, signal?: AbortSignal): Promise<PerturbationResult> {
    return fetchJson<PerturbationResult>('/api/simulations/perturb', {
      method: 'POST',
      body: JSON.stringify(req),
      signal,
    });
  },

  async evaluateSimulation(req: EvaluateRequest, signal?: AbortSignal): Promise<EvaluationResult> {
    return fetchJson<EvaluationResult>('/api/simulations/evaluate', {
      method: 'POST',
      body: JSON.stringify(req),
      signal,
    });
  },

  async getLLMHealth(signal?: AbortSignal): Promise<LLMHealthResponse> {
    return fetchJson<LLMHealthResponse>('/api/llm/health', { signal });
  },

  async getLLMModels(signal?: AbortSignal): Promise<LLMModelsResponse> {
    return fetchJson<LLMModelsResponse>('/api/llm/models', { signal });
  },

  async getLLMProviders(signal?: AbortSignal): Promise<LLMProvidersResponse> {
    return fetchJson<LLMProvidersResponse>('/api/llm/providers', { signal });
  },

  async generateReport(
    simId: string,
    force = false,
    signal?: AbortSignal
  ): Promise<PDFReportResponse> {
    const query = force ? '?force=true' : '';
    return fetchJson<PDFReportResponse>(
      `/api/simulations/${encodeURIComponent(simId)}/report${query}`,
      {
        method: 'POST',
        signal,
      }
    );
  },

  async getReportStatus(
    simId: string,
    signal?: AbortSignal
  ): Promise<PDFReportStatusResponse> {
    return fetchJson<PDFReportStatusResponse>(
      `/api/simulations/${encodeURIComponent(simId)}/report/status`,
      { signal }
    );
  },

  async downloadReportPdf(
    simId: string,
    fallbackFilename?: string
  ): Promise<{ filename: string }> {
    const url = `${API_BASE}/api/simulations/${encodeURIComponent(simId)}/report/download`;
    const response = await fetch(url);
    if (!response.ok) {
      let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorDetail =
            typeof errorJson.detail === 'string'
              ? errorJson.detail
              : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore non-json error bodies
      }
      throw new Error(errorDetail);
    }

    const disposition = response.headers.get('Content-Disposition') || '';
    const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    const plainMatch = disposition.match(/filename="?([^";\n]+)"?/i);
    let filename = fallbackFilename || `Butterfly_Research_Report_${simId}.pdf`;
    if (utf8Match?.[1]) {
      try {
        filename = decodeURIComponent(utf8Match[1].trim());
      } catch {
        filename = utf8Match[1].trim();
      }
    } else if (plainMatch?.[1]) {
      filename = plainMatch[1].trim();
    }

    const blob = await response.blob();
    const blobUrl = window.URL.createObjectURL(blob);
    try {
      const anchor = document.createElement('a');
      anchor.href = blobUrl;
      anchor.download = filename;
      anchor.style.display = 'none';
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
    } finally {
      window.setTimeout(() => window.URL.revokeObjectURL(blobUrl), 1500);
    }

    return { filename };
  },

  async getForecast(simId: string, signal?: AbortSignal): Promise<SimulationForecastResponse> {
    return fetchJson<SimulationForecastResponse>(
      `/api/simulations/${encodeURIComponent(simId)}/forecast`,
      { signal }
    );
  },

  async resolveForecastOutcome(
    simId: string,
    payload: ForecastResolveRequest,
    signal?: AbortSignal
  ): Promise<ForecastTrackingRecord> {
    return fetchJson<ForecastTrackingRecord>(
      `/api/simulations/${encodeURIComponent(simId)}/forecast/resolve`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
        signal,
      }
    );
  },

  async listForecasts(
    includeMock = false,
    signal?: AbortSignal
  ): Promise<{ forecasts: ForecastTrackingRecord[] }> {
    const query = includeMock ? '?include_mock=true' : '';
    return fetchJson<{ forecasts: ForecastTrackingRecord[] }>(`/api/forecasts${query}`, { signal });
  },
};

/**
 * Connect to live WebSocket progress stream strictly scoped to (jobId, expectedSimulationId).
 * Drops events if the stream has been cleaned up or if frame IDs mismatch.
 * Falls back to job polling cleanly on disconnect.
 */
export function connectSimulationWebSocket(
  jobId: string,
  expectedSimulationId: string | null,
  callbacks: {
    onProgress: (message: string, jobId: string, simulationId?: string | null) => void;
    onComplete: (result: SimulationResult, jobId: string, simulationId?: string | null) => void;
    onError: (
      error: string,
      jobId: string,
      simulationId?: string | null,
      result?: SimulationResult | null
    ) => void;
  }
): () => void {
  const wsUrl = API_BASE.startsWith('https')
    ? `wss://${API_BASE.replace(/^https:\/\//, '')}/ws/${jobId}`
    : `ws://${API_BASE.replace(/^http:\/\//, '')}/ws/${jobId}`;

  let isClosed = false;
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  let ws: WebSocket | null = null;
  let pollController: AbortController | null = null;
  let seenProgressCount = 0;

  const matchesIdentity = (frameJobId?: string, frameSimId?: string | null) => {
    if (isClosed) return false;
    if (frameJobId && frameJobId !== jobId) return false;
    if (expectedSimulationId && frameSimId && frameSimId !== expectedSimulationId) return false;
    return true;
  };

  const stopPolling = () => {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
    if (pollController) {
      pollController.abort();
      pollController = null;
    }
  };

  const startPolling = () => {
    if (pollInterval || isClosed) return;
    pollInterval = setInterval(async () => {
      if (isClosed) {
        stopPolling();
        return;
      }
      pollController?.abort();
      pollController = new AbortController();
      try {
        const job = await api.getJob(jobId, pollController.signal);
        if (!matchesIdentity(job.job_id, job.simulation_id)) return;

        if (job.progress && job.progress.length > seenProgressCount) {
          const newMsgs = job.progress.slice(seenProgressCount);
          seenProgressCount = job.progress.length;
          newMsgs.forEach((msg) => callbacks.onProgress(msg, jobId, job.simulation_id));
        }

        if (
          (job.status === 'completed' ||
            job.status === 'complete' ||
            job.status === 'partially_completed') &&
          job.result
        ) {
          stopPolling();
          callbacks.onComplete(job.result, jobId, job.simulation_id);
        } else if (job.status === 'failed') {
          stopPolling();
          callbacks.onError(
            job.error || 'Job failed during execution',
            jobId,
            job.simulation_id,
            job.result || null
          );
        }
      } catch {
        // Ignore aborts / transient poll drops unless closed
      }
    }, 1500);
  };

  try {
    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      if (isClosed) return;
      try {
        const data = JSON.parse(event.data);
        if (!matchesIdentity(data.job_id, data.simulation_id)) return;

        if (data.type === 'progress') {
          seenProgressCount += 1;
          callbacks.onProgress(data.data, jobId, data.simulation_id);
        } else if (data.type === 'complete') {
          isClosed = true;
          stopPolling();
          callbacks.onComplete(data.data, jobId, data.simulation_id);
        } else if (data.type === 'error') {
          isClosed = true;
          stopPolling();
          callbacks.onError(data.data, jobId, data.simulation_id, data.result || null);
        }
      } catch {
        if (!isClosed) callbacks.onProgress(String(event.data), jobId, expectedSimulationId);
      }
    };

    ws.onerror = () => {
      if (!isClosed) startPolling();
    };

    ws.onclose = () => {
      if (!isClosed) startPolling();
    };
  } catch {
    startPolling();
  }

  return () => {
    isClosed = true;
    stopPolling();
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close();
    }
  };
}
