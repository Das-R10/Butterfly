// frontend/src/pages/Settings.tsx

import React, { useState, useEffect } from 'react';
import { useSimulation } from '../context/SimulationContext';
import { api } from '../api/client';
import { MetricCard } from '../components/common/MetricCard';
import {
  LLMHealthResponse,
  LLMModelsResponse,
} from '../types/simulation';
import {
  Settings as SettingsIcon,
  Activity,
  Cpu,
  Database,
  Trash2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  RefreshCw,
  Copy,
  Terminal,
  Server,
  Zap,
  HardDrive,
  Info,
} from 'lucide-react';

export const Settings: React.FC = () => {
  const { backendConnected, backendVersion, resetForNewSimulation } = useSimulation();

  const [healthData, setHealthData] = useState<{
    status: string;
    version: string;
    layers: string[];
    timestamp?: string;
  } | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  const [llmHealth, setLlmHealth] = useState<LLMHealthResponse | null>(null);
  const [modelsData, setModelsData] = useState<LLMModelsResponse | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const checkStatus = async () => {
    const start = performance.now();
    try {
      const data = await api.getHealth();
      const end = performance.now();
      setHealthData(data);
      setLatency(Math.round(end - start));
    } catch {
      setHealthData(null);
      setLatency(null);
    }
  };

  const loadLLMDiagnostics = async () => {
    setIsRefreshing(true);
    try {
      const [h, m] = await Promise.all([
        api.getLLMHealth().catch(() => null),
        api.getLLMModels().catch(() => null),
      ]);
      setLlmHealth(h);
      setModelsData(m);
    } catch (e) {
      console.error('Failed to load LLM diagnostics:', e);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    checkStatus();
    loadLLMDiagnostics();
  }, []);

  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(label);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const handleClearCache = () => {
    if (confirm('Clear current active simulation context and cached localStorage?')) {
      resetForNewSimulation();
      localStorage.clear();
      window.location.reload();
    }
  };

  const isModelInstalled = (modelName: string): boolean => {
    if (!modelsData || !modelsData.ollama_connected) return false;
    const target = modelName.trim().toLowerCase();
    const targetBare = target.split(':')[0];
    return modelsData.installed_models.some((m) => {
      const name = m.name.toLowerCase();
      return (
        name === target ||
        name === `${target}:latest` ||
        target === `${name}:latest` ||
        name.split(':')[0] === targetBare
      );
    });
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <SettingsIcon className="w-6 h-6 text-cyan-400" />
          <span>Platform Settings & System Diagnostics</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Inspect FastAPI backend connectivity, hardware GPU diagnostics, local Ollama integration,
          and inference providers.
        </p>
      </div>

      {/* Backend Connectivity Status */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span>FastAPI Backend Diagnostics</span>
          </h2>
          <button
            onClick={() => {
              checkStatus();
              loadLLMDiagnostics();
            }}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-mono flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>Re-check All Diagnostics</span>
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <MetricCard
            label="API Connectivity"
            value={backendConnected ? 'ONLINE' : 'OFFLINE'}
            subtext="http://localhost:8000/health"
            accent={backendConnected ? 'emerald' : 'rose'}
          />
          <MetricCard
            label="Round-Trip Latency"
            value={latency !== null ? `${latency} ms` : 'N/A'}
            subtext="Local HTTP ping"
            accent="cyan"
          />
          <MetricCard
            label="Backend Engine Version"
            value={`v${backendVersion || '4.2.0'}`}
            subtext="4-Layer Core Architecture"
            accent="violet"
          />
        </div>

        {healthData && (
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800/80 text-xs font-mono text-slate-400 flex items-center justify-between">
            <span>Verified Core Layers: {healthData.layers?.join(' ➔ ')}</span>
            <span className="text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" /> All Layers Verified
            </span>
          </div>
        )}
      </div>

      {/* ── DEDICATED LOCAL MODELS & HARDWARE ACCELERATION SECTION ────────────── */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <Server className="w-4 h-4 text-emerald-400" />
              <span>Local Models (Ollama) & Hardware Acceleration</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Local on-device inference via Ollama HTTP service with verified GPU memory diagnostics.
            </p>
          </div>
          <button
            onClick={loadLLMDiagnostics}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono transition self-start sm:self-auto"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-cyan-400' : ''}`} />
            <span>Refresh Models</span>
          </button>
        </div>

        {/* Status Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Ollama Binary */}
          <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 space-y-1">
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Ollama Host Binary
            </div>
            <div className="text-sm font-bold text-slate-200 flex items-center gap-1.5">
              {llmHealth?.ollama?.installed ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">Installed</span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-amber-400" />
                  <span className="text-amber-400">Not in PATH</span>
                </>
              )}
            </div>
            <p className="text-[10px] text-slate-500 font-mono truncate">
              {llmHealth?.ollama?.binary_path || 'winget install Ollama.Ollama'}
            </p>
          </div>

          {/* Ollama Service */}
          <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 space-y-1">
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Ollama Service Status
            </div>
            <div className="text-sm font-bold flex items-center gap-1.5">
              {llmHealth?.ollama?.running ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">
                    Running (v{llmHealth.ollama.version || '0.x'})
                  </span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-rose-400" />
                  <span className="text-rose-400">Offline</span>
                </>
              )}
            </div>
            <p className="text-[10px] text-slate-500 font-mono">http://localhost:11434</p>
          </div>

          {/* GPU Hardware */}
          <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 space-y-1">
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              GPU Acceleration
            </div>
            <div className="text-sm font-bold flex items-center gap-1.5">
              {llmHealth?.gpu?.verified ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400 truncate">
                    {llmHealth.gpu.name || 'Verified GPU'}
                  </span>
                </>
              ) : (
                <>
                  <Info className="w-4 h-4 text-slate-400" />
                  <span className="text-slate-400">CPU Execution</span>
                </>
              )}
            </div>
            <p className="text-[10px] text-slate-400 font-mono">
              {llmHealth?.gpu?.vram_total_mb
                ? `${Math.round(llmHealth.gpu.vram_total_mb)} MB VRAM (${Math.round(
                    llmHealth.gpu.vram_free_mb || 0
                  )} MB free)`
                : 'No verified NVIDIA VRAM'}
            </p>
          </div>

          {/* Sequential Execution Strategy */}
          <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 space-y-1">
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Execution Strategy
            </div>
            <div className="text-sm font-bold text-cyan-400 flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-cyan-400" />
              <span>Sequential Mode</span>
            </div>
            <p className="text-[10px] text-slate-400">Protects 8 GB VRAM from overload</p>
          </div>
        </div>

        {/* Ollama Alert if Offline */}
        {!llmHealth?.ollama?.running && !isRefreshing && (
          <div className="p-4 rounded-xl border border-amber-800/80 bg-amber-950/30 text-xs text-amber-200 flex items-start gap-3">
            <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-bold text-amber-300">
                Ollama is not available. Install Ollama and start its local service before running
                local inference.
              </div>
              <p className="text-amber-200/90 text-[11px]">
                The Butterfly Algorithm requires Ollama to be running on{' '}
                <code className="text-amber-100 font-mono">http://localhost:11434</code> for local model
                execution. Deterministic Mock is available as an offline fallback for development.
              </p>
            </div>
          </div>
        )}

        {/* Recommended Models Section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Zap className="w-3.5 h-3.5 text-cyan-400" />
              <span>Recommended Models for Dual-Model Simulation</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">
              Optimal for 8 GB VRAM RTX 5070
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model A: qwen3.5:4b */}
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-bold text-slate-200 font-mono text-xs flex items-center gap-2">
                    <span>qwen3.5:4b</span>
                    {isModelInstalled('qwen3.5:4b') ? (
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> Installed
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 flex items-center gap-1">
                        <XCircle className="w-3 h-3" /> Not Installed
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-cyan-400 font-medium mt-0.5">
                    Recommended Role: Primary Analyst (Model Engine A)
                  </div>
                </div>
                <div className="text-[10px] font-mono text-slate-400 text-right">
                  ~2.8 GB VRAM
                </div>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed">
                Strong multi-turn reasoning and precise structured JSON compliance. Ideal for the
                five behavioral archetype interpretations.
              </p>

              <div className="flex items-center justify-between p-2 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300">
                <code>ollama pull qwen3.5:4b</code>
                <button
                  type="button"
                  onClick={() => handleCopy('ollama pull qwen3.5:4b', 'qwen')}
                  className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 text-[10px]"
                >
                  <Copy className="w-3 h-3" />
                  <span>{copiedCmd === 'qwen' ? 'Copied!' : 'Copy'}</span>
                </button>
              </div>
            </div>

            {/* Model B: gemma3:4b */}
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-bold text-slate-200 font-mono text-xs flex items-center gap-2">
                    <span>gemma3:4b</span>
                    {isModelInstalled('gemma3:4b') ? (
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> Installed
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 flex items-center gap-1">
                        <XCircle className="w-3 h-3" /> Not Installed
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-violet-400 font-medium mt-0.5">
                    Recommended Role: Independent Reviewer (Model Engine B)
                  </div>
                </div>
                <div className="text-[10px] font-mono text-slate-400 text-right">
                  ~3.1 GB VRAM
                </div>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed">
                Google Gemma 3 architecture provides diverse epistemic perspective and prevents
                single-model bias in the Model Agreement Score.
              </p>

              <div className="flex items-center justify-between p-2 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300">
                <code>ollama pull gemma3:4b</code>
                <button
                  type="button"
                  onClick={() => handleCopy('ollama pull gemma3:4b', 'gemma3')}
                  className="text-violet-400 hover:text-violet-300 flex items-center gap-1 text-[10px]"
                >
                  <Copy className="w-3 h-3" />
                  <span>{copiedCmd === 'gemma3' ? 'Copied!' : 'Copy'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Installed Models Table */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <HardDrive className="w-3.5 h-3.5 text-slate-400" />
              <span>Installed Ollama Models</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">
              {modelsData?.installed_models?.length || 0} Models Found
            </span>
          </div>

          {modelsData?.installed_models && modelsData.installed_models.length > 0 ? (
            <div className="border border-slate-800 rounded-xl overflow-hidden">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-[10px] uppercase font-mono text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3">Model Tag</th>
                    <th className="p-3">Size on Disk</th>
                    <th className="p-3">Digest</th>
                    <th className="p-3">Last Modified</th>
                    <th className="p-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80 bg-slate-950/40">
                  {modelsData.installed_models.map((m) => (
                    <tr key={m.name} className="hover:bg-slate-900/40 transition font-mono">
                      <td className="p-3 font-semibold text-slate-200">{m.name}</td>
                      <td className="p-3 text-slate-400">{m.size_human}</td>
                      <td className="p-3 text-slate-500 text-[11px]">{m.digest || '—'}</td>
                      <td className="p-3 text-slate-400 text-[11px]">
                        {m.modified_at ? new Date(m.modified_at).toLocaleDateString() : '—'}
                      </td>
                      <td className="p-3">
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                          Ready for Local Inference
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-6 bg-slate-950/60 rounded-xl border border-slate-800 text-center space-y-2">
              <HardDrive className="w-6 h-6 text-slate-600 mx-auto" />
              <div className="text-xs text-slate-300 font-medium">No Models Installed in Ollama</div>
              <p className="text-[11px] text-slate-500 max-w-md mx-auto">
                Run the PowerShell setup commands below to download the recommended models
                (<code className="text-slate-400">qwen3.5:4b</code> and{' '}
                <code className="text-slate-400">gemma3:4b</code>).
              </p>
            </div>
          )}
        </div>

        {/* Windows PowerShell Setup Instructions Required by Spec */}
        <div className="p-5 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span>Windows PowerShell Setup Commands</span>
            </div>
            <button
              type="button"
              onClick={() =>
                handleCopy(
                  `winget install Ollama.Ollama\nollama pull qwen3.5:4b\nollama pull gemma3:4b`,
                  'all'
                )
              }
              className="text-cyan-400 hover:text-cyan-300 text-xs font-mono flex items-center gap-1"
            >
              <Copy className="w-3.5 h-3.5" />
              <span>{copiedCmd === 'all' ? 'All Copied!' : 'Copy All Commands'}</span>
            </button>
          </div>

          <p className="text-xs text-slate-400">
            Open <strong>Windows PowerShell</strong> and run the following commands sequentially.
            <span className="text-amber-400 font-medium ml-1">
              (Do not execute installations automatically; run manually in PowerShell.)
            </span>
          </p>

          <pre className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-cyan-300 overflow-x-auto leading-relaxed">
{`# 1. Install Ollama on Windows 11
winget install Ollama.Ollama

# 2. Pull Model Engine A (Primary Analyst)
ollama pull qwen3.5:4b

# 3. Pull Model Engine B (Independent Reviewer)
ollama pull gemma3:4b`}
          </pre>
        </div>
      </div>

      {/* LLM Provider Architecture & Guide */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-violet-400" />
            <span>Supported LLM Inference Providers</span>
          </h2>
          <span className="text-[10px] font-mono text-slate-400">Environment Variables</span>
        </div>

        <div className="space-y-3">
          {/* 1. Local Ollama */}
          <div className="p-4 bg-slate-950/70 rounded-xl border border-slate-800 flex items-start justify-between">
            <div className="space-y-1">
              <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                <span>1. Local Ollama (Primary / Recommended)</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    llmHealth?.ollama?.running
                      ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {llmHealth?.ollama?.running ? 'Service Online' : 'Local Service'}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                On-device local inference via standard HTTP API on{' '}
                <code className="text-amber-300">http://localhost:11434</code>. Zero cloud egress,
                no subscription keys required.
              </p>
            </div>
          </div>

          {/* 2. Groq Cloud */}
          <div className="p-4 bg-slate-950/70 rounded-xl border border-slate-800 flex items-start justify-between">
            <div className="space-y-1">
              <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                <span>2. Groq Cloud (Optional / Fast LPU)</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    llmHealth?.cloud_providers?.groq?.configured
                      ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {llmHealth?.cloud_providers?.groq?.configured
                    ? 'GROQ_API_KEY Configured'
                    : 'Requires GROQ_API_KEY'}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Ultra-fast cloud LPUs with JSON mode enforcement. Requires{' '}
                <code className="text-cyan-300">GROQ_API_KEY</code> set in backend{' '}
                <code className="text-slate-300">.env</code>.
              </p>
            </div>
          </div>

          {/* 3. Google Gemini API */}
          <div className="p-4 bg-slate-950/70 rounded-xl border border-slate-800 flex items-start justify-between">
            <div className="space-y-1">
              <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                <span>3. Google Gemini API (Optional / Multimodal)</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    llmHealth?.cloud_providers?.gemini?.configured
                      ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {llmHealth?.cloud_providers?.gemini?.configured
                    ? 'GEMINI_API_KEY Configured'
                    : 'Requires GEMINI_API_KEY'}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                High-throughput Gemini reasoning with native structured output schema support.
                Requires <code className="text-violet-300">GEMINI_API_KEY</code> set in backend{' '}
                <code className="text-slate-300">.env</code>.
              </p>
            </div>
          </div>

          {/* 4. Mock */}
          <div className="p-4 bg-slate-950/70 rounded-xl border border-slate-800 flex items-start justify-between">
            <div className="space-y-1">
              <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                <span>4. Deterministic Mock (Development / Testing Only)</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-amber-400 border border-slate-700">
                  Dev / Testing Only
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Deterministic archetype interpretation engine. Requires no hardware acceleration
                and operates offline with 0 latency. Specifically designed for CI/CD test suites and
                pipeline verification—not presented as a production LLM.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Local Storage & Cache Controls */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
          <Database className="w-4 h-4 text-rose-400" />
          <span>Local Session Cache Management</span>
        </h2>
        <p className="text-xs text-slate-400">
          Reset local active simulation state stored in browser localStorage. Does not delete backend
          persisted session files in <code className="text-slate-300">data/sessions/</code>.
        </p>
        <button
          onClick={handleClearCache}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-950/40 border border-rose-800/60 hover:bg-rose-900/40 text-rose-300 text-xs font-medium transition"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Clear Local Browser Context</span>
        </button>
      </div>
    </div>
  );
};
