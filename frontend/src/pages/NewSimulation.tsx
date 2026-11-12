// frontend/src/pages/NewSimulation.tsx

import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSimulation } from '../context/SimulationContext';
import { api } from '../api/client';
import {
  SimulationConfig,
  LLMHealthResponse,
  LLMModelsResponse,
  LLMProvidersResponse,
} from '../types/simulation';
import {
  Play,
  Sparkles,
  Sliders,
  AlertCircle,
  ArrowRight,
  Cpu,
  CheckCircle2,
  XCircle,
  Zap,
  RefreshCw,
  Info,
} from 'lucide-react';
import { AttentionAdoptionNotice } from '../components/common/AttentionAdoptionNotice';

const SIGNAL_TEMPLATES = [
  {
    title: 'Autonomous AI Breakthrough',
    signal:
      'Major research lab claims new autonomous AI model demonstrated recursive self-improvement on complex math proofs, triggering debates on technological singularity and imminent workforce displacement.',
  },
  {
    title: 'Regional Bank Liquidity Run Rumor',
    signal:
      'Social media rumors claim a prominent mid-sized regional bank is facing severe uninsured deposit outflows following commercial real-estate write-downs.',
  },
  {
    title: 'Breakthrough Sodium Battery Commercialization',
    signal:
      'Automaker announces commercial production of zero-lithium sodium-ion batteries with 300 Wh/kg density and $40/kWh cell cost, promising $15,000 electric vehicles by next year.',
  },
  {
    title: 'Synthetic Food Additive Alarm',
    signal:
      'A viral documentary asserts that widely used synthetic food stabilizer E-471 causes irreversible microbiome damage, sparking calls for an immediate consumer boycott.',
  },
];

export const NewSimulation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { startAsyncSimulationWorkspace, resetForNewSimulation } = useSimulation();

  const dupState = (location.state as { duplicateConfig?: Partial<SimulationConfig> } | null)
    ?.duplicateConfig;

  const [eventTitle, setEventTitle] = useState(dupState?.event_title || '');
  const [signal, setSignal] = useState(dupState?.signal || '');
  const [sourceUrl, setSourceUrl] = useState(dupState?.source_url || '');
  const [maxIterations, setMaxIterations] = useState(dupState?.max_iterations || 3);
  const [duplicatedBanner, setDuplicatedBanner] = useState(Boolean(dupState));

  // Model A configuration (Default recommendation: qwen3.5:4b Primary Analyst)
  const [providerA, setProviderA] = useState(dupState?.provider_a || 'ollama');
  const [modelA, setModelA] = useState(dupState?.model_a || 'qwen3.5:4b');
  const [customModelA, setCustomModelA] = useState(false);

  // Model B configuration (Default recommendation: gemma3:4b Independent Reviewer)
  const [providerB, setProviderB] = useState(dupState?.provider_b || 'ollama');
  const [modelB, setModelB] = useState(dupState?.model_b || 'gemma3:4b');
  const [customModelB, setCustomModelB] = useState(false);

  // Macro configuration
  const [populationCeiling, setPopulationCeiling] = useState(dupState?.population_ceiling ?? 1.0);
  const [churnRate, setChurnRate] = useState(dupState?.churn_rate ?? 0.08);
  const [externalWeight, setExternalWeight] = useState(dupState?.external_weight ?? 0.4);

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Diagnostics state
  const [health, setHealth] = useState<LLMHealthResponse | null>(null);
  const [modelsData, setModelsData] = useState<LLMModelsResponse | null>(null);
  const [providersData, setProvidersData] = useState<LLMProvidersResponse | null>(null);
  const [loadingDiagnostics, setLoadingDiagnostics] = useState(true);

  const fetchDiagnostics = async () => {
    try {
      setLoadingDiagnostics(true);
      const [healthRes, modelsRes, providersRes] = await Promise.all([
        api.getLLMHealth().catch(() => null),
        api.getLLMModels().catch(() => null),
        api.getLLMProviders().catch(() => null),
      ]);
      setHealth(healthRes);
      setModelsData(modelsRes);
      setProvidersData(providersRes);
    } catch (e) {
      console.error('Failed to load diagnostics:', e);
    } finally {
      setLoadingDiagnostics(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
    if (dupState) {
      // Isolate workspace when duplicating only configuration from Past Simulations
      resetForNewSimulation();
      window.history.replaceState({}, document.title);
    }
  }, []);

  const handleProviderAChange = (prov: string) => {
    setProviderA(prov);
    setCustomModelA(false);
    if (prov === 'ollama') setModelA('qwen3.5:4b');
    else if (prov === 'groq') setModelA('llama-3.3-70b-versatile');
    else if (prov === 'gemini') setModelA('gemini-1.5-flash');
    else if (prov === 'mock') setModelA('mock-analyst-alpha');
  };

  const handleProviderBChange = (prov: string) => {
    setProviderB(prov);
    setCustomModelB(false);
    if (prov === 'ollama') setModelB('gemma3:4b');
    else if (prov === 'groq') setModelB('llama-3.1-8b-instant');
    else if (prov === 'gemini') setModelB('gemini-2.0-flash');
    else if (prov === 'mock') setModelB('mock-debunker-beta');
  };

  const handleApplyRecommendedLocal = () => {
    const recA = providersData?.recommended_local_config?.model_a;
    const recB = providersData?.recommended_local_config?.model_b;
    setProviderA(recA?.provider || 'ollama');
    setModelA(recA?.model || 'qwen3.5:4b');
    setCustomModelA(false);
    setProviderB(recB?.provider || 'ollama');
    setModelB(recB?.model || 'gemma3:4b');
    setCustomModelB(false);
  };

  const handleApplyMockPreset = () => {
    setProviderA('mock');
    setModelA('mock-analyst-alpha');
    setProviderB('mock');
    setModelB('mock-debunker-beta');
  };

  const handleApplyTemplate = (tmpl: string) => {
    setSignal(tmpl);
    setErrorMessage(null);
  };

  // Check if a model is installed in local Ollama
  const checkIsOllamaModelInstalled = (modelName: string): boolean => {
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

  // Validate readiness before allowing start
  const getValidationIssue = (): string | null => {
    // Model A validation
    if (providerA === 'ollama') {
      if (!health?.ollama?.running) {
        return 'Local Ollama service is not running at http://localhost:11434. Start Ollama before running local inference, or switch to Deterministic Mock.';
      }
      if (!checkIsOllamaModelInstalled(modelA)) {
        return `Model Engine A ('${modelA}') is not installed in local Ollama. Run 'ollama pull ${modelA}' in PowerShell or switch to an installed model.`;
      }
    } else if (providerA === 'groq') {
      if (!health?.cloud_providers?.groq?.configured) {
        return 'Model Engine A is set to Groq Cloud, but GROQ_API_KEY is not configured in the backend environment.';
      }
    } else if (providerA === 'gemini') {
      if (!health?.cloud_providers?.gemini?.configured) {
        return 'Model Engine A is set to Google Gemini API, but GEMINI_API_KEY is not configured in the backend environment.';
      }
    }

    // Model B validation
    if (providerB === 'ollama') {
      if (!health?.ollama?.running) {
        return 'Local Ollama service is not running at http://localhost:11434. Start Ollama before running local inference, or switch to Deterministic Mock.';
      }
      if (!checkIsOllamaModelInstalled(modelB)) {
        return `Model Engine B ('${modelB}') is not installed in local Ollama. Run 'ollama pull ${modelB}' in PowerShell or switch to an installed model.`;
      }
    } else if (providerB === 'groq') {
      if (!health?.cloud_providers?.groq?.configured) {
        return 'Model Engine B is set to Groq Cloud, but GROQ_API_KEY is not configured in the backend environment.';
      }
    } else if (providerB === 'gemini') {
      if (!health?.cloud_providers?.gemini?.configured) {
        return 'Model Engine B is set to Google Gemini API, but GEMINI_API_KEY is not configured in the backend environment.';
      }
    }

    return null;
  };

  const validationIssue = getValidationIssue();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!signal.trim() || signal.trim().length < 5) {
      setErrorMessage('Please enter a signal of at least 5 characters.');
      return;
    }

    if (validationIssue) {
      setErrorMessage(validationIssue);
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    const config: SimulationConfig = {
      signal: signal.trim(),
      event_title: eventTitle.trim() || undefined,
      source_url: sourceUrl.trim() || undefined,
      max_iterations: Number(maxIterations),
      model_a: modelA.trim(),
      model_b: modelB.trim(),
      provider_a: providerA,
      provider_b: providerB,
      population_ceiling: Number(populationCeiling),
      churn_rate: Number(churnRate),
      external_weight: Number(externalWeight),
      run_mode: 'simulation',
    };

    try {
      const response = await api.runSimulationAsync(config);
      startAsyncSimulationWorkspace(response.job_id, response.simulation_id);
      navigate('/live');
    } catch (err: any) {
      setIsSubmitting(false);
      setErrorMessage(err.message || 'Failed to start simulation.');
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <Play className="w-6 h-6 text-cyan-400" />
          <span>Configure Butterfly-Effect Forecast Simulation</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Provide an initial event to model 1st/2nd/3rd-order consequences, stakeholder cascades,
          four future scenarios, and a reasoned Final Verdict across coupled behavioral layers.
        </p>
      </div>

      <AttentionAdoptionNotice />

      {/* Hardware & GPU Diagnostics Banner */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-950/80 border border-emerald-800 flex items-center justify-center flex-shrink-0">
            <Cpu className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <div className="font-semibold text-slate-200 flex items-center gap-2">
              <span>Hardware Acceleration:</span>
              {health?.gpu?.verified ? (
                <span className="text-emerald-400 font-mono text-[11px] flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {health.gpu.name || 'NVIDIA GPU Detected'}
                </span>
              ) : (
                <span className="text-slate-400 font-mono text-[11px]">CPU Host Environment</span>
              )}
            </div>
            <p className="text-[11px] text-slate-400">
              {health?.gpu?.verified && health.gpu.vram_total_mb ? (
                <>
                  <span className="text-slate-300 font-mono font-medium">
                    {Math.round(health.gpu.vram_total_mb)} MB VRAM
                  </span>{' '}
                  ({Math.round(health.gpu.vram_free_mb || 0)} MB free) • Sequential local inference
                  active to preserve 8 GB VRAM
                </>
              ) : (
                'GPU acceleration will only be used if verified on localhost.'
              )}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={fetchDiagnostics}
          disabled={loadingDiagnostics}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono self-start sm:self-auto transition"
        >
          <RefreshCw
            className={`w-3.5 h-3.5 ${loadingDiagnostics ? 'animate-spin text-cyan-400' : ''}`}
          />
          <span>Refresh Hardware Status</span>
        </button>
      </div>

      {errorMessage && (
        <div className="rounded-xl border border-rose-800/60 bg-rose-950/30 p-4 text-xs text-rose-300 flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Signal Templates Pill Selector */}
      <div className="space-y-2">
        <div className="text-xs font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Quick Benchmark Templates</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {SIGNAL_TEMPLATES.map((tmpl, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleApplyTemplate(tmpl.signal)}
              className="text-left p-3 rounded-lg bg-slate-900/60 border border-slate-800 hover:border-cyan-500/40 text-xs transition group"
            >
              <div className="font-semibold text-slate-200 group-hover:text-cyan-400 flex items-center justify-between">
                <span>{tmpl.title}</span>
                <ArrowRight className="w-3 h-3 text-slate-500 group-hover:text-cyan-400 transition" />
              </div>
              <p className="text-slate-400 line-clamp-1 mt-1 text-[11px]">{tmpl.signal}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Duplicated Config Notice */}
      {duplicatedBanner && (
        <div className="rounded-xl border border-cyan-800/60 bg-cyan-950/30 p-3.5 text-xs text-cyan-200 flex items-center justify-between gap-3">
          <span>
            <strong>Configuration Duplicated:</strong> Prompt and model settings were copied into a
            clean isolated simulation workspace without previous outputs.
          </span>
          <button
            type="button"
            onClick={() => {
              setEventTitle('');
              setSignal('');
              setSourceUrl('');
              setDuplicatedBanner(false);
            }}
            className="text-[11px] font-mono px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
          >
            Clear All
          </button>
        </div>
      )}

      {/* Main Configuration Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Signal Input Card */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Event Title (Optional — auto-derived from signal if blank)
            </label>
            <input
              type="text"
              maxLength={100}
              value={eventTitle}
              onChange={(e) => setEventTitle(e.target.value)}
              placeholder="e.g. Regional Bank Liquidity Rumor"
              className="w-full rounded-lg bg-slate-950 border border-slate-800 focus:border-cyan-500 text-xs text-slate-100 px-3 py-2 placeholder-slate-600 outline-none"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
              Initial Event / Exogenous Shock Description <span className="text-rose-400">*</span>
            </label>
            <textarea
              rows={4}
              required
              value={signal}
              onChange={(e) => setSignal(e.target.value)}
              placeholder="Describe the initial event, policy announcement, disruption, or rumor to forecast downstream 1st/2nd/3rd-order consequences..."
              className="w-full rounded-lg bg-slate-950 border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 text-xs text-slate-100 p-3.5 placeholder-slate-600 outline-none transition"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Source URL / Reference (Optional)
            </label>
            <input
              type="url"
              value={sourceUrl}
              onChange={(e) => setSourceUrl(e.target.value)}
              placeholder="https://example.com/news/article-source"
              className="w-full rounded-lg bg-slate-950 border border-slate-800 focus:border-cyan-500 text-xs text-slate-100 px-3 py-2 placeholder-slate-600 outline-none"
            />
          </div>
        </div>

        {/* Dual LLM Configuration */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Dual LLM Interpretation Engines (5 Archetypes × 2 Models = 10 Interpretations)
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Sequential execution across Model A and Model B prevents GPU memory contention.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleApplyRecommendedLocal}
                className="text-[11px] px-2.5 py-1 rounded bg-cyan-950/80 border border-cyan-800 text-cyan-300 hover:bg-cyan-900/80 transition flex items-center gap-1 font-mono"
              >
                <Zap className="w-3 h-3 text-cyan-400" />
                <span>Use Recommended Local (Qwen 4B + Gemma 4B)</span>
              </button>
              <button
                type="button"
                onClick={handleApplyMockPreset}
                className="text-[11px] px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 transition font-mono"
              >
                Use Mock (Dev)
              </button>
            </div>
          </div>

          {/* Informational Note Required by Spec */}
          <div className="p-3 bg-slate-950/90 rounded-lg border border-slate-800 flex items-start gap-2 text-xs text-slate-400">
            <Info className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
            <span>
              <strong>Note:</strong> Local inference uses your computer&apos;s hardware. Cloud
              providers require their respective API credentials configured in{' '}
              <code className="text-slate-300">.env</code>.
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model Engine A Card */}
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-3.5">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold text-cyan-400 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  <span>Model Engine A</span>
                </div>
                <div className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-900 text-cyan-300">
                  Role: Primary Analyst
                </div>
              </div>

              {/* Provider Selector */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <label className="text-slate-400 font-medium">Provider</label>
                  {providerA === 'ollama' && (
                    <span
                      className={`text-[10px] font-mono flex items-center gap-1 ${
                        health?.ollama?.running ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          health?.ollama?.running ? 'bg-emerald-400' : 'bg-rose-400'
                        }`}
                      />
                      {health?.ollama?.running
                        ? `Ollama Online ${health.ollama.version ? `(v${health.ollama.version})` : ''}`
                        : 'Ollama Offline'}
                    </span>
                  )}
                  {providerA === 'groq' && (
                    <span
                      className={`text-[10px] font-mono ${
                        health?.cloud_providers?.groq?.configured
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {health?.cloud_providers?.groq?.configured
                        ? 'API Key Ready'
                        : 'Requires GROQ_API_KEY'}
                    </span>
                  )}
                  {providerA === 'gemini' && (
                    <span
                      className={`text-[10px] font-mono ${
                        health?.cloud_providers?.gemini?.configured
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {health?.cloud_providers?.gemini?.configured
                        ? 'API Key Ready'
                        : 'Requires GEMINI_API_KEY'}
                    </span>
                  )}
                </div>

                <select
                  value={providerA}
                  onChange={(e) => handleProviderAChange(e.target.value)}
                  className="w-full rounded bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-2 outline-none focus:border-cyan-500"
                >
                  <option value="ollama">1. Local Ollama (On-Device Inference)</option>
                  <option value="groq">2. Groq Cloud (Requires API Key)</option>
                  <option value="gemini">3. Google Gemini API (Requires API Key)</option>
                  <option value="mock">
                    4. Deterministic Mock (Development / Testing Only)
                  </option>
                </select>
              </div>

              {/* Ollama Warning if offline */}
              {providerA === 'ollama' && !health?.ollama?.running && !loadingDiagnostics && (
                <div className="p-2.5 rounded bg-amber-950/40 border border-amber-800/80 text-[11px] text-amber-300 flex items-start gap-2">
                  <AlertCircle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <span>
                    Ollama is not available. Install Ollama and start its local service before
                    running local inference.
                  </span>
                </div>
              )}

              {/* Model Identifier Selector */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <label className="text-slate-400 font-medium">Model Identifier</label>
                  {providerA === 'ollama' && (
                    <div className="flex items-center gap-2">
                      {checkIsOllamaModelInstalled(modelA) ? (
                        <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" /> Installed in Ollama
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono text-amber-400 flex items-center gap-1">
                          <XCircle className="w-3 h-3" /> Not Installed
                        </span>
                      )}
                      <button
                        type="button"
                        onClick={() => setCustomModelA(!customModelA)}
                        className="text-[10px] text-cyan-400 hover:underline"
                      >
                        {customModelA ? 'Pick list' : 'Manual text'}
                      </button>
                    </div>
                  )}
                </div>

                {providerA === 'ollama' && !customModelA ? (
                  <select
                    value={modelA}
                    onChange={(e) => setModelA(e.target.value)}
                    className="w-full rounded bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-2 font-mono outline-none focus:border-cyan-500"
                  >
                    <optgroup label="Recommended Local Suggestions">
                      <option value="qwen3.5:4b">
                        qwen3.5:4b (Recommended Primary Analyst){' '}
                        {checkIsOllamaModelInstalled('qwen3.5:4b')
                          ? '— [Installed]'
                          : '— [Not Installed]'}
                      </option>
                      <option value="gemma3:4b">
                        gemma3:4b (Secondary Recommendation){' '}
                        {checkIsOllamaModelInstalled('gemma3:4b')
                          ? '— [Installed]'
                          : '— [Not Installed]'}
                      </option>
                    </optgroup>
                    {modelsData?.installed_models && modelsData.installed_models.length > 0 && (
                      <optgroup label="Other Installed Local Models">
                        {modelsData.installed_models
                          .filter((m) => m.name !== 'qwen3.5:4b' && m.name !== 'gemma3:4b')
                          .map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.name} ({m.size_human})
                            </option>
                          ))}
                      </optgroup>
                    )}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={modelA}
                    onChange={(e) => setModelA(e.target.value)}
                    placeholder={
                      providerA === 'ollama'
                        ? 'e.g. qwen3.5:4b'
                        : providerA === 'groq'
                        ? 'llama-3.3-70b-versatile'
                        : providerA === 'gemini'
                        ? 'gemini-1.5-flash'
                        : 'mock-analyst-alpha'
                    }
                    className="w-full rounded bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-2 font-mono outline-none focus:border-cyan-500"
                  />
                )}
              </div>

              {/* Quick suggestions pills */}
              {providerA === 'ollama' && (
                <div className="flex items-center gap-1.5 pt-1 text-[10px]">
                  <span className="text-slate-500">Quick set:</span>
                  <button
                    type="button"
                    onClick={() => {
                      setModelA('qwen3.5:4b');
                      setCustomModelA(false);
                    }}
                    className={`px-2 py-0.5 rounded border transition font-mono ${
                      modelA === 'qwen3.5:4b'
                        ? 'bg-cyan-950 border-cyan-800 text-cyan-300'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    qwen3.5:4b
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setModelA('gemma3:4b');
                      setCustomModelA(false);
                    }}
                    className={`px-2 py-0.5 rounded border transition font-mono ${
                      modelA === 'gemma3:4b'
                        ? 'bg-cyan-950 border-cyan-800 text-cyan-300'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    gemma3:4b
                  </button>
                </div>
              )}
            </div>

            {/* Model Engine B Card */}
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-3.5">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold text-violet-400 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
                  <span>Model Engine B</span>
                </div>
                <div className="text-[10px] font-mono px-2 py-0.5 rounded bg-violet-950/60 border border-violet-900 text-violet-300">
                  Role: Independent Reviewer
                </div>
              </div>

              {/* Provider Selector */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <label className="text-slate-400 font-medium">Provider</label>
                  {providerB === 'ollama' && (
                    <span
                      className={`text-[10px] font-mono flex items-center gap-1 ${
                        health?.ollama?.running ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          health?.ollama?.running ? 'bg-emerald-400' : 'bg-rose-400'
                        }`}
                      />
                      {health?.ollama?.running
                        ? `Ollama Online ${health.ollama.version ? `(v${health.ollama.version})` : ''}`
                        : 'Ollama Offline'}
                    </span>
                  )}
                  {providerB === 'groq' && (
                    <span
                      className={`text-[10px] font-mono ${
                        health?.cloud_providers?.groq?.configured
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {health?.cloud_providers?.groq?.configured
                        ? 'API Key Ready'
                        : 'Requires GROQ_API_KEY'}
                    </span>
                  )}
                  {providerB === 'gemini' && (
                    <span
                      className={`text-[10px] font-mono ${
                        health?.cloud_providers?.gemini?.configured
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {health?.cloud_providers?.gemini?.configured
                        ? 'API Key Ready'
                        : 'Requires GEMINI_API_KEY'}
                    </span>
                  )}
                </div>

                <select
                  value={providerB}
                  onChange={(e) => handleProviderBChange(e.target.value)}
                  className="w-full rounded bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-2 outline-none focus:border-violet-500"
                >
                  <option value="ollama">1. Local Ollama (On-Device Inference)</option>
                  <option value="groq">2. Groq Cloud (Requires API Key)</option>
                  <option value="gemini">3. Google Gemini API (Requires API Key)</option>
                  <option value="mock">
                    4. Deterministic Mock (Development / Testing Only)
                  </option>
                </select>
              </div>

              {/* Ollama Warning if offline */}
              {providerB === 'ollama' && !health?.ollama?.running && !loadingDiagnostics && (
                <div className="p-2.5 rounded bg-amber-950/40 border border-amber-800/80 text-[11px] text-amber-300 flex items-start gap-2">
                  <AlertCircle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <span>
                    Ollama is not available. Install Ollama and start its local service before
                    running local inference.
                  </span>
                </div>
              )}

              {/* Model Identifier Selector */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <label className="text-slate-400 font-medium">Model Identifier</label>
                  {providerB === 'ollama' && (
                    <div className="flex items-center gap-2">
                      {checkIsOllamaModelInstalled(modelB) ? (
                        <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" /> Installed in Ollama
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono text-amber-400 flex items-center gap-1">
                          <XCircle className="w-3 h-3" /> Not Installed
                        </span>
                      )}
                      <button
                        type="button"
                        onClick={() => setCustomModelB(!customModelB)}
                        className="text-[10px] text-violet-400 hover:underline"
                      >
                        {customModelB ? 'Pick list' : 'Manual text'}
                      </button>
                    </div>
                  )}
                </div>

                {providerB === 'ollama' && !customModelB ? (
                  <select
                    value={modelB}
                    onChange={(e) => setModelB(e.target.value)}
                    className="w-full rounded bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-2 font-mono outline-none focus:border-violet-500"
                  >
                    <optgroup label="Recommended Local Suggestions">
                      <option value="gemma3:4b">
                        gemma3:4b (Recommended Independent Reviewer){' '}
                        {checkIsOllamaModelInstalled('gemma3:4b')
                          ? '— [Installed]'
                          : '— [Not Installed]'}
                      </option>
                      <option value="qwen3.5:4b">
                        qwen3.5:4b (Primary Analyst Recommendation){' '}
                        {checkIsOllamaModelInstalled('qwen3.5:4b')
                          ? '— [Installed]'
                          : '— [Not Installed]'}
                      </option>
                    </optgroup>
                    {modelsData?.installed_models && modelsData.installed_models.length > 0 && (
                      <optgroup label="Other Installed Local Models">
                        {modelsData.installed_models
                          .filter((m) => m.name !== 'gemma3:4b' && m.name !== 'qwen3.5:4b')
                          .map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.name} ({m.size_human})
                            </option>
                          ))}
                      </optgroup>
                    )}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={modelB}
                    onChange={(e) => setModelB(e.target.value)}
                    placeholder={
                      providerB === 'ollama'
                        ? 'e.g. gemma3:4b'
                        : providerB === 'groq'
                        ? 'llama-3.1-8b-instant'
                        : providerB === 'gemini'
                        ? 'gemini-2.0-flash'
                        : 'mock-debunker-beta'
                    }
                    className="w-full rounded bg-slate-900 border border-slate-700 text-xs text-slate-200 px-3 py-2 font-mono outline-none focus:border-violet-500"
                  />
                )}
              </div>

              {/* Quick suggestions pills */}
              {providerB === 'ollama' && (
                <div className="flex items-center gap-1.5 pt-1 text-[10px]">
                  <span className="text-slate-500">Quick set:</span>
                  <button
                    type="button"
                    onClick={() => {
                      setModelB('gemma3:4b');
                      setCustomModelB(false);
                    }}
                    className={`px-2 py-0.5 rounded border transition font-mono ${
                      modelB === 'gemma3:4b'
                        ? 'bg-violet-950 border-violet-800 text-violet-300'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    gemma3:4b
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setModelB('qwen3.5:4b');
                      setCustomModelB(false);
                    }}
                    className={`px-2 py-0.5 rounded border transition font-mono ${
                      modelB === 'qwen3.5:4b'
                        ? 'bg-violet-950 border-violet-800 text-violet-300'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    qwen3.5:4b
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Iterations & Parameter Controls */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Feedback Iterations (Bounded 3–5)
              </h3>
              <p className="text-[11px] text-slate-400">
                Number of recurrent macro-to-micro loops executed per simulation run.
              </p>
            </div>
            <div className="text-lg font-bold font-mono text-cyan-400 px-3 py-1 rounded bg-slate-950 border border-slate-800">
              {maxIterations} Rounds
            </div>
          </div>

          <div className="space-y-1">
            <input
              type="range"
              min={3}
              max={5}
              step={1}
              value={maxIterations}
              onChange={(e) => setMaxIterations(Number(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>3 Iterations (Standard)</span>
              <span>4 Iterations</span>
              <span>5 Iterations (Extended)</span>
            </div>
          </div>

          {/* Advanced Macro Parameters Toggle */}
          <div className="pt-2">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-slate-400 hover:text-cyan-400 flex items-center gap-1.5 transition"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>
                {showAdvanced
                  ? 'Hide Advanced Macro Parameters'
                  : 'Show Advanced Macro Parameters (Bass & Churn)'}
              </span>
            </button>
          </div>

          {showAdvanced && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-3 border-t border-slate-800">
              {/* Churn Rate (Fatigue c) */}
              <div className="space-y-2 p-3 bg-slate-950/60 rounded-lg border border-slate-800/80">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-300">Fatigue / Churn (c)</span>
                  <span className="font-mono text-amber-400 font-bold">{churnRate}</span>
                </div>
                <input
                  type="range"
                  min={0.01}
                  max={0.25}
                  step={0.01}
                  value={churnRate}
                  onChange={(e) => setChurnRate(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-400"
                />
                <p className="text-[10px] text-slate-400">Adoption decay rate per cycle.</p>
              </div>

              {/* External Influence Weight (alpha) */}
              <div className="space-y-2 p-3 bg-slate-950/60 rounded-lg border border-slate-800/80">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-300">External Weight (α)</span>
                  <span className="font-mono text-cyan-400 font-bold">{externalWeight}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={0.8}
                  step={0.05}
                  value={externalWeight}
                  onChange={(e) => setExternalWeight(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                />
                <p className="text-[10px] text-slate-400">Ratio of broadcast vs peer contagion.</p>
              </div>

              {/* Population Ceiling (M) */}
              <div className="space-y-2 p-3 bg-slate-950/60 rounded-lg border border-slate-800/80">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-300">Population Ceiling (M)</span>
                  <span className="font-mono text-emerald-400 font-bold">{populationCeiling}</span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={1.0}
                  step={0.05}
                  value={populationCeiling}
                  onChange={(e) => setPopulationCeiling(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-emerald-400"
                />
                <p className="text-[10px] text-slate-400">Normalized maximum market size.</p>
              </div>
            </div>
          )}
        </div>

        {/* Validation Warning & Clear Reason */}
        {validationIssue && (
          <div className="rounded-xl border border-amber-800/60 bg-amber-950/30 p-4 text-xs text-amber-200 flex items-start gap-3">
            <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-amber-300">Start Simulation Unavailable</div>
              <div className="text-[11px] text-amber-200/90 mt-0.5">{validationIssue}</div>
            </div>
          </div>
        )}

        {/* Submit Action */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
          <div className="text-[11px] text-slate-400">
            {validationIssue ? (
              <span className="text-amber-400 flex items-center gap-1 font-mono">
                <XCircle className="w-3.5 h-3.5" /> Resolve provider requirements to execute
              </span>
            ) : (
              <span className="text-emerald-400 flex items-center gap-1 font-mono">
                <CheckCircle2 className="w-3.5 h-3.5" /> Both model engines verified and ready
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => navigate('/')}
              className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !!validationIssue}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 text-xs font-bold shadow-md shadow-cyan-500/20 transition transform hover:-translate-y-0.5 disabled:transform-none"
            >
              {isSubmitting ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Initializing Pipeline...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Start Simulation</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};
