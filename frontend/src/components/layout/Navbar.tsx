// frontend/src/components/layout/Navbar.tsx

import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useSimulation } from '../../context/SimulationContext';
import { getStatusLabel } from '../../utils/modelSlots';
import { Play, Activity, CheckCircle2, AlertCircle } from 'lucide-react';

export const Navbar: React.FC = () => {
  const {
    backendConnected,
    backendVersion,
    currentSimulation,
    isSimulating,
    activeJobId,
    resetForNewSimulation,
  } = useSimulation();
  const navigate = useNavigate();
  const location = useLocation();

  const handleNewSimulationClick = () => {
    resetForNewSimulation();
    navigate('/new');
  };

  return (
    <header className="h-16 border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md sticky top-0 z-40 px-4 md:px-6 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 via-indigo-600 to-violet-500 flex items-center justify-center p-0.5 shadow-md shadow-cyan-900/20 group-hover:shadow-cyan-500/20 transition-all">
            <div className="w-full h-full bg-slate-950 rounded-[7px] flex items-center justify-center">
              <span className="text-cyan-400 font-mono font-bold text-sm">🦋</span>
            </div>
          </div>
          <div>
            <div className="text-sm font-bold tracking-tight text-slate-100 flex items-center gap-2">
              The Butterfly Algorithm
              <span className="text-[10px] font-mono font-normal px-1.5 py-0.5 rounded bg-cyan-950/60 text-cyan-400 border border-cyan-800/40">
                v{backendVersion || '4.0.0'}
              </span>
            </div>
            <div className="text-[11px] text-slate-400 hidden sm:block">
              4-Layer Behavioral Dynamics Simulation Engine
            </div>
          </div>
        </Link>
      </div>

      <div className="flex items-center gap-3">
        {/* Backend Status indicator */}
        <div
          title={backendConnected ? 'FastAPI Backend Online' : 'Backend Disconnected (Port 8000)'}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono border ${
            backendConnected
              ? 'bg-emerald-950/40 text-emerald-400 border-emerald-800/50'
              : 'bg-rose-950/40 text-rose-400 border-rose-800/50'
          }`}
        >
          {backendConnected ? (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="hidden md:inline">API: 8000</span>
              <CheckCircle2 className="w-3.5 h-3.5" />
            </>
          ) : (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
              <span>Offline</span>
              <AlertCircle className="w-3.5 h-3.5" />
            </>
          )}
        </div>

        {/* Live Simulation active indicator */}
        {isSimulating && activeJobId && (
          <Link
            to="/live"
            className="flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/50 border border-cyan-500/40 text-cyan-300 text-xs animate-pulse hover:bg-cyan-900/40 transition"
          >
            <Activity className="w-3.5 h-3.5 animate-spin" />
            <span>Job {activeJobId} Running</span>
          </Link>
        )}

        {/* Current Active Simulation Pill (hidden cleanly on /new blank config workspace) */}
        {currentSimulation && !isSimulating && location.pathname !== '/new' && (
          <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs text-slate-300 max-w-sm truncate">
            <span
              className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
                currentSimulation.status === 'failed'
                  ? 'bg-rose-950 text-rose-400'
                  : currentSimulation.status === 'partially_completed'
                  ? 'bg-amber-950 text-amber-400'
                  : 'bg-cyan-950 text-cyan-400'
              }`}
            >
              {getStatusLabel(currentSimulation.status)}
            </span>
            <span className="truncate text-slate-200">
              {currentSimulation.event_title || currentSimulation.signal_text}
            </span>
          </div>
        )}

        {/* Action Button */}
        <button
          onClick={handleNewSimulationClick}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-semibold shadow-sm shadow-cyan-500/20 transition-all duration-150"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>New Simulation</span>
        </button>
      </div>
    </header>
  );
};

