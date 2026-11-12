// frontend/src/components/layout/Sidebar.tsx

import React from 'react';
import { NavLink } from 'react-router-dom';
import { useSimulation } from '../../context/SimulationContext';
import { getStatusLabel, isValidInterpretation } from '../../utils/modelSlots';
import {
  LayoutDashboard,
  PlusCircle,
  Activity,
  Target,
  MessageSquareText,
  Network,
  TrendingUp,
  History,
  FlaskConical,
  Settings,
} from 'lucide-react';

export const Sidebar: React.FC = () => {
  const { currentSimulation, isSimulating, resetForNewSimulation } = useSimulation();

  const validInterpCount = currentSimulation
    ? currentSimulation.successful_interpretations_count ??
      (currentSimulation.iterations || []).reduce(
        (acc, it) => acc + (it.micro_interpretations || []).filter((m) => isValidInterpretation(m)).length,
        0
      )
    : 0;
  const expectedInterpCount = currentSimulation
    ? currentSimulation.expected_interpretations_count ??
      (currentSimulation.iterations || []).length * 10
    : 0;

  const navItems = [
    {
      to: '/',
      label: 'Overview',
      icon: LayoutDashboard,
      badge: null,
    },
    {
      to: '/new',
      label: 'New Simulation',
      icon: PlusCircle,
      badge: null,
      onClick: () => resetForNewSimulation(),
    },
    {
      to: '/live',
      label: 'Live Simulation',
      icon: Activity,
      badge: isSimulating ? 'LIVE' : null,
      badgeColor: 'bg-cyan-500 text-slate-950 animate-pulse',
    },
    {
      to: '/forecast',
      label: 'Forecast & Verdict',
      icon: Target,
      badge: currentSimulation?.final_verdict?.scenario_probabilities?.leading_scenario
        ? String(currentSimulation.final_verdict.scenario_probabilities.leading_scenario).toUpperCase()
        : 'Verdict',
      badgeColor: currentSimulation?.final_verdict
        ? 'bg-cyan-950 text-cyan-300 border border-cyan-800/70'
        : undefined,
    },
    {
      to: '/narrative',
      label: 'Narrative Analysis',
      icon: MessageSquareText,
      badge: currentSimulation ? `${validInterpCount}/${expectedInterpCount}` : null,
    },
    {
      to: '/network',
      label: 'Influence Network',
      icon: Network,
      badge: 'Meso',
    },
    {
      to: '/trajectory',
      label: 'Macro Trajectory',
      icon: TrendingUp,
      badge: 'Bass dN/dt',
    },
    {
      to: '/history',
      label: 'Past Simulations',
      icon: History,
      badge: null,
    },
    {
      to: '/lab',
      label: 'Experiment Lab',
      icon: FlaskConical,
      badge: 'Perturb',
    },
    {
      to: '/settings',
      label: 'Settings',
      icon: Settings,
      badge: null,
    },
  ];

  return (
    <aside className="w-64 border-r border-slate-800/80 bg-slate-950/60 backdrop-blur-sm flex flex-col justify-between p-3 select-none">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[10px] font-mono uppercase tracking-widest text-slate-400 font-medium">
          Research Modules
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              onClick={item.onClick}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-slate-900 border border-slate-800 text-cyan-400 font-semibold shadow-inner'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
                }`
              }
            >
              <div className="flex items-center gap-2.5">
                <Icon className="w-4 h-4 flex-shrink-0" />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    item.badgeColor || 'bg-slate-800 text-slate-400 border border-slate-700/60'
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </NavLink>
          );
        })}
      </div>

      {/* Persistent simulation status widget */}
      <div className="pt-4 border-t border-slate-800/60">
        <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-2.5">
          <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400 mb-1 flex items-center justify-between">
            <span>Simulation Context</span>
            <span
              className={`w-2 h-2 rounded-full ${
                currentSimulation ? 'bg-cyan-400' : 'bg-slate-600'
              }`}
            />
          </div>
          {currentSimulation ? (
            <div className="space-y-1">
              <div className="text-xs font-medium text-slate-200 truncate">
                {currentSimulation.event_title || currentSimulation.signal_text}
              </div>
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span>
                  Iters: {currentSimulation.iterations?.length || 0} •{' '}
                  {getStatusLabel(currentSimulation.status)}
                </span>
                <span>
                  Agr:{' '}
                  {currentSimulation.model_agreement_score !== null &&
                  currentSimulation.model_agreement_score !== undefined
                    ? `${Math.round(currentSimulation.model_agreement_score * 100)}%`
                    : 'N/A'}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-400 italic">No simulation loaded</div>
          )}
        </div>
      </div>
    </aside>
  );
};

