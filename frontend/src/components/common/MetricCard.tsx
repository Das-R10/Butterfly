// frontend/src/components/common/MetricCard.tsx

import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon?: LucideIcon;
  badge?: string;
  accent?: 'cyan' | 'violet' | 'emerald' | 'amber' | 'rose' | 'slate';
  tooltip?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  icon: Icon,
  badge,
  accent = 'slate',
  tooltip,
}) => {
  const accentStyles = {
    cyan: {
      border: 'border-cyan-500/20 hover:border-cyan-500/40',
      iconBg: 'bg-cyan-500/10 text-cyan-400',
      valueColor: 'text-cyan-400',
    },
    violet: {
      border: 'border-violet-500/20 hover:border-violet-500/40',
      iconBg: 'bg-violet-500/10 text-violet-400',
      valueColor: 'text-violet-400',
    },
    emerald: {
      border: 'border-emerald-500/20 hover:border-emerald-500/40',
      iconBg: 'bg-emerald-500/10 text-emerald-400',
      valueColor: 'text-emerald-400',
    },
    amber: {
      border: 'border-amber-500/20 hover:border-amber-500/40',
      iconBg: 'bg-amber-500/10 text-amber-400',
      valueColor: 'text-amber-400',
    },
    rose: {
      border: 'border-rose-500/20 hover:border-rose-500/40',
      iconBg: 'bg-rose-500/10 text-rose-400',
      valueColor: 'text-rose-400',
    },
    slate: {
      border: 'border-slate-800 hover:border-slate-700',
      iconBg: 'bg-slate-800/60 text-slate-400',
      valueColor: 'text-slate-100',
    },
  }[accent];

  return (
    <div
      title={tooltip}
      className={`bg-slate-900/80 backdrop-blur-sm rounded-xl p-4 border transition-all duration-200 ${accentStyles.border}`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">{label}</span>
        {Icon && (
          <div className={`p-1.5 rounded-lg ${accentStyles.iconBg}`}>
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        <div className={`text-2xl font-bold font-mono tracking-tight ${accentStyles.valueColor}`}>
          {value}
        </div>
        {badge && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300">
            {badge}
          </span>
        )}
      </div>
      {subtext && <p className="mt-1 text-xs text-slate-500 font-normal">{subtext}</p>}
    </div>
  );
};

