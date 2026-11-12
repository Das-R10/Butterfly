// frontend/src/components/common/ArchetypeBadge.tsx

import React from 'react';
import { ArchetypeKey } from '../../types/simulation';
import { ShieldCheck, Search, Eye, AlertTriangle, Megaphone } from 'lucide-react';

interface ArchetypeBadgeProps {
  archetype: ArchetypeKey | string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

const archetypeConfig: Record<
  string,
  { label: string; bg: string; text: string; border: string; icon: React.ComponentType<{ className?: string }> }
> = {
  institutional_analyst: {
    label: 'Institutional Analyst',
    bg: 'bg-cyan-950/40',
    text: 'text-cyan-400',
    border: 'border-cyan-800/60',
    icon: ShieldCheck,
  },
  skeptic_debunker: {
    label: 'Skeptic Debunker',
    bg: 'bg-rose-950/40',
    text: 'text-rose-400',
    border: 'border-rose-800/60',
    icon: Search,
  },
  pattern_seeker: {
    label: 'Pattern Seeker',
    bg: 'bg-purple-950/40',
    text: 'text-purple-400',
    border: 'border-purple-800/60',
    icon: Eye,
  },
  doomer: {
    label: 'Doomer',
    bg: 'bg-amber-950/40',
    text: 'text-amber-400',
    border: 'border-amber-800/60',
    icon: AlertTriangle,
  },
  amplifier_influencer: {
    label: 'Amplifier Influencer',
    bg: 'bg-emerald-950/40',
    text: 'text-emerald-400',
    border: 'border-emerald-800/60',
    icon: Megaphone,
  },
};

export const ArchetypeBadge: React.FC<ArchetypeBadgeProps> = ({
  archetype,
  size = 'md',
  showIcon = true,
}) => {
  const config = archetypeConfig[archetype] || {
    label: archetype,
    bg: 'bg-slate-800',
    text: 'text-slate-300',
    border: 'border-slate-700',
    icon: Eye,
  };

  const Icon = config.icon;

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-xs px-2.5 py-1 gap-1.5',
    lg: 'text-sm px-3 py-1.5 gap-2',
  }[size];

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-3.5 h-3.5',
    lg: 'w-4 h-4',
  }[size];

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full border ${config.bg} ${config.text} ${config.border} ${sizeClasses}`}
    >
      {showIcon && <Icon className={iconSizes} />}
      <span>{config.label}</span>
    </span>
  );
};

