// frontend/src/components/common/LayerFlowStepper.tsx

import React from 'react';
import { UserCheck, Share2, TrendingUp, RefreshCw, ChevronRight } from 'lucide-react';

interface LayerFlowStepperProps {
  activeLayer?: 'micro' | 'meso' | 'macro' | 'feedback' | 'all';
  onSelectLayer?: (layer: 'micro' | 'meso' | 'macro' | 'feedback') => void;
  iteration?: number;
  maxIterations?: number;
}

export const LayerFlowStepper: React.FC<LayerFlowStepperProps> = ({
  activeLayer = 'all',
  onSelectLayer,
  iteration = 1,
  maxIterations = 3,
}) => {
  const layers = [
    {
      id: 'micro',
      title: '1. Micro Layer',
      subtitle: '5 Archetypes × 2 LLMs (10 Interpretations)',
      icon: UserCheck,
      color: 'cyan',
    },
    {
      id: 'meso',
      title: '2. Meso Layer',
      subtitle: 'Weighted Influence Graph & Contagion',
      icon: Share2,
      color: 'violet',
    },
    {
      id: 'macro',
      title: '3. Macro Layer',
      subtitle: 'Bass Diffusion Curve & Churn (dN/dt)',
      icon: TrendingUp,
      color: 'emerald',
    },
    {
      id: 'feedback',
      title: '4. Feedback Layer',
      subtitle: `Iteration ${iteration}/${maxIterations} Recurrent State Injection`,
      icon: RefreshCw,
      color: 'amber',
    },
  ];

  return (
    <div className="w-full bg-slate-900/60 border border-slate-800 rounded-xl p-3 backdrop-blur-sm">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
        {layers.map((layer, idx) => {
          const Icon = layer.icon;
          const isCurrent = activeLayer === layer.id || activeLayer === 'all';
          const isClickable = !!onSelectLayer;

          return (
            <div
              key={layer.id}
              onClick={() => isClickable && onSelectLayer(layer.id as any)}
              className={`flex items-center gap-3 p-2.5 rounded-lg border transition-all duration-150 ${
                isClickable ? 'cursor-pointer' : ''
              } ${
                activeLayer === layer.id
                  ? 'bg-slate-800/90 border-cyan-500/50 shadow-sm shadow-cyan-500/10'
                  : isCurrent
                  ? 'bg-slate-900/40 border-slate-800/80 hover:border-slate-700'
                  : 'bg-slate-950/40 border-transparent opacity-60'
              }`}
            >
              <div
                className={`p-2 rounded-md ${
                  layer.color === 'cyan'
                    ? 'bg-cyan-500/10 text-cyan-400'
                    : layer.color === 'violet'
                    ? 'bg-violet-500/10 text-violet-400'
                    : layer.color === 'emerald'
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-amber-500/10 text-amber-400'
                }`}
              >
                <Icon className="w-4 h-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold text-slate-200 truncate">{layer.title}</div>
                <div className="text-[10px] text-slate-400 truncate">{layer.subtitle}</div>
              </div>
              {idx < layers.length - 1 && (
                <ChevronRight className="w-4 h-4 text-slate-600 hidden md:block flex-shrink-0" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

