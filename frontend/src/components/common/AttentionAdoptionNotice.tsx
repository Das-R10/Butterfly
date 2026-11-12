// frontend/src/components/common/AttentionAdoptionNotice.tsx

import React from 'react';
import { Info } from 'lucide-react';

interface AttentionAdoptionNoticeProps {
  className?: string;
  compact?: boolean;
}

export const AttentionAdoptionNotice: React.FC<AttentionAdoptionNoticeProps> = ({
  className = '',
  compact = false,
}) => {
  if (compact) {
    return (
      <div
        className={`inline-flex items-center gap-1.5 text-xs text-amber-400/90 bg-amber-950/30 border border-amber-800/40 px-2.5 py-1 rounded-md ${className}`}
      >
        <Info className="w-3.5 h-3.5 flex-shrink-0" />
        <span>
          <strong>Scientific Distinction:</strong> Simulated media attention salience S(t) is strictly separated from simulated active adoption N(t) (Modified Bass proxy, NOT verified real-world adoption).
        </span>
      </div>
    );
  }

  return (
    <div
      className={`rounded-lg border border-amber-800/40 bg-amber-950/20 p-3.5 text-xs text-amber-300/90 flex items-start gap-3 ${className}`}
    >
      <Info className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
      <div className="space-y-1">
        <p className="font-semibold text-amber-200">
          Core Methodological Rule: Simulated Attention Salience ≠ Verified Adoption
        </p>
        <p className="text-slate-400 leading-relaxed">
          The Butterfly Algorithm models <span className="text-cyan-300 font-medium">Simulated Media Attention S(t)</span> (salience, virality, emotional amplification) independently from the <span className="text-emerald-300 font-medium">Simulated Diffusion Trajectory N(t)</span> (Modified Bass proxy governed by innovation p, imitation q, and churn c — NOT verified empirical adoption). High attention volume does not imply persistent behavioral adoption.
        </p>
      </div>
    </div>
  );
};

