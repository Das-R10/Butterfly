// frontend/src/utils/modelSlots.ts

import { MicroInterpretation, SimulationResult, ArchetypeKey } from '../types/simulation';

export type ModelSlot = 'model_a' | 'model_b';

/**
 * Resolves whether a MicroInterpretation belongs to Model A or Model B.
 * Prioritizes backend `model_slot` ("model_a" | "model_b") and gracefully
 * falls back for pre-v4.1 historical session JSONs.
 */
export function partitionInterpretationsBySlot(
  interpretations: MicroInterpretation[],
  simulation?: SimulationResult | null
): {
  modelA: MicroInterpretation[];
  modelB: MicroInterpretation[];
  slotByItem: Map<MicroInterpretation, ModelSlot>;
} {
  const modelA: MicroInterpretation[] = [];
  const modelB: MicroInterpretation[] = [];
  const slotByItem = new Map<MicroInterpretation, ModelSlot>();

  const hasExplicitSlots = interpretations.some(
    (m) => m.model_slot === 'model_a' || m.model_slot === 'model_b'
  );

  if (hasExplicitSlots) {
    interpretations.forEach((m) => {
      const slot: ModelSlot = m.model_slot === 'model_b' ? 'model_b' : 'model_a';
      slotByItem.set(m, slot);
      if (slot === 'model_a') modelA.push(m);
      else modelB.push(m);
    });
    return { modelA, modelB, slotByItem };
  }

  // Legacy fallback using config models, alpha/beta naming, or per-archetype occurrence
  const cfgModelA = simulation?.config?.model_a;
  const cfgModelB = simulation?.config?.model_b;
  const distinctCfg = Boolean(cfgModelA && cfgModelB && cfgModelA !== cfgModelB);
  const perArchetypeSeen: Record<string, number> = {};

  interpretations.forEach((m) => {
    let slot: ModelSlot;
    const lowerName = (m.model_name || '').toLowerCase();

    if (distinctCfg && m.model_name === cfgModelA) {
      slot = 'model_a';
    } else if (distinctCfg && m.model_name === cfgModelB) {
      slot = 'model_b';
    } else if (lowerName.includes('alpha')) {
      slot = 'model_a';
    } else if (lowerName.includes('beta')) {
      slot = 'model_b';
    } else {
      const count = perArchetypeSeen[m.archetype] || 0;
      slot = count === 0 ? 'model_a' : 'model_b';
      perArchetypeSeen[m.archetype] = count + 1;
    }

    slotByItem.set(m, slot);
    if (slot === 'model_a') modelA.push(m);
    else modelB.push(m);
  });

  return { modelA, modelB, slotByItem };
}

export function groupByArchetypeAndSlot(
  interpretations: MicroInterpretation[],
  simulation?: SimulationResult | null
): Array<{
  archetypeKey: ArchetypeKey;
  name: string;
  modelA?: MicroInterpretation;
  modelB?: MicroInterpretation;
}> {
  const { slotByItem } = partitionInterpretationsBySlot(interpretations, simulation);
  const map: Record<
    string,
    {
      archetypeKey: ArchetypeKey;
      name: string;
      modelA?: MicroInterpretation;
      modelB?: MicroInterpretation;
    }
  > = {};

  interpretations.forEach((item) => {
    if (!map[item.archetype]) {
      map[item.archetype] = {
        archetypeKey: item.archetype,
        name: item.archetype_name,
      };
    }
    const slot = slotByItem.get(item);
    if (slot === 'model_a' && !map[item.archetype].modelA) {
      map[item.archetype].modelA = item;
    } else if (!map[item.archetype].modelB) {
      map[item.archetype].modelB = item;
    } else if (!map[item.archetype].modelA) {
      map[item.archetype].modelA = item;
    }
  });

  return Object.values(map);
}

export function isValidInterpretation(item?: MicroInterpretation | null): boolean {
  if (!item) return false;
  return !item.execution_status || item.execution_status === 'success';
}

export function getStatusLabel(status?: string | null): string {
  switch (status) {
    case 'queued':
      return 'QUEUED';
    case 'running':
      return 'RUNNING';
    case 'completed':
    case 'complete':
      return 'COMPLETED';
    case 'partially_completed':
      return 'PARTIAL';
    case 'failed':
      return 'FAILED';
    default:
      return (status || 'UNKNOWN').toUpperCase();
  }
}

export function getExecutionStatusBadge(status?: string | null): {
  label: string;
  className: string;
} {
  switch (status) {
    case undefined:
    case null:
    case 'success':
      return {
        label: 'Success',
        className: 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60',
      };
    case 'timeout':
      return {
        label: 'Timeout',
        className: 'bg-amber-950/60 text-amber-400 border-amber-800/60',
      };
    case 'invalid_json':
      return {
        label: 'Invalid JSON',
        className: 'bg-rose-950/60 text-rose-400 border-rose-800/60',
      };
    case 'validation_failure':
      return {
        label: 'Schema Error',
        className: 'bg-rose-950/60 text-rose-400 border-rose-800/60',
      };
    case 'provider_error':
    default:
      return {
        label: 'Provider Error',
        className: 'bg-rose-950/60 text-rose-400 border-rose-800/60',
      };
  }
}

