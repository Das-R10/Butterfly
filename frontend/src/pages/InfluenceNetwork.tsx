// frontend/src/pages/InfluenceNetwork.tsx

import React, { useState, useMemo } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  Node,
  Edge,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useSimulation } from '../context/SimulationContext';
import { MetricCard } from '../components/common/MetricCard';
import { CrossLayerExplanationPanel } from '../components/common/CrossLayerExplanationPanel';
import { MesoNode } from '../types/simulation';
import { Network, Share2, Activity, Gauge, Info, X, AlertTriangle } from 'lucide-react';

const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  external_media_broadcast: { x: 350, y: 25 },
  external_media: { x: 350, y: 25 },
  institutional_analyst: { x: 100, y: 140 },
  skeptic_debunker: { x: 350, y: 140 },
  amplifier_influencer: { x: 600, y: 140 },
  pattern_seeker: { x: 130, y: 290 },
  doomer: { x: 570, y: 290 },
  engaged_subcultures: { x: 230, y: 430 },
  public_early_adopters: { x: 230, y: 430 },
  mainstream_public: { x: 470, y: 430 },
  general_public: { x: 470, y: 430 },
};

const NODE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  external_media_broadcast: { bg: '#1E1B4B', border: '#6366F1', text: '#A5B4FC' },
  external_media: { bg: '#1E1B4B', border: '#6366F1', text: '#A5B4FC' },
  institutional_analyst: { bg: '#082F49', border: '#0284C7', text: '#38BDF8' },
  skeptic_debunker: { bg: '#4C0519', border: '#BE123C', text: '#FB7185' },
  amplifier_influencer: { bg: '#064E3B', border: '#059669', text: '#34D399' },
  pattern_seeker: { bg: '#3B0764', border: '#7E22CE', text: '#C084FC' },
  doomer: { bg: '#451A03', border: '#D97706', text: '#FBBF24' },
  engaged_subcultures: { bg: '#14271d', border: '#10B981', text: '#6EE7B7' },
  public_early_adopters: { bg: '#14271d', border: '#10B981', text: '#6EE7B7' },
  mainstream_public: { bg: '#0F172A', border: '#38BDF8', text: '#94A3B8' },
  general_public: { bg: '#0F172A', border: '#38BDF8', text: '#94A3B8' },
};

export const InfluenceNetwork: React.FC = () => {
  const { currentSimulation, selectedIteration, setSelectedIteration } = useSimulation();
  const [selectedNode, setSelectedNode] = useState<MesoNode | null>(null);

  const iterations = currentSimulation?.iterations || [];
  const currentIterData =
    iterations.find((it) => it.iteration === selectedIteration) || iterations[iterations.length - 1];
  const mesoState = currentIterData?.meso_state;

  // Transform backend meso nodes & edges into ReactFlow structures
  const { nodes, edges } = useMemo(() => {
    if (!mesoState || !mesoState.nodes) {
      return { nodes: [], edges: [] };
    }

    const flowNodes: Node[] = mesoState.nodes.map((node) => {
      const pos = NODE_POSITIONS[node.id] || { x: 250, y: 250 };
      const color = NODE_COLORS[node.id] || { bg: '#0F172A', border: '#475569', text: '#E2E8F0' };
      const isLegacyOverflow = node.current_activation > 1.0001;
      const activationPct = Math.round(node.current_activation * 100);

      return {
        id: node.id,
        position: pos,
        data: {
          label: (
            <div className="text-center p-1">
              <div className="font-bold text-xs truncate" style={{ color: color.text }}>
                {node.label}
              </div>
              <div className="text-[10px] font-mono text-slate-300 mt-0.5">
                Act: {activationPct}%
              </div>
              {node.raw_influence != null && (
                <div className="text-[9px] font-mono text-slate-400">
                  Raw: {node.raw_influence.toFixed(2)}
                </div>
              )}
              {isLegacyOverflow && (
                <div className="text-[8px] font-mono text-amber-300 uppercase">Legacy &gt;100%</div>
              )}
            </div>
          ),
          rawNode: node,
        },
        style: {
          background: color.bg,
          border: `2px solid ${color.border}`,
          borderRadius: '10px',
          width: 140,
          boxShadow: `0 4px 12px ${color.border}33`,
          cursor: 'pointer',
        },
      };
    });

    const flowEdges: Edge[] = mesoState.edges.map((edge, idx) => {
      const isPeer = edge.channel_type === 'peer_contagion';
      const isExternal = edge.channel_type === 'external_broadcast';
      const edgeColor = isExternal ? '#38BDF8' : isPeer ? '#A855F7' : '#94A3B8';

      return {
        id: `e-${edge.source}-${edge.target}-${idx}`,
        source: edge.source,
        target: edge.target,
        animated: edge.weight > 0.4,
        style: {
          stroke: edgeColor,
          strokeWidth: Math.max(1.5, Math.min(5, edge.weight * 6)),
          opacity: 0.8,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edgeColor,
          width: 14,
          height: 14,
        },
        label: `${edge.weight.toFixed(2)}`,
        labelStyle: { fill: '#CBD5E1', fontSize: 10, fontFamily: 'monospace' },
        labelBgStyle: { fill: '#0A0E17', fillOpacity: 0.85, stroke: '#1E293B' },
      };
    });

    return { nodes: flowNodes, edges: flowEdges };
  }, [mesoState]);

  const handleNodeClick = (_: any, node: Node) => {
    if (node.data?.rawNode) {
      setSelectedNode(node.data.rawNode as MesoNode);
    }
  };

  if (!currentSimulation) {
    return (
      <div className="py-16 text-center space-y-4">
        <Network className="w-12 h-12 text-slate-600 mx-auto" />
        <h2 className="text-lg font-bold text-slate-200">No Simulation Active</h2>
        <p className="text-xs text-slate-400 max-w-sm mx-auto">
          Execute a simulation to generate and explore the weighted Meso influence network.
        </p>
      </div>
    );
  }

  const hasLegacyOverflow =
    mesoState?.nodes?.some((n) => (n.current_activation ?? 0) > 1.0001) ?? false;

  return (
    <div className="space-y-6 animate-fade-in">
      {hasLegacyOverflow && (
        <div className="p-3.5 rounded-xl bg-amber-950/35 border border-amber-800/60 text-xs text-amber-200 flex items-center gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span>
            <strong>Legacy Pre-v4.1 Archived Session:</strong> Historical node activations exceed
            100% because capacity normalization and separate raw inflow fields were introduced in
            v4.1. New simulations strictly guarantee normalized activation in [0, 1].
          </span>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
            <Network className="w-6 h-6 text-violet-400" />
            <span>Meso Layer Influence Network</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Weighted directed graph separating external broadcast media from peer-to-peer social
            contagion.
          </p>
        </div>

        {/* Iteration Selector */}
        <div className="flex items-center gap-1.5 bg-slate-900 p-1.5 rounded-xl border border-slate-800 self-start md:self-auto">
          <span className="text-[10px] font-mono uppercase text-slate-400 px-2">Iteration:</span>
          {iterations.map((it) => (
            <button
              key={it.iteration}
              onClick={() => setSelectedIteration(it.iteration)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition ${
                selectedIteration === it.iteration
                  ? 'bg-violet-500 text-slate-950 font-bold shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Round {it.iteration}
            </button>
          ))}
        </div>
      </div>

      {/* Meso Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard
          label="External Influence Score"
          value={mesoState?.external_influence_score.toFixed(3) || '0.000'}
          subtext="Aggregate broadcast media pressure"
          icon={Share2}
          accent="cyan"
        />
        <MetricCard
          label="Internal Contagion Score"
          value={mesoState?.internal_contagion_score.toFixed(3) || '0.000'}
          subtext="Peer contagion transmission coefficient"
          icon={Activity}
          accent="violet"
        />
        <MetricCard
          label="Effective Spread Potential"
          value={mesoState?.effective_spread_potential.toFixed(3) || '0.000'}
          subtext="Deduplicated network transmission capacity"
          icon={Gauge}
          accent="emerald"
        />
      </div>

      {/* Main Canvas and Details Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* ReactFlow Interactive Canvas */}
        <div className="lg:col-span-3 h-[520px] bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden relative shadow-inner">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodeClick={handleNodeClick}
            fitView
            minZoom={0.5}
            maxZoom={1.5}
          >
            <Background color="#1E293B" gap={20} size={1} variant={BackgroundVariant.Dots} />
            <Controls className="bg-slate-900 border border-slate-800 text-slate-300 rounded-lg overflow-hidden" />
          </ReactFlow>

          {/* Legend Overlay */}
          <div className="absolute bottom-3 left-3 bg-slate-950/90 border border-slate-800 rounded-lg p-2.5 backdrop-blur-sm text-[10px] font-mono space-y-1.5 pointer-events-none">
            <div className="text-slate-400 uppercase font-bold tracking-wider">Edge Channel Types</div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-cyan-400" />
              <span className="text-slate-300">External Broadcast</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-purple-400" />
              <span className="text-slate-300">Peer Contagion</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-slate-400" />
              <span className="text-slate-300">Institutional Trust</span>
            </div>
          </div>
        </div>

        {/* Node Detail Drawer */}
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Node Inspector
              </div>
              {selectedNode && (
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-slate-500 hover:text-slate-300"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {selectedNode ? (
              <div className="space-y-4">
                <div>
                  <div className="text-base font-bold text-slate-100">{selectedNode.label}</div>
                  <div className="text-xs font-mono text-cyan-400 mt-0.5">ID: {selectedNode.id}</div>
                  <div className="text-[11px] font-mono text-slate-400">Type: {selectedNode.node_type}</div>
                </div>

                <div className="space-y-2 p-3 bg-slate-950 rounded-lg border border-slate-800">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-400">Normalized Activation [0,1]</span>
                    <span className="text-cyan-400 font-bold">
                      {(selectedNode.current_activation * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-cyan-400 rounded-full transition-all"
                      style={{
                        width: `${Math.min(Math.max(selectedNode.current_activation * 100, 0), 100)}%`,
                      }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] font-mono text-slate-400 pt-1">
                    <span>Semantics: {selectedNode.activation_mode || 'instantaneous_per_iteration'}</span>
                    <span>Base Wt: {selectedNode.base_weight.toFixed(2)}</span>
                  </div>
                </div>

                {/* External vs Internal Breakdown & Raw Unnormalized Inflows */}
                {(() => {
                  const breakdown = mesoState?.node_influence_breakdown?.[selectedNode.id];
                  const extVal = selectedNode.external_contribution ?? breakdown?.external_contribution;
                  const intVal = selectedNode.internal_contribution ?? breakdown?.internal_contribution;
                  const rawInf = selectedNode.raw_influence ?? breakdown?.raw_influence;
                  const rawExt = selectedNode.raw_external_inflow ?? breakdown?.raw_external_inflow;
                  const rawInt = selectedNode.raw_internal_inflow ?? breakdown?.raw_internal_inflow;
                  const wSum = selectedNode.inbound_weight_sum ?? breakdown?.inbound_weight_sum;
                  const microInputs = breakdown?.input_contributions;
                  const basis = selectedNode.calculation_basis ?? breakdown?.calculation_basis;

                  return (
                    <div className="space-y-2.5 p-3 bg-slate-950/70 rounded-lg border border-slate-800/80 text-xs">
                      <div className="text-[10px] uppercase font-mono text-slate-400 font-bold">
                        Influence Attribution &amp; Raw Inputs
                      </div>
                      {rawInf != null && (
                        <div className="flex justify-between font-mono text-[11px]">
                          <span className="text-amber-300">Raw Unnormalized Inflow:</span>
                          <span className="text-slate-100 font-bold">{rawInf.toFixed(3)}</span>
                        </div>
                      )}
                      {rawExt != null && rawInt != null && (
                        <div className="flex justify-between font-mono text-[10px] text-slate-400">
                          <span>Raw Ext: {rawExt.toFixed(3)}</span>
                          <span>Raw Peer: {rawInt.toFixed(3)}</span>
                          {wSum != null && <span>Σw: {wSum.toFixed(2)}</span>}
                        </div>
                      )}
                      {extVal !== undefined && (
                        <div className="flex justify-between font-mono text-[11px]">
                          <span className="text-cyan-400">Normalized Ext Channel:</span>
                          <span className="text-slate-200">{(extVal * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {intVal !== undefined && (
                        <div className="flex justify-between font-mono text-[11px]">
                          <span className="text-purple-400">Normalized Peer Contagion:</span>
                          <span className="text-slate-200">{(intVal * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {microInputs && (
                        <div className="pt-2 border-t border-slate-800/60 space-y-1 text-[10px] font-mono">
                          <div className="text-slate-300 font-semibold">Micro Layer Input Profile</div>
                          {microInputs.valid_model_count === 0 ? (
                            <div className="text-rose-400">{microInputs.note || '0 valid models'}</div>
                          ) : (
                            <>
                              <div className="flex justify-between text-slate-400">
                                <span>Mean Stance / Conviction:</span>
                                <span className="text-slate-200">
                                  {microInputs.mean_stance} / {microInputs.conviction_factor}
                                </span>
                              </div>
                              <div className="flex justify-between text-slate-400">
                                <span>Virality / Emotion:</span>
                                <span className="text-slate-200">
                                  {microInputs.mean_virality} / {microInputs.mean_emotional_charge}
                                </span>
                              </div>
                              <div className="flex justify-between text-slate-400">
                                <span>Grounding / Penalty:</span>
                                <span className="text-emerald-300">
                                  {microInputs.mean_evidence_grounding_score} ({microInputs.mean_evidence_penalty}x)
                                </span>
                              </div>
                              <div className="flex justify-between text-cyan-300">
                                <span>Transmission Weight (T):</span>
                                <strong>{microInputs.transmission_weight}</strong>
                              </div>
                            </>
                          )}
                        </div>
                      )}
                      {basis && (
                        <div className="pt-1.5 border-t border-slate-800/60 text-[10px] text-slate-400 leading-snug">
                          <span className="font-semibold text-slate-300">Basis: </span>
                          {basis}
                        </div>
                      )}
                    </div>
                  );
                })()}

                {/* Connected Edges */}
                <div className="space-y-2 pt-2 border-t border-slate-800">
                  <div className="text-[11px] uppercase font-mono text-slate-400">
                    Connected Channels
                  </div>
                  <div className="space-y-1.5 max-h-40 overflow-y-auto">
                    {mesoState?.edges
                      .filter((e) => e.source === selectedNode.id || e.target === selectedNode.id)
                      .map((e, idx) => (
                        <div
                          key={idx}
                          className="p-2 bg-slate-950 rounded border border-slate-800/80 text-[10px] font-mono flex items-center justify-between"
                        >
                          <span className="text-slate-300">
                            {e.source === selectedNode.id ? `➔ ${e.target}` : `⬅ ${e.source}`}
                          </span>
                          <span className="text-cyan-400 font-bold">{e.weight.toFixed(2)}</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-slate-500 text-xs space-y-2">
                <Info className="w-8 h-8 mx-auto text-slate-600" />
                <p>Click any node on the graph to inspect activation values, influence attribution, and channel flows.</p>
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-slate-800 text-[10px] font-mono text-slate-500">
            Deduplicated propagation avoids double-counting social influence.
          </div>
        </div>
      </div>

      {/* Network Mathematical Validity & Traceability Card */}
      <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2 text-slate-200">
          <Info className="w-4 h-4 text-violet-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider">
            Network Mathematical Foundation &amp; Epistemic Distinctions
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-400 leading-relaxed">
          <div className="space-y-1.5">
            <h4 className="text-[11px] font-bold text-slate-300">
              External Broadcast Node vs External Score
            </h4>
            <p>
              <code className="text-cyan-400">external_media_broadcast</code> is an exogenous source
              with activation <code className="text-cyan-300">0.5·T(Amp) + 0.5·T(Inst)</code>.{' '}
              <code className="text-cyan-400">external_influence_score</code> combines that
              broadcast channel with direct Amplifier reach:{' '}
              <code className="text-cyan-300">0.5·Act(Ext) + 0.5·T(Amp)</code>.
            </p>
          </div>
          <div className="space-y-1.5">
            <h4 className="text-[11px] font-bold text-slate-300">
              Capacity-Normalized Activation [0,1]
            </h4>
            <p>
              Raw inflows (<code className="text-amber-300">Σ w_i·x_i</code>) can exceed 1.0 when
              multiple peers fire concurrently and are reported separately. Normalized{' '}
              <code className="text-emerald-400">current_activation</code> divides by inbound weight
              sums instantaneously per iteration so activations remain strictly in [0.0, 1.0].
            </p>
          </div>
          <div className="space-y-1.5">
            <h4 className="text-[11px] font-bold text-slate-300">
              Unbiased Conviction &amp; Evidence Penalty
            </h4>
            <p>
              Negative skepticism and positive endorsements carry equal conviction via{' '}
              <code className="text-violet-400">0.20 + 0.80·|stance|</code>. Claims marked{' '}
              <code className="text-rose-400">unsupported</code> receive a{' '}
              <code className="text-rose-300">0.40x</code> transmission penalty before propagating
              to public nodes.
            </p>
          </div>
        </div>
      </div>

      {mesoState?.event_stakeholder_mapping && mesoState.event_stakeholder_mapping.length > 0 && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <Share2 className="w-4 h-4 text-cyan-400" />
              Event-Specific Stakeholder Channel Mapping
            </h3>
            <span className="text-[10px] font-mono text-slate-400">
              Domain-grounded cohorts mapped to canonical Meso influence channels
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] font-mono uppercase text-slate-400">
                  <th className="py-2 pr-3">Stakeholder Cohort</th>
                  <th className="py-2 pr-3">Meso Channel</th>
                  <th className="py-2 pr-3">Relevance</th>
                  <th className="py-2 pr-3">Node Activation</th>
                  <th className="py-2">Causal Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {mesoState.event_stakeholder_mapping.map((item, idx) => (
                  <tr key={`${item.stakeholder}-${idx}`} className="hover:bg-slate-800/30">
                    <td className="py-2 pr-3 font-medium text-slate-200">{item.stakeholder}</td>
                    <td className="py-2 pr-3 font-mono text-[11px] text-cyan-300">
                      {item.meso_node_id.replace(/_/g, ' ')}
                    </td>
                    <td className="py-2 pr-3 font-mono text-emerald-300">
                      {Math.round((item.relevance_weight ?? 0) * 100)}%
                    </td>
                    <td className="py-2 pr-3 font-mono text-slate-300">
                      {Math.round((item.meso_node_activation ?? 0) * 100)}%
                    </td>
                    <td className="py-2 text-slate-400">
                      {item.causal_role || 'Event-grounded transmission channel'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <CrossLayerExplanationPanel
        simulation={currentSimulation}
        selectedIteration={selectedIteration}
        highlightLayer="meso"
      />
    </div>
  );
};

