"""
src/core/meso_layer.py

Layer 2: MESO — Social Influence and Information Propagation
Models information flow through a weighted 8-node influence graph.
Features:
  - Instantaneous per-iteration activation semantics strictly normalized to [0.0, 1.0]
  - Separate preservation of raw unnormalized inbound inflows (which can exceed 1.0)
  - Symmetric stance-conviction transmission weight (no positive-stance bias)
  - Explicit evidence grounding penalty preventing unverified speculation from dominating
  - Clear mathematical separation between edge weights, node activations, and aggregate scores
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple
from loguru import logger

from src.core.schemas import (
    Archetype,
    MicroInterpretation,
    MesoNode,
    MesoEdge,
    MesoGraphState,
    ModelExecutionStatus,
)
from src.core.micro_layer import compute_evidence_and_transmission_scores
from src.core.event_grounding import extract_event_stakeholders


# Canonical Meso Network Topology (Preserved 8-node structure)
DEFAULT_NODES: List[Tuple[str, str, str, float]] = [
    # (id, label, type, base_weight)
    ("institutional_analyst", "Institutional Analyst", "archetype", 1.2),
    ("skeptic_debunker", "Skeptic Debunker", "archetype", 1.0),
    ("pattern_seeker", "Pattern Seeker", "archetype", 0.9),
    ("doomer", "Doomer", "archetype", 0.9),
    ("amplifier_influencer", "Amplifier Influencer", "archetype", 1.4),
    ("mainstream_public", "Mainstream Public", "public_segment", 1.5),
    ("engaged_subcultures", "Engaged Subcultures", "public_segment", 1.1),
    ("external_media_broadcast", "External Media Broadcast", "external_channel", 1.3),
]

# Directed edge definitions: (source, target, weight, channel_type)
DEFAULT_EDGES: List[Tuple[str, str, float, str]] = [
    # External media broadcasting into nodes
    ("external_media_broadcast", "institutional_analyst", 0.85, "external_broadcast"),
    ("external_media_broadcast", "amplifier_influencer", 0.75, "external_broadcast"),
    ("external_media_broadcast", "mainstream_public", 0.40, "external_broadcast"),

    # Amplifier broadcast
    ("amplifier_influencer", "mainstream_public", 0.80, "external_broadcast"),
    ("amplifier_influencer", "engaged_subcultures", 0.85, "external_broadcast"),

    # Institutional trust channel
    ("institutional_analyst", "skeptic_debunker", 0.65, "institutional_trust"),
    ("institutional_analyst", "mainstream_public", 0.70, "institutional_trust"),

    # Skeptic dampening / peer verification
    ("skeptic_debunker", "mainstream_public", 0.55, "peer_contagion"),
    ("skeptic_debunker", "pattern_seeker", 0.40, "peer_contagion"),

    # Peer contagion from pattern seekers & doomers
    ("pattern_seeker", "doomer", 0.65, "peer_contagion"),
    ("pattern_seeker", "engaged_subcultures", 0.70, "peer_contagion"),
    ("doomer", "engaged_subcultures", 0.65, "peer_contagion"),
    ("doomer", "mainstream_public", 0.35, "peer_contagion"),

    # Public cross-talk
    ("engaged_subcultures", "mainstream_public", 0.45, "peer_contagion"),
]


class MesoInfluenceGraph:
    """Deterministic weighted network influence engine with bounded instantaneous activations."""

    def __init__(self, external_weight: float = 0.40):
        self.alpha = float(np.clip(external_weight, 0.0, 1.0))
        self.node_ids = [n[0] for n in DEFAULT_NODES]
        self.node_index = {nid: i for i, nid in enumerate(self.node_ids)}
        self.n_nodes = len(self.node_ids)

        # Build raw adjacency matrix
        self.adj_matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for src, dst, weight, _ in DEFAULT_EDGES:
            if src in self.node_index and dst in self.node_index:
                self.adj_matrix[self.node_index[src], self.node_index[dst]] = weight

        # Normalized row-stochastic matrix
        row_sums = self.adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.norm_adj_matrix = self.adj_matrix / row_sums

    def compute_propagation(
        self,
        interpretations: List[MicroInterpretation],
        signal_text: str = "",
    ) -> MesoGraphState:
        """
        Calculates instantaneous network influence propagation for iteration t
        strictly from valid Micro interpretations in iteration t.
        """
        # Filter strictly to successfully executed interpretations (never include failed slots)
        valid_interps = [
            m for m in interpretations
            if getattr(m, "execution_status", ModelExecutionStatus.SUCCESS.value) == ModelExecutionStatus.SUCCESS.value
        ]

        arch_transmissions: Dict[str, float] = {}
        arch_input_contributions: Dict[str, Dict[str, Any]] = {}

        for arch in Archetype:
            matches = [m for m in valid_interps if m.archetype == arch]
            if matches:
                stances = [float(m.stance) for m in matches]
                convictions = [0.20 + 0.80 * abs(float(m.stance)) for m in matches]
                viralities = [float(m.virality) for m in matches]
                emotions = [float(m.emotional_charge) for m in matches]
                uncertainties = [float(getattr(m, "uncertainty", 0.20)) for m in matches]
                llm_creds = [
                    float(m.model_estimated_credibility)
                    if getattr(m, "model_estimated_credibility", None) is not None
                    else float(m.credibility)
                    for m in matches
                ]

                t_weights: List[float] = []
                grounding_scores: List[float] = []
                penalties: List[float] = []

                for m in matches:
                    if getattr(m, "transmission_weight", None) is not None and getattr(m, "evidence_grounding_score", None) is not None:
                        t_weights.append(float(m.transmission_weight))
                        grounding_scores.append(float(m.evidence_grounding_score))
                        penalties.append(float(getattr(m, "evidence_penalty_applied", 1.0) or 1.0))
                    else:
                        _, _, ev_g, ev_p, t_w = compute_evidence_and_transmission_scores(
                            stance=float(m.stance),
                            virality=float(m.virality),
                            emotional_charge=float(m.emotional_charge),
                            model_estimated_credibility=float(
                                m.model_estimated_credibility
                                if getattr(m, "model_estimated_credibility", None) is not None
                                else m.credibility
                            ),
                            evidence_status=str(getattr(m, "evidence_status", "inferred")),
                            uncertainty=float(getattr(m, "uncertainty", 0.20)),
                            supporting_evidence=list(getattr(m, "supporting_evidence", []) or []),
                        )
                        t_weights.append(t_w)
                        grounding_scores.append(ev_g)
                        penalties.append(ev_p)

                mean_t = float(np.clip(np.mean(t_weights), 0.0, 1.0))
                arch_transmissions[arch.value] = mean_t
                arch_input_contributions[arch.value] = {
                    "valid_model_count": len(matches),
                    "mean_stance": round(float(np.mean(stances)), 4),
                    "conviction_factor": round(float(np.mean(convictions)), 4),
                    "stance_symmetry_rule": "0.20 + 0.80 * |stance| (negative and positive conviction weighted identically)",
                    "mean_virality": round(float(np.mean(viralities)), 4),
                    "mean_emotional_charge": round(float(np.mean(emotions)), 4),
                    "mean_model_estimated_credibility": round(float(np.mean(llm_creds)), 4),
                    "mean_evidence_grounding_score": round(float(np.mean(grounding_scores)), 4),
                    "mean_evidence_penalty": round(float(np.mean(penalties)), 4),
                    "mean_uncertainty": round(float(np.mean(uncertainties)), 4),
                    "transmission_weight": round(mean_t, 4),
                    "evidence_statuses": [str(getattr(m, "evidence_status", "supported")) for m in matches],
                }
            else:
                arch_transmissions[arch.value] = 0.0
                arch_input_contributions[arch.value] = {
                    "valid_model_count": 0,
                    "mean_stance": None,
                    "conviction_factor": 0.0,
                    "transmission_weight": 0.0,
                    "note": "No valid model interpretation produced for this archetype in this iteration.",
                }

        # Instantaneous baseline vector for iteration t (never accumulates iteration t-1 values)
        initial_vector = np.zeros(self.n_nodes, dtype=np.float64)
        for nid, val in arch_transmissions.items():
            if nid in self.node_index:
                initial_vector[self.node_index[nid]] = val

        # Exogenous source baseline for external_media_broadcast
        amp_val = arch_transmissions.get(Archetype.AMPLIFIER_INFLUENCER.value, 0.0)
        inst_val = arch_transmissions.get(Archetype.INSTITUTIONAL_ANALYST.value, 0.0)
        ext_channel_val = float(np.clip(0.5 * amp_val + 0.5 * inst_val, 0.0, 1.0))
        ext_idx = self.node_index["external_media_broadcast"]
        initial_vector[ext_idx] = ext_channel_val

        # Step 2: Aggregate External Influence Score
        # Distinct from external_media_broadcast activation alone: combines exogenous broadcast node + direct amplifier reach
        amp_idx = self.node_index[Archetype.AMPLIFIER_INFLUENCER.value]
        external_influence = float(
            np.clip(0.5 * initial_vector[ext_idx] + 0.5 * initial_vector[amp_idx], 0.0, 1.0)
        )

        # Step 3: Compute both raw inflows (can exceed 1.0) and capacity-normalized activations [0, 1]
        node_activations: Dict[str, float] = {}
        node_breakdown: Dict[str, Dict[str, Any]] = {}

        for nid, lbl, ntype, bweight in DEFAULT_NODES:
            idx = self.node_index[nid]
            base_potency = float(initial_vector[idx])

            in_ext = [
                (src, w) for src, dst, w, ctype in DEFAULT_EDGES
                if dst == nid and ctype == "external_broadcast"
            ]
            ext_w_sum = sum(w for _, w in in_ext)
            raw_ext_inflow = sum(float(initial_vector[self.node_index[src]]) * w for src, w in in_ext)
            ext_contrib = (raw_ext_inflow / ext_w_sum) if ext_w_sum > 0 else 0.0

            in_peer = [
                (src, w, ctype) for src, dst, w, ctype in DEFAULT_EDGES
                if dst == nid and ctype != "external_broadcast"
            ]
            peer_w_sum = sum(w for _, w, _ in in_peer)
            raw_peer_inflow = sum(float(initial_vector[self.node_index[src]]) * w for src, w, _ in in_peer)
            int_contrib = (raw_peer_inflow / peer_w_sum) if peer_w_sum > 0 else 0.0

            total_weight_sum = ext_w_sum + peer_w_sum
            if ntype == "external_channel":
                raw_total = base_potency
                activation = float(np.clip(base_potency, 0.0, 1.0))
                calc_basis = (
                    "Instantaneous exogenous broadcast activation: 0.5·T(Amplifier) + 0.5·T(Institutional). "
                    "Bounded in [0.0, 1.0]; has 0 inbound network edges."
                )
            elif ntype == "public_segment":
                raw_total = raw_ext_inflow + raw_peer_inflow
                if ext_w_sum > 0 and peer_w_sum > 0:
                    blended = self.alpha * ext_contrib + (1.0 - self.alpha) * int_contrib
                elif ext_w_sum > 0:
                    blended = ext_contrib
                else:
                    blended = int_contrib
                activation = float(np.clip(blended, 0.0, 1.0))
                calc_basis = (
                    f"Instantaneous capacity-normalized absorption: "
                    f"{self.alpha:.2f}·NormExt({ext_contrib:.3f}) + {1.0 - self.alpha:.2f}·NormPeer({int_contrib:.3f}) "
                    f"[Raw unnormalized inflow={raw_total:.3f}, inbound weight sum={total_weight_sum:.2f}]."
                )
            else:
                raw_total = base_potency + raw_ext_inflow + raw_peer_inflow
                if ext_w_sum > 0 and peer_w_sum > 0:
                    inbound_mix = self.alpha * ext_contrib + (1.0 - self.alpha) * int_contrib
                elif ext_w_sum > 0:
                    inbound_mix = ext_contrib
                else:
                    inbound_mix = int_contrib
                activation = float(np.clip(0.70 * base_potency + 0.30 * inbound_mix, 0.0, 1.0))
                calc_basis = (
                    f"Instantaneous archetype activation: 0.70·IntrinsicTransmission({base_potency:.3f}) "
                    f"+ 0.30·NormInboundFeedback({inbound_mix:.3f}) [Raw inflow={raw_total:.3f}]."
                )

            act_rounded = round(activation, 4)
            node_activations[nid] = act_rounded

            node_breakdown[nid] = {
                "node_id": nid,
                "label": lbl,
                "node_type": ntype,
                "activation": act_rounded,
                "activation_pct": round(act_rounded * 100.0, 1),
                "raw_influence": round(float(raw_total), 4),
                "raw_external_inflow": round(float(raw_ext_inflow), 4),
                "raw_internal_inflow": round(float(raw_peer_inflow), 4),
                "inbound_weight_sum": round(float(total_weight_sum), 4),
                "activation_mode": "instantaneous_per_iteration",
                "base_weight": bweight,
                "source_baseline": round(base_potency, 4),
                "external_contribution": round(float(np.clip(ext_contrib, 0.0, 1.0)), 4),
                "internal_contribution": round(float(np.clip(int_contrib, 0.0, 1.0)), 4),
                "input_contributions": arch_input_contributions.get(nid),
                "calculation_basis": calc_basis,
                "incoming_sources": [src for src, dst, _, _ in DEFAULT_EDGES if dst == nid],
                "outgoing_targets": [dst for src, dst, _, _ in DEFAULT_EDGES if src == nid],
                "incoming_influences": [
                    {
                        "source": src,
                        "weight": w,
                        "channel": ctype,
                        "source_activation": round(float(initial_vector[self.node_index[src]]), 4),
                        "raw_weighted_signal": round(float(initial_vector[self.node_index[src]]) * w, 4),
                    }
                    for src, dst, w, ctype in DEFAULT_EDGES if dst == nid
                ],
                "outgoing_influences": [
                    {
                        "target": dst,
                        "weight": w,
                        "channel": ctype,
                    }
                    for src, dst, w, ctype in DEFAULT_EDGES if src == nid
                ],
            }

        # Step 4: Internal contagion score (average normalized peer & public activation)
        peer_indices_keys = [
            "mainstream_public",
            "engaged_subcultures",
            Archetype.SKEPTIC_DEBUNKER.value,
            Archetype.PATTERN_SEEKER.value,
            Archetype.DOOMER.value,
        ]
        internal_contagion = float(np.clip(np.mean([node_activations[k] for k in peer_indices_keys]), 0.0, 1.0))

        # Step 5: Combined effective spread potential (convex combination avoiding double-counting)
        effective_spread = float(
            np.clip(self.alpha * external_influence + (1.0 - self.alpha) * internal_contagion, 0.0, 1.0)
        )

        nodes = [
            MesoNode(
                id=nid,
                label=lbl,
                node_type=ntype,
                base_weight=bweight,
                current_activation=node_activations[nid],
                raw_influence=node_breakdown[nid]["raw_influence"],
                raw_external_inflow=node_breakdown[nid]["raw_external_inflow"],
                raw_internal_inflow=node_breakdown[nid]["raw_internal_inflow"],
                inbound_weight_sum=node_breakdown[nid]["inbound_weight_sum"],
                external_contribution=node_breakdown[nid]["external_contribution"],
                internal_contribution=node_breakdown[nid]["internal_contribution"],
                activation_mode="instantaneous_per_iteration",
                calculation_basis=node_breakdown[nid]["calculation_basis"],
            )
            for nid, lbl, ntype, bweight in DEFAULT_NODES
        ]

        edges = [
            MesoEdge(
                source=src,
                target=dst,
                weight=w,
                channel_type=ctype,
            )
            for src, dst, w, ctype in DEFAULT_EDGES
        ]

        valid_micro_dicts = [
            m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m)
            for m in valid_interps
        ]
        event_stakeholder_mapping = extract_event_stakeholders(
            signal_text=signal_text,
            valid_micros=valid_micro_dicts,
            meso_activations=node_activations,
        )
        for nid in node_breakdown:
            node_breakdown[nid]["mapped_event_stakeholders"] = [
                row["stakeholder"]
                for row in event_stakeholder_mapping
                if row.get("meso_node_id") == nid
            ]

        explanation = {
            "activation_semantics": "instantaneous_per_iteration",
            "edge_weights": "Static directed tie strength w_ij in [0.35, 0.85].",
            "raw_inflow": "Unnormalized sum sum(w_ij * T_i) of active incoming links; can exceed 1.0 when total inbound edge weights > 1.0.",
            "node_activation": "Capacity-normalized instantaneous node state strictly bounded to [0.0, 1.0] (never accumulates prior iteration activations).",
            "external_media_vs_external_score": (
                "external_media_broadcast activation = 0.5*T(amp) + 0.5*T(inst) (exogenous broadcast output; 0 inbound edges). "
                "Aggregate external_influence_score = 0.5*a(external_media) + 0.5*T(amp) = 0.75*T(amp) + 0.25*T(inst). "
                "They share exogenous broadcast terms but are mathematically distinct metrics."
            ),
            "stance_symmetry": "Symmetric conviction 0.20 + 0.80 * |stance| treats strong skepticism and strong endorsement equally without positive-stance bias.",
            "event_stakeholder_mapping_count": len(event_stakeholder_mapping),
        }

        return MesoGraphState(
            nodes=nodes,
            edges=edges,
            external_influence_score=round(external_influence, 4),
            internal_contagion_score=round(internal_contagion, 4),
            effective_spread_potential=round(effective_spread, 4),
            activation_semantics="instantaneous_per_iteration",
            meso_metrics_explanation=explanation,
            node_activations=node_activations,
            node_influence_breakdown=node_breakdown,
            event_stakeholder_mapping=event_stakeholder_mapping,
        )


def run_meso_layer(
    interpretations: List[MicroInterpretation],
    external_weight: float = 0.40,
    signal_text: str = "",
) -> MesoGraphState:
    """Convenience function running the deterministic Meso layer."""
    graph = MesoInfluenceGraph(external_weight=external_weight)
    state = graph.compute_propagation(interpretations, signal_text=signal_text)
    logger.info(
        f"[Meso Layer] External: {state.external_influence_score:.3f} | "
        f"Internal: {state.internal_contagion_score:.3f} | "
        f"Effective Spread: {state.effective_spread_potential:.3f}"
    )
    return state
