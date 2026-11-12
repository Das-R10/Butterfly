"""
tests/test_meso.py

Unit tests for Layer 2: MESO.
Validates:
  - Influence graph topology
  - Separation of external vs internal influence
  - Convex combination preventing double-counting
  - Bounded effective spread potential [0, 1]
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.schemas import Archetype
from src.core.llm_client import create_dual_providers
from src.core.micro_layer import run_micro_layer
from src.core.meso_layer import MesoInfluenceGraph, run_meso_layer


def test_meso_graph_initialization():
    graph = MesoInfluenceGraph(external_weight=0.4)
    assert graph.n_nodes == 8
    assert graph.alpha == 0.4
    # Check normalized row stochastic property (sums to <= 1.0)
    row_sums = graph.norm_adj_matrix.sum(axis=1)
    for s in row_sums:
        assert round(s, 4) in (0.0, 1.0)


def test_meso_propagation_and_separation():
    client_a, client_b = create_dual_providers()
    interps = run_micro_layer("Tariff escalation announcement", client_a, client_b)

    state = run_meso_layer(interps, external_weight=0.4)

    assert 0.0 <= state.external_influence_score <= 1.0
    assert 0.0 <= state.internal_contagion_score <= 1.0
    assert 0.0 <= state.effective_spread_potential <= 1.0

    # Ensure no double counting: spread = alpha * ext + (1 - alpha) * int
    expected_spread = 0.4 * state.external_influence_score + 0.6 * state.internal_contagion_score
    assert round(state.effective_spread_potential, 3) == round(expected_spread, 3)

    # Check node activations
    assert "mainstream_public" in state.node_activations
    assert "engaged_subcultures" in state.node_activations

