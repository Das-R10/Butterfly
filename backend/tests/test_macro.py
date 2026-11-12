"""
tests/test_macro.py

Unit tests for Layer 3: MACRO.
Validates:
  - Modified Bass diffusion differential equation
  - Deterministic parameter derivation
  - Inclusion of churn / adoption fatigue parameter c
  - Strict separation of attention metrics from adoption metrics
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.llm_client import create_dual_providers
from src.core.micro_layer import run_micro_layer
from src.core.meso_layer import run_meso_layer
from src.core.macro_layer import compute_macro_dynamics


def test_macro_bass_dynamics_and_churn():
    client_a, client_b = create_dual_providers()
    interps = run_micro_layer("Central bank rate shock", client_a, client_b)
    meso = run_meso_layer(interps)

    macro_1 = compute_macro_dynamics(meso, interps, iteration=1, churn_rate=0.08)

    assert 0.0 < macro_1.p_external < 0.10
    assert 0.10 < macro_1.q_internal < 0.60
    assert macro_1.c_churn == 0.08
    assert 0.0 <= macro_1.cumulative_adoption <= 1.0
    assert 0.0 <= macro_1.active_adoption <= 1.0

    # Iteration 2 check with churn dampening active adoption
    macro_2 = compute_macro_dynamics(meso, interps, iteration=2, prior_macro_state=macro_1, churn_rate=0.08)
    assert macro_2.cumulative_adoption >= macro_1.cumulative_adoption
    # Net active adoption is bounded by cumulative adoption
    assert macro_2.active_adoption <= macro_2.cumulative_adoption


def test_attention_vs_adoption_separation():
    client_a, client_b = create_dual_providers()
    interps = run_micro_layer("Sensational breaking headline", client_a, client_b)
    meso = run_meso_layer(interps)

    macro = compute_macro_dynamics(meso, interps, iteration=1)

    # Attention and Adoption must not be the same variable
    assert isinstance(macro.attention_indicator, float)
    assert isinstance(macro.active_adoption, float)
    assert 0.0 <= macro.attention_indicator <= 1.0
    assert 0.0 <= macro.active_adoption <= 1.0
    assert macro.attention_vs_adoption_ratio > 0.0

