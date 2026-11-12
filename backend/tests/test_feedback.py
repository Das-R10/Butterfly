"""
tests/test_feedback.py

Unit tests for Layer 4: FEEDBACK.
Validates:
  - Context packaging for round t+1
  - Hard bounded recursion (3 to 5 iterations)
  - Termination flag on final cycle
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.llm_client import create_dual_providers
from src.core.micro_layer import run_micro_layer
from src.core.meso_layer import run_meso_layer
from src.core.macro_layer import compute_macro_dynamics
from src.core.feedback_layer import compute_feedback_state


def test_feedback_state_creation_and_bounds():
    client_a, client_b = create_dual_providers()
    interps = run_micro_layer("Geopolitical summit deadlock", client_a, client_b)
    meso = run_meso_layer(interps)
    macro = compute_macro_dynamics(meso, interps, iteration=1)

    # Round 1 out of 3 -> Should generate prompt context for round 2
    fb_1 = compute_feedback_state(interps, meso, macro, iteration=1, max_iterations=3)
    assert fb_1.iteration == 1
    assert "MACRO FEEDBACK FROM ROUND 1" in fb_1.prompt_context_for_next_round
    assert len(fb_1.dominant_framing) > 5

    # Round 3 out of 3 -> Terminal cycle
    fb_3 = compute_feedback_state(interps, meso, macro, iteration=3, max_iterations=3)
    assert fb_3.prompt_context_for_next_round == "SIMULATION_FINAL_CYCLE"

