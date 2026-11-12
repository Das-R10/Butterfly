"""
tests/test_micro.py

Unit tests for Layer 1: MICRO.
Validates:
  - 5 archetypes present
  - 2 independent LLM clients
  - Exact 10 interpretations per cycle
  - Validated Pydantic schema with bounded stances [-1, 1], virality [0, 1]
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.schemas import Archetype, MicroInterpretation
from src.core.llm_client import LLMClient, create_dual_providers
from src.core.micro_layer import run_micro_layer, run_archetype_interpretation


def test_dual_providers_creation():
    client_a, client_b = create_dual_providers(
        provider_a="mock",
        model_a="model_a_test",
        provider_b="mock",
        model_b="model_b_test",
    )
    assert client_a.model_name == "model_a_test"
    assert client_b.model_name == "model_b_test"


def test_micro_archetype_interpretation():
    client = LLMClient(provider="mock", model_name="test_model")
    res = run_archetype_interpretation(
        client=client,
        archetype=Archetype.SKEPTIC_DEBUNKER,
        signal="Unverified satellite rumor of border mobilization",
    )
    assert isinstance(res, MicroInterpretation)
    assert res.archetype == Archetype.SKEPTIC_DEBUNKER
    assert -1.0 <= res.stance <= 1.0
    assert 0.0 <= res.virality <= 1.0
    assert 0.0 <= res.credibility <= 1.0
    assert 0.0 <= res.emotional_charge <= 1.0
    assert len(res.primary_narrative) > 10


def test_micro_layer_exact_10_interpretations():
    client_a, client_b = create_dual_providers(provider_a="mock", provider_b="mock")
    signal = "Major power grid outage reported in metropolitan district"
    interpretations = run_micro_layer(
        signal=signal,
        client_a=client_a,
        client_b=client_b,
        iteration=1,
    )
    assert len(interpretations) == 10

    # Ensure all 5 archetypes are present for both models
    arch_a = [m.archetype for m in interpretations if m.model_name == client_a.model_name]
    arch_b = [m.archetype for m in interpretations if m.model_name == client_b.model_name]

    assert len(arch_a) == 5
    assert len(arch_b) == 5
    for a in Archetype:
        assert a in arch_a
        assert a in arch_b

