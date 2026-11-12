"""
src/core

The core 4-layer simulation engine of The Butterfly Algorithm:
  - Micro: Individual interpretations (5 archetypes × 2 models)
  - Meso: Weighted social influence graph (external + internal contagion)
  - Macro: Modified Bass diffusion model with churn / adoption fatigue
  - Feedback: Closed-loop iteration feeding macro state into round t+1
"""

from src.core.schemas import (
    Archetype,
    InterpretationOutput,
    MicroInterpretation,
    MesoNode,
    MesoEdge,
    MesoGraphState,
    MacroState,
    FeedbackState,
    IterationResult,
    SimulationConfig,
    SimulationResult,
)

__all__ = [
    "Archetype",
    "InterpretationOutput",
    "MicroInterpretation",
    "MesoNode",
    "MesoEdge",
    "MesoGraphState",
    "MacroState",
    "FeedbackState",
    "IterationResult",
    "SimulationConfig",
    "SimulationResult",
]

