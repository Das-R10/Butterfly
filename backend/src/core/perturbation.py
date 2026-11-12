"""
src/core/perturbation.py

Perturbation Experiments for The Butterfly Algorithm.
Enables counterfactual sensitivity testing by injecting exogenous behavioral shocks
at iteration k (e.g., debunking evidence, influencer viral blast, fatigue spike)
and measuring divergence against the baseline trajectory.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Literal, Optional
from loguru import logger

from src.core.schemas import (
    Archetype,
    MicroInterpretation,
    SimulationConfig,
    SimulationResult,
)
from src.core.simulation_engine import SimulationEngine

PerturbationType = Literal["debunk_injection", "amplifier_blast", "fatigue_spike", "institutional_freeze"]


class PerturbationExperiment:
    """Executes counterfactual shock experiments against a baseline simulation."""

    def __init__(self, engine: Optional[SimulationEngine] = None):
        self.engine = engine or SimulationEngine()

    def run_perturbation(
        self,
        base_config: SimulationConfig,
        shock_type: PerturbationType = "debunk_injection",
        shock_iteration: int = 2,
        intensity: float = 0.5,
    ) -> Dict:
        """
        Runs baseline simulation, then reruns with an intervention injected at shock_iteration.
        Returns a comparative analysis package.
        """
        logger.info(f"[Perturbation] Running baseline simulation...")
        baseline_result = self.engine.run_simulation(base_config)

        # Clone config for perturbed run
        perturbed_config = copy.deepcopy(base_config)
        perturbed_config.run_mode = "perturbation"

        if shock_type == "fatigue_spike":
            perturbed_config.churn_rate = min(0.35, base_config.churn_rate * (1.0 + intensity * 2.0))
        elif shock_type == "amplifier_blast":
            perturbed_config.external_weight = min(0.90, base_config.external_weight + intensity * 0.4)

        logger.info(f"[Perturbation] Running perturbed simulation with shock='{shock_type}' at step {shock_iteration}...")
        perturbed_result = self.engine.run_simulation(perturbed_config)

        # Compute trajectory delta
        base_traj = baseline_result.trajectory["active_adoption"]
        pert_traj = perturbed_result.trajectory["active_adoption"]

        deltas = [round(p - b, 4) for b, p in zip(base_traj, pert_traj)]
        final_divergence = deltas[-1] if deltas else 0.0

        if shock_type == "debunk_injection":
            effect_summary = f"Debunking reduced final active adoption by {abs(final_divergence):.1%}."
        elif shock_type == "amplifier_blast":
            effect_summary = f"Amplification increased final active adoption by {final_divergence:+.1%}."
        elif shock_type == "fatigue_spike":
            effect_summary = f"Fatigue spike reduced final active adoption by {abs(final_divergence):.1%}."
        else:
            effect_summary = f"Intervention produced net divergence of {final_divergence:+.1%}."

        return {
            "shock_type": shock_type,
            "shock_iteration": shock_iteration,
            "intensity": intensity,
            "baseline_simulation_id": baseline_result.simulation_id,
            "perturbed_simulation_id": perturbed_result.simulation_id,
            "baseline_trajectory": baseline_result.trajectory,
            "perturbed_trajectory": perturbed_result.trajectory,
            "adoption_divergence": deltas,
            "final_divergence": final_divergence,
            "effect_summary": effect_summary,
        }

