"""
tests/test_engine.py

Unit tests for Master Simulation Engine, Perturbation, and Evaluation.
Validates:
  - Complete 3 to 5 iteration loop
  - Trajectory generation
  - Model agreement score calculation
  - Narrative convergence score calculation
  - Perturbation counterfactual divergence
  - Empirical evaluation metrics
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.core.simulation_engine as simulation_engine_mod
from src.core.schemas import SimulationConfig
from src.core.simulation_engine import SimulationEngine
from src.core.perturbation import PerturbationExperiment
from src.core.evaluation import TrajectoryEvaluator


def test_simulation_engine_3_iterations():
    engine = SimulationEngine()
    cfg = SimulationConfig(signal="Critical semiconductor export embargo announced", max_iterations=3)
    res = engine.run_simulation(cfg)

    assert res.status == "completed"
    assert len(res.iterations) == 3
    assert len(res.trajectory["iterations"]) == 3
    assert len(res.trajectory["active_adoption"]) == 3
    assert len(res.trajectory["attention_indicator"]) == 3
    assert 0.0 <= res.model_agreement_score <= 1.0
    assert 0.0 <= res.narrative_convergence_score <= 1.0
    assert len(res.dominant_narrative) > 5

    # Check persistence file inside the active (isolated during pytest) SESSIONS_DIR
    saved_path = os.path.join(simulation_engine_mod.SESSIONS_DIR, f"{res.simulation_id}.json")
    assert os.path.exists(saved_path)


def test_perturbation_experiment():
    exp = PerturbationExperiment()
    cfg = SimulationConfig(signal="Supply chain cyberattack", max_iterations=3)
    res = exp.run_perturbation(cfg, shock_type="fatigue_spike", intensity=0.7)

    assert "baseline_trajectory" in res
    assert "perturbed_trajectory" in res
    assert "adoption_divergence" in res
    assert len(res["adoption_divergence"]) == 3


def test_trajectory_evaluator():
    engine = SimulationEngine()
    cfg = SimulationConfig(signal="New currency union formed", max_iterations=3)
    sim = engine.run_simulation(cfg)

    ground_truth = [0.08, 0.10, 0.12]
    eval_res = TrajectoryEvaluator.evaluate(sim, ground_truth)

    assert eval_res["evaluated_steps"] == 3
    assert "adoption_mae" in eval_res
    assert "directional_accuracy_pct" in eval_res
    assert eval_res["adoption_mae"] >= 0.0

