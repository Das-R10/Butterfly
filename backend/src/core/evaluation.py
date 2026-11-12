"""
src/core/evaluation.py

Basic evaluation module for The Butterfly Algorithm.
Compares simulated adoption and attention trajectories against held-out empirical observations.
Calculates Mean Squared Error (MSE), Mean Absolute Error (MAE), Peak Timing Error,
and Directional Trend Accuracy.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from src.core.schemas import SimulationResult


class TrajectoryEvaluator:
    """Evaluates simulation accuracy against held-out observation benchmarks."""

    @staticmethod
    def evaluate(
        sim_result: SimulationResult,
        observed_adoption: List[float],
        observed_attention: Optional[List[float]] = None,
    ) -> Dict:
        """
        Compares sim_result.trajectory against ground truth observation arrays.
        """
        predicted_adoption = sim_result.trajectory.get("active_adoption", [])
        n_points = min(len(predicted_adoption), len(observed_adoption))

        if n_points == 0:
            return {"error": "Insufficient trajectory points for evaluation"}

        p_adopt = np.array(predicted_adoption[:n_points])
        o_adopt = np.array(observed_adoption[:n_points])

        # Adoption error metrics
        adoption_mse = float(np.mean((p_adopt - o_adopt) ** 2))
        adoption_mae = float(np.mean(np.abs(p_adopt - o_adopt)))

        # Peak timing accuracy (index of max value)
        pred_peak_idx = int(np.argmax(p_adopt))
        obs_peak_idx = int(np.argmax(o_adopt))
        peak_timing_error = abs(pred_peak_idx - obs_peak_idx)

        # Directional trend accuracy (did it go up/down in the same direction?)
        if n_points > 1:
            pred_diffs = np.diff(p_adopt)
            obs_diffs = np.diff(o_adopt)
            matches = (pred_diffs * obs_diffs) >= 0
            directional_accuracy = float(np.mean(matches))
        else:
            directional_accuracy = 1.0

        eval_summary = {
            "evaluated_steps": n_points,
            "adoption_mse": round(adoption_mse, 6),
            "adoption_mae": round(adoption_mae, 4),
            "adoption_rmse": round(float(np.sqrt(adoption_mse)), 4),
            "peak_step_predicted": pred_peak_idx + 1,
            "peak_step_observed": obs_peak_idx + 1,
            "peak_timing_error": peak_timing_error,
            "peak_timing_error_steps": peak_timing_error,
            "directional_accuracy": round(directional_accuracy, 4),
            "directional_accuracy_pct": f"{directional_accuracy:.0%}",
            "benchmark_passed": adoption_mae <= 0.15,
        }

        # Optional attention comparison
        if observed_attention:
            pred_attn = np.array(sim_result.trajectory.get("attention_indicator", [])[:n_points])
            o_attn = np.array(observed_attention[:n_points])
            attn_mae = float(np.mean(np.abs(pred_attn - o_attn)))
            eval_summary["attention_mae"] = round(attn_mae, 4)

        logger.info(
            f"[Evaluation] Steps: {n_points} | "
            f"Adoption MAE: {eval_summary['adoption_mae']} | "
            f"Directional Acc: {eval_summary['directional_accuracy_pct']}"
        )
        return eval_summary

