"""
src/core/macro_layer.py

Layer 3: MACRO — Aggregate Trend and Adoption Dynamics
Implements a modified Bass Diffusion Model with adoption fatigue/churn:
  dN/dt = [p + q * (N/M)] * (M - N) - c * N

Strictly separates Attention Indicators (simulated media buzz / conversational volume)
from Simulated Behavioral Adoption (modified Bass diffusion proxy, NOT verified empirical adoption).
100% deterministic mathematical calculations.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from loguru import logger

from src.core.schemas import (
    Archetype,
    MicroInterpretation,
    MesoGraphState,
    MacroState,
)


def compute_macro_dynamics(
    meso_state: MesoGraphState,
    micro_interpretations: List[MicroInterpretation],
    iteration: int = 1,
    prior_macro_state: Optional[MacroState] = None,
    population_ceiling: float = 1.0,
    churn_rate: float = 0.08,
) -> MacroState:
    """
    Deterministically computes Bass diffusion dynamics and attention indicators.
    
    Bass Parameters:
      p (innovation / external): Derived from Meso external influence. Scale: [0.01, 0.06]
      q (imitation / contagion): Derived from Meso internal contagion. Scale: [0.15, 0.45]
      M (potential ceiling): Default 1.0 (100% of target demographic)
      c (fatigue / churn): Rate at which adopters abandon the narrative per round
    """
    # ── Step 1: Deterministic derivation of Bass parameters from Meso graph ───
    ext_score = meso_state.external_influence_score
    int_score = meso_state.internal_contagion_score

    # Empirical parameter scaling
    # Standard Bass p ranges from 0.01 to 0.05; q ranges from 0.20 to 0.50
    p = float(np.clip(0.01 + 0.05 * ext_score, 0.005, 0.08))
    q = float(np.clip(0.15 + 0.35 * int_score, 0.10, 0.55))
    M = float(population_ceiling)
    c = float(np.clip(churn_rate, 0.01, 0.30))

    # ── Step 2: Cumulative Adoption Integration ──────────────────────────────
    initial_n0 = (
        float(np.clip(p * 0.5, 0.005, 0.05))
        if (prior_macro_state is None or iteration == 1)
        else (prior_macro_state.initial_n0 or float(np.clip(p * 0.5, 0.005, 0.05)))
    )

    if prior_macro_state is None or iteration == 1:
        # Initial baseline innovators at t=0
        n_prev = initial_n0
        prior_active = n_prev
    else:
        n_prev = prior_macro_state.cumulative_adoption
        prior_active = prior_macro_state.active_adoption

    # Modified Bass Equation:
    # dN_gross = (p + q * (n_prev / M)) * (M - n_prev)
    # churn_loss = c * prior_active
    # dN_net = dN_gross - churn_loss
    n_ratio = min(n_prev / M, 1.0)
    adoption_pressure = p + q * n_ratio
    remaining_potential = max(M - n_prev, 0.0)
    
    gross_new_adopters = adoption_pressure * remaining_potential
    churn_loss = c * prior_active
    net_velocity = gross_new_adopters - churn_loss

    # Cumulative adoption monotonically accumulates gross adoptions
    cumulative_adoption = float(np.clip(n_prev + gross_new_adopters, 0.0, M))
    # Active adoption reflects net adopters currently holding the stance: strictly A(t) <= N(t) <= M
    active_adoption = float(np.clip(min(prior_active + net_velocity, cumulative_adoption), 0.0, M))

    # ── Step 3: Compute Attention Indicator (Strictly Separated from Adoption) ─
    # Filter strictly to valid interpretations (exclude failed slots)
    valid_micro = [
        m for m in micro_interpretations
        if getattr(m, "execution_status", "success") == "success"
    ]
    viralities = [m.virality for m in valid_micro]
    emotions = [m.emotional_charge for m in valid_micro]

    mean_virality = float(np.mean(viralities)) if viralities else 0.0
    mean_emotion = float(np.mean(emotions)) if emotions else 0.0

    # Check amplifier specifically
    amp_records = [m for m in valid_micro if m.archetype == Archetype.AMPLIFIER_INFLUENCER]
    amp_boost = float(np.mean([m.virality * m.emotional_charge for m in amp_records])) if amp_records else mean_virality * mean_emotion

    # Attention index = 40% virality + 35% emotional intensity + 25% amplifier blast
    attention_raw = 0.40 * mean_virality + 0.35 * mean_emotion + 0.25 * amp_boost
    attention_indicator = float(np.clip(attention_raw, 0.0, 1.0))

    # Attention vs Adoption Ratio:
    # High ratio (> 1.5) = Viral hype bubble with low real commitment
    # Low ratio (< 0.8) = Deep behavioral penetration with low sensationalism
    denom = max(active_adoption, 0.02)
    ratio = round(attention_indicator / denom, 2)

    derivation_str = (
        f"p={p:.4f} (from Meso ext_score {ext_score:.3f}: 0.01 + 0.05*ext); "
        f"q={q:.4f} (from Meso int_score {int_score:.3f}: 0.15 + 0.35*int); "
        f"M={M:.2f}; c={c:.3f}; dt=1.0; N(0)={initial_n0:.4f}; valid_micro={len(valid_micro)}"
    )
    assumptions_str = (
        "Mean-field modified Bass differential equation assuming homogeneous sub-segment mixing; "
        "parameters p and q are derived from synthetic Micro/Meso states and represent model sensitivity projections rather than empirical certainty."
    )

    macro_state = MacroState(
        iteration=iteration,
        p_external=round(p, 4),
        q_internal=round(q, 4),
        M_potential=round(M, 4),
        c_churn=round(c, 4),
        cumulative_adoption=round(cumulative_adoption, 4),
        active_adoption=round(active_adoption, 4),
        adoption_velocity=round(net_velocity, 4),
        attention_indicator=round(attention_indicator, 4),
        attention_vs_adoption_ratio=ratio,
        dt=1.0,
        initial_n0=round(initial_n0, 4),
        formula_derivation=derivation_str,
        assumptions_and_limitations=assumptions_str,
    )

    logger.info(
        f"[Macro Layer] Iteration {iteration} | "
        f"Cumul. Sim. Diffusion: {macro_state.cumulative_adoption:.1%} | "
        f"Active Sim. Adoption: {macro_state.active_adoption:.1%} | "
        f"Velocity (dN/dt): {macro_state.adoption_velocity:+.3f} | "
        f"Sim. Attention Index: {macro_state.attention_indicator:.1%} | "
        f"Attention/Adoption Ratio: {macro_state.attention_vs_adoption_ratio}x"
    )
    return macro_state

