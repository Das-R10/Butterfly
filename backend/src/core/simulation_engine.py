"""
src/core/simulation_engine.py

Master Orchestrator for The Butterfly Algorithm.
Executes the closed-loop 4-layer simulation:
  Micro (10 interpretation slots) -> Meso (Network Graph) -> Macro (Bass Diffusion) -> Feedback (Iterative Loop)
Enforces:
  - Bounded 3 to 5 iterations
  - Valid-pair-only Model Agreement Score (returning None/N/A when <2 valid models exist)
  - Honest Narrative Convergence tracking (stance variance evolution over valid outputs)
  - Explicit execution statuses (queued | running | completed | partially_completed | failed)
  - Full telemetry diagnostics without secret leakage
  - JSON persistence in data/sessions/ keyed strictly by unique simulation_id
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from src.core.schemas import (
    Archetype,
    CORE_RESEARCH_QUESTION,
    CausalPathway,
    EpistemicBreakdown,
    FinalVerdict,
    FutureScenario,
    MicroInterpretation,
    MesoGraphState,
    MacroState,
    FeedbackState,
    IterationResult,
    ModelExecutionStatus,
    OrderEffectsSummary,
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    StakeholderReaction,
    TimeHorizonForecast,
)
from src.core.llm_client import LLMClient, create_dual_providers
from src.core.micro_layer import run_micro_layer
from src.core.meso_layer import run_meso_layer
from src.core.macro_layer import compute_macro_dynamics
from src.core.feedback_layer import compute_feedback_state
from src.core.forecasting_engine import build_forecast_bundle
from src.core.forecast_tracking import save_forecast_snapshot
from src.core.gpu_diagnostics import get_gpu_diagnostics
from src.core.ollama_service import check_ollama_health, is_model_installed

PRODUCTION_SESSIONS_DIR = os.getenv("BUTTERFLY_SESSIONS_DIR", "data/sessions")
TEST_SESSIONS_DIR = os.getenv("BUTTERFLY_TEST_SESSIONS_DIR", "data/test_sessions")
REPORTS_DIR = os.getenv("BUTTERFLY_REPORTS_DIR", "data/reports")
BACKUPS_DIR = os.getenv("BUTTERFLY_BACKUPS_DIR", "data/backups")
SESSIONS_DIR = PRODUCTION_SESSIONS_DIR
OUTPUTS_DIR = os.getenv("BUTTERFLY_OUTPUTS_DIR", "data/outputs")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "have", "has", "was", "were", "are", "been", "not", "but", "they", "their",
    "there", "which", "would", "could", "should", "will", "more", "than",
}


def derive_event_title(signal_text: str, explicit_title: Optional[str] = None) -> str:
    """Returns explicit title or derives a concise event title from the signal text."""
    if explicit_title and explicit_title.strip():
        return explicit_title.strip()[:100]
    cleaned = re.sub(r"\s+", " ", (signal_text or "").strip())
    if not cleaned:
        return "Untitled Simulation"
    first_clause = re.split(r"[.!?;\n]", cleaned, maxsplit=1)[0].strip()
    candidate = first_clause or cleaned
    if len(candidate) > 64:
        candidate = candidate[:61].rsplit(" ", 1)[0] + "..."
    return candidate


def _extract_semantic_tokens(interp: MicroInterpretation) -> set[str]:
    tokens: set[str] = set()
    for tag in (interp.narrative_tags or []):
        for part in re.split(r"[^a-zA-Z0-9]+", tag.lower()):
            if len(part) >= 3 and part not in _STOPWORDS:
                tokens.add(part)
    text_blob = f"{interp.primary_narrative or ''} {' '.join(interp.claims or [])}"
    for part in re.split(r"[^a-zA-Z0-9]+", text_blob.lower()):
        if len(part) >= 4 and part not in _STOPWORDS:
            tokens.add(part)
    return tokens


def _extract_causal_reasoning_tokens(interp: MicroInterpretation) -> set[str]:
    """Extracts tokens representing analytical question, stakeholder focus, causal mechanisms, opposing/supporting logic, actions, shifts, consequences, and triggers."""
    tokens: set[str] = set()
    parts_list: List[str] = []
    for attr in (
        "analytical_question",
        "stakeholder_focus",
        "causal_mechanism",
        "strongest_supporting_consideration",
        "strongest_opposing_consideration",
    ):
        val = getattr(interp, attr, "")
        if val:
            parts_list.append(str(val))
    parts_list.extend(str(x) for x in (getattr(interp, "expected_stakeholder_actions", None) or []))
    parts_list.extend(str(x) for x in (getattr(interp, "behavioural_changes", None) or []))
    parts_list.extend(str(x) for x in (getattr(interp, "emergent_consequences", None) or []))
    parts_list.extend(str(x) for x in (getattr(interp, "trigger_conditions", None) or []))
    parts_list.extend(str(x) for x in (getattr(interp, "invalidation_conditions", None) or []))
    blob = " ".join(parts_list).strip()
    if not blob:
        return tokens
    for part in re.split(r"[^a-zA-Z0-9]+", blob.lower()):
        if len(part) >= 4 and part not in _STOPWORDS:
            tokens.add(part)
    return tokens


def _partition_models(interpretations: List[MicroInterpretation]) -> Tuple[Dict[Archetype, MicroInterpretation], Dict[Archetype, MicroInterpretation]]:
    """
    Separates valid interpretations into (model_a_by_arch, model_b_by_arch).
    Prioritizes explicit `model_slot`; falls back to distinct model names for legacy files.
    """
    valid = [
        m for m in interpretations
        if getattr(m, "execution_status", ModelExecutionStatus.SUCCESS.value) == ModelExecutionStatus.SUCCESS.value
    ]
    slot_a: Dict[Archetype, MicroInterpretation] = {}
    slot_b: Dict[Archetype, MicroInterpretation] = {}

    has_slots = any(getattr(m, "model_slot", None) in ("model_a", "model_b") for m in valid)
    if has_slots:
        for m in valid:
            slot = getattr(m, "model_slot", None)
            if slot == "model_a" and m.archetype not in slot_a:
                slot_a[m.archetype] = m
            elif slot == "model_b" and m.archetype not in slot_b:
                slot_b[m.archetype] = m
        return slot_a, slot_b

    # Legacy fallback (e.g. historical JSONs lacking model_slot)
    distinct_models: List[str] = []
    for m in valid:
        if m.model_name not in distinct_models:
            distinct_models.append(m.model_name)
    if len(distinct_models) < 2:
        # Check if duplicate model name appeared twice per archetype
        counts: Dict[Archetype, int] = {}
        for m in valid:
            cnt = counts.get(m.archetype, 0)
            if cnt == 0:
                slot_a[m.archetype] = m
            elif cnt == 1:
                slot_b[m.archetype] = m
            counts[m.archetype] = cnt + 1
        return slot_a, slot_b

    name_a, name_b = distinct_models[0], distinct_models[1]
    for m in valid:
        if m.model_name == name_a and m.archetype not in slot_a:
            slot_a[m.archetype] = m
        elif m.model_name == name_b and m.archetype not in slot_b:
            slot_b[m.archetype] = m
    return slot_a, slot_b


def compute_model_agreement_details(interpretations: List[MicroInterpretation]) -> Dict[str, Any]:
    """
    Computes multi-metric cross-model agreement strictly on valid matched (Model A, Model B) pairs.
    Distinguishes:
      - stance_agreement: 1 - |stance_A - stance_B| / 2.0
      - stance_correlation: Pearson r across matched pairs (when n>=2 & variance > 0)
      - tag_semantic_similarity: Jaccard similarity of narrative tags, claims & primary narrative
      - causal_semantic_similarity: Jaccard similarity of causal mechanism, actions, behaviours, consequences & triggers
      - reasoning_agreement: 0.40 * tag_semantic_similarity + 0.60 * causal_semantic_similarity
    Penalizes superficial stance agreement when causal mechanisms & narrative themes diverge.
    Returns score=None (N/A) if <2 valid models or 0 matched archetype pairs exist.
    """
    slot_a, slot_b = _partition_models(interpretations)
    pairs: List[Tuple[MicroInterpretation, MicroInterpretation]] = []
    for arch in Archetype:
        if arch in slot_a and arch in slot_b:
            pairs.append((slot_a[arch], slot_b[arch]))

    total_archetypes = len(Archetype)
    if not slot_a or not slot_b or len(pairs) == 0:
        missing_info = []
        if not slot_a:
            missing_info.append("Model A produced 0 valid outputs")
        if not slot_b:
            missing_info.append("Model B produced 0 valid outputs")
        reason = (
            "Model Agreement N/A: " + "; ".join(missing_info)
            if missing_info
            else "Model Agreement N/A: No matched archetype pairs completed by both models."
        )
        return {
            "available": False,
            "score": None,
            "valid_pairs": 0,
            "total_archetypes": total_archetypes,
            "stance_agreement": None,
            "stance_correlation": None,
            "tag_semantic_similarity": None,
            "causal_semantic_similarity": None,
            "reasoning_agreement": None,
            "stance_without_reasoning_warning": False,
            "formula": "Requires valid matched (Model A, Model B) archetype pairs.",
            "reason": reason,
        }

    stance_agreements: List[float] = []
    semantic_sims: List[float] = []
    causal_sims: List[float] = []
    reasoning_agrs: List[float] = []
    stances_a: List[float] = []
    stances_b: List[float] = []

    for a, b in pairs:
        stances_a.append(float(a.stance))
        stances_b.append(float(b.stance))
        dist = abs(float(a.stance) - float(b.stance)) / 2.0
        stance_agreements.append(float(np.clip(1.0 - dist, 0.0, 1.0)))

        tok_a = _extract_semantic_tokens(a)
        tok_b = _extract_semantic_tokens(b)
        jacc = (len(tok_a & tok_b) / len(tok_a | tok_b)) if (tok_a or tok_b) else 0.0
        semantic_sims.append(jacc)

        ctok_a = _extract_causal_reasoning_tokens(a)
        ctok_b = _extract_causal_reasoning_tokens(b)
        if ctok_a or ctok_b:
            c_jacc = len(ctok_a & ctok_b) / len(ctok_a | ctok_b)
        else:
            c_jacc = jacc
        causal_sims.append(c_jacc)
        reasoning_agrs.append(0.40 * jacc + 0.60 * c_jacc)

    mean_stance_agr = float(np.mean(stance_agreements))
    mean_sem_sim = float(np.mean(semantic_sims))
    mean_causal_sim = float(np.mean(causal_sims))
    mean_reasoning_agr = float(np.mean(reasoning_agrs))

    # Pearson r on stances (only meaningful if >= 2 pairs and non-zero variance in both)
    stance_corr: Optional[float] = None
    if len(pairs) >= 2 and float(np.std(stances_a)) > 1e-6 and float(np.std(stances_b)) > 1e-6:
        corr_val = float(np.corrcoef(stances_a, stances_b)[0, 1])
        if not np.isnan(corr_val):
            stance_corr = round(float(np.clip(corr_val, -1.0, 1.0)), 4)

    stance_without_reasoning = bool(mean_stance_agr >= 0.75 and mean_reasoning_agr < 0.35)
    divergence_penalty = 0.25 * max(0.0, mean_stance_agr - mean_reasoning_agr - 0.25)
    composite_score = round(
        float(np.clip(0.45 * mean_stance_agr + 0.55 * mean_reasoning_agr - divergence_penalty, 0.0, 1.0)),
        4,
    )

    reason_notes: List[str] = []
    if len(pairs) < total_archetypes:
        reason_notes.append(f"Partial comparison based on {len(pairs)}/{total_archetypes} valid archetype pairs.")
    if stance_without_reasoning:
        reason_notes.append(
            "Diagnostic warning: Models share similar stance scores despite divergent causal reasoning and thematic tokens."
        )

    return {
        "available": True,
        "score": composite_score,
        "valid_pairs": len(pairs),
        "total_archetypes": total_archetypes,
        "stance_agreement": round(mean_stance_agr, 4),
        "stance_correlation": stance_corr,
        "tag_semantic_similarity": round(mean_sem_sim, 4),
        "causal_semantic_similarity": round(mean_causal_sim, 4),
        "reasoning_agreement": round(mean_reasoning_agr, 4),
        "stance_without_reasoning_warning": stance_without_reasoning,
        "formula": "0.45 * StanceAgr + 0.55 * ReasoningAgr (0.4*TagClaimSim + 0.6*CausalSim) - divergence_penalty",
        "reason": " ".join(reason_notes) if reason_notes else None,
    }


def compute_model_agreement_score(interpretations: List[MicroInterpretation]) -> Optional[float]:
    """
    Backwards-compatible wrapper returning aggregate agreement score in [0.0, 1.0],
    or None (N/A) when fewer than 2 valid models completed matched archetypes.
    """
    return compute_model_agreement_details(interpretations)["score"]


def _inter_archetype_similarity(interps: List[MicroInterpretation]) -> float:
    """Computes average pairwise Jaccard similarity across distinct valid archetypes in one round."""
    by_arch: Dict[Archetype, set[str]] = {}
    for m in interps:
        if getattr(m, "execution_status", ModelExecutionStatus.SUCCESS.value) != ModelExecutionStatus.SUCCESS.value:
            continue
        toks = _extract_semantic_tokens(m) | _extract_causal_reasoning_tokens(m)
        if m.archetype not in by_arch:
            by_arch[m.archetype] = set(toks)
        else:
            by_arch[m.archetype] |= toks
    archs = list(by_arch.keys())
    if len(archs) < 2:
        return 0.0
    sims: List[float] = []
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            ta, tb = by_arch[archs[i]], by_arch[archs[j]]
            sims.append((len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0)
    return float(np.mean(sims)) if sims else 0.0


def _inter_archetype_reasoning_similarity(interps: List[MicroInterpretation]) -> float:
    """Computes average pairwise Jaccard similarity strictly on causal/analytical reasoning tokens across archetypes."""
    by_arch: Dict[Archetype, set[str]] = {}
    for m in interps:
        if getattr(m, "execution_status", ModelExecutionStatus.SUCCESS.value) != ModelExecutionStatus.SUCCESS.value:
            continue
        toks = _extract_causal_reasoning_tokens(m)
        if not toks:
            toks = _extract_semantic_tokens(m)
        if m.archetype not in by_arch:
            by_arch[m.archetype] = set(toks)
        else:
            by_arch[m.archetype] |= toks
    archs = list(by_arch.keys())
    if len(archs) < 2:
        return 0.0
    sims: List[float] = []
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            ta, tb = by_arch[archs[i]], by_arch[archs[j]]
            sims.append((len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0)
    return float(np.mean(sims)) if sims else 0.0


def compute_narrative_convergence_details(iterations: List[IterationResult]) -> Dict[str, Any]:
    """
    Measures whether valid archetype interpretations converged or polarized across iterations
    by comparing stance variance at round 1 vs round K, plus semantic convergence,
    causal reasoning differentiation across archetypes, and true role-collapse detection.
    """
    if len(iterations) < 2:
        return {
            "available": False,
            "score": None,
            "var_initial": None,
            "var_final": None,
            "direction": "insufficient_iterations",
            "semantic_convergence": None,
            "archetype_stance_std_final": None,
            "inter_archetype_similarity_final": None,
            "inter_archetype_reasoning_similarity": None,
            "reasoning_differentiation_score": None,
            "legitimate_consensus_detected": False,
            "archetype_collapse_warning": False,
            "formula": "1 - (Var(stance_K) / Var(stance_1)) when Var(stance_K) <= Var(stance_1), else 0.0",
            "reason": "Convergence requires at least 2 completed iterations.",
        }

    initial_valid = [
        m for m in iterations[0].micro_interpretations
        if getattr(m, "execution_status", ModelExecutionStatus.SUCCESS.value) == ModelExecutionStatus.SUCCESS.value
    ]
    final_valid = [
        m for m in iterations[-1].micro_interpretations
        if getattr(m, "execution_status", ModelExecutionStatus.SUCCESS.value) == ModelExecutionStatus.SUCCESS.value
    ]
    initial_stances = [float(m.stance) for m in initial_valid]
    final_stances = [float(m.stance) for m in final_valid]

    if len(initial_stances) < 2 or len(final_stances) < 2:
        return {
            "available": False,
            "score": None,
            "var_initial": None,
            "var_final": None,
            "direction": "insufficient_valid_outputs",
            "semantic_convergence": None,
            "archetype_stance_std_final": None,
            "inter_archetype_similarity_final": None,
            "inter_archetype_reasoning_similarity": None,
            "reasoning_differentiation_score": None,
            "legitimate_consensus_detected": False,
            "archetype_collapse_warning": False,
            "formula": "1 - (Var(stance_K) / Var(stance_1))",
            "reason": "Convergence N/A: fewer than 2 valid interpretations in initial or final round.",
        }

    var_initial = float(np.var(initial_stances))
    var_final = float(np.var(final_stances))

    if var_initial <= 1e-4 and var_final <= 1e-4:
        convergence = 1.0
        direction = "stable_consensus"
    elif var_initial <= 1e-4:
        convergence = 0.0
        direction = "diverging_polarizing"
    elif var_final <= var_initial:
        ratio = var_final / var_initial
        convergence = round(float(np.clip(1.0 - ratio, 0.0, 1.0)), 4)
        direction = "converging" if ratio < 0.98 else "stable"
    else:
        # Variance expanded across iterations -> archetypes polarized/diverged
        convergence = 0.0
        direction = "diverging_polarizing"

    # Archetype-level separation & collapse audit in final round (distinguishing legitimate consensus from reasoning collapse)
    arch_means: Dict[Archetype, List[float]] = {}
    for m in final_valid:
        arch_means.setdefault(m.archetype, []).append(float(m.stance))
    per_arch_stance = [float(np.mean(v)) for v in arch_means.values()]
    arch_stance_std = round(float(np.std(per_arch_stance)), 4) if len(per_arch_stance) >= 2 else 0.0

    sim_initial = _inter_archetype_similarity(initial_valid)
    sim_final = _inter_archetype_similarity(final_valid)
    reasoning_sim_final = _inter_archetype_reasoning_similarity(final_valid)
    reasoning_diff_score = round(float(np.clip(1.0 - reasoning_sim_final, 0.0, 1.0)), 4)
    sem_conv = round(float(np.clip(sim_final - sim_initial, -1.0, 1.0)), 4)

    stances_close = len(per_arch_stance) >= 3 and arch_stance_std < 0.08
    collapse_warning = bool(stances_close and (sim_final >= 0.45 or reasoning_sim_final >= 0.45))
    legitimate_consensus = bool(stances_close and not collapse_warning)

    reason_msg: Optional[str] = None
    if collapse_warning:
        reason_msg = (
            "Archetype collapse warning: distinct archetypes collapsed onto near-identical stances AND overlapping causal reasoning."
        )
    elif legitimate_consensus:
        reason_msg = (
            "Directional consensus without role collapse: archetypes share similar overall impact stance while maintaining distinct causal reasoning."
        )

    return {
        "available": True,
        "score": convergence,
        "var_initial": round(var_initial, 4),
        "var_final": round(var_final, 4),
        "direction": direction,
        "semantic_convergence": sem_conv,
        "archetype_stance_std_final": arch_stance_std,
        "inter_archetype_similarity_final": round(sim_final, 4),
        "inter_archetype_reasoning_similarity": round(reasoning_sim_final, 4),
        "reasoning_differentiation_score": reasoning_diff_score,
        "legitimate_consensus_detected": legitimate_consensus,
        "archetype_collapse_warning": collapse_warning,
        "formula": "1 - (Var(stance_K) / Var(stance_1)) for shrinking variance; 0.0 if variance expands (polarization).",
        "reason": reason_msg,
    }


def compute_narrative_convergence_score(iterations: List[IterationResult]) -> Optional[float]:
    """Returns stance convergence score in [0.0, 1.0], or None if insufficient valid outputs."""
    return compute_narrative_convergence_details(iterations)["score"]


class SimulationEngine:
    """Master orchestrator executing the 4-layer Butterfly Algorithm simulation."""

    def __init__(
        self,
        client_a: Optional[LLMClient] = None,
        client_b: Optional[LLMClient] = None,
    ):
        self._explicit_clients = bool(client_a and client_b)
        if client_a and client_b:
            self.client_a = client_a
            self.client_b = client_b
        else:
            self.client_a, self.client_b = create_dual_providers()

    def run_simulation(
        self,
        config: SimulationConfig,
        progress_callback: Optional[Callable[[str, int, Dict], None]] = None,
        simulation_id: Optional[str] = None,
    ) -> SimulationResult:
        """
        Runs the full closed-loop 4-layer simulation for 3 to 5 iterations.
        Assigns/preserves explicit `simulation_id`, tracks accurate execution statuses,
        and never substitutes failed models or returns fake 100% agreement.
        """
        start_time = time.time()
        sim_id = simulation_id or f"sim_{uuid.uuid4().hex[:10]}_{int(time.time())}"
        signal_text = config.signal.strip()
        event_title = derive_event_title(signal_text, config.event_title)
        max_iter = int(np.clip(config.max_iterations, 3, 5))

        client_a = self.client_a
        client_b = self.client_b
        if (not self._explicit_clients and (config.provider_a or config.model_a or config.provider_b or config.model_b)) or (
            config.provider_a is not None or config.provider_b is not None
        ):
            client_a, client_b = create_dual_providers(
                provider_a=config.provider_a,
                model_a=config.model_a,
                provider_b=config.provider_b,
                model_b=config.model_b,
            )

        # Pre-flight availability check for real providers (fail fast, no silent mock fallback)
        for client in (client_a, client_b):
            if client.provider == "ollama":
                health = check_ollama_health()
                if not health.get("running"):
                    raise RuntimeError("Ollama is not available. Install Ollama and start its local service before running local inference.")
                if not is_model_installed(client.model_name):
                    raise RuntimeError(f"Ollama model '{client.model_name}' is not installed. Run 'ollama pull {client.model_name}' before running local inference.")
            elif client.provider == "groq":
                if not os.getenv("GROQ_API_KEY"):
                    raise RuntimeError("Groq Cloud provider selected but GROQ_API_KEY is not configured in backend environment.")
            elif client.provider in ("gemini", "google"):
                if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                    raise RuntimeError("Google Gemini API provider selected but GEMINI_API_KEY is not configured in backend environment.")

        logger.info(f"=== Starting Simulation {sim_id} ({max_iter} iterations) ===")
        logger.info(f"Signal: {signal_text[:80]}...")
        logger.info(f"Model A: {client_a.provider} / {client_a.model_name} | Model B: {client_b.provider} / {client_b.model_name}")

        iteration_results: List[IterationResult] = []
        diagnostics_log: List[Dict[str, Any]] = []
        current_feedback_context = ""
        prior_macro: Optional[MacroState] = None

        trajectory: Dict[str, List[Any]] = {
            "iterations": [],
            "steps": [],
            "cumulative_adoption": [],
            "active_adoption": [],
            "adoption_velocity": [],
            "attention_indicator": [],
            "churn_loss": [],
            "churn_volume": [],
        }

        per_iter_agreement_details: List[Dict[str, Any]] = []

        for step in range(1, max_iter + 1):
            logger.info(f"--- Iteration {step}/{max_iter} ---")

            if progress_callback:
                progress_callback("micro", step, {"simulation_id": sim_id, "status": "running_micro", "interpretations_count": 10})

            # ── Layer 1: MICRO (10 interpretation slots) ──
            micro_interps = run_micro_layer(
                signal=signal_text,
                client_a=client_a,
                client_b=client_b,
                feedback_context=current_feedback_context,
                iteration=step,
                source_url=config.source_url or "",
                diagnostics_out=diagnostics_log,
            )

            iter_agr_detail = compute_model_agreement_details(micro_interps)
            per_iter_agreement_details.append(iter_agr_detail)

            if progress_callback:
                progress_callback("meso", step, {
                    "simulation_id": sim_id,
                    "status": "running_meso",
                    "agreement": iter_agr_detail["score"],
                    "valid_pairs": iter_agr_detail["valid_pairs"],
                })

            # ── Layer 2: MESO (Weighted Influence Graph) ──
            meso_state = run_meso_layer(
                interpretations=micro_interps,
                external_weight=config.external_weight,
                signal_text=signal_text,
            )

            if progress_callback:
                progress_callback("macro", step, {
                    "simulation_id": sim_id,
                    "status": "running_macro",
                    "effective_spread": meso_state.effective_spread_potential,
                })

            # ── Layer 3: MACRO (Modified Bass Diffusion) ──
            macro_state = compute_macro_dynamics(
                meso_state=meso_state,
                micro_interpretations=micro_interps,
                iteration=step,
                prior_macro_state=prior_macro,
                population_ceiling=config.population_ceiling,
                churn_rate=config.churn_rate,
            )
            prior_macro = macro_state

            # Record trajectory (dual keys for backward/frontend compatibility)
            churn_val = round(macro_state.c_churn * macro_state.active_adoption, 4)
            trajectory["iterations"].append(step)
            trajectory["steps"].append(step)
            trajectory["cumulative_adoption"].append(macro_state.cumulative_adoption)
            trajectory["active_adoption"].append(macro_state.active_adoption)
            trajectory["adoption_velocity"].append(macro_state.adoption_velocity)
            trajectory["attention_indicator"].append(macro_state.attention_indicator)
            trajectory["churn_loss"].append(churn_val)
            trajectory["churn_volume"].append(churn_val)

            if progress_callback:
                progress_callback("feedback", step, {
                    "simulation_id": sim_id,
                    "status": "running_feedback",
                    "active_adoption": macro_state.active_adoption,
                })

            # ── Layer 4: FEEDBACK (Closed Loop State) ──
            feedback_state = compute_feedback_state(
                micro_interpretations=micro_interps,
                meso_state=meso_state,
                macro_state=macro_state,
                iteration=step,
                max_iterations=max_iter,
                prior_interpretations=iteration_results[-1].micro_interpretations if iteration_results else None,
                prior_meso_state=iteration_results[-1].meso_state if iteration_results else None,
                prior_macro_state=iteration_results[-1].macro_state if iteration_results else None,
                original_signal=signal_text,
            )
            current_feedback_context = feedback_state.prompt_context_for_next_round

            iter_result = IterationResult(
                iteration=step,
                micro_interpretations=micro_interps,
                meso_state=meso_state,
                macro_state=macro_state,
                feedback_state=feedback_state,
            )
            iteration_results.append(iter_result)

            if progress_callback:
                progress_callback("completed_iteration", step, {
                    "simulation_id": sim_id,
                    "iteration": step,
                    "active_adoption": macro_state.active_adoption,
                    "attention": macro_state.attention_indicator,
                })

        # Overall Agreement Aggregation (strictly over valid pairs; None/N/A if unavailable)
        valid_iter_agrs = [d for d in per_iter_agreement_details if d.get("available") and d.get("score") is not None]
        if valid_iter_agrs:
            mean_agreement = round(float(np.mean([d["score"] for d in valid_iter_agrs])), 4)
            stance_agr_vals = [d["stance_agreement"] for d in valid_iter_agrs if d.get("stance_agreement") is not None]
            sem_sim_vals = [d["tag_semantic_similarity"] for d in valid_iter_agrs if d.get("tag_semantic_similarity") is not None]
            causal_sim_vals = [d["causal_semantic_similarity"] for d in valid_iter_agrs if d.get("causal_semantic_similarity") is not None]
            reason_agr_vals = [d["reasoning_agreement"] for d in valid_iter_agrs if d.get("reasoning_agreement") is not None]
            corr_vals = [d["stance_correlation"] for d in valid_iter_agrs if d.get("stance_correlation") is not None]
            has_reasoning_warn = any(bool(d.get("stance_without_reasoning_warning")) for d in valid_iter_agrs)
            overall_agr_details: Dict[str, Any] = {
                "available": True,
                "score": mean_agreement,
                "valid_pairs": valid_iter_agrs[-1]["valid_pairs"],
                "total_archetypes": len(Archetype),
                "stance_agreement": round(float(np.mean(stance_agr_vals)), 4) if stance_agr_vals else None,
                "stance_correlation": round(float(np.mean(corr_vals)), 4) if corr_vals else None,
                "tag_semantic_similarity": round(float(np.mean(sem_sim_vals)), 4) if sem_sim_vals else None,
                "causal_semantic_similarity": round(float(np.mean(causal_sim_vals)), 4) if causal_sim_vals else None,
                "reasoning_agreement": round(float(np.mean(reason_agr_vals)), 4) if reason_agr_vals else None,
                "stance_without_reasoning_warning": has_reasoning_warn,
                "formula": valid_iter_agrs[-1]["formula"],
                "reason": valid_iter_agrs[-1].get("reason"),
            }
        else:
            mean_agreement = None
            overall_agr_details = per_iter_agreement_details[-1] if per_iter_agreement_details else {
                "available": False,
                "score": None,
                "valid_pairs": 0,
                "total_archetypes": len(Archetype),
                "causal_semantic_similarity": None,
                "reasoning_agreement": None,
                "stance_without_reasoning_warning": False,
                "reason": "Model Agreement N/A: fewer than 2 valid models completed matched archetypes.",
            }

        convergence_details = compute_narrative_convergence_details(iteration_results)
        convergence_score = convergence_details["score"]

        # Execution accounting & honest overall status
        expected_count = max_iter * 10
        all_interps = [m for it in iteration_results for m in it.micro_interpretations]
        success_count = sum(1 for m in all_interps if m.execution_status == ModelExecutionStatus.SUCCESS.value)

        failed_by_model: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for m in all_interps:
            if m.execution_status != ModelExecutionStatus.SUCCESS.value:
                key = (m.model_slot or "unknown", m.model_name, m.provider)
                if key not in failed_by_model:
                    failed_by_model[key] = {
                        "model_slot": m.model_slot or "unknown",
                        "model_name": m.model_name,
                        "provider": m.provider,
                        "failure_count": 0,
                        "execution_statuses": [],
                        "error_categories": [],
                        "sample_error": m.error_message,
                    }
                failed_by_model[key]["failure_count"] += 1
                if m.execution_status not in failed_by_model[key]["execution_statuses"]:
                    failed_by_model[key]["execution_statuses"].append(m.execution_status)
                if m.error_category and m.error_category not in failed_by_model[key]["error_categories"]:
                    failed_by_model[key]["error_categories"].append(m.error_category)

        failed_models_list = list(failed_by_model.values())

        if success_count == expected_count and expected_count > 0:
            sim_status = SimulationStatus.COMPLETED.value
            status_details = f"All {success_count}/{expected_count} archetype interpretations succeeded."
        elif success_count > 0:
            sim_status = SimulationStatus.PARTIALLY_COMPLETED.value
            failed_names = ", ".join(f"{f['model_slot']} ({f['model_name']})" for f in failed_models_list)
            status_details = (
                f"Partially completed: {success_count}/{expected_count} interpretations succeeded. "
                f"Failures in: {failed_names}."
            )
        else:
            sim_status = SimulationStatus.FAILED.value
            status_details = (
                f"Simulation failed: 0/{expected_count} valid model interpretations generated."
            )

        final_dominant = (
            iteration_results[-1].feedback_state.dominant_framing
            if success_count > 0
            else "Simulation Failed — No valid model interpretations generated."
        )
        total_duration = round(time.time() - start_time, 2)

        diagnostics_data = {
            "gpu": get_gpu_diagnostics(),
            "selected_models": {
                "model_a": {"provider": client_a.provider, "model": client_a.model_name},
                "model_b": {"provider": client_b.provider, "model": client_b.model_name},
            },
            "is_mock_only": (client_a.provider == "mock" and client_b.provider == "mock"),
            "inference_duration_sec": total_duration,
            "sequential_execution": True,
            "provider_errors": [
                {
                    "model_slot": d["model_slot"],
                    "model": d["model_name"],
                    "iteration": d["iteration"],
                    "archetype": d["archetype"],
                    "status": d["execution_status"],
                    "category": d.get("error_category"),
                    "message": d.get("error_message"),
                    "retries": d.get("retry_count", 0),
                    "finish_reason": d.get("finish_reason"),
                }
                for d in diagnostics_log
                if d.get("execution_status") != ModelExecutionStatus.SUCCESS.value
            ],
        }

        sim_result = SimulationResult(
            simulation_id=sim_id,
            event_title=event_title,
            signal_text=signal_text,
            research_question=CORE_RESEARCH_QUESTION,
            config=config,
            iterations=iteration_results,
            model_agreement_score=mean_agreement,
            model_agreement_details=overall_agr_details,
            narrative_convergence_score=convergence_score,
            narrative_convergence_details=convergence_details,
            dominant_narrative=final_dominant,
            trajectory=trajectory,
            timestamp=datetime.utcnow().isoformat() + "Z",
            status=sim_status,
            status_details=status_details,
            successful_interpretations_count=success_count,
            expected_interpretations_count=expected_count,
            failed_models=failed_models_list,
            model_diagnostics=diagnostics_log,
            diagnostics=diagnostics_data,
        )

        # Build Causal Butterfly-Effect Pathways, 4 Scenarios & Final Verdict
        forecast_bundle = build_forecast_bundle(sim_result)
        sim_result = sim_result.model_copy(update={
            "research_question": forecast_bundle["research_question"],
            "causal_pathways": [CausalPathway(**pw) for pw in forecast_bundle["causal_pathways"]],
            "order_effects_summary": OrderEffectsSummary(**forecast_bundle["order_effects_summary"]),
            "stakeholder_reactions": [StakeholderReaction(**sr) for sr in forecast_bundle["stakeholder_reactions"]],
            "scenarios": [FutureScenario(**sc) for sc in forecast_bundle["scenarios"]],
            "time_horizon_forecast": TimeHorizonForecast(**forecast_bundle["time_horizon_forecast"]),
            "epistemic_breakdown": EpistemicBreakdown(**forecast_bundle["epistemic_breakdown"]),
            "final_verdict": FinalVerdict(**forecast_bundle["final_verdict"]),
        })

        # Persist JSON record isolated strictly under simulation_id.json
        self._save_simulation_json(sim_result)
        save_forecast_snapshot(sim_result.model_dump())
        logger.info(
            f"=== Simulation {sim_id} Finished [{sim_status}] | "
            f"Valid: {success_count}/{expected_count} | Final Active Adoption: {trajectory['active_adoption'][-1]:.1%} ==="
        )
        return sim_result

    @staticmethod
    def _is_mock_only_result(result: SimulationResult) -> bool:
        diag = result.diagnostics or {}
        if "is_mock_only" in diag:
            return bool(diag["is_mock_only"])
        selected = diag.get("selected_models") or {}
        pa = (selected.get("model_a") or {}).get("provider") or result.config.provider_a or "mock"
        pb = (selected.get("model_b") or {}).get("provider") or result.config.provider_b or "mock"
        return str(pa).lower() == "mock" and str(pb).lower() == "mock"

    def _save_simulation_json(self, result: SimulationResult) -> str:
        """
        Persists simulation record to SESSIONS_DIR preserving existing JSON conventions.
        Enforces strict storage hygiene: pure mock runs NEVER write to production SESSIONS_DIR.
        """
        if self._is_mock_only_result(result) and (
            os.path.abspath(SESSIONS_DIR) == os.path.abspath(PRODUCTION_SESSIONS_DIR)
        ):
            logger.info(
                f"[Storage Hygiene] Deterministic mock simulation {result.simulation_id} skipped writing "
                f"to production '{PRODUCTION_SESSIONS_DIR}' (isolated mock/test storage boundary enforced)."
            )
            return ""

        os.makedirs(SESSIONS_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

        session_path = os.path.join(SESSIONS_DIR, f"{result.simulation_id}.json")
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"Simulation record preserved at {session_path}")
        except Exception as e:
            logger.error(f"Failed to save simulation JSON: {e}")

        return session_path
