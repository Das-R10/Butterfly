"""
src/ensemble/prediction_registry.py

Cold Storage Prediction Registry.

Every prediction is locked to a timestamped non-editable JSON snapshot
BEFORE the outcome is known. This is what separates legitimate forecasting
from retrofitted storytelling.

Workflow:
  1. Simulation runs → full result produced
  2. lock_prediction() → saves immutable snapshot with mode tag
  3. 7 days later → resolve_prediction() marks ground truth
  4. score_prediction() → calculates Brier Score at resolution

The fixed 7 buckets are used here (not dynamic buckets) because Brier
Score requires consistent categories across predictions for comparison.

Also maintains:
  - Archetype Leaderboard: per-archetype Brier Score over time
  - Shadow Bench: ButterflyAI vs single-prompt GPT-4o comparison
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timedelta
from loguru import logger

REGISTRY_DIR = "data/registry"
BRIER_DIR    = os.path.join(REGISTRY_DIR, "brier")
LOCKED_DIR   = os.path.join(REGISTRY_DIR, "locked")
RESOLVED_DIR = os.path.join(REGISTRY_DIR, "resolved")
LEADERBOARD_FILE = os.path.join(REGISTRY_DIR, "archetype_leaderboard.json")

# Fixed 7 buckets for cross-signal Brier Score comparison.
# DO NOT use these for in-run analysis — dynamic buckets handle that.
FIXED_BUCKETS_ORDERED = [
    "conflict_escalation",
    "economic_chaos",
    "diplomatic_resolution",
    "conspiracy_elite_control",
    "routine_overblown",
    "historical_precedent",
    "tech_innovation_opportunity",
]


def _ensure_dirs():
    for d in [REGISTRY_DIR, BRIER_DIR, LOCKED_DIR, RESOLVED_DIR]:
        os.makedirs(d, exist_ok=True)


def _prediction_path(signal_id: str) -> str:
    return os.path.join(LOCKED_DIR, f"{signal_id}.json")


def _resolved_path(signal_id: str) -> str:
    return os.path.join(RESOLVED_DIR, f"{signal_id}_resolved.json")


# ── Locking ───────────────────────────────────────────────────────────────────

def lock_prediction(
    result: dict,
    mode: str,
    fixed_bucket_map: dict = None,
) -> str:
    """
    Lock a prediction snapshot immutably before outcome is known.

    Args:
        result:           Full simulation result dict
        mode:             "pure" or "support" — MUST be stored
        fixed_bucket_map: Optional manual mapping to fixed 7 buckets.
                          If None, auto-mapped from dynamic clusters.

    Returns:
        Path to the locked snapshot file.

    CRITICAL: mode tag must be stored. Without it, cross-mode Brier Score
    comparison is impossible.
    """
    _ensure_dirs()

    signal_id  = result.get("signal_id", f"pred_{int(datetime.utcnow().timestamp())}")
    locked_at  = datetime.utcnow().isoformat() + "Z"
    resolve_by = (datetime.utcnow() + timedelta(days=7)).isoformat() + "Z"

    # Build fixed-bucket probability vector from dynamic clusters
    if fixed_bucket_map is None:
        fixed_bucket_map = _map_dynamic_to_fixed(result.get("clusters", []))

    # Normalise to probability vector ordered by FIXED_BUCKETS_ORDERED
    prob_vector = [
        fixed_bucket_map.get(bucket, 0.0)
        for bucket in FIXED_BUCKETS_ORDERED
    ]
    # Renormalize
    total = sum(prob_vector)
    if total > 0:
        prob_vector = [round(p / total, 6) for p in prob_vector]

    snapshot = {
        "signal_id":         signal_id,
        "signal_text":       result.get("signal_text", "")[:500],
        "mode":              mode,           # "pure" or "support"
        "locked_at":         locked_at,
        "resolve_by":        resolve_by,
        "status":            "locked",
        "domain":            result.get("domain", "unknown"),

        # The prediction itself
        "dominant_narrative":result.get("dominant_narrative", {}).get("label", "unknown"),
        "dominant_bucket":   result.get("dominant_narrative", {}).get("bucket", "unknown"),
        "dominant_prob":     result.get("dominant_narrative", {}).get("probability", 0.0),

        # Fixed-bucket probability vector for Brier Score
        "fixed_bucket_labels":  FIXED_BUCKETS_ORDERED,
        "fixed_bucket_probs":   prob_vector,
        "fixed_bucket_map":     fixed_bucket_map,

        # Dynamic cluster summary (readable)
        "dynamic_clusters": [
            {
                "bucket": c.get("bucket"),
                "label":  c.get("label"),
                "prob":   c.get("probability"),
            }
            for c in result.get("clusters", [])[:7]
        ],

        # Convergence gap at lock time
        "convergence_gap": result.get("convergence_gap", {}),

        # Volatility at lock time
        "volatility_score": result.get("debate_result", {}).get("volatility_score", None),
        "high_octane":      result.get("debate_result", {}).get("high_octane_narrative", False),

        # Ground truth — filled at resolution
        "ground_truth_bucket": None,
        "ground_truth_label":  None,
        "brier_score":         None,
        "resolved_at":         None,
        "resolution_notes":    None,
    }

    path = _prediction_path(signal_id)

    # Never overwrite a locked prediction
    if os.path.exists(path):
        logger.warning(f"Prediction {signal_id} already locked — not overwriting")
        return path

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info(
        f"Prediction locked: {signal_id} | mode={mode} | "
        f"dominant='{snapshot['dominant_narrative']}' ({snapshot['dominant_prob']:.0%}) | "
        f"resolve_by={resolve_by}"
    )
    return path


def _map_dynamic_to_fixed(clusters: list[dict]) -> dict[str, float]:
    """
    Map dynamic bucket probabilities to the fixed 7 buckets.
    Uses keyword similarity between dynamic bucket keys and fixed bucket keys.
    """
    if not clusters:
        return {b: 0.0 for b in FIXED_BUCKETS_ORDERED}

    # Keyword mapping from dynamic to fixed
    DYNAMIC_TO_FIXED = {
        # Conflict / escalation
        "military_escalation": "conflict_escalation",
        "escalation":          "conflict_escalation",
        "war":                 "conflict_escalation",
        "conflict":            "conflict_escalation",
        "attack":              "conflict_escalation",
        "strike":              "conflict_escalation",
        # Economic
        "market_crash":        "economic_chaos",
        "economic_crisis":     "economic_chaos",
        "recession":           "economic_chaos",
        "financial":           "economic_chaos",
        "sanctions":           "economic_chaos",
        "trade_war":           "economic_chaos",
        # Diplomatic
        "diplomatic_resolution":"diplomatic_resolution",
        "peace_talks":         "diplomatic_resolution",
        "ceasefire":           "diplomatic_resolution",
        "negotiation":         "diplomatic_resolution",
        "diplomatic":          "diplomatic_resolution",
        # Conspiracy
        "conspiracy":          "conspiracy_elite_control",
        "conspiracy_narrative":"conspiracy_elite_control",
        "false_flag":          "conspiracy_elite_control",
        # Routine
        "media_overreaction":  "routine_overblown",
        "no_real_impact":      "routine_overblown",
        "no_impact":           "routine_overblown",
        "routine":             "routine_overblown",
        "overblown":           "routine_overblown",
        # Historical
        "historical_precedent":"historical_precedent",
        "resolves":            "historical_precedent",
        # Tech / innovation
        "tech_opportunity":    "tech_innovation_opportunity",
        "innovation":          "tech_innovation_opportunity",
        "breakthrough":        "tech_innovation_opportunity",
    }

    fixed_weights = {b: 0.0 for b in FIXED_BUCKETS_ORDERED}

    for cluster in clusters:
        dyn_key = cluster.get("bucket", "").lower()
        prob    = cluster.get("probability", 0.0)

        # Try direct match first
        if dyn_key in DYNAMIC_TO_FIXED:
            fixed_weights[DYNAMIC_TO_FIXED[dyn_key]] += prob
            continue

        # Try partial keyword match
        matched = False
        for kw, fixed in DYNAMIC_TO_FIXED.items():
            if kw in dyn_key or dyn_key in kw:
                fixed_weights[fixed] += prob
                matched = True
                break

        # If no match found, dump into routine_overblown as catch-all
        if not matched:
            fixed_weights["routine_overblown"] += prob * 0.5

    return {k: round(v, 6) for k, v in fixed_weights.items()}


# ── Resolution ────────────────────────────────────────────────────────────────

def resolve_prediction(
    signal_id: str,
    ground_truth_bucket: str,
    resolution_notes: str = "",
) -> dict:
    """
    Mark the ground truth outcome for a locked prediction and calculate Brier Score.

    Args:
        signal_id:            The signal_id from the locked prediction
        ground_truth_bucket:  One of FIXED_BUCKETS_ORDERED
        resolution_notes:     Optional notes about how outcome was determined

    Returns:
        Updated snapshot dict with brier_score filled.
    """
    _ensure_dirs()

    path = _prediction_path(signal_id)
    if not os.path.exists(path):
        logger.error(f"No locked prediction found for {signal_id}")
        return {}

    with open(path) as f:
        snapshot = json.load(f)

    if snapshot.get("status") == "resolved":
        logger.warning(f"Prediction {signal_id} already resolved")
        return snapshot

    if ground_truth_bucket not in FIXED_BUCKETS_ORDERED:
        logger.error(
            f"Invalid ground truth bucket: {ground_truth_bucket}. "
            f"Must be one of {FIXED_BUCKETS_ORDERED}"
        )
        return snapshot

    # Build ground truth vector (1 for correct bucket, 0 for others)
    truth_vector = [
        1.0 if b == ground_truth_bucket else 0.0
        for b in FIXED_BUCKETS_ORDERED
    ]

    prob_vector = snapshot.get("fixed_bucket_probs", [])
    brier_score = _calculate_brier(prob_vector, truth_vector)

    snapshot.update({
        "status":              "resolved",
        "ground_truth_bucket": ground_truth_bucket,
        "ground_truth_label":  ground_truth_bucket.replace("_", " ").title(),
        "brier_score":         brier_score,
        "resolved_at":         datetime.utcnow().isoformat() + "Z",
        "resolution_notes":    resolution_notes,
    })

    # Save resolved copy
    resolved_path = _resolved_path(signal_id)
    with open(resolved_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Update leaderboard
    _update_leaderboard(snapshot)

    logger.info(
        f"Prediction resolved: {signal_id} | "
        f"ground_truth={ground_truth_bucket} | "
        f"Brier Score={brier_score:.4f}"
    )
    return snapshot


def _calculate_brier(prob_vector: list, truth_vector: list) -> float:
    """
    Multi-class Brier Score.
    BS = (1/N) * sum((p_i - o_i)^2)
    Lower is better. 0.0 = perfect. 1.0 = maximally wrong.
    """
    if len(prob_vector) != len(truth_vector):
        return 1.0
    n = len(prob_vector)
    if n == 0:
        return 1.0
    bs = sum((p - o) ** 2 for p, o in zip(prob_vector, truth_vector)) / n
    return round(bs, 6)


# ── Leaderboard ───────────────────────────────────────────────────────────────

def _update_leaderboard(resolved_snapshot: dict):
    """
    Update the archetype leaderboard with per-archetype Brier contribution.
    Tracks which archetypes are most predictive over time.
    """
    _ensure_dirs()

    leaderboard = {}
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE) as f:
            leaderboard = json.load(f)

    # For each archetype in this prediction, record whether their bucket
    # matched the ground truth and update running stats
    ground_truth = resolved_snapshot.get("ground_truth_bucket", "")
    brier        = resolved_snapshot.get("brier_score", 1.0)
    signal_id    = resolved_snapshot.get("signal_id", "")
    mode         = resolved_snapshot.get("mode", "unknown")

    # We don't have per-archetype ground truth here — track at signal level
    # Per-archetype tracking requires the full outputs, done in score_archetypes()
    entry = {
        "signal_id":    signal_id,
        "mode":         mode,
        "brier_score":  brier,
        "ground_truth": ground_truth,
        "resolved_at":  resolved_snapshot.get("resolved_at", ""),
    }

    if "signal_history" not in leaderboard:
        leaderboard["signal_history"] = []
    leaderboard["signal_history"].append(entry)

    # Calculate running stats
    scores = [e["brier_score"] for e in leaderboard["signal_history"] if e.get("brier_score") is not None]
    leaderboard["overall_brier_mean"] = round(sum(scores) / len(scores), 6) if scores else None
    leaderboard["n_predictions"] = len(scores)
    leaderboard["last_updated"] = datetime.utcnow().isoformat() + "Z"

    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard, f, indent=2)


def score_archetypes(
    resolved_snapshot: dict,
    full_outputs: list[dict],
) -> dict[str, float]:
    """
    Calculate per-archetype Brier Score for a resolved prediction.

    Each archetype's primary bucket assignment is compared to ground truth.
    Returns dict of personality_key → archetype_brier_score.

    Archetype scores accumulate in the leaderboard over time — this is
    how we learn which archetypes are most predictive per domain.
    """
    ground_truth = resolved_snapshot.get("ground_truth_bucket", "")
    if not ground_truth or not full_outputs:
        return {}

    archetype_scores = {}
    for output in full_outputs:
        key    = output.get("personality", "")
        bucket = output.get("bucket", "")  # set by dynamic_bucket_scorer

        # Map archetype's dynamic bucket to fixed bucket
        fixed_bucket = _map_single_dynamic_to_fixed(bucket)

        # Simple: 0 if correct, 1 if wrong (binary Brier at archetype level)
        score = 0.0 if fixed_bucket == ground_truth else 1.0
        archetype_scores[key] = score

    return archetype_scores


def _map_single_dynamic_to_fixed(dynamic_key: str) -> str:
    """Map a single dynamic bucket key to the closest fixed bucket."""
    if not dynamic_key:
        return "routine_overblown"

    MAPPING = {
        "escalation": "conflict_escalation", "war": "conflict_escalation",
        "conflict": "conflict_escalation", "attack": "conflict_escalation",
        "military": "conflict_escalation", "strike": "conflict_escalation",
        "crash": "economic_chaos", "economic": "economic_chaos",
        "recession": "economic_chaos", "financial": "economic_chaos",
        "sanctions": "economic_chaos", "trade": "economic_chaos",
        "diplomatic": "diplomatic_resolution", "peace": "diplomatic_resolution",
        "ceasefire": "diplomatic_resolution", "negotiation": "diplomatic_resolution",
        "conspiracy": "conspiracy_elite_control", "manipulation": "conspiracy_elite_control",
        "staged": "conspiracy_elite_control",
        "routine": "routine_overblown", "overblown": "routine_overblown",
        "normal": "routine_overblown", "no_impact": "routine_overblown",
        "historical": "historical_precedent", "precedent": "historical_precedent",
        "tech": "tech_innovation_opportunity", "innovation": "tech_innovation_opportunity",
    }

    dk = dynamic_key.lower()
    for kw, fixed in MAPPING.items():
        if kw in dk:
            return fixed
    return "routine_overblown"


# ── Query / reporting ─────────────────────────────────────────────────────────

def get_pending_resolutions() -> list[dict]:
    """Return all locked predictions past their resolve_by date."""
    _ensure_dirs()
    pending = []
    now = datetime.utcnow()

    for fname in os.listdir(LOCKED_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(LOCKED_DIR, fname)
        with open(path) as f:
            snap = json.load(f)
        if snap.get("status") == "locked":
            resolve_by = snap.get("resolve_by", "")
            try:
                if datetime.fromisoformat(resolve_by.rstrip("Z")) < now:
                    pending.append(snap)
            except Exception:
                pass

    return sorted(pending, key=lambda x: x.get("locked_at", ""))


def get_leaderboard_summary() -> dict:
    """Return current leaderboard stats."""
    if not os.path.exists(LEADERBOARD_FILE):
        return {"n_predictions": 0, "overall_brier_mean": None}
    with open(LEADERBOARD_FILE) as f:
        return json.load(f)


def compare_modes() -> dict:
    """
    Compare Brier Scores between Pure Sim and Support Sim.
    Core research question: does ambient context improve prediction accuracy?
    """
    if not os.path.exists(LEADERBOARD_FILE):
        return {}
    with open(LEADERBOARD_FILE) as f:
        leaderboard = json.load(f)

    history = leaderboard.get("signal_history", [])
    pure    = [e["brier_score"] for e in history if e.get("mode") == "pure" and e.get("brier_score") is not None]
    support = [e["brier_score"] for e in history if e.get("mode") == "support" and e.get("brier_score") is not None]

    return {
        "pure_mean":    round(sum(pure) / len(pure), 6) if pure else None,
        "support_mean": round(sum(support) / len(support), 6) if support else None,
        "pure_n":       len(pure),
        "support_n":    len(support),
        "narrative_pressure_affects_accuracy": (
            "support better" if (pure and support and
                                 sum(support)/len(support) < sum(pure)/len(pure))
            else "pure better" if (pure and support)
            else "insufficient data"
        ),
    }
