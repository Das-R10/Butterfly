"""
src/ensemble/precedent_store.py

Session Memory — Bounded Precedent Store.

ButterflyAI agents are stateless — they reset on every signal.
Memory lives at the SESSION level, not the agent level.

Architecture (legal precedent model):
  - Stores the last 2 related predictions per signal cluster
  - Checks every incoming signal against stored precedents via cosine similarity
  - If similarity > 0.75: inject precedent context into pipeline (not into agents)
  - Agents stay pure. Pipeline gets institutional memory.

Session detection:
  - Similarity > 0.75 → resume existing session (developing story)
  - Similarity < 0.75 → fork new session (new topic)

What memory carries:
  - Probability map time series per session
  - MBTI reaction trajectory (how population emotion evolved)
  - Archetype position history (which archetypes shifted)
  - Debate layer record (what critics found per signal)
  - Session topic fingerprint for similarity matching
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from loguru import logger

SESSION_DIR  = "data/sessions"
SESSION_SIMILARITY_THRESHOLD = 0.75
MAX_PRECEDENTS_PER_CLUSTER   = 2

# ── Embedding helper ──────────────────────────────────────────────────────────

def _embed(text: str):
    """Return a sentence-transformer embedding. Keyword fallback if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode([text], normalize_embeddings=True)[0]
    except ImportError:
        return None


def _cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np
        return float(np.dot(a, b))
    except Exception:
        return 0.0


def _text_similarity_fallback(a: str, b: str) -> float:
    """Jaccard similarity on word sets — fallback when embeddings unavailable."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute similarity between two signal texts."""
    emb_a = _embed(text_a)
    emb_b = _embed(text_b)
    if emb_a is not None and emb_b is not None:
        return _cosine_similarity(emb_a, emb_b)
    return _text_similarity_fallback(text_a, text_b)


# ── Session management ────────────────────────────────────────────────────────

def _session_path(session_id: str) -> str:
    os.makedirs(SESSION_DIR, exist_ok=True)
    return os.path.join(SESSION_DIR, f"{session_id}.json")


def _load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_session(session: dict):
    sid  = session["session_id"]
    path = _session_path(sid)
    with open(path, "w") as f:
        json.dump(session, f, indent=2)


def _list_active_sessions() -> list[dict]:
    """Load all sessions that have signals in them."""
    os.makedirs(SESSION_DIR, exist_ok=True)
    sessions = []
    for fname in os.listdir(SESSION_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SESSION_DIR, fname)
        try:
            with open(path) as f:
                sessions.append(json.load(f))
        except Exception:
            pass
    return sessions


# ── Session detection ─────────────────────────────────────────────────────────

def detect_session(signal_text: str, force_new: bool = False) -> dict:
    """
    Check incoming signal against active sessions.

    Returns:
        {
            "session_id":   str,
            "action":       "resume" | "new",
            "similarity":   float,
            "prior_signals": list of prior signal summaries in this session,
            "precedent_context": str (formatted text for pipeline injection),
        }
    """
    if force_new:
        return _create_new_session(signal_text)

    sessions = _list_active_sessions()
    best_match_score = 0.0
    best_session     = None

    for session in sessions:
        if not session.get("signals"):
            continue
        # Compare against the session's topic fingerprint (last signal text)
        fingerprint = session.get("topic_fingerprint", "")
        if not fingerprint:
            continue
        sim = compute_similarity(signal_text, fingerprint)
        if sim > best_match_score:
            best_match_score = sim
            best_session     = session

    if best_match_score >= SESSION_SIMILARITY_THRESHOLD and best_session:
        logger.info(
            f"Session resumed: {best_session['session_id']} "
            f"(similarity={best_match_score:.3f})"
        )
        return {
            "session_id":        best_session["session_id"],
            "action":            "resume",
            "similarity":        round(best_match_score, 4),
            "prior_signals":     best_session.get("signals", []),
            "precedent_context": _build_precedent_context(best_session),
        }
    else:
        new = _create_new_session(signal_text)
        logger.info(
            f"New session created: {new['session_id']} "
            f"(best similarity was {best_match_score:.3f})"
        )
        return new


def _create_new_session(signal_text: str) -> dict:
    session_id = f"sess_{hashlib.md5(signal_text.encode()).hexdigest()[:10]}_{int(datetime.utcnow().timestamp())}"
    return {
        "session_id":        session_id,
        "action":            "new",
        "similarity":        0.0,
        "prior_signals":     [],
        "precedent_context": "",
    }


def _build_precedent_context(session: dict) -> str:
    """
    Build the formatted precedent context string that gets injected
    into the pipeline (NOT into individual agents).
    """
    signals = session.get("signals", [])
    if not signals:
        return ""

    lines = [
        "═" * 55,
        "PRECEDENT CONTEXT (Session Memory)",
        f"Session: {session['session_id']}",
        f"Prior signals in this thread: {len(signals)}",
        "═" * 55,
        "",
    ]

    for i, sig in enumerate(signals[-MAX_PRECEDENTS_PER_CLUSTER:], 1):
        lines += [
            f"Signal {i} ({sig.get('timestamp', 'unknown time')[:16]}):",
            f"  {sig.get('signal_text', '')[:150]}...",
            f"  Dominant narrative: {sig.get('dominant_narrative', 'unknown')} "
            f"({sig.get('dominant_prob', 0):.0%})",
            f"  Convergence gap: {sig.get('convergence_gap_status', 'unknown')}",
        ]
        if sig.get("adversarial_flags"):
            lines.append(f"  ⚠ Adversarial flags: {sig['adversarial_flags']}")
        lines.append("")

    # Show probability evolution if multiple signals
    if len(signals) >= 2:
        lines += ["Probability trajectory:", ""]
        for sig in signals[-3:]:
            dom = sig.get("dominant_narrative", "?")
            prob = sig.get("dominant_prob", 0)
            ts   = sig.get("timestamp", "")[:10]
            lines.append(f"  {ts}  {dom:<35} {prob:.0%}")
        lines.append("")

    lines += ["═" * 55, "END PRECEDENT CONTEXT", "═" * 55]
    return "\n".join(lines)


# ── Session update ────────────────────────────────────────────────────────────

def update_session(
    session_id: str,
    signal_text: str,
    result: dict,
    mbti_report: dict = None,
    debate_result: dict = None,
    adversarial_flags: list = None,
) -> dict:
    """
    Add a new signal result to an existing or new session.

    Call this AFTER the simulation completes and results are available.

    Returns the updated session dict.
    """
    session = _load_session(session_id)

    if not session:
        # Initialize new session
        session = {
            "session_id":        session_id,
            "created_at":        datetime.utcnow().isoformat() + "Z",
            "signals":           [],
            "topic_fingerprint": signal_text[:300],
            "probability_timeline": [],
            "mbti_trajectory":   [],
        }

    # Build signal summary to store
    clusters  = result.get("clusters", [])
    dominant  = clusters[0] if clusters else {}
    conv_gap  = result.get("convergence_gap", {})

    signal_entry = {
        "signal_id":             result.get("signal_id", ""),
        "signal_text":           signal_text[:200],
        "timestamp":             datetime.utcnow().isoformat() + "Z",
        "dominant_narrative":    dominant.get("label", "unknown"),
        "dominant_bucket":       dominant.get("bucket", "unknown"),
        "dominant_prob":         dominant.get("probability", 0.0),
        "convergence_gap_status":conv_gap.get("gap_status", "unknown"),
        "adversarial_flags":     adversarial_flags or [],
        "volatility_score":      debate_result.get("volatility_score") if debate_result else None,
        "high_octane":           debate_result.get("high_octane_narrative") if debate_result else False,
    }

    session["signals"].append(signal_entry)

    # Keep only last 10 signals per session
    session["signals"] = session["signals"][-10:]

    # Update topic fingerprint to latest signal
    session["topic_fingerprint"] = signal_text[:300]

    # Update probability timeline
    session["probability_timeline"].append({
        "timestamp":   signal_entry["timestamp"],
        "dominant":    dominant.get("label", "unknown"),
        "probability": dominant.get("probability", 0.0),
        "clusters":    [
            {"bucket": c.get("bucket"), "prob": c.get("probability")}
            for c in clusters[:4]
        ],
    })

    # Update MBTI trajectory
    if mbti_report:
        session["mbti_trajectory"].append({
            "timestamp":       signal_entry["timestamp"],
            "dominant_emotion":mbti_report.get("dominant_emotion", ""),
            "sharing_velocity":mbti_report.get("sharing_velocity", 0),
        })

    session["last_updated"] = datetime.utcnow().isoformat() + "Z"
    session["n_signals"]    = len(session["signals"])

    _save_session(session)
    logger.info(
        f"Session updated: {session_id} | "
        f"n_signals={session['n_signals']} | "
        f"dominant='{signal_entry['dominant_narrative']}'"
    )
    return session


# ── Precedent store (bounded per cluster) ────────────────────────────────────

def store_precedent(
    cluster_bucket: str,
    signal_text: str,
    probability_map: list[dict],
    gdelt_goldstein: float = None,
    final_confidence: float = None,
    status: str = "unverified",
    archetype_outliers: list = None,
) -> str:
    """
    Store a bounded precedent for a narrative cluster.
    Only the last MAX_PRECEDENTS_PER_CLUSTER predictions per cluster are kept.

    Precedent schema (legal model — stores dissenting opinions too):
      - signal text + timestamp
      - pure and support probability maps
      - GDELT Goldstein score at ingestion
      - final confidence score
      - resolution status: unverified / validated / flagged / adversarial
      - archetype outlier flags (the dissenting opinions)
    """
    os.makedirs(os.path.join(SESSION_DIR, "precedents"), exist_ok=True)
    cluster_file = os.path.join(SESSION_DIR, "precedents", f"{cluster_bucket}.json")

    precedents = []
    if os.path.exists(cluster_file):
        with open(cluster_file) as f:
            precedents = json.load(f)

    entry = {
        "signal_text":       signal_text[:300],
        "timestamp":         datetime.utcnow().isoformat() + "Z",
        "probability_map":   probability_map[:5],
        "gdelt_goldstein":   gdelt_goldstein,
        "final_confidence":  final_confidence,
        "status":            status,
        "archetype_outliers":archetype_outliers or [],
    }

    precedents.append(entry)

    # Keep only last MAX_PRECEDENTS_PER_CLUSTER per cluster
    precedents = precedents[-MAX_PRECEDENTS_PER_CLUSTER:]

    with open(cluster_file, "w") as f:
        json.dump(precedents, f, indent=2)

    return cluster_file


def get_precedents(cluster_bucket: str) -> list[dict]:
    """Retrieve stored precedents for a cluster bucket."""
    cluster_file = os.path.join(SESSION_DIR, "precedents", f"{cluster_bucket}.json")
    if not os.path.exists(cluster_file):
        return []
    with open(cluster_file) as f:
        return json.load(f)


def check_adversarial_update(
    incoming_signal: str,
    referenced_signal_id: str = None,
) -> dict:
    """
    Adversarial update detection.

    If Signal A was flagged as Unanchored Pending and Signal B arrives
    claiming to confirm it, Signal B cannot be treated as independent
    corroboration — it is evaluated as a potential adversarial confirmation.

    Returns:
        {
            "is_potential_adversarial_update": bool,
            "referenced_signal_status": str,
            "warning": str,
        }
    """
    if not referenced_signal_id:
        return {"is_potential_adversarial_update": False}

    # Check if referenced signal exists and was flagged
    locked_path = os.path.join("data/registry/locked", f"{referenced_signal_id}.json")
    if not os.path.exists(locked_path):
        return {"is_potential_adversarial_update": False}

    with open(locked_path) as f:
        prior = json.load(f)

    prior_status = prior.get("status", "locked")

    if prior_status in ("flagged", "adversarial"):
        return {
            "is_potential_adversarial_update": True,
            "referenced_signal_status": prior_status,
            "warning": (
                f"Signal {referenced_signal_id} was previously flagged as '{prior_status}'. "
                f"This incoming signal cannot be treated as independent corroboration. "
                f"Evaluating as potential adversarial confirmation attempt."
            ),
        }

    return {
        "is_potential_adversarial_update": False,
        "referenced_signal_status": prior_status,
    }
