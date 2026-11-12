"""
src/personalities/mbti_layer.py  — L2A

MBTI Population Reaction Layer.

Runs 16 MBTI types against the signal and L1 world-state brief.
Each type produces: emotional state, sharing intention, trust level,
and an instinctive narrative frame scored against the dynamic buckets.

Outputs are weighted by real population distribution and compiled into
a structured population reaction report that is prepended to every
analyst archetype prompt in L2B.

This is the bottom-up pass — simulating population behavioral reality,
not analytical interpretation.

Population weights sourced from Myers-Briggs Foundation estimates:
  ISTJ 14%, ISFJ 12%, INFJ 2%, INTJ 2%
  ISTP 5%, ISFP 6%, INFP 4%, INTP 3%
  ESTP 4%, ESFP 7%, ENFP 8%, ENTP 3%
  ESTJ 9%, ESFJ 12%, ENFJ 2%, ENTJ 2%

Fixes applied:
  v1.1 — sentence-transformer model loaded once as module-level singleton
          instead of being reloaded on every MBTI type call (was 16 reloads,
          ~30s wasted, 16 BertModel LOAD REPORT spam lines)
  v1.1 — _compute_dominant_emotion now skips bare numeric tokens (e.g. "1",
          "1.", "2)") that the LLM sometimes prepends to numbered responses,
          fixing the dominant_emotion=1 display bug
"""

from __future__ import annotations

import re
import requests
import time
import threading
from collections import Counter, defaultdict
from datetime import datetime
from loguru import logger

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"
TEMPERATURE  = 0.3   # locked — same as analyst layer

# ── MBTI type definitions ─────────────────────────────────────────────────────

MBTI_TYPES: dict[str, dict] = {
    "ISTJ": {
        "population_pct": 0.14,
        "label": "Inspector / Duty Fulfiller",
        "cognitive_style": "factual, procedural, skeptical of unverified claims, trusts institutions",
        "sharing_behavior": "moderate broadcast — shares only when verified by trusted sources",
        "emotional_baseline": "calm, controlled, resists alarm until evidence is overwhelming",
        "trust_hierarchy": "official institutions, established media, government statements",
        "activation_domains": ["security", "economics", "policy", "historical"],
    },
    "ISFJ": {
        "population_pct": 0.12,
        "label": "Protector / Nurturer",
        "cognitive_style": "community-focused, prioritizes safety of known people, concrete not abstract",
        "sharing_behavior": "high broadcast within close networks — family, community groups, local channels",
        "emotional_baseline": "anxious when community safety threatened, high alarm on personal threat signals",
        "trust_hierarchy": "community figures, local news, people they know personally",
        "activation_domains": ["safety", "community", "health", "family"],
    },
    "INFJ": {
        "population_pct": 0.02,
        "label": "Counselor / Advocate",
        "cognitive_style": "pattern-seeking with moral dimension, sees long-term systemic implications",
        "sharing_behavior": "low broadcast but high depth — writes long considered posts to small audiences",
        "emotional_baseline": "deep concern, tends toward worst-case moral framing",
        "trust_hierarchy": "independent thinkers, whistleblowers, activists",
        "activation_domains": ["humanitarian", "political", "social justice", "environment"],
    },
    "INTJ": {
        "population_pct": 0.02,
        "label": "Mastermind / Strategist",
        "cognitive_style": "systemic, strategic, skeptical of emotional framing, seeks underlying structure",
        "sharing_behavior": "very low broadcast — high depth influence in specialist networks",
        "emotional_baseline": "detached, analytical, intellectually engaged not emotionally alarmed",
        "trust_hierarchy": "primary sources, data, expert analysis",
        "activation_domains": ["geopolitical", "economics", "technology", "strategy"],
    },
    "ISTP": {
        "population_pct": 0.05,
        "label": "Craftsman / Mechanic",
        "cognitive_style": "technical, practical, skeptical of narrative framing, focuses on mechanism",
        "sharing_behavior": "low broadcast — contrarian influence in technical communities",
        "emotional_baseline": "disengaged from drama, interested in how things work",
        "trust_hierarchy": "technical experts, engineers, people with direct experience",
        "activation_domains": ["technology", "security", "military_technical", "infrastructure"],
    },
    "ISFP": {
        "population_pct": 0.06,
        "label": "Composer / Artist",
        "cognitive_style": "values-driven, empathetic, focuses on human cost not strategic picture",
        "sharing_behavior": "moderate broadcast — shares emotionally resonant content, images, personal stories",
        "emotional_baseline": "moved by human suffering, disengaged from abstract strategic analysis",
        "trust_hierarchy": "personal stories, eyewitness accounts, humanitarian organizations",
        "activation_domains": ["humanitarian", "cultural", "personal impact"],
    },
    "INFP": {
        "population_pct": 0.04,
        "label": "Idealist / Mediator",
        "cognitive_style": "values-first, sees moral dimension of every event, resists cynical framing",
        "sharing_behavior": "moderate broadcast — emotionally resonant posts, counter-narratives to aggression",
        "emotional_baseline": "high alarm on moral failures, hopeful about human capacity for change",
        "trust_hierarchy": "activists, independent journalists, moral voices",
        "activation_domains": ["humanitarian", "political", "social justice"],
    },
    "INTP": {
        "population_pct": 0.03,
        "label": "Architect / Thinker",
        "cognitive_style": "logical, theoretical, demands evidence, challenges assumptions",
        "sharing_behavior": "low broadcast — high credibility node in skeptical communities",
        "emotional_baseline": "intellectually engaged, low emotional reactivity",
        "trust_hierarchy": "peer-reviewed sources, logical argument, technical documentation",
        "activation_domains": ["technology", "scientific", "economics", "geopolitical"],
    },
    "ESTP": {
        "population_pct": 0.04,
        "label": "Promoter / Entrepreneur",
        "cognitive_style": "pragmatic, action-oriented, interested in immediate practical implications",
        "sharing_behavior": "high broadcast — frames events in terms of immediate opportunity or risk",
        "emotional_baseline": "energized by events, quick to act on implications",
        "trust_hierarchy": "direct experience, market signals, people who have done it",
        "activation_domains": ["economics", "corporate", "security"],
    },
    "ESFP": {
        "population_pct": 0.07,
        "label": "Performer / Entertainer",
        "cognitive_style": "social, present-focused, interested in immediate human drama",
        "sharing_behavior": "very high broadcast — emotional contagion, amplifies most dramatic version",
        "emotional_baseline": "high reactivity, strong emotional amplification",
        "trust_hierarchy": "celebrities, influencers, emotionally compelling content",
        "activation_domains": ["cultural", "entertainment", "social"],
    },
    "ENFP": {
        "population_pct": 0.08,
        "label": "Champion / Inspirer",
        "cognitive_style": "imaginative, connects ideas across domains, sees possibilities",
        "sharing_behavior": "very high broadcast — generates novel framings that spread widely",
        "emotional_baseline": "enthusiastic or alarmed, swings between hope and concern",
        "trust_hierarchy": "visionary thinkers, alternative media, innovators",
        "activation_domains": ["political", "cultural", "technology", "social"],
    },
    "ENTP": {
        "population_pct": 0.03,
        "label": "Inventor / Debater",
        "cognitive_style": "devil's advocate, challenges consensus, generates counter-arguments",
        "sharing_behavior": "moderate broadcast — high engagement through provocative framing",
        "emotional_baseline": "intellectually excited by controversy, not emotionally alarmed",
        "trust_hierarchy": "unconventional thinkers, contrarian analysts",
        "activation_domains": ["technology", "political", "economics", "conspiracy"],
    },
    "ESTJ": {
        "population_pct": 0.09,
        "label": "Supervisor / Executive",
        "cognitive_style": "procedural, authority-respecting, black-and-white, action-oriented",
        "sharing_behavior": "high broadcast through authority channels — frames events in terms of order/disorder",
        "emotional_baseline": "alarmed by disorder, reassured by institutional response",
        "trust_hierarchy": "government officials, military, law enforcement, established media",
        "activation_domains": ["security", "policy", "economics", "geopolitical"],
    },
    "ESFJ": {
        "population_pct": 0.12,
        "label": "Provider / Caregiver",
        "cognitive_style": "community-focused, harmony-seeking, high sensitivity to social threat",
        "sharing_behavior": "very high broadcast — emotional amplification in community networks",
        "emotional_baseline": "high alarm when community or social harmony threatened",
        "trust_hierarchy": "community leaders, local institutions, people they know",
        "activation_domains": ["community", "safety", "health", "family"],
    },
    "ENFJ": {
        "population_pct": 0.02,
        "label": "Teacher / Protagonist",
        "cognitive_style": "people-focused, inspirational framing, draws moral lessons from events",
        "sharing_behavior": "high broadcast — motivating, rallying, call-to-action framing",
        "emotional_baseline": "moved to action by injustice, optimistic about collective response",
        "trust_hierarchy": "community leaders, educators, moral authorities",
        "activation_domains": ["humanitarian", "political", "social", "cultural"],
    },
    "ENTJ": {
        "population_pct": 0.02,
        "label": "Commander / Field Marshal",
        "cognitive_style": "strategic, command-oriented, analyzes power dynamics and decision leverage",
        "sharing_behavior": "high broadcast in professional/leadership networks",
        "emotional_baseline": "decisive, action-demanding, impatient with indecision",
        "trust_hierarchy": "strategic analysts, military experts, executive decision-makers",
        "activation_domains": ["geopolitical", "economics", "corporate", "security"],
    },
}

REACTION_CLUSTERS = {
    "HIGH_ALARM_HIGH_SHARING":      ["ESFJ", "ISFJ", "ENFJ", "ESFP"],
    "STRATEGIC_THREAT_ASSESSMENT":  ["ENTJ", "ESTJ", "INTJ", "ISTJ"],
    "SKEPTICAL_EVIDENCE_DEMANDING": ["INTP", "ISTP", "ENTP", "INTJ"],
    "EMOTIONAL_MORAL_FRAMING":      ["INFP", "INFJ", "ISFP", "ENFP"],
    "ACTION_PRAGMATIC":             ["ESTP", "ESTJ", "ENTJ", "ENFJ"],
}

# ── MBTI reaction prompt ──────────────────────────────────────────────────────

MBTI_REACTION_PROMPT = """You are modeling the instinctive reaction of an {mbti_type} personality type ({label}) to a news signal.

MBTI cognitive profile:
- Cognitive style: {cognitive_style}
- Emotional baseline: {emotional_baseline}
- Trust hierarchy: {trust_hierarchy}
- Sharing behavior: {sharing_behavior}

This is a behavioral psychology simulation. Stay true to this cognitive profile.

Signal: {signal}

{world_state_section}

Respond as this MBTI type would instinctively react. Answer these four things in order, one sentence each:
1. EMOTIONAL STATE: What is the immediate emotional response? (one word + one sentence explanation)
2. NARRATIVE FRAME: What instinctive story does this type reach for? (one sentence)
3. SHARING INTENTION: Will they share this? Where? How widely? (one sentence)
4. TRUST LEVEL: Do they trust this signal as real? Why or why not? (one sentence)

Keep each answer to exactly one sentence. No preamble. Plain text only."""


# ── Ollama caller ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, max_tokens: int = 300) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
            },
            timeout=45,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        logger.warning(f"Ollama returned {resp.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _parse_mbti_response(raw: str, mbti_type: str) -> dict:
    if not raw:
        return {
            "emotional_state": "unknown",
            "narrative_frame": "unknown",
            "sharing_intention": "unknown",
            "trust_level": "unknown",
            "raw": "",
        }

    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]

    def extract(lines, keyword):
        for line in lines:
            if keyword.upper() in line.upper():
                for sep in [":", "—", "-"]:
                    if sep in line:
                        return line.split(sep, 1)[1].strip()
                return line.strip()
        idx = {"EMOTIONAL": 0, "NARRATIVE": 1, "SHARING": 2, "TRUST": 3}
        pos = idx.get(keyword.upper(), 0)
        return lines[pos].strip() if pos < len(lines) else "unknown"

    return {
        "emotional_state":    extract(lines, "EMOTIONAL"),
        "narrative_frame":    extract(lines, "NARRATIVE"),
        "sharing_intention":  extract(lines, "SHARING"),
        "trust_level":        extract(lines, "TRUST"),
        "raw":                raw,
    }


# ── Progress tracker ──────────────────────────────────────────────────────────

class MBTIProgressTracker:
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, total: int):
        self.total        = total
        self.current      = 0
        self.current_name = "starting..."
        self.start_time   = time.time()
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def start(self):  self._thread.start()

    def update(self, index: int, name: str):
        self.current      = index
        self.current_name = name

    def stop(self):
        self._stop.set()
        self._thread.join()
        print("\r" + " " * 80 + "\r", end="", flush=True)

    def _run(self):
        spin_i = 0
        while not self._stop.is_set():
            elapsed          = int(time.time() - self.start_time)
            mins, secs       = divmod(elapsed, 60)
            time_str         = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            done             = self.current
            bar              = "█" * done + "░" * (self.total - done)
            spinner          = self.SPINNER[spin_i % len(self.SPINNER)]
            spin_i          += 1
            print(
                f"\r  {spinner} MBTI Layer  [{bar}] {done}/{self.total}  "
                f"elapsed: {time_str}  current: {self.current_name[:28]:<28}",
                end="", flush=True,
            )
            time.sleep(0.2)


# ── Sentence-transformer singleton ────────────────────────────────────────────
# FIX v1.1: model is loaded ONCE here and reused for all 16 MBTI types.
# Previously it was loaded inside _score_mbti_against_buckets on every call,
# causing 16 BertModel LOAD REPORT lines and wasting ~30 seconds per run.

_EMBED_MODEL = None


def _get_embed_model():
    """Return the singleton embedding model. Loads on first call, then cached."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded (singleton, will not reload)")
        except ImportError:
            _EMBED_MODEL = False   # sentinel — skip on every subsequent call
    return _EMBED_MODEL if _EMBED_MODEL is not False else None


# ── Score MBTI output against dynamic buckets ─────────────────────────────────

def _score_mbti_against_buckets(
    reaction: dict,
    buckets: list[dict],
) -> dict[str, float]:
    """
    Score an MBTI reaction's narrative_frame against dynamic buckets.
    Uses the module-level singleton — no reloading per call.
    """
    if not buckets:
        return {}

    frame_text = (
        reaction.get("narrative_frame", "") + " " +
        reaction.get("emotional_state", "")
    ).strip()
    if not frame_text:
        return {}

    model = _get_embed_model()

    if model is not None:
        import numpy as np
        frame_emb = model.encode([frame_text], normalize_embeddings=True)[0]
        ref_embs  = model.encode(
            [b["reference"] for b in buckets], normalize_embeddings=True
        )
        return {
            b["key"]: round(float(np.dot(frame_emb, ref_embs[i])), 4)
            for i, b in enumerate(buckets)
        }

    # Keyword fallback (sentence-transformers not installed)
    frame_lower = frame_text.lower()
    return {
        b["key"]: round(
            sum(1 for w in b["reference"].lower().split() if w in frame_lower)
            / max(len(b["reference"].split()), 1),
            4,
        )
        for b in buckets
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_mbti_layer(
    signal: str,
    world_state_brief: str = "",
    dynamic_buckets: list[dict] = None,
) -> dict:
    """
    Run all 16 MBTI types against the signal.

    Args:
        signal:            The raw signal text (or enriched signal from L1)
        world_state_brief: L1 behavioral synthesis brief (empty in Pure Mode)
        dynamic_buckets:   L0-generated bucket list for scoring reactions

    Returns:
        population_report dict containing reactions, clusters, dominant_emotion,
        sharing_velocity, population_buckets, report_text, timestamp.
    """
    dynamic_buckets = dynamic_buckets or []

    # Pre-warm the singleton so the one load happens before the progress bar starts.
    if dynamic_buckets:
        _get_embed_model()

    logger.info(f"MBTI Layer: running 16 types on signal ({len(signal)} chars)")
    print(f"\n  ── MBTI Population Reaction Layer ───────────────────────────────")

    world_state_section = (
        f"\nCurrent information environment:\n{world_state_brief[:400]}"
        if world_state_brief else ""
    )

    tracker  = MBTIProgressTracker(total=len(MBTI_TYPES))
    tracker.start()
    reactions = {}
    run_start = time.time()

    try:
        for i, (mbti_type, config) in enumerate(MBTI_TYPES.items()):
            tracker.update(i, mbti_type + "...")

            prompt = MBTI_REACTION_PROMPT.format(
                mbti_type=mbti_type,
                label=config["label"],
                cognitive_style=config["cognitive_style"],
                emotional_baseline=config["emotional_baseline"],
                trust_hierarchy=config["trust_hierarchy"],
                sharing_behavior=config["sharing_behavior"],
                signal=signal[:500],
                world_state_section=world_state_section,
            )

            raw    = _call_ollama(prompt, max_tokens=250)
            parsed = _parse_mbti_response(raw, mbti_type)

            bucket_scores  = _score_mbti_against_buckets(parsed, dynamic_buckets)
            primary_bucket = (
                max(bucket_scores, key=bucket_scores.get)
                if bucket_scores else "unknown"
            )

            reactions[mbti_type] = {
                "mbti_type":         mbti_type,
                "label":             config["label"],
                "population_pct":    config["population_pct"],
                "sharing_behavior":  config["sharing_behavior"],
                "emotional_state":   parsed["emotional_state"],
                "narrative_frame":   parsed["narrative_frame"],
                "sharing_intention": parsed["sharing_intention"],
                "trust_level":       parsed["trust_level"],
                "primary_bucket":    primary_bucket,
                "bucket_scores":     bucket_scores,
                "raw_response":      parsed["raw"],
            }

            tracker.update(i + 1, f"{mbti_type} done ({round(time.time()-run_start,1)}s)")

    finally:
        tracker.stop()

    total_time = round(time.time() - run_start, 1)
    print(f"  ✓ MBTI Layer complete — 16 types — {total_time}s")

    clusters           = _build_reaction_clusters(reactions)
    population_buckets = _compute_population_bucket_weights(reactions, dynamic_buckets)
    sharing_velocity   = _compute_sharing_velocity(reactions)
    dominant_emotion   = _compute_dominant_emotion(reactions)
    report_text        = _build_report_text(
        reactions, clusters, population_buckets, sharing_velocity, dominant_emotion,
    )

    logger.info(
        f"MBTI Layer complete: dominant_emotion={dominant_emotion}, "
        f"sharing_velocity={sharing_velocity:.2f}"
    )

    return {
        "reactions":           reactions,
        "reaction_clusters":   clusters,
        "dominant_emotion":    dominant_emotion,
        "sharing_velocity":    sharing_velocity,
        "population_buckets":  population_buckets,
        "report_text":         report_text,
        "timestamp":           datetime.utcnow().isoformat() + "Z",
    }


# ── Population analytics ──────────────────────────────────────────────────────

def _build_reaction_clusters(reactions: dict) -> dict:
    clusters = {}
    for cluster_name, types in REACTION_CLUSTERS.items():
        if not any(t in reactions for t in types):
            continue
        population_weight = sum(reactions[t]["population_pct"] for t in types if t in reactions)
        emotional_states  = [reactions[t]["emotional_state"]  for t in types if t in reactions]
        narrative_frames  = [reactions[t]["narrative_frame"]  for t in types if t in reactions]
        clusters[cluster_name] = {
            "types":            types,
            "population_pct":   round(population_weight, 3),
            "population_label": f"{round(population_weight * 100, 1)}% of population",
            "emotional_states": emotional_states,
            "narrative_frames": narrative_frames,
            "summary":          _summarize_cluster(emotional_states, narrative_frames),
        }
    return clusters


def _summarize_cluster(emotional_states: list, narrative_frames: list) -> str:
    if not emotional_states:
        return "No data"
    words = []
    for s in emotional_states:
        words.extend(s.lower().split()[:2])
    most_common = Counter(words).most_common(1)
    mood  = most_common[0][0] if most_common else "mixed"
    frame = narrative_frames[0][:80] if narrative_frames else "no frame"
    return f"{mood.capitalize()} — e.g.: {frame}"


def _compute_population_bucket_weights(
    reactions: dict,
    dynamic_buckets: list[dict],
) -> dict[str, float]:
    if not dynamic_buckets:
        return {}
    weights = defaultdict(float)
    for reaction in reactions.values():
        primary = reaction.get("primary_bucket", "unknown")
        if primary and primary != "unknown":
            weights[primary] += reaction["population_pct"]
    total = sum(weights.values())
    if total == 0:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(weights.items(), key=lambda x: -x[1])}


def _compute_sharing_velocity(reactions: dict) -> float:
    HIGH   = {"ESFJ", "ISFJ", "ESFP", "ENFP", "ENFJ", "ESTJ"}
    MEDIUM = {"ESTP", "ENTP", "ENTJ", "ISFP", "INFP"}
    velocity = sum(
        r["population_pct"] * (1.0 if t in HIGH else 0.5 if t in MEDIUM else 0.0)
        for t, r in reactions.items()
    )
    return round(min(velocity, 1.0), 3)


def _compute_dominant_emotion(reactions: dict) -> str:
    """
    Find the dominant emotional state weighted by population percentage.

    FIX v1.1: skips bare numeric tokens (e.g. "1", "1.", "2)") that the LLM
    sometimes prepends to numbered responses.

    FIX v1.2: also skips common stopwords ("the", "a", "an", "is", "this",
    "immediate", "emotional", "response") that appear when the LLM returns
    a full sentence instead of leading with the emotion word itself.
    The first substantive word (length > 3, not a stopword) is used.
    """
    # Words to skip — articles, filler words, prompt echoes
    SKIP = {
        "the", "a", "an", "is", "are", "was", "this", "that", "it", "its",
        "my", "our", "their", "i", "we", "they", "he", "she",
        "immediate", "emotional", "emotion", "response", "reaction",
        "state", "feeling", "feel", "type", "mbti", "would", "will",
        "instinctive", "instinct", "typical", "likely", "probably",
    }

    emotion_weights: Counter = Counter()
    for reaction in reactions.values():
        pop   = reaction["population_pct"]
        state = reaction.get("emotional_state", "").lower().strip()
        core  = "unknown"
        for token in state.split():
            cleaned = token.rstrip(".,;:)")
            # Skip pure numeric tokens like "1", "1.", "2)"
            if re.match(r'^\d+$', cleaned):
                continue
            # Skip stopwords and short filler
            if cleaned in SKIP or len(cleaned) <= 2:
                continue
            core = cleaned
            break
        emotion_weights[core] += pop

    return emotion_weights.most_common(1)[0][0] if emotion_weights else "unknown"


# ── Report builder ────────────────────────────────────────────────────────────

def _build_report_text(
    reactions: dict,
    clusters: dict,
    population_buckets: dict,
    sharing_velocity: float,
    dominant_emotion: str,
) -> str:
    lines = [
        "═" * 60,
        "POPULATION REACTION REPORT (L2A — MBTI Layer)",
        "═" * 60,
        "",
        f"Dominant emotional state: {dominant_emotion.upper()}",
        f"Sharing velocity: {sharing_velocity:.0%} of population will actively broadcast",
        "",
        "── Reaction clusters ──────────────────────────────────────",
        "",
    ]

    for cluster_name, data in clusters.items():
        readable = cluster_name.replace("_", " ").title()
        lines.append(f"{readable} ({data['population_label']}):")
        lines.append(f"  {data['summary']}")
        if data["narrative_frames"]:
            lines.append(f"  Sample frame: \"{data['narrative_frames'][0][:100]}\"")
        lines.append("")

    lines += [
        "── Per-type reactions (population weight) ─────────────────",
        "",
        f"  {'Type':<6} {'Pop%':<6} {'Emotion':<20} {'Shares?'}",
        "  " + "─" * 50,
    ]
    for mbti_type, r in sorted(reactions.items(), key=lambda x: -x[1]["population_pct"]):
        pop       = f"{r['population_pct']*100:.0f}%"
        emotion   = r["emotional_state"][:18] if r["emotional_state"] else "—"
        sharing   = r["sharing_intention"][:28] if r["sharing_intention"] else "—"
        share_flag = (
            "HIGH"     if "widely" in sharing.lower() or "broadcast" in sharing.lower()
            else "low" if "not" in sharing.lower() or "unlikely" in sharing.lower()
            else "moderate"
        )
        lines.append(f"  {mbti_type:<6} {pop:<6} {emotion:<20} {share_flag}")

    if population_buckets:
        lines += ["", "── Population bucket alignment (bottom-up) ────────────────", ""]
        for bucket_key, weight in list(population_buckets.items())[:5]:
            bar = "█" * int(weight * 30) + "░" * (30 - int(weight * 30))
            lines.append(f"  {bucket_key:<28} {bar} {weight:.0%}")

    lines += ["", "═" * 60, "END POPULATION REPORT — analysts: consider this context.", "═" * 60]
    return "\n".join(lines)


# ── Convergence gap calculator ────────────────────────────────────────────────

def compute_convergence_gap(
    population_buckets: dict,
    analyst_probability_map: list[dict],
) -> dict:
    """
    Calculate the convergence gap between the MBTI bottom-up map
    and the analyst top-down probability map. Called from L4.
    """
    if not population_buckets or not analyst_probability_map:
        return {"gap_status": "insufficient_data", "gap_score": 0.0}

    analyst_top    = analyst_probability_map[0]
    analyst_bucket = analyst_top.get("bucket", "unknown")
    analyst_prob   = analyst_top.get("probability", 0.0)
    pop_weight     = population_buckets.get(analyst_bucket, 0.0)
    gap_score      = round(abs(analyst_prob - pop_weight), 4)

    if gap_score <= 0.15:
        status  = "Validated Narrative"
        meaning = "Spreading for the right reasons — genuine cognitive resonance."
        action  = "High confidence prediction."
    elif pop_weight > analyst_prob + 0.15:
        status  = "Artificially Amplified"
        meaning = "Population reaction exceeds analytical merit. Possible coordination, bots, or agenda-setting."
        action  = "Flag for investigation. Lower prediction confidence."
    else:
        status  = "Latent Narrative"
        meaning = "Analytically strong but not yet spreading. Suppressed or ahead of its time."
        action  = "Watch for emergence. Likely incoming."

    return {
        "gap_status":          status,
        "gap_score":           gap_score,
        "meaning":             meaning,
        "action":              action,
        "analyst_top_bucket":  analyst_bucket,
        "analyst_probability": analyst_prob,
        "population_weight":   pop_weight,
    }
