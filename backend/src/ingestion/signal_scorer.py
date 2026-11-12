"""
src/ingestion/signal_scorer.py

Scores raw articles/posts from all sources and converts them
into clean signal dicts ready for the personality engine.

Scoring criteria (v1.2):
  - Geopolitical relevance gate  (MUST pass — hard filter)
  - Ambiguity / interpretability (0.40 weight)
  - Emotional charge / sentiment (0.30 weight)
  - Content richness / length    (0.30 weight)

v1.2 changes:
  - Added hard relevance gate — article must mention at least one
    geopolitical/economic keyword to enter scoring at all.
    Eliminates false positives like "Drugwars calculator" and
    "Bill Maher Mark Twain Prize" that score high on sentiment alone.
  - Rebalanced weights: ambiguity up, sentiment down.
  - Added topic penalty for clearly off-topic content.
"""

import os
import re
import json
import hashlib
from datetime import datetime
from collections import defaultdict
from loguru import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    logger.warning("vaderSentiment not installed — sentiment scoring disabled")

THRESHOLD = float(os.getenv("SIGNAL_SCORE_THRESHOLD", "0.45"))

# ── Hard relevance gate ───────────────────────────────────────────────────────
# Article must mention at least 1 of these to enter scoring.
# This is a GATE not a score — if it fails, score = 0 immediately.
RELEVANCE_KEYWORDS = [
    # Conflict
    "war", "invasion", "military", "troops", "missile", "bomb", "attack",
    "nato", "nuclear", "drone", "airstrike", "ceasefire", "offensive",
    # Geopolitics
    "sanctions", "tariff", "geopolitical", "ukraine", "russia", "china",
    "iran", "israel", "taiwan", "nato", "north korea", "pakistan",
    "middle east", "hormuz", "border dispute", "coup", "protest",
    # Economy
    "inflation", "recession", "market crash", "supply chain", "oil price",
    "trade", "trade war", "rare earth", "federal reserve", "interest rate",
    "economic crisis", "stock market", "gdp", "tariffs", "embargo",
    # Crisis signals
    "breaking", "escalation", "tensions", "crisis", "emergency",
    "blockade", "siege", "collapse", "unrest",
]

# ── Ambiguity words ───────────────────────────────────────────────────────────
# Present = interpretable signal (higher value for Butterfly)
AMBIGUITY_WORDS = [
    "may", "could", "might", "reportedly", "sources say", "alleged",
    "unclear", "possible", "potential", "rumor", "unconfirmed",
    "signals", "suggests", "indicates", "appears", "tensions",
    "escalat", "crisis", "warning", "threat", "concerns",
    "unexpected", "surprise", "shock", "sudden", "breaking",
    "warned", "feared", "risks", "at stake", "imminent",
]

# ── Resolved words ────────────────────────────────────────────────────────────
# Present = factual settled statement (lower interpretive value)
RESOLVED_WORDS = [
    "confirmed", "officially", "announced", "signed", "completed",
    "finalized", "agreed", "concluded", "ended", "resolved",
    "voted", "passed", "approved", "rejected",
]

# ── Off-topic penalty triggers ────────────────────────────────────────────────
# If title/text matches these patterns, apply a score penalty.
# Catches cultural/entertainment content that sneaks through.
OFF_TOPIC_PATTERNS = [
    r"\b(calculator|ti-\d+|ti\d+)\b",           # calculator models
    r"\b(award|prize|grammy|oscar|emmy|bafta)\b", # entertainment awards
    r"\b(movie|film|album|song|concert|tour)\b",  # entertainment
    r"\b(recipe|cooking|food|restaurant)\b",       # food
    r"\b(sports|nba|nfl|soccer|football|cricket)\b", # sports (not military)
    r"drugwar",   # catches Drugwars even though war is in RELEVANCE_KEYWORDS                                  # specific false positive
    r"\bgambl",                                    # gambling
]
_OFF_TOPIC_RE = re.compile("|".join(OFF_TOPIC_PATTERNS), re.IGNORECASE)


def _passes_relevance_gate(text: str) -> bool:
    """Hard gate — must mention at least 1 geopolitical/economic keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in RELEVANCE_KEYWORDS)


def _has_off_topic_penalty(text: str) -> bool:
    """Check if content is clearly off-topic entertainment/lifestyle."""
    return bool(_OFF_TOPIC_RE.search(text))


def _sentiment_score(text: str) -> float:
    """Absolute VADER compound score 0.0 to 1.0"""
    if not HAS_VADER or not text:
        return 0.3
    scores = VADER.polarity_scores(text)
    return abs(scores["compound"])


def _ambiguity_score(text: str) -> float:
    """How ambiguous / interpretable is this signal? 0.0 to 1.0"""
    if not text:
        return 0.0
    text_lower = text.lower()
    ambig_hits = sum(1 for w in AMBIGUITY_WORDS if w in text_lower)
    resolved_hits = sum(1 for w in RESOLVED_WORDS if w in text_lower)
    raw = (ambig_hits * 2 - resolved_hits) / max(len(AMBIGUITY_WORDS), 1)
    return max(0.0, min(1.0, raw))


def _length_score(text: str) -> float:
    """Longer signals have more content to analyse. 0.0 to 1.0"""
    length = len(text.strip())
    if length < 50:
        return 0.1
    if length < 150:
        return 0.4
    if length < 400:
        return 0.7
    if length < 800:
        return 0.9
    return 1.0


def _signal_id(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


def build_signal_text(article: dict) -> str:
    """
    Combine article fields into a rich signal text
    that gives personalities enough context to analyse.
    """
    parts = []

    title = article.get("title", "").strip()
    if title:
        parts.append(title)

    description = article.get("description", "").strip()
    if description and description != title:
        parts.append(description)

    content = article.get("content", "").strip()
    if content and len(content) > 100:
        parts.append(content[:400])

    selftext = article.get("selftext", "").strip()
    if selftext and len(selftext) > 50:
        parts.append(selftext[:300])

    for comment in article.get("top_comments", [])[:2]:
        if comment:
            parts.append(f"[Community reaction: {comment[:200]}]")

    source = article.get("source", "") or article.get("subreddit", "")
    country = article.get("country", "")
    if source:
        ctx = f"[Source: {source}"
        if country:
            ctx += f" | {country}"
        ctx += "]"
        parts.append(ctx)

    return " ".join(parts).strip()


def score_article(article: dict) -> dict:
    """
    Score a single article.
    Returns enriched dict with score fields.

    Hard gate first — if article is not geopolitically relevant,
    score = 0 and passes_threshold = False immediately.
    """
    # Use existing signal_text if already built (from signal_builder)
    signal_text = article.get("signal_text") or build_signal_text(article)

    # ── User-submitted signals always pass ───────────────────────────────────
    is_user_input = article.get("source_type") in ("user_text", "url")

    # ── Hard relevance gate (auto-ingest only) ────────────────────────────────
    title = article.get("title", "")
    full_text = f"{title} {signal_text}"

    if not is_user_input and not _passes_relevance_gate(full_text):
        return {
            **article,
            "signal_text": signal_text,
            "signal_id": _signal_id(signal_text),
            "score": 0.0,
            "score_breakdown": {
                "sentiment": 0.0,
                "ambiguity": 0.0,
                "length": 0.0,
                "gate": "failed_relevance",
            },
            "passes_threshold": False,
            "scored_at": datetime.utcnow().isoformat() + "Z",
        }

    # ── Off-topic penalty ─────────────────────────────────────────────────────
    off_topic = _has_off_topic_penalty(full_text)

    # ── Component scores ──────────────────────────────────────────────────────
    sentiment = _sentiment_score(signal_text)
    ambiguity = _ambiguity_score(signal_text)
    length = _length_score(signal_text)

    # Rebalanced weights v1.2:
    # Ambiguity is the most important — we want interpretable signals
    # Length matters — short tweets are less useful than full articles
    # Sentiment is last — emotional charge alone is not enough
    composite = (
        ambiguity * 0.40 +
        length    * 0.30 +
        sentiment * 0.30
    )

    # Apply off-topic penalty
    if off_topic:
        composite *= 0.4

    return {
        **article,
        "signal_text": signal_text,
        "signal_id": _signal_id(signal_text),
        "score": round(composite, 4),
        "score_breakdown": {
            "sentiment": round(sentiment, 4),
            "ambiguity": round(ambiguity, 4),
            "length":    round(length, 4),
            "off_topic": off_topic,
        },
        "passes_threshold": is_user_input or composite >= THRESHOLD,
        "scored_at": datetime.utcnow().isoformat() + "Z",
    }


def filter_and_rank(articles: list[dict]) -> list[dict]:
    """
    Score all articles, filter by threshold, deduplicate,
    return ranked list (highest score first).
    """
    scored = [score_article(a) for a in articles]
    passing = [a for a in scored if a["passes_threshold"]]

    seen_ids = set()
    unique = []
    for a in passing:
        sid = a["signal_id"]
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique.append(a)

    ranked = sorted(unique, key=lambda x: x["score"], reverse=True)

    logger.info(
        f"Signal scorer: {len(articles)} in → "
        f"{len(passing)} passed threshold → "
        f"{len(ranked)} unique signals"
    )
    return ranked


def save_signals(signals: list[dict], output_dir: str = "data/signals") -> list[str]:
    """Save each signal as a JSON file. Returns list of saved paths."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for s in signals:
        path = os.path.join(output_dir, f"{s['signal_id']}.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2, ensure_ascii=False)
            saved.append(path)
    return saved