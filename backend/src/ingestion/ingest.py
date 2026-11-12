"""
src/ingestion/ingest.py

Main ingestion runner — handles all three input types:
  1. User text / query        → signal_builder.from_user_text()
  2. URL / article link       → signal_builder.from_url()
  3. Auto-ingest (background) → all scrapers → signal_builder.from_ingested()

All three paths produce the same enriched signal dict:
  entities extracted → dynamic search enrichment → scorer → ready for engine

Run:
    python src/ingestion/ingest.py                  # auto-ingest mode
    python src/ingestion/ingest.py --sources        # show source stats
"""

import sys
import os
from loguru import logger

from src.ingestion.signal_builder import build_signal, from_user_text, from_url
from src.ingestion.entity_extractor import extract_entities
from src.ingestion.dynamic_search import enrich_signal
from src.ingestion.signal_scorer import filter_and_rank, save_signals, score_article

from src.ingestion.gdelt_scraper import fetch_all_gdelt_signals
from src.ingestion.hackernews_scraper import fetch_hn_signals
from src.ingestion.bluesky_scraper import fetch_bluesky_signals
from src.ingestion.lemmy_scraper import fetch_lemmy_signals
from src.ingestion.newsapi_scraper import fetch_all_news_signals

AUTO_SOURCES = [
    {"name": "GDELT",        "fn": lambda: fetch_all_gdelt_signals(hours_back=6)},
    {"name": "Hacker News",  "fn": lambda: fetch_hn_signals(max_stories=30)},
    {"name": "Bluesky",      "fn": lambda: fetch_bluesky_signals(max_results=40)},
    {"name": "Lemmy",        "fn": lambda: fetch_lemmy_signals()},
    {"name": "News APIs",    "fn": lambda: fetch_all_news_signals()},
]


def process_signal(signal: dict, enrich: bool = True) -> dict:
    """
    Full processing pipeline for any signal dict:
      1. Extract entities (spaCy or regex)
      2. Enrich with related articles (dynamic search)
      3. Score (sentiment + ambiguity + length)
    """
    signal = extract_entities(signal)
    if enrich:
        signal = enrich_signal(signal)
    signal = score_article(signal)
    return signal


def ingest_user_input(text_or_url: str, enrich: bool = True) -> dict:
    """
    Entry point for user-submitted signals — text or URL.
    Returns a fully processed, enriched signal dict
    ready to pass to run_all_models().

    Usage:
        signal = ingest_user_input("I heard Russia is mobilising near Finland")
        signal = ingest_user_input("https://www.bbc.com/news/article-xyz")
    """
    logger.info(f"User input: {text_or_url[:80]}...")
    signal = build_signal(text_or_url)
    signal = process_signal(signal, enrich=enrich)

    logger.info(f"Signal score: {signal['score']:.3f} | "
                f"entities: {len(signal.get('entities', []))} | "
                f"enriched: {signal.get('enriched', False)}")
    return signal


def run_ingestion(max_signals: int = 10, verbose: bool = True) -> list[dict]:
    """
    Auto-ingestion mode — pulls from all background sources,
    processes, scores, returns top signals.
    """
    if verbose:
        logger.info("=== Butterfly Algorithm — Signal Ingestion ===")

    all_raw = []
    source_counts = {}

    for source in AUTO_SOURCES:
        name = source["name"]
        try:
            logger.info(f"Fetching {name}...")
            articles = source["fn"]()
            all_raw.extend(articles)
            source_counts[name] = len(articles)
            logger.info(f"  {name}: {len(articles)} items")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            source_counts[name] = 0

    logger.info(f"Total raw: {len(all_raw)} items")

    # Normalise all raw articles into signal dicts
    signals = []
    for raw in all_raw:
        try:
            s = build_signal(raw)
            s = extract_entities(s)
            # Skip enrichment in batch mode — too slow for 50+ articles
            # Enrichment happens when a specific signal is selected for simulation
            s = score_article(s)
            signals.append(s)
        except Exception as e:
            logger.warning(f"Signal processing failed: {e}")

    # Filter, deduplicate, rank
    ranked = filter_and_rank(signals)
    top = ranked[:max_signals]

    saved = save_signals(top)
    if saved:
        logger.info(f"Saved {len(saved)} new signals → data/signals/")

    return top


def print_signals(signals: list[dict]) -> None:
    if not signals:
        print("\n  No signals passed threshold.")
        print("  Tip: lower SIGNAL_SCORE_THRESHOLD in .env (default 0.45)\n")
        return

    print(f"\n  {'#':<4} {'Score':<7} {'Source':<14} {'Title/Signal'}")
    print("  " + "─" * 70)
    for i, s in enumerate(signals):
        src = s.get("source_type", "?")
        title = s.get("title", s["signal_text"])[:60]
        bd = s.get("score_breakdown", {})
        ents = [e["text"] for e in s.get("entities", [])[:3]]
        print(f"  {i+1:<4} {s['score']:.3f}  {src:<14} {title}")
        print(
            f"       sent={bd.get('sentiment',0):.2f}  "
            f"ambig={bd.get('ambiguity',0):.2f}  "
            f"enriched={s.get('enriched', False)}  "
            f"entities={ents}"
        )
    print()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    if len(sys.argv) > 1 and sys.argv[1] == "--user":
        # Test user input mode
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "US and China are in a trade war"
        signal = ingest_user_input(text)
        print(f"\nSignal text:\n{signal['signal_text']}\n")
        print(f"Entities: {signal.get('entities', [])}")
        print(f"Score: {signal['score']:.3f}")
        print(f"Enriched: {signal.get('enriched', False)}")
    else:
        signals = run_ingestion(max_signals=10)
        print_signals(signals)
