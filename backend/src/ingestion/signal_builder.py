"""
src/ingestion/signal_builder.py

Normalises all three input types into a clean signal dict:
  1. Raw text / user query   — "what I heard about X"
  2. URL / article link      — fetches and extracts text
  3. Auto-ingested article   — passes through with enrichment

Output is always:
  {
    "signal_id": str,
    "signal_text": str,        # clean text for personality engine
    "title": str,              # short title for display
    "source_type": str,        # "user_text" | "url" | "gdelt" | "hackernews" | ...
    "source": str,             # URL or source name
    "raw_input": str,          # original input unchanged
    "entities": [],            # filled by entity_extractor
    "enriched": False,         # True after dynamic_search enrichment
    "ingested_at": str,
  }
"""

import re
import hashlib
import requests
from datetime import datetime
from loguru import logger

# Max chars for signal text fed to personality engine
MAX_SIGNAL_CHARS = 1200


def _signal_id(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


def _clean_text(text: str) -> str:
    """Remove excess whitespace, HTML fragments, and normalize quotes."""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    return text.strip()


def _fetch_url(url: str) -> dict:
    """
    Fetch article text from a URL.
    Uses a simple extraction heuristic — good enough for news articles.
    For production, swap with newspaper3k or trafilatura.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; ButterflyAlgorithm/1.0; "
                "+https://github.com/butterfly-algorithm)"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        # Try to extract <title>
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = _clean_text(title_match.group(1)) if title_match else url

        # Extract <article> or <main> content first, fall back to <p> tags
        article_match = re.search(
            r"<(?:article|main)[^>]*>(.*?)</(?:article|main)>",
            html, re.IGNORECASE | re.DOTALL
        )
        if article_match:
            raw = article_match.group(1)
        else:
            raw = html

        # Extract all paragraph text
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", raw, re.IGNORECASE | re.DOTALL)
        text_parts = [_clean_text(p) for p in paragraphs if len(_clean_text(p)) > 40]
        body = " ".join(text_parts[:12])  # first 12 paragraphs

        if not body:
            logger.warning(f"Could not extract text from {url}")
            body = title

        return {
            "title": title[:200],
            "body": body[:MAX_SIGNAL_CHARS],
            "url": url,
        }

    except Exception as e:
        logger.error(f"URL fetch failed for {url}: {e}")
        return {"title": url, "body": "", "url": url}


def from_user_text(text: str, title: str = None) -> dict:
    """
    Build a signal from raw user text.
    This is what gets called when a user types "I heard that..." or
    pastes a paragraph they saw somewhere.
    """
    clean = _clean_text(text)
    signal_text = clean[:MAX_SIGNAL_CHARS]

    return {
        "signal_id": _signal_id(signal_text),
        "signal_text": signal_text,
        "title": title or signal_text[:80] + ("..." if len(signal_text) > 80 else ""),
        "source_type": "user_text",
        "source": "user_input",
        "raw_input": text,
        "entities": [],
        "enriched": False,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }


def from_url(url: str) -> dict:
    """
    Build a signal by fetching and extracting text from a URL.
    Handles news articles, Reddit threads, blog posts.
    """
    logger.info(f"Fetching URL: {url}")
    fetched = _fetch_url(url)

    title = fetched["title"]
    body = fetched["body"]

    # Combine title + body for a rich signal
    if body and body.lower() not in title.lower():
        signal_text = f"{title}. {body}"
    else:
        signal_text = title

    signal_text = signal_text[:MAX_SIGNAL_CHARS]

    return {
        "signal_id": _signal_id(signal_text),
        "signal_text": signal_text,
        "title": title[:200],
        "source_type": "url",
        "source": url,
        "raw_input": url,
        "entities": [],
        "enriched": False,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }


def from_ingested(article: dict) -> dict:
    """
    Normalise an auto-ingested article (from GDELT, HN, Lemmy etc.)
    into the standard signal dict format.
    """
    # Build rich signal text from available fields
    parts = []
    title = article.get("title", "").strip()
    description = article.get("description", "").strip()
    content = article.get("content", "").strip()
    selftext = article.get("selftext", "").strip()

    if title:
        parts.append(title)
    if description and description != title:
        parts.append(description)
    if content and len(content) > 80:
        parts.append(content[:600])
    elif selftext and len(selftext) > 80:
        parts.append(selftext[:400])

    # Add source context
    source = article.get("source", "") or article.get("subreddit", "")
    country = article.get("country", "")
    if source:
        ctx = f"[Source: {source}"
        if country:
            ctx += f" | {country}"
        ctx += "]"
        parts.append(ctx)

    signal_text = " ".join(parts).strip()[:MAX_SIGNAL_CHARS]

    return {
        **article,
        "signal_id": article.get("signal_id") or _signal_id(signal_text),
        "signal_text": signal_text,
        "title": title or signal_text[:80],
        "source_type": article.get("source_type", "unknown"),
        "source": article.get("url", article.get("source", "")),
        "raw_input": signal_text,
        "entities": [],
        "enriched": False,
        "ingested_at": article.get("ingested_at", datetime.utcnow().isoformat() + "Z"),
    }


def build_signal(input_data: str | dict) -> dict:
    """
    Universal entry point — auto-detects input type:
      - dict  → from_ingested
      - str starting with http → from_url
      - str  → from_user_text
    """
    if isinstance(input_data, dict):
        return from_ingested(input_data)
    if isinstance(input_data, str):
        stripped = input_data.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            return from_url(stripped)
        return from_user_text(stripped)
    raise ValueError(f"Unsupported input type: {type(input_data)}")
