"""
src/ingestion/gdelt_scraper.py

Pulls live news signals from GDELT 2.0 DOC API.
Zero API key required — fully free and open.
Updates every 15 minutes from 100+ languages worldwide.

Fix v1.1: gdeltdoc library has a date format bug — always use
direct HTTP requests instead of the library.
"""

import requests
from datetime import datetime
from loguru import logger

SIGNAL_KEYWORDS = [
    "military conflict escalation",
    "war declaration sanctions",
    "economic crisis inflation",
    "geopolitical tensions border",
    "nuclear threat deterrence",
    "trade war tariff retaliation",
    "coup protest uprising",
    "terrorist attack security",
    "energy crisis oil supply",
    "diplomatic breakdown summit",
]

GDELT_API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_articles_direct(keyword: str, hours_back: int = 6) -> list[dict]:
    """
    Fetch articles from GDELT DOC API using direct HTTP.
    Avoids the gdeltdoc library date format bug entirely.
    Uses timespan parameter — no date formatting required.
    """
    params = {
        "query": keyword,
        "mode": "artlist",
        "maxrecords": 10,
        "timespan": f"{hours_back}h",
        "sort": "datedesc",
        "format": "json",
    }
    try:
        resp = requests.get(GDELT_API_BASE, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"GDELT returned {resp.status_code} for '{keyword}'")
            return []

        data = resp.json()
        articles = data.get("articles", [])
        results = []
        for a in articles:
            results.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("domain", ""),
                "published": a.get("seendate", ""),
                "language": a.get("language", "English"),
                "country": a.get("sourcecountry", ""),
                "keyword_query": keyword,
                "source_type": "gdelt",
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            })
        return results

    except Exception as e:
        logger.error(f"GDELT fetch failed for '{keyword}': {e}")
        return []


def fetch_all_gdelt_signals(hours_back: int = 6) -> list[dict]:
    """
    Fetch articles for all monitored keyword groups.
    Returns deduplicated list of articles.
    Always uses direct HTTP — never the gdeltdoc library.
    """
    seen_urls = set()
    all_articles = []

    for keyword in SIGNAL_KEYWORDS:
        logger.info(f"GDELT: fetching '{keyword}'...")
        articles = fetch_gdelt_articles_direct(keyword, hours_back)

        for a in articles:
            url = a.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(a)

    logger.info(f"GDELT: {len(all_articles)} unique articles fetched")
    return all_articles