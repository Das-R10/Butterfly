"""
src/ingestion/dynamic_search.py

Takes LLM-extracted entities from a signal and builds live search
queries for GDELT and HN. No hardcoded keywords anywhere.
"""

import requests
from datetime import datetime
from loguru import logger

from src.ingestion.entity_extractor import entities_to_queries

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"


def search_gdelt_dynamic(queries: list[str], hours_back: int = 12) -> list[dict]:
    """Run dynamic GDELT queries built from signal entities."""
    seen_urls = set()
    articles = []

    for query in queries:
        try:
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": 8,
                "timespan": f"{hours_back}h",
                "sort": "datedesc",
                "format": "json",
            }
            resp = requests.get(GDELT_API, params=params, timeout=12)
            if resp.status_code != 200:
                continue

            for a in resp.json().get("articles", []):
                url = a.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append({
                        "title": a.get("title", ""),
                        "url": url,
                        "source": a.get("domain", ""),
                        "published": a.get("seendate", ""),
                        "language": a.get("language", "English"),
                        "country": a.get("sourcecountry", ""),
                        "keyword_query": query,
                        "source_type": "gdelt",
                        "ingested_at": datetime.utcnow().isoformat() + "Z",
                    })
        except Exception as e:
            logger.warning(f"Dynamic GDELT query '{query}' failed: {e}")

    logger.info(f"Dynamic GDELT: {len(articles)} articles for {len(queries)} queries")
    return articles


def search_hn_dynamic(queries: list[str], min_points: int = 20) -> list[dict]:
    """Search HN Algolia index with dynamic queries."""
    seen_ids = set()
    stories = []

    for query in queries:
        try:
            resp = requests.get(
                HN_SEARCH_API,
                params={
                    "query": query,
                    "tags": "story",
                    "numericFilters": f"points>{min_points}",
                    "hitsPerPage": 8,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                continue

            for hit in resp.json().get("hits", []):
                story_id = hit.get("objectID", "")
                if story_id in seen_ids:
                    continue
                seen_ids.add(story_id)

                title = hit.get("title", "")
                if not title:
                    continue

                stories.append({
                    "title": title,
                    "description": hit.get("story_text", ""),
                    "url": hit.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                    "source": "news.ycombinator.com",
                    "source_type": "hackernews",
                    "hn_score": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", 0),
                    "keyword_query": query,
                    "published": hit.get("created_at", ""),
                    "ingested_at": datetime.utcnow().isoformat() + "Z",
                })
        except Exception as e:
            logger.warning(f"Dynamic HN query '{query}' failed: {e}")

    logger.info(f"Dynamic HN: {len(stories)} stories for {len(queries)} queries")
    return stories


def enrich_signal(signal: dict, hours_back: int = 12) -> dict:
    """
    Enrich a signal with related articles from GDELT + HN.
    Uses LLM-extracted entities to build targeted queries.
    """
    # Build queries from extracted entities
    queries = entities_to_queries(signal, max_queries=4)

    if not queries:
        logger.info("No queries generated — using signal as-is")
        signal["enriched"] = False
        return signal

    logger.info(f"Enrichment queries: {queries}")

    gdelt_articles = search_gdelt_dynamic(queries, hours_back=hours_back)
    hn_stories = search_hn_dynamic(queries)
    related = gdelt_articles[:3] + hn_stories[:2]

    if not related:
        signal["enriched"] = False
        return signal

    # Append related coverage as context block
    context_parts = ["\n[Related coverage:"]
    for a in related[:4]:
        title = a.get("title", "").strip()
        source = a.get("source", "")
        if title:
            context_parts.append(f"  • {title} [{source}]")
    context_parts.append("]")
    context = "\n".join(context_parts)

    current = signal.get("signal_text", "")
    max_total = 1500
    available = max_total - len(current)
    if available > 100:
        signal["signal_text"] = current + context[:available]

    signal["related_articles"] = [
        {"title": a.get("title"), "source": a.get("source"), "url": a.get("url")}
        for a in related
    ]
    signal["enriched"] = True
    signal["enrichment_queries"] = queries

    logger.info(f"Signal enriched with {len(related)} related articles")
    return signal
