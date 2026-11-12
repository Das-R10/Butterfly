"""
src/ingestion/newsapi_scraper.py

Pulls news from two free sources:
  1. NewsData.io  — 200 credits/day free, no CC required
     Sign up at: https://newsdata.io/register
     Add to .env: NEWSDATA_API_KEY=your_key

  2. GNews        — 100 req/day free
     Sign up at: https://gnews.io
     Add to .env: GNEWS_API_KEY=your_key

Both are optional — system works without them using GDELT alone.
"""

import os
import requests
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

NEWSDATA_KEY = os.getenv("NEWSDATA_API_KEY", "")
GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")

# Keywords to monitor
GEOPOLITICAL_QUERIES = [
    "war military conflict",
    "economic crisis recession",
    "sanctions trade war",
    "nuclear weapons threat",
    "political crisis coup",
    "terrorism attack security",
    "energy oil prices",
    "border dispute territory",
]


def fetch_newsdata(query: str, max_results: int = 10) -> list[dict]:
    """
    Fetch from NewsData.io free tier.
    200 credits/day — each request uses 1 credit.
    Docs: https://newsdata.io/documentation
    """
    if not NEWSDATA_KEY:
        return []

    try:
        params = {
            "apikey": NEWSDATA_KEY,
            "q": query,
            "language": "en",
            "category": "politics,world,business",
            "size": min(max_results, 10),
        }
        resp = requests.get(
            "https://newsdata.io/api/1/news",
            params=params,
            timeout=15
        )
        if resp.status_code != 200:
            logger.warning(f"NewsData returned {resp.status_code}")
            return []

        data = resp.json()
        articles = []
        for item in data.get("results", []):
            articles.append({
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "content": item.get("content", ""),
                "url": item.get("link", ""),
                "source": item.get("source_id", ""),
                "published": item.get("pubDate", ""),
                "country": ", ".join(item.get("country", [])),
                "keyword_query": query,
                "source_type": "newsdata",
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            })
        return articles

    except Exception as e:
        logger.error(f"NewsData fetch failed: {e}")
        return []


def fetch_gnews(query: str, max_results: int = 10) -> list[dict]:
    """
    Fetch from GNews free tier.
    100 requests/day free.
    Docs: https://gnews.io/docs/v4
    """
    if not GNEWS_KEY:
        return []

    try:
        params = {
            "q": query,
            "lang": "en",
            "country": "any",
            "max": min(max_results, 10),
            "apikey": GNEWS_KEY,
        }
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params=params,
            timeout=15
        )
        if resp.status_code != 200:
            logger.warning(f"GNews returned {resp.status_code}")
            return []

        data = resp.json()
        articles = []
        for item in data.get("articles", []):
            articles.append({
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("name", ""),
                "published": item.get("publishedAt", ""),
                "country": "",
                "keyword_query": query,
                "source_type": "gnews",
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            })
        return articles

    except Exception as e:
        logger.error(f"GNews fetch failed: {e}")
        return []


def fetch_all_news_signals() -> list[dict]:
    """
    Fetch from all available news APIs.
    Deduplicates by URL.
    """
    seen_urls = set()
    all_articles = []

    for query in GEOPOLITICAL_QUERIES:
        for article in fetch_newsdata(query) + fetch_gnews(query):
            url = article.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(article)

    logger.info(f"News APIs: {len(all_articles)} unique articles")
    return all_articles
