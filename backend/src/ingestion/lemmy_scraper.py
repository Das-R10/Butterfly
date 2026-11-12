"""
src/ingestion/lemmy_scraper.py

Pulls signals from Lemmy — the decentralized Reddit alternative.
Zero API key required. Completely open ActivityPub-based API.
48,000+ monthly active users as of late 2025, high-quality discussions.

Covers communities equivalent to:
  r/worldnews, r/geopolitics, r/collapse, r/economics, r/conspiracy

Docs: https://join-lemmy.org/api/
"""

import requests
from datetime import datetime
from loguru import logger

# Most active Lemmy instances for geopolitical content
INSTANCES = [
    "lemmy.world",
    "lemmy.ml",
    "beehaw.org",
    "lemmy.blahaj.zone",
]

# Communities to monitor (community@instance format)
TARGET_COMMUNITIES = [
    ("worldnews", "lemmy.ml"),
    ("geopolitics", "lemmy.world"),
    ("news", "lemmy.world"),
    ("technology", "lemmy.world"),
    ("collapse", "lemmy.world"),
    ("economics", "lemmy.ml"),
]

MIN_SCORE = 10
MIN_COMMENTS = 3

SIGNAL_KEYWORDS = [
    "war", "conflict", "sanctions", "tariff", "military", "invasion",
    "nuclear", "missile", "inflation", "recession", "crisis", "collapse",
    "coup", "protest", "election", "geopolitical", "trade", "oil",
    "china", "russia", "iran", "ukraine", "taiwan", "nato",
    "breaking", "market", "supply chain", "energy",
]


def _is_relevant(title: str) -> bool:
    return any(kw in title.lower() for kw in SIGNAL_KEYWORDS)


def fetch_community_posts(
    community: str,
    instance: str,
    limit: int = 20
) -> list[dict]:
    """Fetch posts from a specific Lemmy community."""
    try:
        url = f"https://{instance}/api/v3/post/list"
        params = {
            "community_name": community,
            "sort": "Hot",
            "limit": limit,
            "type_": "Local",
        }
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []

        posts = []
        for item in r.json().get("posts", []):
            post = item.get("post", {})
            counts = item.get("counts", {})

            title = post.get("name", "")
            score = counts.get("score", 0)
            comments = counts.get("comments", 0)

            if score < MIN_SCORE or comments < MIN_COMMENTS:
                continue
            if not _is_relevant(title):
                continue

            posts.append({
                "title": title,
                "description": post.get("body", "")[:400],
                "url": post.get("url", f"https://{instance}/post/{post.get('id')}"),
                "source": f"lemmy/{community}@{instance}",
                "score": score,
                "num_comments": comments,
                "published": post.get("published", ""),
                "source_type": "lemmy",
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            })

        return posts

    except Exception as e:
        logger.warning(f"Lemmy {community}@{instance} failed: {e}")
        return []


def fetch_lemmy_signals() -> list[dict]:
    """Fetch signals from all target Lemmy communities."""
    seen_urls = set()
    all_posts = []

    for community, instance in TARGET_COMMUNITIES:
        posts = fetch_community_posts(community, instance)
        for p in posts:
            url = p.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_posts.append(p)

    logger.info(f"Lemmy: {len(all_posts)} relevant posts fetched")
    return all_posts
