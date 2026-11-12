"""
src/ingestion/hackernews_scraper.py

Pulls signals from Hacker News via the official Firebase API.
Zero API key required. Completely free and open.

Fix v1.1: tightened keyword filter — requires at least 2 keyword
hits OR one strong geopolitical keyword to reduce false positives
like "Drugwars calculator" and "gambling harm" passing the filter.
"""

import re
import requests
from datetime import datetime
from loguru import logger

HN_API = "https://hacker-news.firebaseio.com/v0"

HN_ENDPOINTS = {
    "top":  f"{HN_API}/topstories.json",
    "new":  f"{HN_API}/newstories.json",
    "best": f"{HN_API}/beststories.json",
}

# Strong geopolitical keywords — one hit is enough
STRONG_KEYWORDS = [
    "war", "invasion", "nato", "nuclear", "missile", "sanctions",
    "tariff", "geopolitical", "ukraine", "russia", "china", "iran",
    "taiwan", "coup", "troops", "blockade", "embargo", "escalation",
    "military strike", "trade war", "oil supply", "supply chain crisis",
    "rare earth", "semiconductor ban",
]

# Weak keywords — need 2+ hits to count
WEAK_KEYWORDS = [
    "conflict", "inflation", "recession", "crisis", "collapse",
    "protest", "election", "oil", "energy", "market crash",
    "border", "drone", "attack", "breaking",
]

MIN_SCORE = 80       # raised from 50 — higher bar for HN
MIN_COMMENTS = 15    # raised from 10


def _is_relevant(title: str) -> bool:
    """
    Require either:
    - 1 strong geopolitical keyword, OR
    - 2+ weak keywords
    This prevents false positives like 'Drugwars' or 'gambling bombs'
    """
    title_lower = title.lower()
    strong_hits = sum(1 for kw in STRONG_KEYWORDS if kw in title_lower)
    if strong_hits >= 1:
        return True
    weak_hits = sum(1 for kw in WEAK_KEYWORDS if kw in title_lower)
    return weak_hits >= 2


def fetch_story(story_id: int) -> dict | None:
    try:
        r = requests.get(f"{HN_API}/item/{story_id}.json", timeout=8)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_top_comments(story: dict, max_comments: int = 3) -> list[str]:
    comments = []
    kids = story.get("kids", [])[:max_comments]
    for kid_id in kids:
        try:
            r = requests.get(f"{HN_API}/item/{kid_id}.json", timeout=6)
            if r.status_code == 200:
                item = r.json()
                text = item.get("text", "")
                if text and len(text) > 30:
                    text = re.sub(r"<[^>]+>", " ", text).strip()
                    comments.append(text[:300])
        except Exception:
            continue
    return comments


def fetch_hn_signals(max_stories: int = 30) -> list[dict]:
    """
    Fetch top HN stories, filter by relevance and score.
    Tightened filter prevents false positives.
    """
    seen_ids = set()
    all_stories = []

    for feed_name, endpoint in HN_ENDPOINTS.items():
        try:
            r = requests.get(endpoint, timeout=10)
            if r.status_code != 200:
                continue
            story_ids = r.json()[:80]  # check top 80 per feed

            for sid in story_ids:
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)

                story = fetch_story(sid)
                if not story:
                    continue

                title = story.get("title", "")
                score = story.get("score", 0)
                num_comments = story.get("descendants", 0)

                if score < MIN_SCORE:
                    continue
                if num_comments < MIN_COMMENTS:
                    continue
                if not _is_relevant(title):
                    continue

                top_comments = fetch_top_comments(story, max_comments=2)

                all_stories.append({
                    "title": title,
                    "description": story.get("text", ""),
                    "url": story.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                    "source": "news.ycombinator.com",
                    "source_type": "hackernews",
                    "hn_score": score,
                    "num_comments": num_comments,
                    "feed": feed_name,
                    "top_comments": top_comments,
                    "published": datetime.utcfromtimestamp(
                        story.get("time", 0)
                    ).isoformat() + "Z",
                    "ingested_at": datetime.utcnow().isoformat() + "Z",
                })

                if len(all_stories) >= max_stories:
                    break

        except Exception as e:
            logger.warning(f"HN {feed_name} feed failed: {e}")
            continue

    logger.info(f"Hacker News: {len(all_stories)} relevant stories fetched")
    return all_stories