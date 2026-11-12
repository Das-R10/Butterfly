"""
src/ingestion/bluesky_scraper.py

Pulls signals from Bluesky via the public AT Protocol API.
Fix v1.1: public search endpoint now returns 403 without auth.
Solution: use the public AppView search which requires a guest token,
OR fall back to fetching from known geopolitical feeds/starter packs
via the public timeline endpoint which has no auth requirement.
"""

import requests
from datetime import datetime
from loguru import logger

BLUESKY_API = "https://public.api.bsky.app/xrpc"
BSKY_AUTH_URL = "https://bsky.social/xrpc/com.atproto.server.createSession"

# Known high-signal Bluesky accounts for geopolitics/news
# Public profiles — timeline fetch requires no auth
SIGNAL_FEEDS = [
    "did:plc:z72i7hdynmk6r22z27h6tvur",   # bsky.app news feed
]

GEOPOLITICAL_HANDLES = [
    "foreignpolicy.com",
    "theatlantic.com",
    "reuters.com",
    "apnews.com",
    "bbcnews.bsky.social",
]

SIGNAL_KEYWORDS = [
    "war", "conflict", "sanctions", "tariff", "military", "invasion",
    "nuclear", "missile", "inflation", "recession", "crisis", "collapse",
    "coup", "protest", "election", "geopolitical", "trade", "oil",
    "china", "russia", "iran", "ukraine", "taiwan", "nato",
    "breaking", "market", "supply chain", "energy", "attack",
    "drone", "troops", "border", "blockade", "embargo",
]

MIN_LIKES = 3
MIN_REPLIES = 1


def _is_relevant(text: str) -> bool:
    return any(kw in text.lower() for kw in SIGNAL_KEYWORDS)


def _get_guest_token() -> str | None:
    """
    Attempt to get a guest/anonymous session token.
    Bluesky allows unauthenticated reads of public content
    via the AppView but search now requires a token.
    Returns None if unavailable — caller falls back gracefully.
    """
    try:
        # Use the public AppView directly — no credentials
        resp = requests.get(
            f"{BLUESKY_API}/app.bsky.feed.getFeedGenerators",
            timeout=5
        )
        # If we get a valid response, the AppView is reachable
        if resp.status_code == 200:
            return "public"
        return None
    except Exception:
        return None


def fetch_author_feed(handle: str, limit: int = 20) -> list[dict]:
    """
    Fetch recent posts from a specific Bluesky author.
    Author feeds are publicly readable without authentication.
    """
    try:
        params = {
            "actor": handle,
            "limit": min(limit, 20),
        }
        r = requests.get(
            f"{BLUESKY_API}/app.bsky.feed.getAuthorFeed",
            params=params,
            timeout=10,
            headers={"Accept": "application/json"}
        )
        if r.status_code != 200:
            return []

        posts = []
        for item in r.json().get("feed", []):
            post = item.get("post", {})
            record = post.get("record", {})
            text = record.get("text", "")

            like_count = post.get("likeCount", 0)
            reply_count = post.get("replyCount", 0)

            if not _is_relevant(text):
                continue
            if len(text) < 40:
                continue
            if like_count < MIN_LIKES and reply_count < MIN_REPLIES:
                continue

            author = post.get("author", {})
            uri = post.get("uri", "")
            post_id = uri.split("/")[-1] if uri else ""

            posts.append({
                "title": text[:120],
                "description": text,
                "url": f"https://bsky.app/profile/{author.get('handle','')}/post/{post_id}",
                "source": f"bsky/{author.get('handle', 'unknown')}",
                "source_type": "bluesky",
                "like_count": like_count,
                "reply_count": reply_count,
                "author_handle": author.get("handle", ""),
                "published": record.get("createdAt", datetime.utcnow().isoformat() + "Z"),
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            })

        return posts

    except Exception as e:
        logger.warning(f"Bluesky author feed failed for {handle}: {e}")
        return []


def search_bluesky_authenticated(query: str, limit: int = 10) -> list[dict]:
    """
    Search Bluesky using the search endpoint.
    Falls back gracefully if 403 — search requires auth now.
    """
    try:
        params = {
            "q": query,
            "limit": min(limit, 25),
            "sort": "top",
        }
        r = requests.get(
            f"{BLUESKY_API}/app.bsky.feed.searchPosts",
            params=params,
            timeout=10,
            headers={"Accept": "application/json"}
        )
        if r.status_code == 403:
            # Search requires auth — not available without credentials
            return []
        if r.status_code != 200:
            return []

        posts = []
        for item in r.json().get("posts", []):
            record = item.get("record", {})
            text = record.get("text", "")
            author = item.get("author", {})
            like_count = item.get("likeCount", 0)
            reply_count = item.get("replyCount", 0)

            if not _is_relevant(text) or len(text) < 40:
                continue
            if like_count < MIN_LIKES and reply_count < MIN_REPLIES:
                continue

            uri = item.get("uri", "")
            post_id = uri.split("/")[-1] if uri else ""

            posts.append({
                "title": text[:120],
                "description": text,
                "url": f"https://bsky.app/profile/{author.get('handle','')}/post/{post_id}",
                "source": f"bsky/{author.get('handle', 'unknown')}",
                "source_type": "bluesky",
                "like_count": like_count,
                "reply_count": reply_count,
                "author_handle": author.get("handle", ""),
                "keyword_query": query,
                "published": record.get("createdAt", datetime.utcnow().isoformat() + "Z"),
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            })

        return posts

    except Exception as e:
        logger.warning(f"Bluesky search failed for '{query}': {e}")
        return []


def fetch_bluesky_signals(max_results: int = 40) -> list[dict]:
    """
    Fetch Bluesky signals — tries search first, falls back to
    author feeds if search returns 403.
    """
    seen_urls = set()
    all_posts = []

    # Try search on a single query to check if it works
    test = search_bluesky_authenticated("geopolitical conflict", limit=3)

    if test:
        # Search is working — use it across all queries
        queries = [
            "war conflict military",
            "economic crisis sanctions tariff",
            "nuclear iran ukraine nato",
            "market crash supply chain oil",
            "coup protest breaking news",
        ]
        for query in queries:
            posts = search_bluesky_authenticated(query, limit=10)
            for p in posts:
                url = p.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_posts.append(p)
            if len(all_posts) >= max_results:
                break
    else:
        # Search blocked — fall back to author feeds
        logger.info("Bluesky search blocked (403) — falling back to author feeds")
        for handle in GEOPOLITICAL_HANDLES:
            posts = fetch_author_feed(handle, limit=15)
            for p in posts:
                url = p.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_posts.append(p)
            if len(all_posts) >= max_results:
                break

    logger.info(f"Bluesky: {len(all_posts)} relevant posts fetched")
    return all_posts