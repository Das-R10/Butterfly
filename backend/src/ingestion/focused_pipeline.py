"""
src/ingestion/focused_pipeline.py

User-focused pipeline. Takes one input (text or URL), finds only
articles related to that specific story, synthesizes them into
one enriched signal, runs personality engine, returns verdict.

This is the primary entry point for user-submitted signals.
The broad auto-ingest (ingest.py) is for background monitoring only.

Usage:
    from src.ingestion.focused_pipeline import run_focused_signal
    result = run_focused_signal("Britain condemns Iran missile strikes on Diego Garcia")
"""

import json
import re
import requests
from datetime import datetime
from loguru import logger

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"
GDELT_API   = "https://api.gdeltproject.org/api/v2/doc/doc"
HN_SEARCH   = "https://hn.algolia.com/api/v1/search"


# ── Step 1: Extract focused keywords from user input ─────────────────────────

KEYWORD_PROMPT = """Extract search keywords from this news signal for finding related articles.
Return ONLY valid JSON, no explanation:
{{
  "queries": ["query1", "query2", "query3"],
  "entities": ["entity1", "entity2"],
  "topic": "one sentence describing what this is about"
}}

Rules:
- queries: 3 short search phrases (2-4 words each) that would find related news articles
- entities: key people, countries, organisations, locations mentioned
- topic: plain English description of the story

Signal: {signal}

JSON:"""


def _ollama(prompt: str, max_tokens: int = 300) -> str | None:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": max_tokens},
            },
            timeout=30,
        )
        return r.json().get("response", "").strip() if r.status_code == 200 else None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _parse_json(raw: str) -> dict | list | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def extract_keywords(user_input: str) -> dict:
    """
    Ask Ollama to extract focused search queries from user input.
    Returns dict with queries, entities, topic.
    """
    raw = _ollama(KEYWORD_PROMPT.format(signal=user_input[:600]))
    parsed = _parse_json(raw) if raw else None

    if parsed and "queries" in parsed:
        logger.info(f"Queries: {parsed['queries']}")
        logger.info(f"Entities: {parsed.get('entities', [])}")
        logger.info(f"Topic: {parsed.get('topic', '')}")
        return parsed

    # Fallback: use the raw input as a single query
    logger.warning("Keyword extraction failed — using raw input as query")
    words = user_input.split()[:6]
    return {
        "queries": [" ".join(words)],
        "entities": [],
        "topic": user_input[:100],
    }


# ── Step 2: Fetch related articles using those specific queries ───────────────

def fetch_related_gdelt(queries: list[str], hours_back: int = 24) -> list[dict]:
    """Fetch articles from GDELT using the focused queries only."""
    seen_urls = set()
    articles = []

    for query in queries:
        try:
            # Try up to 2 times with increasing timeout
            r = None
            for attempt, tout in enumerate([15, 25]):
                try:
                    r = requests.get(
                        GDELT_API,
                        params={
                            "query": query,
                            "mode": "artlist",
                            "maxrecords": 10,
                            "timespan": f"{hours_back}h",
                            "sort": "datedesc",
                            "format": "json",
                        },
                        timeout=tout,
                    )
                    break
                except requests.exceptions.Timeout:
                    if attempt == 0:
                        logger.warning(f"GDELT timeout on '{query}' — retrying...")
                    else:
                        raise
            if r is None:
                continue
            if r.status_code != 200:
                continue
            for a in r.json().get("articles", []):
                url = a.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append({
                        "title":  a.get("title", ""),
                        "url":    url,
                        "source": a.get("domain", ""),
                        "query":  query,
                    })
        except Exception as e:
            logger.warning(f"GDELT query '{query}' failed: {e}")

    logger.info(f"GDELT: {len(articles)} related articles")
    return articles


def fetch_related_hn(queries: list[str]) -> list[dict]:
    """Fetch HN stories using the focused queries only."""
    seen_ids = set()
    stories = []

    for query in queries:
        try:
            r = requests.get(
                HN_SEARCH,
                params={"query": query, "tags": "story", "hitsPerPage": 8},
                timeout=10,
            )
            if r.status_code != 200:
                continue
            for h in r.json().get("hits", []):
                sid = h.get("objectID", "")
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    stories.append({
                        "title":  h.get("title", ""),
                        "url":    h.get("url", ""),
                        "source": "news.ycombinator.com",
                        "score":  h.get("points", 0),
                        "query":  query,
                    })
        except Exception as e:
            logger.warning(f"HN query '{query}' failed: {e}")

    logger.info(f"HN: {len(stories)} related stories")
    return stories


def fetch_related_lemmy(queries: list[str]) -> list[dict]:
    """Fetch Lemmy posts as fallback when GDELT is unavailable."""
    from src.ingestion.lemmy_scraper import fetch_community_posts

    COMMUNITIES = [
        ("worldnews", "lemmy.ml"),
        ("geopolitics", "lemmy.world"),
        ("news", "lemmy.world"),
    ]

    seen_urls = set()
    posts = []
    first_query = queries[0] if queries else ""
    keywords = first_query.lower().split() if first_query else []

    for community, instance in COMMUNITIES:
        try:
            raw = fetch_community_posts(community, instance, limit=20)
            for p in raw:
                url = p.get("url", "")
                title = p.get("title", "").lower()
                # Only include if title matches at least one keyword
                if url and url not in seen_urls:
                    if any(kw in title for kw in keywords if len(kw) > 3):
                        seen_urls.add(url)
                        posts.append({
                            "title":  p.get("title", ""),
                            "url":    url,
                            "source": f"lemmy/{community}",
                            "query":  first_query,
                        })
        except Exception as e:
            logger.warning(f"Lemmy fallback failed for {community}: {e}")

    logger.info(f"Lemmy fallback: {len(posts)} related posts")
    return posts


# ── Step 3: Synthesize user input + related articles into one signal ──────────

SYNTHESIS_PROMPT = """You are a news analyst. Combine the user's input and these related articles
into one comprehensive signal paragraph of 4-6 sentences.

Include: what happened, who is involved, locations, key numbers/quotes, current status.
Write as a factual intelligence briefing. No opinions. No markdown. Plain text only.

User's input:
{user_input}

Related articles found:
{articles}

Synthesized briefing:"""


def synthesize_signal(user_input: str, related: list[dict]) -> str:
    """
    Combine user input with related articles into one rich signal.
    If synthesis fails, returns the user input with article titles appended.
    """
    if not related:
        return user_input

    article_lines = "\n".join(
        f"• {a['title']} [{a.get('source', '')}]"
        for a in related[:10]
        if a.get("title")
    )

    prompt = SYNTHESIS_PROMPT.format(
        user_input=user_input,
        articles=article_lines,
    )

    synthesized = _ollama(prompt, max_tokens=400)

    if synthesized and len(synthesized) > 80:
        logger.info(f"Synthesized signal: {len(synthesized)} chars")
        return synthesized.strip()

    # Fallback: append article titles to user input
    logger.warning("Synthesis failed — using fallback concatenation")
    context = "\n[Related: " + " | ".join(
        a["title"][:80] for a in related[:5] if a.get("title")
    ) + "]"
    return user_input + context


# ── Main entry point ──────────────────────────────────────────────────────────

def build_focused_signal(user_input: str) -> dict:
    """
    Full focused pipeline for a single user input.

    1. Extract focused keywords
    2. Search for related articles
    3. Synthesize into one rich signal

    Returns a signal dict ready for run_all_models().
    """
    import hashlib

    logger.info(f"Focused pipeline: {user_input[:80]}")

    # Step 1: Extract keywords
    print("  → Extracting keywords from your input...")
    extraction = extract_keywords(user_input)
    queries  = extraction.get("queries", [user_input[:50]])
    entities = extraction.get("entities", [])
    topic    = extraction.get("topic", user_input[:100])

    print(f"  → Search queries: {queries}")
    print(f"  → Entities: {entities}")

    # Step 2: Fetch related articles
    print("  → Searching for related articles...")
    gdelt_articles = fetch_related_gdelt(queries, hours_back=48)
    hn_stories     = fetch_related_hn(queries)
    related = gdelt_articles[:8] + hn_stories[:5]

    # If GDELT and HN both returned nothing, fall back to Lemmy
    if not related:
        logger.warning("GDELT and HN returned nothing — trying Lemmy fallback")
        print("  → GDELT unavailable — trying Lemmy as fallback source...")
        lemmy_posts = fetch_related_lemmy(queries)
        related = lemmy_posts[:8]

    print(f"  → Found {len(related)} related articles")
    if related:
        print("  → Related articles:")
        for a in related[:8]:
            print(f"       • {a['title'][:80]} [{a.get('source','')}]")

    # Step 3: Synthesize
    print("  → Synthesizing signal...")
    signal_text = synthesize_signal(user_input, related)
    print(f"\n  SYNTHESIZED SIGNAL:\n  {signal_text}\n")

    signal_id = hashlib.md5(signal_text.encode()).hexdigest()[:12]

    # Step 4: Generate dynamic buckets from the synthesized signal (L0)
    # These buckets are generated ONCE here and passed to every downstream layer.
    # All layers score against the same bucket set for this signal.
    print("  → Generating dynamic narrative buckets...")
    try:
        from src.ensemble.dynamic_bucket_scorer import generate_dynamic_buckets
        dynamic_buckets = generate_dynamic_buckets(signal_text)
        bucket_labels = [b["label"] for b in dynamic_buckets]
        print(f"  → Buckets: {bucket_labels}")
    except Exception as e:
        logger.warning(f"Dynamic bucket generation failed: {e} — using empty list")
        dynamic_buckets = []

    return {
        "signal_id":          signal_id,
        "signal_text":        signal_text,
        "user_input":         user_input,
        "topic":              topic,
        "queries_used":       queries,
        "entities":           entities,
        "related_articles":   related,
        "total_related":      len(related),
        "dynamic_buckets":    dynamic_buckets,   # L0 buckets — passed to all layers
        "source_type":        "user_focused",
        "ingested_at":        datetime.utcnow().isoformat() + "Z",
    }


# ── Full orchestration — L0 through L5 ───────────────────────────────────────

def run_focused_signal(
    user_input: str,
    mode: str = "support",
    run_debate: bool = True,
    lock_prediction: bool = False,
) -> dict:
    """
    Full pipeline entry point. Runs L0 → L5 sequentially.

    L0  build_focused_signal()     keyword extraction + article fetch + synthesis + bucket generation
    L2A run_mbti_layer()           16 MBTI types react, produce population report
    L2B run_all_models()           16 archetypes run with MBTI context + domain tag, Ambiguity Analyst
    L4  score_and_map_dynamic()    multi-tag semantic scoring against L0 buckets
    L5  run_debate_layer()         three critics, volatility score, high-octane flag

    Args:
        user_input:       Raw user signal text
        mode:             "support" (live data enriched) or "pure" (isolation only)
        run_debate:       Whether to run L5 Debate Layer (adds ~3 LLM calls)
        lock_prediction:  Whether to lock a Prediction Registry snapshot

    Returns:
        Full result dict with all layer outputs attached.

    Usage:
        result = run_focused_signal("Iran strikes Diego Garcia base")
        result = run_focused_signal("...", mode="pure", run_debate=False)
    """
    from src.personalities.mbti_layer import run_mbti_layer
    from src.personalities.personality_runner import run_all_models
    from src.ensemble.dynamic_bucket_scorer import score_and_map_dynamic

    logger.info(f"run_focused_signal | mode={mode} | input: {user_input[:80]}")
    print(f"\n{'═'*60}")
    print(f"  BUTTERFLY ALGORITHM — FULL PIPELINE")
    print(f"  Mode: {mode.upper()}")
    print(f"{'═'*60}\n")

    # ── L0: Signal ingestion + bucket generation ──────────────────────────────
    print("  [L0] Signal ingestion...")
    signal_dict     = build_focused_signal(user_input)
    signal_text     = signal_dict["signal_text"]
    signal_id       = signal_dict["signal_id"]
    dynamic_buckets = signal_dict.get("dynamic_buckets", [])

    # ── L2A: MBTI population reaction ────────────────────────────────────────
    # Skipped in pure mode — no world-state brief available
    mbti_report = None
    if mode == "support":
        print("\n  [L2A] MBTI population reaction layer...")
        try:
            mbti_report = run_mbti_layer(
                signal=signal_text,
                world_state_brief="",      # L1 brief would go here when gdelt_grounding is built
                dynamic_buckets=dynamic_buckets,
            )
        except Exception as e:
            logger.error(f"MBTI layer failed: {e}")
            print(f"  ✗ MBTI layer failed: {e} — continuing without population context")

    # ── L2B: Analyst archetypes + Ambiguity Analyst ──────────────────────────
    print("\n  [L2B] Analyst archetype layer (all 16)...")
    try:
        result = run_all_models(
            signal=signal_text,
            signal_id=signal_id,
            mbti_report=mbti_report,
            dynamic_buckets=dynamic_buckets,
            domain_context=None,       # auto-classified inside run_all_models
            run_ambiguity=True,
        )
    except Exception as e:
        logger.error(f"Archetype layer failed: {e}")
        raise

    # Attach signal metadata and layer outputs to result
    result["signal_dict"]   = signal_dict
    result["mode"]          = mode
    result["mbti_report"]   = mbti_report
    result["dynamic_buckets"] = dynamic_buckets

    # ── L4: Consensus engine — dynamic multi-tag scoring ─────────────────────
    print("\n  [L4] Consensus engine — dynamic scoring...")
    try:
        result = score_and_map_dynamic(result)
    except Exception as e:
        logger.error(f"Dynamic scoring failed: {e}")
        print(f"  ✗ Scoring failed: {e}")

    # ── Convergence gap (L4 post-step) ────────────────────────────────────────
    if mbti_report and result.get("clusters"):
        try:
            from src.personalities.mbti_layer import compute_convergence_gap
            convergence_gap = compute_convergence_gap(
                population_buckets=mbti_report.get("population_buckets", {}),
                analyst_probability_map=result["clusters"],
            )
            result["convergence_gap"] = convergence_gap
            status = convergence_gap.get("gap_status", "unknown")
            print(f"\n  Convergence gap: {status}")
        except Exception as e:
            logger.warning(f"Convergence gap calculation failed: {e}")

    # ── L5: Debate layer ─────────────────────────────────────────────────────
    if run_debate and result.get("clusters"):
        print("\n  [L5] Debate layer...")
        try:
            from src.ensemble.debate_layer import run_debate_layer
            debate_result = run_debate_layer(
                signal=signal_text,
                probability_map=result["clusters"],
                outputs=result.get("outputs", []),
                mbti_report=mbti_report,
                behavioral_brief="",
                gdelt_data=None,
                narrative_drift_score=0.0,
            )
            result["debate_result"] = debate_result
            # Use post-critique map as the final probability map
            if debate_result.get("adjusted_probability_map"):
                result["clusters"] = debate_result["adjusted_probability_map"]
                result["dominant_narrative"] = result["clusters"][0]
        except Exception as e:
            logger.error(f"Debate layer failed: {e}")
            print(f"  ✗ Debate layer failed: {e}")

    # ── Lock prediction snapshot ──────────────────────────────────────────────
    if lock_prediction:
        try:
            from src.ensemble.prediction_registry import lock_prediction as lock
            lock(result, mode=mode)
            print(f"\n  Prediction locked to registry (mode={mode})")
        except Exception as e:
            logger.warning(f"Prediction locking failed: {e}")

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_result_summary(result)

    return result


def _print_result_summary(result: dict):
    """Print a clean summary of the full pipeline result."""
    clusters  = result.get("clusters", [])
    ambiguity = result.get("ambiguity_report", {})
    debate    = result.get("debate_result", {})
    conv_gap  = result.get("convergence_gap", {})

    print(f"\n{'═'*60}")
    print(f"  RESULT SUMMARY")
    print(f"{'═'*60}")

    if clusters:
        print(f"\n  Narrative probability map:")
        for c in clusters[:5]:
            bar_len = int(c.get("probability", 0) * 25)
            bar     = "█" * bar_len + "░" * (25 - bar_len)
            ho_flag = " 🔥" if c.get("bucket") == clusters[0].get("bucket") and debate.get("high_octane_narrative") else ""
            print(
                f"    {bar} {c.get('probability_pct', '?'):>6}  "
                f"{c.get('label', c.get('bucket', '?'))[:35]}{ho_flag}"
            )

    if conv_gap:
        print(f"\n  Convergence gap: {conv_gap.get('gap_status', '?')}")
        print(f"  {conv_gap.get('meaning', '')}")

    if debate:
        print(f"\n  Volatility score: {debate.get('volatility_score', '?'):.3f}")
        print(f"  High-Octane: {'YES 🔥' if debate.get('high_octane_narrative') else 'no'}")

    if ambiguity and ambiguity.get("has_significant_findings"):
        print(f"\n  Ambiguity Analyst flags:")
        if ambiguity.get("domain_challenge", {}).get("challenge"):
            rec = ambiguity["domain_challenge"].get("recommended_domain", "?")
            print(f"    ⚠ Domain reclassification recommended → {rec}")
        for insight in ambiguity.get("cross_domain_insights", [])[:2]:
            print(f"    ↗ Cross-domain: [{insight.get('archetype')}] → {insight.get('secondary_domain')} ({insight.get('significance', 0):.0%})")

    print(f"\n  Total outputs: {result.get('total_outputs', 0)}")
    print(f"  Mode: {result.get('mode', 'unknown').upper()}")
    print(f"{'═'*60}\n")