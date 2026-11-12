"""
src/ensemble/semantic_scorer.py

Phase 3 consensus engine — open signal discovery mapped to fixed buckets.

Architecture:
  1. Embed all 32-64 responses as vectors
  2. Agglomerative clustering — groups form by actual semantic similarity
     (open discovery, no predefined labels)
  3. Each cluster's centroid compared to 7 fixed bucket reference embeddings
     via cosine similarity
  4. Cluster assigned to the closest fixed bucket
  5. Multiple clusters can map to the same bucket — they stay as sub-clusters
     inside it with their own LLM-generated labels

Result:
  - Outer probability map uses fixed bucket labels (comparable, backtestable)
  - Inner sub-clusters show what actually emerged (granular, signal-specific)
  - Brier Score works on outer labels
  - Narrative divergence visible in sub-clusters

Falls back to keyword scoring if sentence-transformers not installed.
"""

from __future__ import annotations

import json
import re
import requests
import numpy as np
from collections import defaultdict
from loguru import logger

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"

# ── Fixed bucket definitions (reference texts for cosine similarity) ──────────
# These never change — they define the output space for backtesting.
# The embeddings of these texts are the anchors that open clusters map onto.

FIXED_BUCKETS = {
    "conflict_escalation": {
        "label": "Conflict / escalation likely",
        "color": "red",
        "reference": (
            "This situation is escalating toward military conflict, war, or violent confrontation. "
            "Active threats, weapons, troop movements, missile strikes, attacks, or aggressive "
            "military posturing. Historical parallels point to wars that spiralled out of control. "
            "The situation is dangerous, unstable, moving toward open conflict. Escalation is "
            "the most likely outcome."
        ),
    },
    "economic_chaos": {
        "label": "Economic disruption / market panic",
        "color": "coral",
        "reference": (
            "This event causes significant economic disruption, market panic, financial instability, "
            "or supply chain collapse. Stock markets falling, oil prices surging, trade flows "
            "disrupted, inflation rising, recession imminent. Primary impact is economic — GDP, "
            "employment, commodity prices, currency values, bond yields, corporate earnings."
        ),
    },
    "diplomatic_resolution": {
        "label": "Diplomatic / peaceful resolution",
        "color": "green",
        "reference": (
            "This situation can be resolved through diplomacy, dialogue, negotiation, peaceful means. "
            "Diplomatic pathways exist and are underutilized. A ceasefire, agreement, treaty, or "
            "negotiated settlement is achievable. De-escalation is possible. Multilateral institutions "
            "and diplomatic channels offer viable solutions. Peace is the realistic outcome."
        ),
    },
    "conspiracy_elite_control": {
        "label": "Conspiracy / elite manipulation narrative",
        "color": "amber",
        "reference": (
            "This event is engineered, staged, or exploited by powerful hidden actors — the WEF, "
            "deep state, central banks, intelligence agencies, globalists, or elite cabals — to "
            "advance an agenda of centralized control or population manipulation. False flag, "
            "distraction, or controlled operation. The public narrative is deliberately misleading."
        ),
    },
    "routine_overblown": {
        "label": "Routine event / media overblown",
        "color": "gray",
        "reference": (
            "This situation is being exaggerated, sensationalized, blown out of proportion by media "
            "and social amplification. The base rate for this type of event is normal. Insufficient "
            "evidence for dramatic interpretations. Most parsimonious explanation is mundane. "
            "Pattern-matching and confirmation bias driving panic. Routine geopolitical noise."
        ),
    },
    "historical_precedent": {
        "label": "Historical precedent — likely resolves",
        "color": "blue",
        "reference": (
            "Historical precedent shows situations like this typically resolve within weeks or months. "
            "Structurally similar to past crises — Suez Crisis, Cuban Missile Crisis, 1973 oil shock — "
            "that were contained without escalating to catastrophe. History provides specific examples "
            "of how this type of crisis was resolved. The resolution timeline and mechanism are "
            "predictable. Most crises resolve; this one will too."
        ),
    },
    "tech_innovation_opportunity": {
        "label": "Tech / innovation opportunity",
        "color": "teal",
        "reference": (
            "This crisis is a forcing function for technological innovation, automation, and human "
            "ingenuity. Supply chain disruption accelerates reshoring and robotics. Energy crises "
            "drive renewable scale-up. Wars produce important technological breakthroughs. Long-term "
            "opportunity outweighs short-term friction. Technology will solve this faster than "
            "doom scenarios predict. Catalyst for positive transformation."
        ),
    },
}

# ── Sub-cluster LLM labelling ─────────────────────────────────────────────────

SUBLABEL_PROMPT = """You are a narrative analyst. Read these responses from different personality types.
They all share a common narrative frame. Give this group a short specific label.

Return ONLY valid JSON — no explanation, no markdown:
{{
  "label": "specific label (3-5 words)",
  "description": "one sentence describing this specific angle",
  "dominant_emotion": "one word",
  "virality": "low/medium/high/very_high"
}}

Responses:
{responses}

JSON:"""


def _call_ollama(prompt: str) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150},
            },
            timeout=25,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama sub-label call failed: {e}")
        return None


def _parse_json(raw: str) -> dict | None:
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


def _label_subcluster(outputs: list[dict], fallback_label: str) -> dict:
    """Ask LLM to name a sub-cluster from its content."""
    samples = outputs[:3]
    response_texts = "\n---\n".join(
        f"[{o['personality_name']}]: {o.get('response','')[:250]}"
        for o in samples
    )
    raw    = _call_ollama(SUBLABEL_PROMPT.format(responses=response_texts))
    parsed = _parse_json(raw) if raw else None

    if parsed and "label" in parsed:
        return {
            "sub_label":       parsed.get("label", fallback_label),
            "description":     parsed.get("description", ""),
            "dominant_emotion":parsed.get("dominant_emotion", ""),
            "virality":        parsed.get("virality", "medium"),
        }
    return {
        "sub_label":       fallback_label,
        "description":     "",
        "dominant_emotion":"",
        "virality":        "medium",
    }


# ── Keyword extraction ────────────────────────────────────────────────────────

def _extract_keywords(outputs: list[dict], top_n: int = 6) -> list[str]:
    from collections import Counter
    STOP = {
        "the","a","an","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","could","should","may","might","this",
        "that","these","those","i","we","you","they","he","she","it","as","at",
        "by","for","in","of","on","to","and","or","but","not","with","from",
        "into","about","its","their","our","your","which","who","what","when",
        "where","how","all","any","more","most","other","some","than","then",
        "there","very","just","also","however","while","although","because",
    }
    words = []
    for o in outputs:
        words.extend([
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', o.get("response",""))
            if w.lower() not in STOP
        ])
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_n)]


# ── Model loading ─────────────────────────────────────────────────────────────

_embed_model = None
_bucket_embeddings = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded")
    return _embed_model


def _get_bucket_embeddings() -> dict[str, np.ndarray]:
    """Embed fixed bucket reference texts once and cache."""
    global _bucket_embeddings
    if _bucket_embeddings is None:
        model = _get_embed_model()
        keys  = list(FIXED_BUCKETS.keys())
        texts = [FIXED_BUCKETS[k]["reference"] for k in keys]
        vecs  = model.encode(texts, normalize_embeddings=True)
        _bucket_embeddings = {k: vecs[i] for i, k in enumerate(keys)}
        logger.info(f"Bucket reference embeddings cached ({len(keys)} buckets)")
    return _bucket_embeddings


# ── Core functions ────────────────────────────────────────────────────────────

def _embed_responses(outputs: list[dict]) -> np.ndarray:
    model = _get_embed_model()
    texts = [o.get("response", "") for o in outputs]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _cluster_agglomerative(embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings)


def _assign_to_bucket(cluster_embeddings: np.ndarray) -> tuple[str, float]:
    """Find the closest fixed bucket for a cluster centroid."""
    centroid  = cluster_embeddings.mean(axis=0)
    centroid  = centroid / (np.linalg.norm(centroid) + 1e-9)
    bucket_embs = _get_bucket_embeddings()
    scores    = {k: float(np.dot(centroid, v)) for k, v in bucket_embs.items()}
    best      = max(scores, key=scores.get)
    return best, round(scores[best], 4)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_semantic_probability_map(
    outputs: list[dict],
    distance_threshold: float = 0.45,
) -> list[dict]:
    """
    Open signal discovery mapped to fixed buckets.

    1. Embed all responses
    2. Agglomerative cluster (open — finds natural groups)
    3. Map each cluster to the closest fixed bucket via cosine similarity
    4. Build probability map with fixed outer labels + sub-cluster detail inside
    """
    total = len(outputs)
    if total == 0:
        return []

    logger.info(f"Semantic scoring: embedding {total} responses...")
    embeddings = _embed_responses(outputs)

    logger.info(f"Agglomerative clustering (threshold={distance_threshold})...")
    cluster_labels = _cluster_agglomerative(embeddings, distance_threshold)
    n_clusters     = len(set(cluster_labels))
    logger.info(f"Found {n_clusters} open clusters → mapping to fixed buckets...")

    # Group outputs + embeddings by cluster id
    cluster_groups: dict[int, list] = defaultdict(list)
    for i, output in enumerate(outputs):
        cluster_groups[int(cluster_labels[i])].append((output, embeddings[i]))

    # Map each open cluster to a fixed bucket
    # Multiple open clusters can map to the same fixed bucket
    bucket_to_subclusters: dict[str, list] = defaultdict(list)

    for cid, members in sorted(cluster_groups.items(), key=lambda x: -len(x[1])):
        cluster_outs = [m[0] for m in members]
        cluster_embs = np.array([m[1] for m in members])

        # Find closest fixed bucket
        bucket_key, similarity = _assign_to_bucket(cluster_embs)

        # Get LLM sub-label for this specific cluster
        fallback = f"{FIXED_BUCKETS[bucket_key]['label']} — cluster {cid}"
        sub_info = _label_subcluster(cluster_outs, fallback)

        subcluster = {
            "cluster_id":       cid,
            "bucket":           bucket_key,
            "sub_label":        sub_info["sub_label"],
            "description":      sub_info["description"],
            "dominant_emotion": sub_info["dominant_emotion"],
            "virality":         sub_info["virality"],
            "keywords":         _extract_keywords(cluster_outs),
            "similarity":       similarity,
            "count":            len(members),
            "outputs":          cluster_outs,
            "embeddings":       cluster_embs,
        }

        # Assign bucket back to each output
        for o in cluster_outs:
            o["bucket"]      = bucket_key
            o["sub_label"]   = sub_info["sub_label"]
            o["cluster_id"]  = cid

        bucket_to_subclusters[bucket_key].append(subcluster)

        logger.info(
            f"  Cluster {cid} ({len(members)} outputs) "
            f"→ '{FIXED_BUCKETS[bucket_key]['label']}' "
            f"[sim={similarity:.3f}] "
            f"sub: '{sub_info['sub_label']}'"
        )

    # Build probability map — one entry per fixed bucket that appeared
    probability_map = []
    bucket_counts   = {bk: sum(sc["count"] for sc in scs)
                       for bk, scs in bucket_to_subclusters.items()}

    for bucket_key, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        bucket_info   = FIXED_BUCKETS[bucket_key]
        subclusters   = bucket_to_subclusters[bucket_key]
        all_outs      = [o for sc in subclusters for o in sc["outputs"]]

        personalities_in = [o["personality_name"] for o in all_outs]
        models_in        = list(set(o["model_display"] for o in all_outs))

        personality_counts = defaultdict(int)
        for o in all_outs:
            personality_counts[o["personality"]] += 1
        cross_model = [p for p, c in personality_counts.items() if c > 1]

        # Build sub-cluster summary list (strip heavy data before serialising)
        sub_summary = [
            {
                "cluster_id":       sc["cluster_id"],
                "sub_label":        sc["sub_label"],
                "description":      sc["description"],
                "dominant_emotion": sc["dominant_emotion"],
                "virality":         sc["virality"],
                "keywords":         sc["keywords"],
                "similarity":       sc["similarity"],
                "count":            sc["count"],
                "probability_pct":  f"{round(sc['count']/total*100,1)}%",
            }
            for sc in sorted(subclusters, key=lambda x: -x["count"])
        ]

        # Top keywords across all sub-clusters in this bucket
        all_keywords = _extract_keywords(all_outs, top_n=8)

        probability_map.append({
            "bucket":            bucket_key,
            "label":             bucket_info["label"],
            "color":             bucket_info["color"],
            "count":             count,
            "probability":       round(count / total, 4),
            "probability_pct":   f"{round(count / total * 100, 1)}%",
            "confidence":        "high" if len(cross_model) >= 3
                                 else "medium" if len(cross_model) >= 1
                                 else "low",
            "models_represented":models_in,
            "personalities":     personalities_in,
            "cross_model_agreement": cross_model,
            "cross_model_count": len(cross_model),
            "keywords":          all_keywords,
            "sub_clusters":      sub_summary,
            "n_sub_clusters":    len(subclusters),
            "sample_response":   all_outs[0]["response"][:200] + "...",
        })

    return probability_map


# ── Drop-in replacement ───────────────────────────────────────────────────────

def score_and_map_semantic(result: dict, distance_threshold: float = 0.45) -> dict:
    """
    Drop-in replacement for similarity_scorer.score_and_map().

    Fixed outer buckets for comparability + backtesting.
    Open sub-clusters for narrative granularity.
    """
    outputs = result.get("outputs", [])
    if not outputs:
        logger.warning("No outputs to score")
        return result

    logger.info(
        f"Semantic scoring: {len(outputs)} outputs | "
        f"threshold={distance_threshold}"
    )

    try:
        probability_map = compute_semantic_probability_map(
            outputs,
            distance_threshold=distance_threshold,
        )
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — "
            "falling back to keyword scoring."
        )
        from src.ensemble.similarity_scorer import compute_probability_map
        probability_map = compute_probability_map(outputs)

    result["clusters"]           = probability_map
    result["dominant_narrative"] = probability_map[0] if probability_map else None
    result["scoring_method"]     = "semantic_open_to_fixed"

    if probability_map:
        top = probability_map[0]
        logger.info(f"Dominant narrative: '{top['label']}' at {top['probability_pct']}")
        logger.info(f"Cross-model agreement: {top['cross_model_count']} personalities")
        logger.info(f"Sub-clusters inside dominant: {top['n_sub_clusters']}")
        for sc in top.get("sub_clusters", []):
            logger.info(
                f"  └── {sc['probability_pct']:>6}  "
                f"'{sc['sub_label']}' "
                f"[{sc['dominant_emotion']}]"
            )

    return result