"""
src/ensemble/dynamic_bucket_scorer.py

Dynamic bucket generation + multi-tag semantic scoring.

Two-stage process:
  Stage 1 — Bucket generation
    LLM reads the signal and generates 4-7 context-specific outcome tags.
    Economic news → crash, dip, bullish, no change, regulatory action
    Geopolitical  → escalation, peace talks, conspiracy, economic fallout
    Corporate     → restructuring, growth signal, fraud, market reaction
    Cultural      → public outcry, no reaction, viral moment, cancellation

  Stage 2 — Multi-tag scoring
    Each archetype output is scored against ALL generated buckets.
    Semantically related phrases cluster together:
      attack, threat, strike, offensive → escalation bucket
      fine, penalty, charges, lawsuit   → regulatory/economic bucket
    Outputs scoring above threshold on MULTIPLE buckets get fractional
    weight in each — probability mass is split, not hard-assigned.

The fixed 7 buckets (src/ensemble/semantic_scorer.py) are kept for
Brier Score validation only — you need stable categories for cross-signal
comparison. Dynamic buckets are for per-signal analysis depth.
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

MULTI_TAG_THRESHOLD  = 0.45   # outputs scoring above this on a bucket get assigned to it
SPLIT_WEIGHT_MIN     = 0.20   # minimum weight for a secondary tag assignment

# ── Stage 1: Dynamic bucket generation ───────────────────────────────────────

BUCKET_GEN_PROMPT = """You are a narrative analyst. Given the following news signal, generate 4-7 possible narrative outcome tags — the distinct ways this story could be interpreted or develop.

Rules:
- Tags should be SHORT (2-4 words each)
- Tags should be MUTUALLY DISTINCT (no overlaps)
- Tags should be DOMAIN-APPROPRIATE (economic news → market outcomes, not military ones)
- Cover the realistic range from best to worst case, plus conspiracy/fringe if plausible
- Think: what are the distinct narrative buckets people will argue about for THIS signal?

Examples by domain:
  Economic news:    ["market crash", "mild correction", "bullish signal", "regulatory action", "no real impact"]
  Geopolitical:     ["military escalation", "diplomatic resolution", "economic sanctions", "conspiracy narrative", "media overreaction"]
  Corporate:        ["fraud / scandal", "restructuring signal", "growth opportunity", "market punishment", "employee impact"]
  Scientific:       ["breakthrough confirmed", "overhyped result", "regulatory hurdle", "public health concern", "funding battle"]
  Political:        ["policy reversal", "political theater", "genuine reform", "coalition collapse", "grassroots mobilization"]
  Cultural/Media:   ["public outrage", "viral moment", "industry disruption", "no lasting impact", "cancellation attempt"]

Signal: {signal}

Return ONLY valid JSON array of tag strings. No explanation, no markdown.
Example output: ["military escalation", "diplomatic resolution", "economic sanctions", "conspiracy narrative"]

JSON:"""


def generate_dynamic_buckets(signal_text: str) -> list[dict]:
    """
    Stage 1: Ask LLM to generate signal-appropriate narrative outcome tags.
    Returns list of bucket dicts with key, label, and reference text.
    """
    prompt = BUCKET_GEN_PROMPT.format(signal=signal_text[:600])

    raw = _call_ollama(prompt, max_tokens=200)
    tags = _parse_json_list(raw)

    if not tags or len(tags) < 2:
        logger.warning("Dynamic bucket generation failed — using fallback")
        return _fallback_buckets(signal_text)

    # Clean and deduplicate tags
    seen = set()
    buckets = []
    for tag in tags[:7]:
        tag_clean = str(tag).strip().lower()[:40]
        if tag_clean and tag_clean not in seen:
            seen.add(tag_clean)
            key = tag_clean.replace(" ", "_").replace("/", "_")
            buckets.append({
                "key":       key,
                "label":     tag_clean.capitalize(),
                "reference": _build_reference_text(tag_clean),
                "color":     _assign_color(tag_clean),
            })

    logger.info(f"Generated {len(buckets)} dynamic buckets: {[b['label'] for b in buckets]}")
    return buckets


def _build_reference_text(tag: str) -> str:
    """
    Build a rich semantic reference text for a tag.
    This is what outputs get compared against via embeddings.
    The richer this text, the better the semantic clustering.
    """
    expansions = {
        # Conflict
        "military escalation":  "military attack strike offensive troops missile conflict war escalation armed forces battle fighting aggression",
        "escalation":           "escalation attack threat strike force military conflict confrontation aggressive action",
        "war":                  "war combat military conflict armed forces troops fighting battle invasion",
        "conflict":             "conflict fighting confrontation attack threat military forces aggression",
        # Diplomatic
        "diplomatic resolution":"diplomacy negotiation peace ceasefire agreement treaty talks resolution compromise",
        "peace talks":          "peace negotiations diplomacy ceasefire summit dialogue back-channel resolution",
        "diplomatic":           "diplomacy negotiation agreement cooperation dialogue resolution",
        # Economic
        "market crash":         "market crash collapse financial crisis panic sell-off recession economic disruption stocks falling",
        "economic sanctions":   "sanctions embargo trade restrictions economic punishment financial measures penalties",
        "market correction":    "market correction dip adjustment pullback moderate decline temporary",
        "mild correction":      "mild correction small decline temporary pullback normal adjustment",
        "bullish signal":       "bullish growth positive market gain opportunity upside economic growth",
        "no real impact":       "no impact overblown normal routine business as usual unchanged stable",
        "regulatory action":    "regulation fine penalty charges lawsuit enforcement compliance legal action",
        "fraud":                "fraud scandal deception misconduct criminal charges investigation",
        "restructuring":        "restructuring reorganization layoffs cost-cutting pivot new direction",
        # Political
        "policy reversal":      "policy reversal U-turn backtrack change direction political decision",
        "political theater":    "political theater performance optics narrative management spin",
        "genuine reform":       "genuine reform real change policy improvement legislative progress",
        # Cultural
        "public outrage":       "public outrage anger backlash controversy criticism scandal reaction",
        "no lasting impact":    "no impact forgotten temporary news cycle normal routine",
        "viral moment":         "viral trending social media spread amplification attention",
        "conspiracy":           "conspiracy theory hidden agenda orchestration elite manipulation staged false flag",
        "conspiracy narrative": "conspiracy theory hidden actors orchestrated staged elite manipulation deep state",
        "media overreaction":   "media hype overblown exaggeration sensationalism panic disproportionate",
    }
    # Try exact match first
    if tag in expansions:
        return expansions[tag]
    # Try partial match
    for key, text in expansions.items():
        if any(word in tag for word in key.split()):
            return f"{tag} — {text}"
    # Fallback: use the tag itself repeated with synonyms
    return f"narrative outcome scenario where {tag} occurs and dominates the information environment"


def _assign_color(tag: str) -> str:
    """Assign a semantic color to a dynamic bucket."""
    tag_lower = tag.lower()
    if any(w in tag_lower for w in ["war","attack","conflict","escalat","militar","strike","threat"]):
        return "red"
    if any(w in tag_lower for w in ["crash","crisis","recession","sanction","fraud","collapse"]):
        return "coral"
    if any(w in tag_lower for w in ["peace","diplomat","resolut","ceasefire","cooperat"]):
        return "green"
    if any(w in tag_lower for w in ["conspiracy","manipulat","staged","false flag","deep state"]):
        return "amber"
    if any(w in tag_lower for w in ["routine","overblown","hype","no impact","normal","stable"]):
        return "gray"
    if any(w in tag_lower for w in ["growth","bullish","opportunity","breakthrough","positive"]):
        return "teal"
    if any(w in tag_lower for w in ["regulat","fine","charges","lawsuit","legal","reform"]):
        return "blue"
    return "purple"


def _fallback_buckets(signal_text: str) -> list[dict]:
    """Fallback when LLM generation fails — minimal signal-agnostic set."""
    return [
        {"key": "escalation",     "label": "Escalation",     "reference": "conflict escalation attack threat aggressive action", "color": "red"},
        {"key": "resolution",     "label": "Resolution",     "reference": "resolution diplomacy peace negotiation agreement",    "color": "green"},
        {"key": "economic_impact","label": "Economic impact","reference": "economic disruption market crash financial impact",   "color": "coral"},
        {"key": "conspiracy",     "label": "Conspiracy",     "reference": "conspiracy manipulation staged orchestrated agenda",  "color": "amber"},
        {"key": "no_impact",      "label": "No impact",      "reference": "routine normal overblown no real change stable",     "color": "gray"},
    ]


# ── Stage 2: Multi-tag semantic scoring ──────────────────────────────────────

def score_outputs_dynamic(
    outputs: list[dict],
    buckets: list[dict],
    multi_tag: bool = True,
) -> list[dict]:
    """
    Stage 2: Score each archetype output against the dynamic buckets.
    Supports multi-tag assignment — outputs can belong to multiple buckets.

    Returns outputs with bucket_scores and primary/secondary bucket assignments.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning("sentence-transformers not available — using keyword fallback")
        return _keyword_score_outputs(outputs, buckets)

    # Embed all bucket reference texts
    ref_texts   = [b["reference"] for b in buckets]
    ref_embeddings = model.encode(ref_texts, normalize_embeddings=True)
    bucket_emb_map = {b["key"]: ref_embeddings[i] for i, b in enumerate(buckets)}

    # Embed all output responses
    response_texts = [o.get("response", "") for o in outputs]
    if not any(response_texts):
        return outputs
    output_embeddings = model.encode(response_texts, normalize_embeddings=True)

    scored_outputs = []
    for i, output in enumerate(outputs):
        emb = output_embeddings[i]
        scores = {}
        for b in buckets:
            sim = float(np.dot(emb, bucket_emb_map[b["key"]]))
            scores[b["key"]] = round(sim, 4)

        # Primary bucket = highest score
        primary = max(scores, key=scores.get)
        primary_score = scores[primary]

        # Multi-tag: find secondary buckets above threshold
        secondary = []
        if multi_tag:
            for bk, sc in scores.items():
                if bk != primary and sc >= MULTI_TAG_THRESHOLD and sc >= primary_score * 0.65:
                    secondary.append((bk, sc))
            secondary.sort(key=lambda x: -x[1])

        # Build fractional weights for probability mass splitting
        tag_weights = {primary: 1.0}
        if secondary:
            total_secondary_weight = sum(s for _, s in secondary[:2])
            for bk, sc in secondary[:2]:
                weight = max(SPLIT_WEIGHT_MIN, sc / primary_score * 0.5)
                tag_weights[bk] = round(weight, 3)

        # Normalize weights
        total = sum(tag_weights.values())
        tag_weights = {k: round(v / total, 3) for k, v in tag_weights.items()}

        out = dict(output)
        out["bucket"]            = primary
        out["bucket_scores"]     = scores
        out["bucket_weights"]    = tag_weights
        out["secondary_buckets"] = [b for b in secondary[:2]]
        out["multi_tag"]         = len(tag_weights) > 1
        scored_outputs.append(out)

    return scored_outputs


def compute_dynamic_probability_map(
    scored_outputs: list[dict],
    buckets: list[dict],
) -> list[dict]:
    """
    Build probability map from multi-tag scored outputs.
    Probability mass is split across tags for multi-tag outputs.
    Returns sorted probability map.
    """
    total_weight  = defaultdict(float)
    bucket_outputs = defaultdict(list)

    for output in scored_outputs:
        weights = output.get("bucket_weights", {output.get("bucket", "unknown"): 1.0})
        for bk, w in weights.items():
            total_weight[bk] += w
            if w >= SPLIT_WEIGHT_MIN:
                bucket_outputs[bk].append(output)

    total = sum(total_weight.values())
    if total == 0:
        return []

    bucket_map = {b["key"]: b for b in buckets}
    prob_map   = []

    for bk, weight in sorted(total_weight.items(), key=lambda x: -x[1]):
        if weight < 0.01:
            continue
        b_info = bucket_map.get(bk, {"key": bk, "label": bk.replace("_", " ").capitalize(), "color": "gray"})
        outs   = bucket_outputs[bk]

        personalities = list(set(o["personality_name"] for o in outs))
        models_in     = list(set(o.get("model_display", "") for o in outs))

        personality_model_counts = defaultdict(int)
        for o in outs:
            personality_model_counts[o["personality"]] += 1
        cross_model = [p for p, c in personality_model_counts.items() if c > 1]

        prob_map.append({
            "bucket":              bk,
            "label":               b_info["label"],
            "color":               b_info.get("color", "gray"),
            "weight":              round(weight, 4),
            "probability":         round(weight / total, 4),
            "probability_pct":     f"{round(weight / total * 100, 1)}%",
            "confidence":          "high" if len(cross_model) >= 3 else "medium" if len(cross_model) >= 1 else "low",
            "models_represented":  models_in,
            "personalities":       personalities,
            "cross_model_agreement": cross_model,
            "cross_model_count":   len(cross_model),
            "multi_tag_outputs":   sum(1 for o in outs if o.get("multi_tag")),
            "sample_response":     outs[0]["response"][:200] + "..." if outs else "",
        })

    return prob_map


# ── Keyword fallback ──────────────────────────────────────────────────────────

def _keyword_score_outputs(outputs: list[dict], buckets: list[dict]) -> list[dict]:
    """Keyword-based fallback when sentence-transformers unavailable."""
    for output in outputs:
        text   = output.get("response", "").lower()
        scores = {}
        for b in buckets:
            ref_words = b["reference"].lower().split()
            hits = sum(1 for w in ref_words if w in text)
            scores[b["key"]] = round(hits / max(len(ref_words), 1), 4)

        primary = max(scores, key=scores.get) if scores else "unknown"
        output["bucket"]         = primary
        output["bucket_scores"]  = scores
        output["bucket_weights"] = {primary: 1.0}
        output["secondary_buckets"] = []
        output["multi_tag"]      = False
    return outputs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, max_tokens: int = 300) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": max_tokens}},
            timeout=30,
        )
        return resp.json().get("response", "").strip() if resp.status_code == 200 else None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _parse_json_list(raw: str) -> list | None:
    if not raw:
        return None
    for pattern in [r'\[.*?\]']:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def score_and_map_dynamic(result: dict) -> dict:
    """
    Drop-in replacement for score_and_map / score_and_map_semantic.
    Generates signal-specific buckets then scores with multi-tag support.

    Stores generated buckets in result for use by narrative_tree.py
    and the journalist summarizer.
    """
    outputs     = result.get("outputs", [])
    signal_text = result.get("signal_text", "")

    if not outputs:
        logger.warning("No outputs to score dynamically")
        return result

    logger.info("Stage 1: Generating dynamic buckets...")
    buckets = generate_dynamic_buckets(signal_text)
    logger.info(f"Buckets: {[b['label'] for b in buckets]}")

    logger.info("Stage 2: Multi-tag semantic scoring...")
    scored_outputs = score_outputs_dynamic(outputs, buckets, multi_tag=True)

    logger.info("Building probability map...")
    prob_map = compute_dynamic_probability_map(scored_outputs, buckets)

    result["outputs"]            = scored_outputs
    result["clusters"]           = prob_map
    result["dynamic_buckets"]    = buckets
    result["dominant_narrative"] = prob_map[0] if prob_map else None
    result["scoring_method"]     = "dynamic_multi_tag"

    if prob_map:
        top = prob_map[0]
        multi_tag_count = sum(1 for o in scored_outputs if o.get("multi_tag"))
        logger.info(f"Dominant: '{top['label']}' at {top['probability_pct']}")
        logger.info(f"Multi-tag outputs: {multi_tag_count}/{len(outputs)}")

    return result
