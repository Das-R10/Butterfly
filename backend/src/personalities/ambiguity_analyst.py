"""
src/personalities/ambiguity_analyst.py  — L2B post-pass

The Ambiguity Analyst.

Runs after all 16 analyst archetypes have produced outputs.
It does not analyze the signal — it analyzes the analysis.

Three jobs:
  1. Cross-domain insight extraction
     Which archetypes found angles outside the primary domain?
     e.g. Economist surfacing financial implications of a cultural signal.
     e.g. Religious Interpreter finding prophetic framing in a geopolitical event.
     These cross-domain outputs get their own significance score and
     are surfaced to L5 Critics, L7 Summarizer, and L8 Causal Generator.

  2. Domain classification challenge
     Is the primary domain tag the correct lens for this signal?
     If multiple archetypes are clearly processing this as a different
     domain, the Ambiguity Analyst flags a reclassification recommendation.

  3. Missing context flags
     What would need to be true — but is not in the brief — for this
     signal to make complete sense? Gaps that could change the outputs
     if filled.

Output is passed to L5, L7, and L8. It does NOT modify archetype outputs
retroactively — it annotates them.
"""

from __future__ import annotations

import json
import re
import requests
from datetime import datetime
from loguru import logger

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"
TEMPERATURE  = 0.3

# Minimum significance score for a cross-domain insight to be surfaced
CROSS_DOMAIN_MIN_SIGNIFICANCE = 0.3

# Domain taxonomy — used to detect when archetypes stray outside primary domain
DOMAIN_ARCHETYPES = {
    "geopolitical":  ["geopolitical_hawk", "foreign_policy_dove", "institutional_analyst", "historian"],
    "economic":      ["economist", "cynical_realist", "institutional_analyst"],
    "technology":    ["techno_optimist", "pattern_seeker", "skeptic_debunker"],
    "cultural":      ["religious_interpreter", "local_journalist", "anxious_sharer"],
    "humanitarian":  ["idealist_pacifist", "anxious_sharer", "religious_interpreter"],
    "conspiracy":    ["conspiracy_theorist", "pattern_seeker", "cynical_realist"],
}

# Archetypes that are inherently cross-domain by design — always scan all domains
UNIVERSAL_ARCHETYPES = {
    "pattern_seeker", "skeptic_debunker", "amplifier_influencer",
    "anxious_sharer", "conspiracy_theorist",
}


# ── Prompts ───────────────────────────────────────────────────────────────────

CROSS_DOMAIN_PROMPT = """You are a meta-analyst reviewing outputs from 16 different analytical archetypes who have all analyzed the same news signal.

Signal: {signal}
Primary domain classification: {primary_domain}

Here are the 16 archetype outputs (summarized):
{outputs_summary}

Your job: identify which archetypes found CROSS-DOMAIN insights — angles that go significantly beyond the primary domain classification.

For each cross-domain insight you find, assess:
- Which archetype found it
- What secondary domain it belongs to
- How significant is this angle (0.0 to 1.0 — how much does it change the picture)
- One sentence describing the insight

Return ONLY valid JSON array. No explanation, no markdown:
[
  {{
    "archetype": "archetype_key",
    "secondary_domain": "domain_name",
    "significance": 0.0,
    "insight": "one sentence describing what cross-domain angle this archetype found"
  }}
]

Only include insights with significance above 0.2. If none found, return [].

JSON:"""


DOMAIN_CHALLENGE_PROMPT = """You are reviewing whether a signal has been processed by the correct analytical domain.

Signal: {signal}
Assigned primary domain: {primary_domain}

Archetype outputs summary:
{outputs_summary}

Question: Based on what the archetypes actually focused on, is "{primary_domain}" the right primary domain for this signal?

Consider: if most archetypes are talking about economics when the signal was classified as geopolitical, the domain may be wrong.

Return ONLY valid JSON:
{{
  "challenge": true or false,
  "confidence": 0.0 to 1.0,
  "recommended_domain": "domain if challenge is true, else same as primary",
  "reasoning": "one sentence"
}}

JSON:"""


MISSING_CONTEXT_PROMPT = """You are reviewing an analysis of a news signal for critical missing context.

Signal: {signal}

Archetype outputs summary (what analysts actually said):
{outputs_summary}

Identify up to 3 pieces of missing context — information that is NOT in the signal or brief, but if present would significantly change how archetypes interpreted this.

Return ONLY valid JSON array:
[
  {{
    "missing_item": "what is missing",
    "impact": "how this would change the analysis if known",
    "urgency": "high / medium / low"
  }}
]

Return [] if no significant gaps. No explanation, no markdown.

JSON:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, max_tokens: int = 400) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
            },
            timeout=45,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return None
    except Exception as e:
        logger.warning(f"Ambiguity Analyst Ollama call failed: {e}")
        return None


def _parse_json(raw: str) -> list | dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON from response
        for pattern in [r'\[.*\]', r'\{.*\}']:
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
    return None


def _build_outputs_summary(outputs: list[dict], max_chars_per: int = 120) -> str:
    """
    Build a compact summary of all archetype outputs for the Ambiguity Analyst.
    Each line: archetype_name | first N chars of response
    """
    lines = []
    for o in outputs:
        name     = o.get("personality_name", o.get("personality", "unknown"))
        response = o.get("response", "").strip()[:max_chars_per]
        response = response.replace("\n", " ")
        lines.append(f"  [{name}]: {response}")
    return "\n".join(lines)


# ── Cross-domain insight detector ─────────────────────────────────────────────

def _detect_cross_domain_insights(
    signal: str,
    primary_domain: str,
    outputs: list[dict],
) -> list[dict]:
    """
    Ask LLM to identify which archetypes found cross-domain insights.
    Falls back to heuristic detection if LLM fails.
    """
    outputs_summary = _build_outputs_summary(outputs)
    prompt = CROSS_DOMAIN_PROMPT.format(
        signal=signal[:400],
        primary_domain=primary_domain,
        outputs_summary=outputs_summary[:1800],
    )

    raw = _call_ollama(prompt, max_tokens=500)
    parsed = _parse_json(raw)

    if isinstance(parsed, list):
        # Filter by minimum significance
        return [
            item for item in parsed
            if isinstance(item, dict)
            and item.get("significance", 0) >= CROSS_DOMAIN_MIN_SIGNIFICANCE
        ]

    # Heuristic fallback — look for domain keyword mismatches
    logger.debug("Ambiguity Analyst: LLM cross-domain detection failed, using heuristics")
    return _heuristic_cross_domain(primary_domain, outputs)


def _heuristic_cross_domain(primary_domain: str, outputs: list[dict]) -> list[dict]:
    """
    Simple heuristic: if an archetype whose domain doesn't match the primary
    domain produces a response with strong domain keywords, flag it.
    """
    domain_keywords = {
        "economic":     ["market", "price", "economic", "financial", "trade", "gdp", "inflation"],
        "geopolitical": ["military", "war", "conflict", "nato", "sanctions", "troops", "border"],
        "technology":   ["ai", "tech", "software", "digital", "automation", "algorithm"],
        "humanitarian": ["civilian", "refugee", "human rights", "suffering", "aid", "crisis"],
        "cultural":     ["religion", "culture", "society", "community", "belief", "tradition"],
    }

    insights = []
    for o in outputs:
        pname    = o.get("personality", "")
        response = o.get("response", "").lower()

        for domain, keywords in domain_keywords.items():
            if domain == primary_domain:
                continue
            hits = sum(1 for kw in keywords if kw in response)
            if hits >= 3:
                significance = min(hits / len(keywords), 0.8)
                insights.append({
                    "archetype":        pname,
                    "secondary_domain": domain,
                    "significance":     round(significance, 2),
                    "insight":          f"{pname} focused on {domain} aspects despite {primary_domain} classification",
                })
                break  # one cross-domain per archetype

    return insights


# ── Domain classification challenger ─────────────────────────────────────────

def _challenge_domain_classification(
    signal: str,
    primary_domain: str,
    outputs: list[dict],
) -> dict:
    """
    Check whether the primary domain classification is actually correct
    given what archetypes focused on.
    """
    outputs_summary = _build_outputs_summary(outputs, max_chars_per=80)
    prompt = DOMAIN_CHALLENGE_PROMPT.format(
        signal=signal[:400],
        primary_domain=primary_domain,
        outputs_summary=outputs_summary[:1200],
    )

    raw = _call_ollama(prompt, max_tokens=200)
    parsed = _parse_json(raw)

    if isinstance(parsed, dict):
        return {
            "challenge":            parsed.get("challenge", False),
            "confidence":           parsed.get("confidence", 0.0),
            "recommended_domain":   parsed.get("recommended_domain", primary_domain),
            "reasoning":            parsed.get("reasoning", ""),
        }

    return {
        "challenge": False,
        "confidence": 0.0,
        "recommended_domain": primary_domain,
        "reasoning": "Classification challenge assessment unavailable",
    }


# ── Missing context detector ──────────────────────────────────────────────────

def _detect_missing_context(
    signal: str,
    outputs: list[dict],
) -> list[dict]:
    """
    Identify what critical context is missing that would change the analysis.
    """
    outputs_summary = _build_outputs_summary(outputs, max_chars_per=80)
    prompt = MISSING_CONTEXT_PROMPT.format(
        signal=signal[:400],
        outputs_summary=outputs_summary[:1200],
    )

    raw = _call_ollama(prompt, max_tokens=400)
    parsed = _parse_json(raw)

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)][:3]

    return []


# ── Surface framing challenge ─────────────────────────────────────────────────

def _check_surface_framing(outputs: list[dict], signal: str) -> dict:
    """
    Simple heuristic: if archetype outputs diverge wildly on the basic
    nature of the event (what IS this?), flag the surface framing as
    potentially misleading.

    Checks: do archetypes agree on what type of event this is?
    """
    # Count how many outputs contain keywords suggesting different event types
    event_types = {
        "military_action":  ["strike", "attack", "missile", "troops", "bombing", "airstrike"],
        "economic_event":   ["market", "crash", "tariff", "trade", "economic", "sanctions"],
        "political_event":  ["election", "vote", "policy", "legislation", "president", "minister"],
        "information_event":["report", "claim", "allegation", "rumor", "leaked", "anonymous"],
    }

    counts = {etype: 0 for etype in event_types}
    for o in outputs:
        response = o.get("response", "").lower()
        for etype, keywords in event_types.items():
            if sum(1 for kw in keywords if kw in response) >= 2:
                counts[etype] += 1

    total = sum(counts.values())
    if total == 0:
        return {"framing_ambiguous": False}

    dominant_type  = max(counts, key=counts.get)
    dominant_count = counts[dominant_type]
    dominant_pct   = dominant_count / max(total, 1)

    return {
        "framing_ambiguous":    dominant_pct < 0.5,
        "dominant_event_type":  dominant_type,
        "dominant_pct":         round(dominant_pct, 2),
        "event_type_counts":    counts,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_ambiguity_analyst(
    signal: str,
    outputs: list[dict],
    primary_domain: str = "unknown",
    dynamic_buckets: list[dict] = None,
) -> dict:
    """
    Run the Ambiguity Analyst on all 16 archetype outputs.

    Args:
        signal:         The original signal text
        outputs:        All 16 archetype outputs from L2B
        primary_domain: The L0 domain classification tag
        dynamic_buckets: L0 dynamic buckets (for context)

    Returns:
        ambiguity_report dict containing:
          - cross_domain_insights:    list of cross-domain angles found
          - domain_challenge:         whether primary domain should be reclassified
          - missing_context:          list of missing context items
          - surface_framing:          whether the surface framing of the signal is ambiguous
          - report_text:              formatted text for downstream layers
          - has_significant_findings: bool — whether any finding clears threshold
          - timestamp
    """
    dynamic_buckets = dynamic_buckets or []

    logger.info(f"Ambiguity Analyst: analyzing {len(outputs)} archetype outputs")
    print(f"\n  ── Ambiguity Analyst ────────────────────────────────────────────")

    # Run all three checks
    cross_domain = _detect_cross_domain_insights(signal, primary_domain, outputs)
    domain_challenge = _challenge_domain_classification(signal, primary_domain, outputs)
    missing_context  = _detect_missing_context(signal, outputs)
    surface_framing  = _check_surface_framing(outputs, signal)

    # Determine if findings are significant enough to surface
    has_significant = (
        len(cross_domain) > 0
        or domain_challenge.get("challenge") is True
        or any(item.get("urgency") == "high" for item in missing_context)
        or surface_framing.get("framing_ambiguous") is True
    )

    # Annotate outputs with cross-domain flags
    cross_domain_by_archetype = {
        item["archetype"]: item for item in cross_domain
    }
    for o in outputs:
        pkey = o.get("personality", "")
        if pkey in cross_domain_by_archetype:
            o["cross_domain"] = cross_domain_by_archetype[pkey]
        else:
            o["cross_domain"] = None

    # Build report text
    report_text = _build_ambiguity_report(
        cross_domain, domain_challenge, missing_context,
        surface_framing, primary_domain,
    )

    print(f"  ✓ Ambiguity Analyst complete — "
          f"{len(cross_domain)} cross-domain insights, "
          f"domain challenge: {domain_challenge.get('challenge')}, "
          f"{len(missing_context)} missing context items")

    return {
        "cross_domain_insights":  cross_domain,
        "domain_challenge":       domain_challenge,
        "missing_context":        missing_context,
        "surface_framing":        surface_framing,
        "primary_domain":         primary_domain,
        "has_significant_findings": has_significant,
        "report_text":            report_text,
        "timestamp":              datetime.utcnow().isoformat() + "Z",
    }


# ── Report builder ────────────────────────────────────────────────────────────

def _build_ambiguity_report(
    cross_domain: list,
    domain_challenge: dict,
    missing_context: list,
    surface_framing: dict,
    primary_domain: str,
) -> str:
    """
    Build formatted report text passed to L5 Critics, L7 Summarizer, L8 Causal Generator.
    """
    lines = [
        "═" * 60,
        "AMBIGUITY ANALYST REPORT",
        "═" * 60,
        "",
    ]

    # Domain classification
    if domain_challenge.get("challenge"):
        lines += [
            f"⚠  DOMAIN RECLASSIFICATION RECOMMENDED",
            f"   Primary: {primary_domain}",
            f"   Recommended: {domain_challenge.get('recommended_domain', '?')}",
            f"   Confidence: {domain_challenge.get('confidence', 0):.0%}",
            f"   Reason: {domain_challenge.get('reasoning', '')}",
            "",
        ]
    else:
        lines.append(f"Domain classification '{primary_domain}': confirmed by archetype outputs")
        lines.append("")

    # Surface framing
    if surface_framing.get("framing_ambiguous"):
        lines += [
            f"⚠  SURFACE FRAMING AMBIGUOUS",
            f"   Archetypes disagree on the basic nature of this event.",
            f"   Dominant event type: {surface_framing.get('dominant_event_type')} "
            f"({surface_framing.get('dominant_pct', 0):.0%} of outputs)",
            "",
        ]

    # Cross-domain insights
    if cross_domain:
        lines += ["── Cross-domain insights ──────────────────────────────────", ""]
        for insight in sorted(cross_domain, key=lambda x: -x.get("significance", 0)):
            sig = insight.get("significance", 0)
            arch = insight.get("archetype", "unknown")
            domain = insight.get("secondary_domain", "unknown")
            text = insight.get("insight", "")
            lines.append(
                f"  [{arch}] → {domain.upper()} (significance: {sig:.0%})"
            )
            lines.append(f"    {text}")
            lines.append("")
    else:
        lines.append("No significant cross-domain insights detected.")
        lines.append("")

    # Missing context
    if missing_context:
        lines += ["── Missing context ────────────────────────────────────────", ""]
        for item in missing_context:
            urgency = item.get("urgency", "low").upper()
            missing = item.get("missing_item", "")
            impact  = item.get("impact", "")
            lines.append(f"  [{urgency}] {missing}")
            lines.append(f"    Impact if known: {impact}")
            lines.append("")

    lines += ["═" * 60, "END AMBIGUITY ANALYST REPORT", "═" * 60]
    return "\n".join(lines)
