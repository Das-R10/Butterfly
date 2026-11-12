"""
src/ensemble/debate_layer.py  — L5

The Debate Layer. Three Critics stress-test the dominant narrative.

Each Critic receives:
  - The original signal
  - The L1 behavioral brief (if Support Mode)
  - The L2A MBTI population reaction report
  - ALL L2B archetype outputs (not just the dominant cluster)
  - The L0 dynamic buckets
  - GDELT Goldstein data (if available)

Three Critics:
  Fact-Checker  — challenges the logical leap from signal to story.
                  Cross-references archetype claims against GDELT Goldstein.
                  Asks: does the data actually support this extrapolation?

  Cynic         — questions who benefits from the dominant narrative winning.
                  Reads all 16 outputs looking for suspicious alignment.
                  Asks: who benefits if this story dominates?

  Chaos Monkey  — takes the single most divergent minority archetype output
                  and builds the strongest possible case that it is correct
                  and the majority cluster is wrong.
                  Asks: what if the outlier is right?

After all three Critics run:
  - Volatility Score calculated (semantic distance pre vs post critique)
  - High-Octane Narrative flag set if: dominant prob > 50% AND volatility low AND drift low
  - Recalibration signal produced for MBTI layer
"""

from __future__ import annotations

import json
import re
import requests
import numpy as np
from datetime import datetime
from loguru import logger

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"
TEMPERATURE  = 0.3

# Volatility thresholds
HIGH_OCTANE_MIN_PROBABILITY  = 0.50   # dominant cluster must be above this
HIGH_OCTANE_MAX_VOLATILITY   = 0.25   # volatility must be below this
HIGH_OCTANE_MAX_DRIFT        = 0.20   # narrative drift must be below this


# ── Critic prompts ────────────────────────────────────────────────────────────

FACT_CHECKER_PROMPT = """You are the Fact-Checker critic in a narrative intelligence system.

Your job: challenge the logical leap from signal to dominant story. Be rigorous and specific.

Original signal:
{signal}

Dominant narrative that emerged ({dominant_label}, {dominant_pct} probability):
{dominant_sample}

All 16 archetype outputs summary:
{outputs_summary}

{gdelt_section}

{mbti_section}

Challenge this narrative on THREE grounds:
1. EVIDENCE GAP: What specific evidence would actually be required to support this interpretation that is NOT present in the signal?
2. BASE RATE: What is the historical base rate for this type of event leading to this outcome? Is the probability justified?
3. ALTERNATIVE: What is the most parsimonious mundane explanation that most archetypes ignored?

Be specific. Name claims, not generalities. 3-5 sentences per point. Plain text, no markdown."""


CYNIC_PROMPT = """You are the Cynic critic in a narrative intelligence system.

Your job: identify who benefits from the dominant narrative winning, and whether the archetype outputs show suspicious alignment.

Original signal:
{signal}

Dominant narrative ({dominant_label}, {dominant_pct} probability):
{dominant_sample}

All 16 archetype outputs summary:
{outputs_summary}

{mbti_section}

Analyze on THREE grounds:
1. BENEFICIARIES: Which specific actors (governments, corporations, media entities, political factions) benefit if this narrative dominates?
2. ALIGNMENT CHECK: Do any archetype outputs look suspiciously consistent with each other in a way that suggests a shared framing rather than independent analysis?
3. DISTRACTION: What real story might this narrative be displacing or obscuring?

Be specific. Name actors. 3-5 sentences per point. Plain text, no markdown."""


CHAOS_MONKEY_PROMPT = """You are the Chaos Monkey critic in a narrative intelligence system.

Your job: find the most divergent minority archetype output and build the strongest possible case that IT is correct and the dominant majority is wrong.

Original signal:
{signal}

Dominant narrative ({dominant_label}, {dominant_pct} probability) — this is what MOST archetypes believe:
{dominant_sample}

All 16 archetype outputs — find the outlier:
{outputs_summary}

{mbti_section}

Do THREE things:
1. IDENTIFY OUTLIER: Name the specific archetype whose output is most divergent from the dominant narrative. Quote the key divergent claim.
2. STRONGEST CASE: Build the most rigorous argument you can that the outlier is correct. What would need to be true? What evidence is consistent with it?
3. FAILURE MODE: If the outlier IS correct and the majority IS wrong, what is the specific mechanism of failure? How does a false consensus form here?

Be steel-man specific. 3-5 sentences per point. Plain text, no markdown."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, max_tokens: int = 500) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return None
    except Exception as e:
        logger.warning(f"Debate layer Ollama call failed: {e}")
        return None


def _build_outputs_summary(outputs: list[dict], max_chars: int = 100) -> str:
    """Compact summary of all archetype outputs for Critic prompts."""
    lines = []
    for o in outputs:
        name     = o.get("personality_name", o.get("personality", "?"))
        response = o.get("response", "").strip()[:max_chars].replace("\n", " ")
        lines.append(f"  [{name}]: {response}")
    return "\n".join(lines)


def _build_gdelt_section(gdelt_data: dict) -> str:
    if not gdelt_data:
        return ""
    goldstein = gdelt_data.get("goldstein_score")
    location  = gdelt_data.get("location", "named location")
    if goldstein is None:
        return ""
    return (
        f"\nGDELT Goldstein Scale for {location}: {goldstein:.1f} "
        f"(range -10 to +10, negative = destabilizing).\n"
        f"Use this as ground truth for conflict probability calibration."
    )


def _build_mbti_section(mbti_report: dict) -> str:
    if not mbti_report:
        return ""
    dominant  = mbti_report.get("dominant_emotion", "unknown")
    velocity  = mbti_report.get("sharing_velocity", 0)
    return (
        f"\nPopulation context (MBTI Layer): dominant emotion is '{dominant}', "
        f"sharing velocity {velocity:.0%}. "
        f"Factor this into your critique — is the dominant narrative consistent "
        f"with how the population is actually reacting?"
    )


def _find_outlier_archetype(outputs: list[dict], dominant_bucket: str) -> dict | None:
    """
    Find the archetype whose output is most divergent from the dominant bucket.
    Uses primary bucket assignment if available, else falls back to the
    archetype whose response is shortest and most hedged.
    """
    # First try: find archetype assigned to a different bucket
    non_dominant = [
        o for o in outputs
        if o.get("bucket", dominant_bucket) != dominant_bucket
        and o.get("response", "") != "[No output generated]"
    ]
    if non_dominant:
        # Return the one with the most confident non-dominant assignment
        return max(non_dominant, key=lambda o: len(o.get("response", "")))

    # Fallback: find the archetype with most skeptical/questioning language
    skeptical_keywords = ["however", "unclear", "unlikely", "question", "doubt",
                          "actually", "but", "although", "wait", "overblown"]
    best = None
    best_score = -1
    for o in outputs:
        text  = o.get("response", "").lower()
        score = sum(1 for kw in skeptical_keywords if kw in text)
        if score > best_score:
            best_score = score
            best = o
    return best


# ── Volatility calculator ─────────────────────────────────────────────────────

def _compute_volatility(
    pre_map: list[dict],
    post_map: list[dict],
) -> float:
    """
    Volatility Score = semantic distance between pre-critique and post-critique
    probability maps.

    Calculated as the sum of absolute probability changes across all buckets.
    0.0 = no change (armored narrative)
    1.0 = complete redistribution (fragile narrative)
    """
    if not pre_map or not post_map:
        return 0.0

    pre_probs  = {c["bucket"]: c["probability"] for c in pre_map}
    post_probs = {c["bucket"]: c["probability"] for c in post_map}

    all_buckets = set(pre_probs) | set(post_probs)
    total_delta = sum(
        abs(pre_probs.get(b, 0) - post_probs.get(b, 0))
        for b in all_buckets
    )

    # Normalize to 0-1 (max possible delta is 2.0 for complete reversal)
    return round(min(total_delta / 2.0, 1.0), 4)


def _apply_critic_adjustments(
    probability_map: list[dict],
    fact_check: str,
    cynic: str,
    chaos_monkey: str,
    outlier_archetype: dict | None,
) -> list[dict]:
    """
    Simple heuristic: adjust probability weights based on critic findings.

    - If Fact-Checker found major evidence gaps → reduce dominant cluster by 5-10%
    - If Chaos Monkey identified credible outlier → bump outlier's bucket by 3-5%
    - Renormalize after adjustments

    This is intentionally conservative — critics shift weights, not overturn them.
    """
    if not probability_map:
        return probability_map

    import copy
    adjusted = copy.deepcopy(probability_map)

    # Detect if Fact-Checker found serious gaps
    fact_check_lower = (fact_check or "").lower()
    serious_gap = any(
        phrase in fact_check_lower
        for phrase in ["no evidence", "no data", "unverified", "speculation",
                       "insufficient", "contradicts", "inconsistent"]
    )

    # Detect if Chaos Monkey found credible outlier
    chaos_lower = (chaos_monkey or "").lower()
    credible_outlier = any(
        phrase in chaos_lower
        for phrase in ["credible", "compelling", "strong case", "correct",
                       "possible", "cannot rule out", "consistent with"]
    )

    # Apply adjustments
    dominant = adjusted[0]

    if serious_gap:
        dominant["probability"] = round(max(dominant["probability"] - 0.07, 0.01), 4)
        dominant["critic_adjusted"] = True

    if credible_outlier and outlier_archetype and len(adjusted) > 1:
        outlier_bucket = outlier_archetype.get("bucket", "")
        for cluster in adjusted[1:]:
            if cluster.get("bucket") == outlier_bucket:
                cluster["probability"] = round(
                    min(cluster["probability"] + 0.05, 0.99), 4
                )
                cluster["chaos_monkey_flag"] = True
                break

    # Renormalize
    total = sum(c["probability"] for c in adjusted)
    if total > 0:
        for c in adjusted:
            c["probability"] = round(c["probability"] / total, 4)
            c["probability_pct"] = f"{round(c['probability'] * 100, 1)}%"

    return sorted(adjusted, key=lambda x: -x["probability"])


# ── Main entry point ──────────────────────────────────────────────────────────

def run_debate_layer(
    signal: str,
    probability_map: list[dict],
    outputs: list[dict],
    mbti_report: dict = None,
    behavioral_brief: str = "",
    gdelt_data: dict = None,
    narrative_drift_score: float = 0.0,
) -> dict:
    """
    Run all three Critics on the dominant narrative.

    Args:
        signal:              Original signal text
        probability_map:     Current probability map from L4 (pre-critique)
        outputs:             ALL 16 archetype outputs from L2B
        mbti_report:         L2A MBTI population report
        behavioral_brief:    L1 world-state brief (Support Mode only)
        gdelt_data:          GDELT Goldstein data dict (optional)
        narrative_drift_score: L3 drift score

    Returns:
        debate_result dict containing:
          - fact_check_critique
          - cynic_critique
          - chaos_monkey_critique
          - outlier_archetype
          - adjusted_probability_map  (post-critique weights)
          - volatility_score
          - high_octane_narrative      (bool)
          - recalibration_signal       (for MBTI recalibration pass)
          - report_text
          - timestamp
    """
    if not probability_map:
        logger.warning("Debate Layer: no probability map to debate")
        return {"error": "no probability map", "volatility_score": 0.0}

    dominant         = probability_map[0]
    dominant_label   = dominant.get("label", "unknown")
    dominant_pct     = dominant.get("probability_pct", "?")
    dominant_bucket  = dominant.get("bucket", "unknown")
    dominant_sample  = dominant.get("sample_response", "")[:300]

    logger.info(f"Debate Layer: critiquing '{dominant_label}' ({dominant_pct})")
    print(f"\n  ── Debate Layer ─────────────────────────────────────────────────")

    # Build shared context sections
    outputs_summary  = _build_outputs_summary(outputs, max_chars=110)
    gdelt_section    = _build_gdelt_section(gdelt_data)
    mbti_section     = _build_mbti_section(mbti_report)

    # Find outlier before running critics (Chaos Monkey needs to know)
    outlier = _find_outlier_archetype(outputs, dominant_bucket)
    outlier_name = outlier.get("personality_name", "unknown") if outlier else "none"
    logger.info(f"  Outlier archetype identified: {outlier_name}")

    # ── Fact-Checker ──────────────────────────────────────────────────────────
    print(f"  Running Fact-Checker...")
    fact_prompt = FACT_CHECKER_PROMPT.format(
        signal=signal[:400],
        dominant_label=dominant_label,
        dominant_pct=dominant_pct,
        dominant_sample=dominant_sample,
        outputs_summary=outputs_summary[:1500],
        gdelt_section=gdelt_section,
        mbti_section=mbti_section,
    )
    fact_check = _call_ollama(fact_prompt, max_tokens=500) or "[Fact-Checker unavailable]"

    # ── Cynic ─────────────────────────────────────────────────────────────────
    print(f"  Running Cynic...")
    cynic_prompt = CYNIC_PROMPT.format(
        signal=signal[:400],
        dominant_label=dominant_label,
        dominant_pct=dominant_pct,
        dominant_sample=dominant_sample,
        outputs_summary=outputs_summary[:1500],
        mbti_section=mbti_section,
    )
    cynic = _call_ollama(cynic_prompt, max_tokens=500) or "[Cynic unavailable]"

    # ── Chaos Monkey ──────────────────────────────────────────────────────────
    print(f"  Running Chaos Monkey (outlier: {outlier_name})...")
    chaos_prompt = CHAOS_MONKEY_PROMPT.format(
        signal=signal[:400],
        dominant_label=dominant_label,
        dominant_pct=dominant_pct,
        dominant_sample=dominant_sample,
        outputs_summary=outputs_summary[:1800],
        mbti_section=mbti_section,
    )
    chaos_monkey = _call_ollama(chaos_prompt, max_tokens=500) or "[Chaos Monkey unavailable]"

    # ── Apply adjustments and calculate volatility ────────────────────────────
    adjusted_map = _apply_critic_adjustments(
        probability_map, fact_check, cynic, chaos_monkey, outlier
    )
    volatility = _compute_volatility(probability_map, adjusted_map)

    # ── High-Octane Narrative flag ────────────────────────────────────────────
    dominant_prob   = dominant.get("probability", 0.0)
    high_octane     = (
        dominant_prob  >= HIGH_OCTANE_MIN_PROBABILITY
        and volatility <= HIGH_OCTANE_MAX_VOLATILITY
        and narrative_drift_score <= HIGH_OCTANE_MAX_DRIFT
    )

    if high_octane:
        logger.info(
            f"  🔥 HIGH-OCTANE NARRATIVE: '{dominant_label}' "
            f"prob={dominant_prob:.0%} volatility={volatility:.2f} drift={narrative_drift_score:.2f}"
        )
        print(f"  🔥 HIGH-OCTANE NARRATIVE flagged: '{dominant_label}'")
    else:
        logger.info(f"  Volatility: {volatility:.3f} | High-Octane: {high_octane}")

    # ── Recalibration signal for MBTI ─────────────────────────────────────────
    # If Chaos Monkey found a credible outlier, signal which MBTI types
    # should have their weights adjusted upward
    recalibration_signal = _build_recalibration_signal(
        chaos_monkey, outlier, mbti_report
    )

    # ── Build report ──────────────────────────────────────────────────────────
    report_text = _build_debate_report(
        dominant_label, dominant_pct, volatility, high_octane,
        fact_check, cynic, chaos_monkey, outlier_name, recalibration_signal,
    )

    print(
        f"  ✓ Debate Layer complete — volatility={volatility:.3f} "
        f"high_octane={high_octane}"
    )

    return {
        "fact_check_critique":      fact_check,
        "cynic_critique":           cynic,
        "chaos_monkey_critique":    chaos_monkey,
        "outlier_archetype":        outlier,
        "outlier_archetype_name":   outlier_name,
        "pre_critique_map":         probability_map,
        "adjusted_probability_map": adjusted_map,
        "volatility_score":         volatility,
        "high_octane_narrative":    high_octane,
        "recalibration_signal":     recalibration_signal,
        "report_text":              report_text,
        "timestamp":                datetime.utcnow().isoformat() + "Z",
    }


# ── Recalibration signal builder ──────────────────────────────────────────────

def _build_recalibration_signal(
    chaos_monkey_text: str,
    outlier: dict | None,
    mbti_report: dict | None,
) -> dict:
    """
    Build the recalibration signal that tells the MBTI layer which types
    should have their weights adjusted after Critics run.

    If Chaos Monkey found a credible outlier, the MBTI types most aligned
    with that archetype's cognitive style get a weight bump.
    """
    if not outlier or not chaos_monkey_text:
        return {"recalibrate": False}

    chaos_lower = chaos_monkey_text.lower()
    credible = any(
        phrase in chaos_lower
        for phrase in ["credible", "compelling", "correct", "cannot rule out",
                       "strong case", "consistent with", "possible that"]
    )

    if not credible:
        return {"recalibrate": False}

    # Map archetype to MBTI types that share its cognitive style
    ARCHETYPE_MBTI_AFFINITY = {
        "skeptic_debunker":    ["INTP", "ISTP", "INTJ"],
        "institutional_analyst":["ISTJ", "ESTJ"],
        "historian":           ["ISTJ", "INTJ"],
        "economist":           ["INTJ", "ENTJ"],
        "foreign_policy_dove": ["INFJ", "INFP"],
        "idealist_pacifist":   ["INFP", "ENFJ"],
        "pattern_seeker":      ["INTJ", "INTP"],
        "conspiracy_theorist": ["INTP", "INFJ"],
        "cynical_realist":     ["ENTJ", "ENTP"],
        "doomer":              ["INFJ", "INFP"],
    }

    outlier_key  = outlier.get("personality", "")
    affected_types = ARCHETYPE_MBTI_AFFINITY.get(outlier_key, [])

    return {
        "recalibrate":          True,
        "reason":               f"Chaos Monkey found credible case for {outlier.get('personality_name', outlier_key)}",
        "uplift_mbti_types":    affected_types,
        "uplift_factor":        0.15,   # proportional weight increase
        "outlier_archetype":    outlier_key,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def _build_debate_report(
    dominant_label: str,
    dominant_pct: str,
    volatility: float,
    high_octane: bool,
    fact_check: str,
    cynic: str,
    chaos_monkey: str,
    outlier_name: str,
    recalibration_signal: dict,
) -> str:
    lines = [
        "═" * 60,
        "DEBATE LAYER REPORT (L5)",
        "═" * 60,
        "",
        f"Dominant narrative under critique: {dominant_label} ({dominant_pct})",
        f"Volatility Score: {volatility:.3f}  |  "
        f"High-Octane: {'YES 🔥' if high_octane else 'no'}",
        "",
    ]

    if high_octane:
        lines += [
            "⚡ HIGH-OCTANE NARRATIVE",
            "   High probability + survived critique + low drift.",
            "   This story is psychologically armored. Most likely to go viral and persist.",
            "",
        ]

    lines += [
        "── Fact-Checker ───────────────────────────────────────────",
        "",
        fact_check or "[unavailable]",
        "",
        "── Cynic ──────────────────────────────────────────────────",
        "",
        cynic or "[unavailable]",
        "",
        f"── Chaos Monkey (outlier: {outlier_name}) ─────────────────",
        "",
        chaos_monkey or "[unavailable]",
        "",
    ]

    if recalibration_signal.get("recalibrate"):
        lines += [
            "── MBTI Recalibration Signal ───────────────────────────────",
            f"   Uplift types: {recalibration_signal.get('uplift_mbti_types', [])}",
            f"   Reason: {recalibration_signal.get('reason', '')}",
            "",
        ]

    lines += ["═" * 60, "END DEBATE LAYER REPORT", "═" * 60]
    return "\n".join(lines)
