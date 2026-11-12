"""
src/core/forecast_summarizer.py

Grounded Single-Paragraph Executive Forecast Summarizer & Compact 5-Stage Causal
Chain Builder for The Butterfly Algorithm (Phase 6.4).

Produces ONE authoritative structured summary payload per simulation stored on
`FinalVerdict.forecast_summary` (plus `forecast_summary_meta`, `causal_chain_compact`,
and `initial_event_date`) so Backend JSON, Frontend UI, and PDF Page 1 remain 100%
consistent without ever re-running simulation layers during API reads or PDF export.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.core.event_grounding import (
    check_event_relevance,
    classify_event_domain,
    reframe_ungrounded_numbers,
    sanitize_prompt_anchor_leaks,
)


SUMMARY_MIN_WORDS = 120
SUMMARY_MAX_WORDS = 180

_BANNED_BOILERPLATE = (
    "transit lead-time",
    "customs exemption",
    "freight/input",
    "multi-port",
    "retaliatory counter-tariff",
    "procedural waiver",
    "smooth routing",
    "verification norms",
    "workflow restructuring, governance standards",
    "competing benchmark reverses",
    "buffer hoarding compresses margins",
    "pull forward orders",
)

_MONTH_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)


def extract_event_date(signal_text: str) -> Optional[str]:
    """Extracts an explicit event date if mentioned in the input signal, else None."""
    sig = re.sub(r"\s+", " ", (signal_text or "").strip())
    if not sig:
        return None
    patterns = [
        rf"\b({_MONTH_PATTERN}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})\b",
        rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_PATTERN}\.?,?\s+\d{{4}})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        rf"\b({_MONTH_PATTERN}\s+\d{{4}})\b",
    ]
    for pat in patterns:
        m = re.search(pat, sig, flags=re.IGNORECASE)
        if m:
            raw = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".,;")
            return raw
    return None


def _strip_debug_tags(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"^Initial\s+signal\s+text\s+reported:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\[(?:Insufficiency Disclosure|Note|Unverified[^\]]*|Inferred[^\]]*)[^\]]*\]", "", s, flags=re.IGNORECASE)
    s = re.sub(
        r"\[(?:Confirmed / Slot Anchor|Attributed Model Hypothesis[^\]]*|Simulated Layer Telemetry|"
        r"Uncertainty Reduction|Institutional Stabilization|Resilience|Governance|"
        r"Weak Tail Trigger[^\]]*|Single-slot policy risk[^\]]*|Model Hypothesis)\]\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\b(?:Hypothesis|Worst-Case Hypothesis):\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\(Note:\s*benign competition[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\(simulated churn[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\((?:Hypothesis|Tail Hypothesis(?:; unverified)?)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*--\(via [^)]+\)-->\s*", "; ", s)
    s = re.sub(r"\b(?!that\b|had\b|very\b)([A-Za-z]{2,})\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\.\.(?!\.)", ".", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip('"').strip()
    if s.count('"') == 1:
        s = s.replace('"', "")
    return s


def _lowercase_first(text: str) -> str:
    s = _strip_debug_tags(text).strip()
    if not s:
        return ""
    if len(s) >= 2 and s[0].isupper() and not s[:2].isupper():
        first_word = re.split(r"[^A-Za-z0-9\-']", s, maxsplit=1)[0]
        if first_word.lower() not in {
            "nvidia", "jensen", "huang", "deepseek", "openai", "chatgpt", "us", "u.s.", "china",
            "american", "european", "western", "ai", "gpu", "gpus", "sec", "doj", "cfius", "h5n1",
        }:
            return s[0].lower() + s[1:]
    return s


def _complete_clause(text: str, max_words: int = 24) -> str:
    """
    Returns a syntactically complete grammatical clause, splitting ONLY at natural
    clause boundaries (never stranding dangling prepositions, numerals, or adjectives).
    """
    s = _strip_debug_tags(text).strip().rstrip(".,;:")
    if not s:
        return ""
    words = s.split()
    if len(words) <= max_words:
        return s

    for sep_pat in (
        r";\s+",
        r"\s+[—–]\s+",
        r",?\s+(?:while|whereas|potentially|though|unless|particularly|indicating|reflecting|raising|leaving|because)\b",
        r",\s+(?:if|and|but|which|as)\s+",
        r"\s+(?:to\s+prioritize|over\s+raw|capable\s+of|over\s+\d+(?:-\d+)?\s+(?:days|weeks|months|years))\b",
    ):
        parts = re.split(sep_pat, s, maxsplit=1, flags=re.IGNORECASE)
        head = parts[0].strip().rstrip(".,;:")
        if 5 <= len(head.split()) <= max_words:
            return head

    # Prefer a trailing prepositional/clause break rather than cutting an appositive comma or "A, B & C" list
    if "," in s:
        for comma_m in re.finditer(r",\s+", s):
            cand = s[:comma_m.start()].strip()
            tail = s[comma_m.end():]
            prev_tok = (cand.rsplit(" ", 2)[-2:] if cand else [])
            looks_like_series = (
                ("&" in tail[:45] and "," not in tail[:25])
                or any(p.lower() in {"across", "between", "including", "in", "via", "through", "for", "of"} for p in prev_tok[:1])
            )
            if looks_like_series:
                continue
            if 7 <= len(cand.split()) <= max_words:
                return cand

    clipped = " ".join(words[:max_words]).rstrip(",;:-—")
    dangling = {
        "and", "or", "the", "a", "an", "of", "to", "for", "with", "in", "on", "by",
        "as", "that", "if", "at", "from", "has", "have", "had", "is", "are", "was",
        "were", "will", "would", "could", "should", "may", "can", "into", "over",
        "under", "between", "capable", "large", "small", "significant", "many", "several",
        "raw", "gradual", "uneven", "durable", "operational", "private", "public", "new",
    }
    toks = clipped.split()
    while len(toks) > 5 and (
        toks[-1].lower().strip(",;:") in dangling or re.fullmatch(r"\d+(?:[-–]\d+)?", toks[-1])
    ):
        toks.pop()
    # If chopping stranded the tail of a trailing "over ..." / "by ..." prepositional start, drop trailing preposition too
    while len(toks) > 5 and toks[-1].lower() in {"over", "by", "of", "for", "with", "in", "to", "from", "into"}:
        toks.pop()
    return " ".join(toks).rstrip(",;:-—")


def _clip_chars(text: str, max_chars: int) -> str:
    s = _strip_debug_tags(text).strip()
    if len(s) <= max_chars:
        return s
    cut = s[: max_chars - 3].rsplit(" ", 1)[0].rstrip(",;:-—.")
    return (cut or s[: max_chars - 3]) + "..."


def _short_stakeholder_list(stakeholders: List[str], limit: int = 3) -> str:
    cleaned: List[str] = []
    for raw in stakeholders:
        s = re.sub(r"\s*\([^)]+\)", "", str(raw or "")).strip()
        if not s:
            continue
        if len(s) > 36 and "," in s and "&" in s:
            s = re.sub(r",\s*[^,&]+\s+(?=&)", " ", s, count=1).strip()
        elif "," in s and "&" not in s:
            s = s.split(",", 1)[0].strip()
        if "&" in s and len(s) > 36:
            s = s.split("&", 1)[0].strip()
        if len(s.split()) > 5:
            s = _complete_clause(s, max_words=4)
        if s and s.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(s)
        if len(cleaned) >= limit:
            break
    if not cleaned:
        return "key institutional and market stakeholders"
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{cleaned[0]}, {cleaned[1]}, and {cleaned[2]}"


def _is_usable_clause(text: str, domain_ctx: Dict[str, Any], signal_text: str) -> Tuple[bool, str]:
    s = _strip_debug_tags(text)
    if not s or "insufficient grounded" in s.lower() or "unavailable" in s.lower():
        return False, ""
    reframed, _ = reframe_ungrounded_numbers(s, signal_text)
    reframed = sanitize_prompt_anchor_leaks(reframed, domain_ctx)
    ok, _ = check_event_relevance(reframed, domain_ctx, allow_bridged=True)
    if not ok:
        return False, ""
    low = reframed.lower()
    if any(bad in low for bad in _BANNED_BOILERPLATE):
        return False, ""
    return True, _strip_debug_tags(reframed).rstrip(".")


def _format_stabilizer_clause(clause: str) -> str:
    """Formats a stabilizer condition grammatically as either 'if <verb clause>' or 'with <noun phrase>'."""
    s = _strip_debug_tags(clause).strip().rstrip(".")
    low = s.lower()
    if not s:
        return "if independent verification contains escalation"
    rewrites = [
        (r"^verification\s+that\s+", "if independent audits confirm that "),
        (r"^verification\s+of\s+", "with independent verification of "),
        (r"^confirmation\s+or\s+denial\s+of\s+", "with clear official findings regarding "),
        (r"^confirmation\s+that\s+", "if official findings confirm that "),
        (r"^publication\s+of\s+", "if institutions publish "),
        (r"^release\s+of\s+", "if authorities release "),
        (r"^ratio\s+of\s+", "if durable adoption outlasts novelty in "),
    ]
    for pat, rep in rewrites:
        if re.match(pat, low):
            s = re.sub(pat, rep, s, count=1, flags=re.IGNORECASE)
            comp = _complete_clause(s, 14)
            return comp if comp.lower().startswith(("if ", "with ", "through ")) else f"with {comp}"
    if any(
        re.search(rf"\b{v}\b", low)
        for v in (
            "confirm", "clarify", "publish", "release", "hold", "holds", "verify", "stabiliz",
            "remain", "declin", "resolve", "halt", "neutraliz", "adjust", "maintain", "contain",
            "comply", "cooperate", "adopt", "respond", "shift", "reduce", "limit",
        )
    ):
        return f"if {_lowercase_first(_complete_clause(s, 13))}"
    return f"with {_lowercase_first(_complete_clause(s, 12))}"


def build_compact_causal_chain(
    *,
    signal_text: str,
    confirmed_facts: List[str],
    q1_raw: str,
    q2_second_raw: str,
    q2_third_raw: str,
    order_effects: Dict[str, Any],
    leading_scenario: Dict[str, Any],
    domain_ctx: Dict[str, Any],
    is_failed: bool = False,
) -> Dict[str, str]:
    """
    Constructs a concise 5-stage causal chain for PDF Page 1:
      initial_event -> immediate_reaction -> secondary_consequence
      -> third_order_effect -> possible_systemic_outcome
    """
    first_sig = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", (signal_text or "").strip()), maxsplit=1)[0]
    raw_anchor = _strip_debug_tags(confirmed_facts[0] if confirmed_facts else first_sig)
    if len(raw_anchor) > 95:
        raw_anchor = _strip_debug_tags(first_sig)
    initial_event = _clip_chars(_complete_clause(raw_anchor, 15), 94)

    if is_failed:
        return {
            "initial_event": initial_event or "Reported initial event",
            "immediate_reaction": "Unavailable — model execution failed before Micro interpretations completed",
            "secondary_consequence": "Unavailable — requires at least one valid simulation cycle",
            "third_order_effect": "Unavailable — unmodeled without Micro/Meso/Macro telemetry",
            "possible_systemic_outcome": "Forecast withheld until valid model execution is restored",
        }

    imm_cands = [q1_raw] + list(order_effects.get("first_order") or [])
    imm_clean = ""
    for cand in imm_cands:
        ok, cleaned = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and len(cleaned) >= 12:
            imm_clean = cleaned
            break
    immediate_reaction = _clip_chars(_complete_clause(imm_clean or "Early institutional scrutiny and stakeholder reassessment", 14), 92)

    sec_cands = [q2_second_raw] + list(order_effects.get("second_order") or []) + list(leading_scenario.get("medium_term_consequences") or [])
    sec_clean = ""
    for cand in sec_cands:
        ok, cleaned = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and len(cleaned) >= 12 and cleaned.lower() != imm_clean.lower():
            sec_clean = cleaned
            break
    secondary_consequence = _clip_chars(_complete_clause(sec_clean or "Stakeholder hedging and partner due-diligence adjustments", 14), 92)

    third_cands = [q2_third_raw] + list(order_effects.get("third_order_and_systemic") or []) + list(leading_scenario.get("long_term_consequences") or [])
    third_clean = ""
    for cand in third_cands:
        ok, cleaned = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and len(cleaned) >= 12 and cleaned.lower() not in {imm_clean.lower(), sec_clean.lower()}:
            third_clean = cleaned
            break
    third_order_effect = _clip_chars(_complete_clause(third_clean or "Structural recalibration of institutional controls and market posture", 14), 92)

    lead_type = str(leading_scenario.get("scenario_type") or "baseline").lower()
    long_list = list(leading_scenario.get("long_term_consequences") or []) + [str(leading_scenario.get("summary") or "")]
    sys_clean = ""
    for cand in long_list:
        ok, cleaned = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and len(cleaned) >= 12:
            sys_clean = cleaned
            break
    if not sys_clean:
        sys_clean = f"{lead_type.capitalize()} trajectory yields gradual adaptation unless unresolved risks compound"
    possible_systemic_outcome = _clip_chars(_complete_clause(sys_clean, 15), 95)

    return {
        "initial_event": initial_event,
        "immediate_reaction": immediate_reaction,
        "secondary_consequence": secondary_consequence,
        "third_order_effect": third_order_effect,
        "possible_systemic_outcome": possible_systemic_outcome,
    }


def _extract_leading_focus(leading_scenario: Dict[str, Any], domain_ctx: Dict[str, Any], signal_text: str) -> str:
    raw_sum = str(leading_scenario.get("summary") or "").strip()
    ok, cleaned = _is_usable_clause(raw_sum, domain_ctx, signal_text)
    if ok and cleaned:
        comp = _complete_clause(cleaned, 22)
        if comp:
            return _lowercase_first(comp)
    stype = str(leading_scenario.get("scenario_type") or "baseline").lower()
    if stype == "positive":
        return "transparent resolution and contained operational spillover"
    if stype == "negative":
        return "sustained partner hedging and elevated compliance friction"
    if stype == "extreme":
        return "severe tail vulnerability should primary safeguards fail"
    return "contained near-term spillover and gradual institutional adaptation"


def _extract_uncertainty_clause(
    gaps_classified: Dict[str, List[str]],
    domain_ctx: Dict[str, Any],
    signal_text: str,
    is_partial: bool = False,
) -> str:
    ms_gaps = gaps_classified.get("missing_source_evidence") or []
    ul_gaps = gaps_classified.get("unsupported_causal_links") or []
    md_gaps = gaps_classified.get("model_disagreement") or []

    domain = domain_ctx.get("domain", "general")
    subj = domain_ctx.get("org_entity") or domain_ctx.get("subject_label") or "the focal event"
    no_url = any("no external primary source url" in g.lower() for g in ms_gaps)
    has_unsup = any("unsupported" in g.lower() or "missing intermediate" in g.lower() for g in ul_gaps)
    has_reframed = any("ungrounded quantitative" in g.lower() for g in ms_gaps)
    has_div = any("low inter-model" in g.lower() or "agreement scored" in g.lower() or "divergent" in g.lower() for g in md_gaps)

    if is_partial:
        return "incomplete model-slot coverage alongside unverified primary evidence outside the prompt"
    if domain == "corporate_governance":
        if no_url:
            return f"the absence of primary inquiry disclosures separating reported allegations from verified breach at {subj}"
        return f"whether official inquiry findings confirm or refute underlying allegations around {subj}"
    if domain == "ai_software":
        if no_url and has_reframed:
            return "uncorroborated capability and adoption economics lacking primary independent workload audits"
        if no_url:
            return f"whether headline claims around {subj} hold under independent enterprise workload audits"
    if no_url and has_unsup:
        return "the lack of primary source corroboration and unverified multi-hop longer-horizon assumptions"
    if no_url:
        return f"the absence of verified external source evidence beyond the prompt on {subj}"
    if has_div:
        return "divergent causal mechanisms across models paired with unverified longer-horizon assumptions"
    if ul_gaps:
        ok, cleaned = _is_usable_clause(ul_gaps[0], domain_ctx, signal_text)
        if ok and cleaned:
            return _lowercase_first(_complete_clause(cleaned, 14))
    return "unverified longer-horizon assumptions requiring primary monitoring"


def compress_summary_for_pdf(summary: str, max_words: int = 160) -> str:
    """
    Deterministically tightens phrase redundancy cleanly at phrase/clause level without
    ever truncating sentences mid-clause so meaning and epistemic disclosures remain intact.
    """
    s = _strip_debug_tags(summary)
    if len(s.split()) <= max_words:
        return s

    tighten_rules = [
        (", which is a heuristic index rather than an empirical probability", " (heuristic weight, non-empirical)"),
        (", not an empirical probability)", ", non-empirical)"),
        ("Across the stakeholder network, ", "Across stakeholders, "),
        ("while awaiting independent verification", "pending audit"),
        ("while downstream claims await verification", "with downstream claims unverified"),
        ("Limited long-term impact remains plausible ", "Limited lasting impact stays plausible "),
        ("the decisive uncertainty is ", "the key uncertainty is "),
        ("temporarily outpaces durable operational shifts", "outpaces retained operational shifts"),
    ]
    for old, new in tighten_rules:
        if len(s.split()) <= max_words:
            break
        s = s.replace(old, new)

    regex_tighten = [
        (r"^Reporting on [^:]+ establishes that ", "Reportedly, "),
        (r"\s+as institutional moderation tempers escalation", ""),
        (r"\s+as heightened amplification tests governance safeguards", ""),
        (r",?\s*while third-order change depends on verification", ""),
        (r"\s+outpaces retained operational shifts", " outpaces operational shifts"),
        (r"\s+with downstream claims unverified", ""),
        (r"Across stakeholders,\s+[^.]+?\s+adjust posture", "Across stakeholders, key groups adjust posture"),
    ]
    for pat, rep in regex_tighten:
        if len(s.split()) <= max_words:
            break
        s = re.sub(pat, rep, s, count=1, flags=re.IGNORECASE)

    if len(s.split()) > max_words:
        s = re.sub(r"\s+(?:indicating|where)\s+contained near-term spillover and gradual adaptation", "", s, count=1, flags=re.IGNORECASE)
    if len(s.split()) > max_words:
        s = re.sub(r",\s*whereas [^.;]+ could widen disruption", ", barring verified safety or compliance failures", s, count=1, flags=re.IGNORECASE)

    return _strip_debug_tags(s)


def validate_summary_text(
    summary: str,
    *,
    signal_text: str,
    domain_ctx: Dict[str, Any],
    leading_weight_pct: Optional[int],
    is_failed: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Validates strict quality, grounding, single-paragraph formatting, and 120-180
    word target for `forecast_summary`.
    """
    raw = str(summary or "").strip()
    if not raw:
        return False, "empty_summary"
    if "\n\n" in raw or re.search(r"(?m)^\s*[-*•]\s+", raw) or re.search(r"(?m)^#{1,6}\s+", raw):
        return False, "not_single_cohesive_paragraph"
    if "insufficient grounded" in raw.lower():
        return False, "contains_placeholder_text"

    words = raw.split()
    if len(words) < SUMMARY_MIN_WORDS or len(words) > SUMMARY_MAX_WORDS:
        return False, f"word_count_out_of_range:{len(words)}"

    low = raw.lower()
    for bad in _BANNED_BOILERPLATE:
        if bad in low:
            return False, f"banned_boilerplate:{bad}"

    ok_rel, rel_reason = check_event_relevance(raw, domain_ctx, allow_bridged=True)
    if not ok_rel:
        return False, f"off_domain:{rel_reason}"

    check_text = raw
    if leading_weight_pct is not None:
        check_text = re.sub(rf"\b{leading_weight_pct}(?:\.\d+)?\s*%", "HEURISTIC_WEIGHT", check_text)
    _, num_note = reframe_ungrounded_numbers(check_text, signal_text)
    if num_note is not None:
        return False, f"ungrounded_numeric_projection:{num_note}"

    if not is_failed and leading_weight_pct is not None:
        if f"{leading_weight_pct}%" not in raw and f"{leading_weight_pct}.0%" not in raw:
            return False, "missing_leading_scenario_weight"

    if not any(tok in low for tok in ("uncertain", "unverified", "unconfirmed", "gap", "assumption", "corroborat")):
        return False, "missing_uncertainty_or_evidence_distinction"

    return True, None


def build_extractive_summary(
    *,
    event_title: str,
    signal_text: str,
    initial_event_date: Optional[str],
    confirmed_facts: List[str],
    unverified_allegations: List[str],
    q1_immediate: str,
    q2_second: str,
    q2_third: str,
    q3_downside: str,
    q4_stabilizing: str,
    stakeholder_reactions: List[Dict[str, Any]],
    order_effects: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    leading_scenario: Dict[str, Any],
    gaps_classified: Dict[str, List[str]],
    triggers_classified: Dict[str, List[str]],
    macro_state: Dict[str, Any],
    feedback_state: Dict[str, Any],
    domain_ctx: Dict[str, Any],
    model_names: Tuple[str, str],
    valid_slots: int,
    expected_slots: int,
    is_partial: bool,
    is_failed: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Synthesizes a 130-165 word cohesive, grammatical analyst paragraph summarizing
    all four simulation layers and answering all 7 executive forecast questions.
    """
    lead_label = str(leading_scenario.get("scenario_type") or "baseline").capitalize()
    lead_w = leading_scenario.get("conditional_weight")
    lead_pct = int(round(float(lead_w) * 100)) if lead_w is not None else None
    event_label = _complete_clause(event_title or signal_text, 7)
    # Remove leading preposition/date fragments when event_title comes from first words of signal
    event_label_clean = re.sub(r"^(?:On|In|At)\s+", "", event_label, flags=re.IGNORECASE).strip()

    if is_failed or valid_slots <= 0:
        first_sig = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", (signal_text or "").strip()), maxsplit=1)[0]
        anchor_clean = _complete_clause(_strip_debug_tags(confirmed_facts[0] if confirmed_facts else first_sig), 18)
        m_a, m_b = model_names
        failed_para = (
            f"The initial event report concerning {event_label_clean} describes unverified developments "
            f"({anchor_clean}), but this simulation run finished with 0 of {expected_slots} valid model interpretations "
            f"across {m_a} and {m_b}, halting downstream network propagation, Bass diffusion modeling, and iterative feedback. "
            f"Because no valid behavioral interpretations completed, immediate stakeholder reactions, secondary and third-order "
            f"consequences, leading scenario trajectory, and heuristic scenario weights remain unavailable rather than reliably estimated. "
            f"Under healthy execution, stabilizing or limited long-term outcomes would be weighed against adverse tail scenarios based on "
            f"verified institutional responses; at present, neither escalation nor containment can be concluded from simulation outputs. "
            f"The decisive uncertainty blocking this forecast is complete model execution failure paired with uncorroborated primary claims "
            f"that require restored model connectivity and verified source evidence before drawing conclusions."
        )
        failed_para = _strip_debug_tags(failed_para)
        if len(failed_para.split()) < SUMMARY_MIN_WORDS:
            failed_para += " All numerical probability bands remain withheld to preserve strict methodological transparency."
        return failed_para, {
            "method": "deterministic_honest_failure_summary",
            "fallback_used": False,
            "fallback_reason": None,
            "word_count": len(failed_para.split()),
            "leading_scenario": None,
            "leading_weight": None,
            "limited_impact_plausible": False,
            "uncertainty_disclosure": "Model execution failed (0 valid slots); no forecast scenarios synthesized.",
        }

    # Part 1: Confirmed facts vs unverified claims + immediate reaction
    first_sig = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", (signal_text or "").strip()), maxsplit=1)[0]
    raw_fact = str(confirmed_facts[0] if confirmed_facts else "")
    if "..." in raw_fact or raw_fact.lower().startswith("initial signal text reported"):
        raw_conf = _strip_debug_tags(first_sig)
    else:
        raw_conf = _strip_debug_tags(raw_fact or first_sig)
    if len(raw_conf.split()) > 30:
        raw_conf = _strip_debug_tags(first_sig)
    ok_conf, conf_clause = _is_usable_clause(raw_conf, domain_ctx, signal_text)
    conf_clean = _complete_clause(conf_clause if (ok_conf and conf_clause) else first_sig, 30)

    has_real_allegation = bool(
        unverified_allegations
        and not any("none isolated" in u.lower() for u in unverified_allegations)
    )
    domain = domain_ctx.get("domain", "general")
    subj = domain_ctx.get("org_entity") or domain_ctx.get("person_entity") or event_label_clean

    imm_cands = [q1_immediate] + list(order_effects.get("first_order") or [])
    imm_clause = ""
    for cand in imm_cands:
        ok, cl = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and cl:
            imm_clause = _lowercase_first(_complete_clause(cl, 16))
            break
    if not imm_clause:
        imm_clause = f"early scrutiny and precautionary operational reviews begin around {subj}"
    if re.match(r"^(?:issue|publish|conduct|audit|verify|initiate|clarify|review|establish|document|deploy|activate|enforce|accelerate)\b", imm_clause):
        imm_clause = f"authorities and stakeholders move to {imm_clause}"

    def _make_s1(compact: bool = False) -> str:
        c_txt = _complete_clause(event_label_clean if compact else conf_clean, 14 if compact else 25)
        if domain == "corporate_governance" and has_real_allegation:
            return (
                f"Confirmed reporting shows that {_lowercase_first(c_txt)} "
                f"while leaving underlying allegations unverified; immediately, {imm_clause}."
            )
        if has_real_allegation and not compact:
            unv_cl = _lowercase_first(_complete_clause(unverified_allegations[0], 10))
            if "," in unv_cl and not re.search(r",\s*(?:and|or)\b", unv_cl):
                unv_cl = unv_cl.rsplit(",", 1)[0].strip()
            verb_unv = "remain" if re.search(r"\b(?:dependencies|claims|allegations|risks|costs|effects|transfers|impacts|shortages|pressures)\b", unv_cl.lower()) else "remains"
            return (
                f"Reporting on {event_label_clean} establishes that {_lowercase_first(c_txt)} "
                f"while {unv_cl} {verb_unv} unverified; immediately, {imm_clause}."
            )
        return (
            f"Available reporting establishes that {_lowercase_first(c_txt)} "
            f"with downstream claims awaiting verification; immediately, {imm_clause}."
        )

    s1 = _make_s1(compact=False)

    # Part 2: Meso stakeholders + Macro salience vs adoption dynamic
    stake_names = [str(sr.get("stakeholder") or "") for sr in stakeholder_reactions]
    stake_str = _short_stakeholder_list(stake_names, limit=3)
    attn_val = float(macro_state.get("attention_indicator", 0.0) or 0.0)
    act_val = float(macro_state.get("active_adoption", 0.0) or 0.0)
    if attn_val >= max(0.25, act_val * 2.0):
        macro_plain = "as public and media attention outpaces retained operational shifts"
    else:
        macro_plain = "translating peer network signaling into measured behavioral shifts"
    s2 = f"Across stakeholders, {stake_str} adjust posture {macro_plain}."

    # Part 3: 2nd- and 3rd-order consequences
    sec_cands = [q2_second] + list(order_effects.get("second_order") or []) + list(leading_scenario.get("medium_term_consequences") or [])
    sec_clause = ""
    for cand in sec_cands:
        ok, cl = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and cl and cl.lower() != imm_clause.lower():
            sec_clause = _lowercase_first(_complete_clause(cl, 16))
            break
    third_cands = [q2_third] + list(order_effects.get("third_order_and_systemic") or []) + list(leading_scenario.get("long_term_consequences") or [])
    third_clause = ""
    for cand in third_cands:
        if re.search(r"simulated.*\d+(?:\.\d+)?%", str(cand or ""), re.IGNORECASE):
            continue
        ok, cl = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and cl and cl.lower() not in {imm_clause.lower(), sec_clause.lower()}:
            third_clause = _lowercase_first(_complete_clause(cl, 16))
            break

    def _make_s3(compact: bool = False) -> str:
        head_sec = re.split(r"\b(?:if|when|while)\b", sec_clause.lower(), maxsplit=1)[0]
        is_noun_start = bool(
            sec_clause
            and re.match(
                r"^(?:potential|accelerated|increased|stagnation|decline|rise|surge|growth|reduced|wider|extended|persistent)\b",
                sec_clause.lower(),
            )
        )
        finite_sec = bool(
            sec_clause
            and not is_noun_start
            and re.search(
                r"\b(?:compresses|settles|remains|triggers|forces|creates|prompts|widens|accelerates|drives|shifts|shapes|spurs|resets)\b",
                head_sec,
            )
        )
        third_is_finite = bool(
            third_clause
            and re.search(r"\b(?:settles|remains|solidif|shifts|stabilizes|controls)\b", third_clause.lower())
        )
        if sec_clause and third_clause and not compact:
            if finite_sec:
                conn = "while" if third_is_finite else "potentially leading toward"
                return f"Over subsequent months, {sec_clause}, {conn} {third_clause}."
            conn = "while longer-term projections suggest" if third_is_finite else "potentially leading toward"
            return f"Over subsequent months, second-order pressure favors {sec_clause}, {conn} {third_clause}."
        if sec_clause:
            if finite_sec:
                return f"Over subsequent months, {sec_clause}, while third-order change depends on verification."
            return f"Over subsequent months, secondary spillover centers on {sec_clause}, while third-order change depends on verification."
        return "Over subsequent months, stakeholders balance defensive hedging against continuity as longer-term structure remains unconfirmed."

    s3 = _make_s3(compact=False)

    # Part 4: Leading scenario, heuristic weight & Feedback systemic balance
    focus_clause = _extract_leading_focus(leading_scenario, domain_ctx, signal_text)
    sys_st = (feedback_state.get("systemic_state") or {}) if isinstance(feedback_state, dict) else {}
    inst_damp = float(sys_st.get("institutional_posture_dampening", 0.45) or 0.45)
    esc_proxy = float(sys_st.get("escalation_risk", 0.30) or 0.30)
    partial_note = f" ({valid_slots}/{expected_slots} slots)" if is_partial else ""
    if inst_damp >= esc_proxy:
        bal_note = "as institutional moderation tempers escalation"
    else:
        bal_note = "as heightened amplification tests governance safeguards"

    lead_type_low = str(leading_scenario.get("scenario_type") or "baseline").lower()
    compact_focus_map = {
        "baseline": "indicating contained near-term spillover and gradual adaptation",
        "positive": "indicating transparent resolution and constructive containment",
        "negative": "indicating sustained partner hedging and elevated friction",
        "extreme": "indicating severe tail sensitivity if safeguards fail",
    }

    def _make_s4(compact: bool = False) -> str:
        if compact:
            c_focus = compact_focus_map.get(lead_type_low, "indicating contained spillover")
            return f"The {lead_label} trajectory leads{partial_note} ({lead_pct}% heuristic weight, non-empirical), {c_focus} {bal_note}."
        f_txt = _complete_clause(focus_clause, 24)
        if f_txt.lower().startswith("during "):
            return f"The {lead_label} trajectory leads{partial_note} ({lead_pct}% heuristic weight, not an empirical probability), where {f_txt} {bal_note}."
        return (
            f"The {lead_label} trajectory leads{partial_note} ({lead_pct}% heuristic weight, not an empirical probability): "
            f"{f_txt}, {bal_note}."
        )

    s4 = _make_s4(compact=False)

    # Part 5: Better/worse + limited long-term impact + biggest uncertainty
    stab_cands = (
        list(triggers_classified.get("stabilization") or [])
        + list(triggers_classified.get("reversal") or [])
        + list(order_effects.get("prevention_and_reversal_conditions") or [])
        + [q4_stabilizing]
    )
    stab_raw = ""
    for cand in stab_cands:
        ok, cl = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and cl:
            stab_raw = cl
            break
    stab_phrase = _format_stabilizer_clause(stab_raw)

    esc_cands = list(triggers_classified.get("escalation") or []) + [q3_downside]
    esc_clause = ""
    for cand in esc_cands:
        ok, cl = _is_usable_clause(cand, domain_ctx, signal_text)
        if ok and cl:
            esc_clause = _lowercase_first(_complete_clause(cl, 13))
            break
    if not esc_clause:
        esc_clause = "substantiated adverse disclosures or oversight failures"

    unc_clause = _extract_uncertainty_clause(gaps_classified, domain_ctx, signal_text, is_partial=is_partial)

    s5 = (
        f"Limited long-term impact remains plausible {stab_phrase}, whereas {esc_clause} could widen disruption; "
        f"the decisive uncertainty is {unc_clause}."
    )

    para = _strip_debug_tags(" ".join([s1, s2, s3, s4, s5]))
    # Complete-sentence tightening passes if > 162 words (target 135-160 words for identical Website + PDF Page 1 fit)
    if len(para.split()) > 162:
        para = _strip_debug_tags(" ".join([s1, s2, _make_s3(compact=True), s4, s5]))
    if len(para.split()) > 162:
        para = _strip_debug_tags(" ".join([s1, s2, _make_s3(compact=True), _make_s4(compact=True), s5]))
    if len(para.split()) > 168:
        para = _strip_debug_tags(" ".join([_make_s1(compact=True), s2, _make_s3(compact=True), _make_s4(compact=True), s5]))
    if len(para.split()) > SUMMARY_MAX_WORDS:
        para = compress_summary_for_pdf(para, max_words=165)

    wc = len(para.split())
    if wc < SUMMARY_MIN_WORDS:
        add_sent = (
            "Heuristic weights synthesize five behavioral archetypes and eight network channels "
            "rather than empirically calibrated statistical forecasts."
        )
        para = _strip_debug_tags(f"{para} {add_sent}")
        wc = len(para.split())
        if wc < SUMMARY_MIN_WORDS:
            pos_sc = next((sc for sc in scenarios if sc.get("scenario_type") == "positive"), {})
            ext_sc = next((sc for sc in scenarios if sc.get("scenario_type") == "extreme"), {})
            pw = int(round(float(pos_sc.get("conditional_weight") or 0.0) * 100))
            ew = int(round(float(ext_sc.get("conditional_weight") or 0.0) * 100))
            para = _strip_debug_tags(
                f"{para} Alternative Positive ({pw}% weight) and Extreme ({ew}% weight) "
                f"branches remain contingent on incoming evidence."
            )

    wc = len(para.split())
    limited_plausible = str(leading_scenario.get("scenario_type") or "").lower() in ("baseline", "positive")

    return para, {
        "method": "deterministic_grounded_extractive",
        "fallback_used": False,
        "fallback_reason": None,
        "word_count": wc,
        "leading_scenario": leading_scenario.get("scenario_type"),
        "leading_weight": leading_scenario.get("conditional_weight"),
        "limited_impact_plausible": limited_plausible,
        "uncertainty_disclosure": unc_clause,
    }


def polish_summary_with_ollama(
    digest: Dict[str, Any],
    *,
    model_name: str,
    base_url: Optional[str] = None,
    timeout_sec: float = 18.0,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Optional bounded Ollama single-paragraph executive summarizer.
    Never invents external facts and NEVER substitutes Mock inference.
    Returns (polished_summary_or_None, error_reason_or_None).
    """
    host = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    prompt = (
        "You are an intelligence and risk forecasting analyst. Write ONE single cohesive paragraph of "
        "130 to 165 words synthesizing this completed four-layer simulation digest. "
        "Include ONLY facts, entities, and heuristic numbers present in the digest. "
        "Answer all seven points seamlessly in fluent prose (NO bullet points, NO headings, NO metric-list dumps like A(t)/N(t)): "
        "(1) what happened distinguishing confirmed facts from unverified claims, "
        "(2) most likely immediate reaction, (3) key stakeholder reactions, (4) plausible second- and third-order consequences, "
        "(5) leading scenario name and exact heuristic weight percentage explicitly labeled non-empirical, "
        "(6) what could improve or worsen outcomes including whether limited long-term impact is plausible, and "
        "(7) the biggest remaining uncertainty or missing evidence.\n\n"
        f"SIMULATION DIGEST:\n{digest}\n\n"
        "Return ONLY the single paragraph."
    )
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.25,
                    "num_predict": 320,
                },
            },
            timeout=(4.0, timeout_sec),
        )
        if resp.status_code != 200:
            return None, f"ollama_http_{resp.status_code}"
        data = resp.json()
        raw = str(data.get("response") or "").strip()
        raw = re.sub(r"^\*\*[^*]+\*\*\s*", "", raw).strip().strip('"').strip()
        raw = _strip_debug_tags(raw)
        return raw, None
    except Exception as exc:
        return None, f"ollama_exception:{type(exc).__name__}"


def synthesize_forecast_summary_and_chain(
    *,
    event_title: str,
    signal_text: str,
    confirmed_facts: List[str],
    unverified_allegations: List[str],
    q1_immediate: str,
    q2_second: str,
    q2_third: str,
    q3_downside: str,
    q4_stabilizing: str,
    stakeholder_reactions: List[Dict[str, Any]],
    order_effects: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    leading_scenario: Dict[str, Any],
    gaps_classified: Dict[str, List[str]],
    triggers_classified: Dict[str, List[str]],
    macro_state: Dict[str, Any],
    feedback_state: Dict[str, Any],
    domain_ctx: Optional[Dict[str, Any]] = None,
    model_names: Tuple[str, str] = ("Model A", "Model B"),
    provider_names: Tuple[str, str] = ("unknown", "unknown"),
    valid_slots: int = 0,
    expected_slots: int = 0,
    is_partial: bool = False,
    is_failed: bool = False,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Single authoritative entry point producing:
      - `initial_event_date`
      - `causal_chain_compact` (5-stage PDF Page 1 causal strip)
      - `forecast_summary` (120-180 words, single paragraph)
      - `forecast_summary_meta` (method, fallback tracking, uncertainty, word count)
    """
    ctx = domain_ctx or classify_event_domain(signal_text)
    initial_date = extract_event_date(signal_text)
    lead_w = leading_scenario.get("conditional_weight")
    lead_pct = int(round(float(lead_w) * 100)) if lead_w is not None else None

    compact_chain = build_compact_causal_chain(
        signal_text=signal_text,
        confirmed_facts=confirmed_facts,
        q1_raw=q1_immediate,
        q2_second_raw=q2_second,
        q2_third_raw=q2_third,
        order_effects=order_effects,
        leading_scenario=leading_scenario,
        domain_ctx=ctx,
        is_failed=is_failed,
    )

    extractive_summary, meta = build_extractive_summary(
        event_title=event_title,
        signal_text=signal_text,
        initial_event_date=initial_date,
        confirmed_facts=confirmed_facts,
        unverified_allegations=unverified_allegations,
        q1_immediate=q1_immediate,
        q2_second=q2_second,
        q2_third=q2_third,
        q3_downside=q3_downside,
        q4_stabilizing=q4_stabilizing,
        stakeholder_reactions=stakeholder_reactions,
        order_effects=order_effects,
        scenarios=scenarios,
        leading_scenario=leading_scenario,
        gaps_classified=gaps_classified,
        triggers_classified=triggers_classified,
        macro_state=macro_state,
        feedback_state=feedback_state,
        domain_ctx=ctx,
        model_names=model_names,
        valid_slots=valid_slots,
        expected_slots=expected_slots,
        is_partial=is_partial,
        is_failed=is_failed,
    )

    final_summary = extractive_summary

    can_use_ollama = (
        use_llm
        and not is_failed
        and valid_slots > 0
        and any(str(p).lower() == "ollama" for p in provider_names)
    )
    if can_use_ollama:
        chosen_idx = next(i for i, p in enumerate(provider_names) if str(p).lower() == "ollama")
        ollama_model = model_names[chosen_idx]
        digest = {
            "event_title": event_title,
            "initial_event_date": initial_date or "Not specified",
            "confirmed_fact": confirmed_facts[:1],
            "unverified_allegation": unverified_allegations[:1],
            "immediate_reaction": compact_chain["immediate_reaction"],
            "stakeholders": [sr.get("stakeholder") for sr in stakeholder_reactions[:3]],
            "secondary_consequence": compact_chain["secondary_consequence"],
            "third_order_effect": compact_chain["third_order_effect"],
            "leading_scenario": str(leading_scenario.get("scenario_type") or "baseline").capitalize(),
            "leading_weight_pct": lead_pct,
            "stabilizer": (triggers_classified.get("stabilization") or triggers_classified.get("reversal") or [])[:1],
            "downside_risk": (triggers_classified.get("escalation") or [])[:1],
            "decisive_uncertainty": meta.get("uncertainty_disclosure"),
        }
        cand, err = polish_summary_with_ollama(digest, model_name=ollama_model)
        if cand:
            ok_val, val_err = validate_summary_text(
                cand,
                signal_text=signal_text,
                domain_ctx=ctx,
                leading_weight_pct=lead_pct,
                is_failed=is_failed,
            )
            if ok_val:
                final_summary = cand
                meta["method"] = f"ollama_grounded_polish:{ollama_model}"
                meta["word_count"] = len(cand.split())
                meta["fallback_used"] = False
                meta["fallback_reason"] = None
            else:
                meta["fallback_used"] = True
                meta["fallback_reason"] = f"ollama_output_rejected:{val_err}"
        else:
            meta["fallback_used"] = True
            meta["fallback_reason"] = err or "ollama_empty_response"

    return {
        "forecast_summary": final_summary,
        "forecast_summary_meta": meta,
        "causal_chain_compact": compact_chain,
        "initial_event_date": initial_date,
    }
