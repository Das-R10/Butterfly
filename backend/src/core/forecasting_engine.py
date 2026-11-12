"""
src/core/forecasting_engine.py

Causal Butterfly-Effect Forecasting & Final Verdict Engine.
Synthesizes coupled Micro (5 archetypes x 2 models causal hypotheses) ->
Meso (8-node propagation & inflows) -> Macro (Bass N(t)/A(t)/attention) ->
Feedback (systemic state evolution & claim audit) telemetry into:

  1. Competing Multi-Order Causal Pathways (1st -> 2nd -> 3rd -> Systemic -> Feedback)
     with automated weak / unsupported / multi-assumption causal link flagging.
  2. Stakeholder Reaction & Impact Profiles tied to Meso activations.
  3. Four Distinct Future Scenarios (Baseline, Positive, Negative, Extreme) with
     triggers, evidence vs counter-evidence, indicators to monitor, and heuristic
     conditional probability bands clearly distinguished from statistical calibration.
  4. Time-Horizon Forecast (Immediate Days, Short-Term Weeks, Medium-Term Months, Long-Term Quarters)
     plus low-probability / high-impact tail risks & reversal conditions.
  5. Six-Tier Epistemic Breakdown strictly separating confirmed facts from claims,
     unverified allegations, model interpretations, simulated states, and hypothetical futures.
  6. Structured 8-Question Final Verdict with full simulation traceability and honest
     degradation under PARTIALLY COMPLETED or FAILED runs.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.schemas import (
    Archetype,
    CORE_RESEARCH_QUESTION,
    CausalPathway,
    CausalStep,
    EpistemicBreakdown,
    FinalVerdict,
    FutureScenario,
    ModelExecutionStatus,
    OrderEffectsSummary,
    StakeholderReaction,
    TimeHorizonForecast,
)
from src.core.event_grounding import (
    check_event_relevance,
    classify_event_domain,
    classify_trigger_and_condition,
    enforce_horizon_alignment,
    extract_event_entities,
    extract_event_stakeholders,
    extract_horizon_and_strip_subtag,
    is_completed_premise_action,
    is_reassuring_or_stabilizing,
    reframe_ungrounded_numbers,
    sanitize_prompt_anchor_leaks,
)
from src.core.forecast_summarizer import synthesize_forecast_summary_and_chain


CALIBRATION_WARNING = (
    "Heuristic conditional weights and confidence estimates are derived from coupled LLM archetype framing "
    "and Bass/Meso simulation dynamics; they do NOT constitute empirically calibrated statistical probabilities."
)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _is_valid_slot(m: Dict[str, Any]) -> bool:
    st = m.get("execution_status")
    if st is not None:
        return st == ModelExecutionStatus.SUCCESS.value
    return bool(str(m.get("primary_narrative") or "").strip())


def _extract_short_signal_anchor(signal_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (signal_text or "").strip())
    if not cleaned:
        return "Reported initial event"
    first = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    return first[:180] + ("..." if len(first) > 180 else "")


def _clean_sentence(text: str) -> str:
    """Normalizes whitespace, removes accidental adjacent duplicates (e.g. 'raw raw'), and fixes trailing punctuation."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    # Collapse accidental duplicated alphabetic words while preserving valid 'that that' / 'had had'
    s = re.sub(r"\b(?!that\b|had\b|very\b)([A-Za-z]{2,})\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    # Collapse trailing '..' (not '...') into single '.'
    while s.endswith("..") and not s.endswith("..."):
        s = s[:-1]
    s = re.sub(r"[;,\s]+\.$", ".", s)
    return s.strip()


def _strip_order_prefix(text: str) -> Tuple[Optional[str], str]:
    """
    Detects leading order/horizon tags such as '1st-order:', 'First-order:', '2nd-order:',
    'Medium-term:', as well as inner temporal sub-tags such as 'Short-term (7 days) - ...',
    returning (normalized_order_or_None, clean_body).
    """
    cleaned = _clean_sentence(text)
    patterns: List[Tuple[str, Optional[str]]] = [
        (r"^(?:1st|first)[-\s]?order\s*[:\-–]\s*", "1st_order"),
        (r"^(?:2nd|second)[-\s]?order\s*[:\-–]\s*", "2nd_order"),
        (r"^(?:3rd|third)[-\s]?order\s*[:\-–]\s*", "3rd_order"),
        (r"^systemic(?:\s+long[-\s]?term)?\s*[:\-–]\s*", "3rd_order"),
        (r"^immediate[-\s]?term\s*[:\-–]\s*", "1st_order"),
        (r"^short[-\s]?term\s*[:\-–]\s*", "1st_order"),
        (r"^medium[-\s]?term\s*[:\-–]\s*", "2nd_order"),
        (r"^long[-\s]?term\s*[:\-–]\s*", "3rd_order"),
    ]
    mapped: Optional[str] = None
    body = cleaned
    for pat, mapped_order in patterns:
        m = re.match(pat, cleaned, flags=re.IGNORECASE)
        if m:
            mapped = mapped_order
            body = cleaned[m.end():].strip() or cleaned
            break

    sub_order, sub_body = extract_horizon_and_strip_subtag(body)
    final_order = mapped or sub_order
    return final_order, (sub_body or body)


def _dedupe_strs(items: List[str], limit: int = 6) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        s = _clean_sentence(str(raw or ""))
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _collect_by_archetype(valid_micros: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {a.value: [] for a in Archetype}
    for m in valid_micros:
        arch = str(m.get("archetype") or "")
        if hasattr(m.get("archetype"), "value"):
            arch = m["archetype"].value
        grouped.setdefault(arch, []).append(m)
    return grouped


def _first_str(items: List[str], fallback: str) -> str:
    return items[0] if items else fallback


_DOMAIN_STOPWORDS = {
    "that", "this", "with", "from", "into", "about", "have", "were", "been",
    "they", "their", "there", "which", "would", "could", "should", "will",
    "more", "than", "also", "when", "what", "where", "while", "after", "before",
    "between", "through", "such", "many", "some", "other", "over", "under",
    "publicly", "initial", "signal", "event", "reported", "users", "people",
    "lead", "leading", "increased", "widespread", "tool", "system", "impact",
}

_FALSIFICATION_CUES = (
    "never occurred",
    "never happened",
    "did not occur",
    "did not happen",
    "did not release",
    "did not announce",
    "was not launched",
    "was not released",
    "false report",
    "hoax",
    "fabricated",
    "retracted",
    "non-existent",
    "no release occurred",
    "does not exist",
)

_MITIGATION_CUES = (
    "waiver",
    "exemption",
    "audit",
    "verification",
    "verify",
    "verified benchmark",
    "verified protocol",
    "verified standard",
    "fact-check",
    "guideline",
    "standard",
    "guardrail",
    "safeguard",
    "clarif",
    "compliance",
    "buffer",
    "patch",
    "backstop",
    "stabiliz",
    "mitigat",
    "contain",
    "transparen",
    "benchmark",
    "oversight",
    "framework",
    "protocol",
    "governance",
    "review",
    "calibration",
    "human-in-the-loop",
    "moderation",
    "cooperation",
    "coordinat",
    "disclosure",
)

_ESCALATION_TRIGGER_CUES = (
    "harm",
    "incident",
    "failure",
    "fail ",
    "failed",
    "collapse",
    "crisis",
    "misinformation",
    "disinformation",
    "deepfake",
    "fraud",
    "outcry",
    "backlash",
    "unrest",
    "volatility",
    "displacement",
    "layoff",
    "job loss",
    "loss ",
    "losses",
    "breach",
    "outage",
    "erosion",
    "unacceptable risk",
    "breakdown",
    "ban ",
    "bans",
    "banning",
    "prohibit",
    "shortage",
    "spike in",
    "spikes",
    "manipulation",
    "sabotage",
    "copyright dispute",
    "lawsuit",
)

_ABSENCE_OR_NEG_MITIGATION_GUARDS = (
    "lack of",
    "absence of",
    "without any",
    "without functioning",
    "ineffective",
    "remain negligible",
    "statistically negligible",
    "purely theoretical",
    "purely advisory",
    "decline in public interest",
    "no practical utility",
    "fail to meet",
    "fails to",
    "lags behind",
    "lagging behind",
)

_VULNERABILITY_CUES = (
    "friction",
    "disrupt",
    "erosion",
    "erod",
    "polariz",
    "liability",
    "dispute",
    "overload",
    "anxiety",
    "displacement",
    "lag",
    "fragment",
    "bias",
    "misus",
    "uncertainty",
    "manipul",
    "declin",
    "bottleneck",
    "volatile",
    "misinform",
    "risk",
    "vulnerability",
    "pressure",
    "harm",
)

_POSITIVE_DEV_CUES = (
    "productiv",
    "benefi",
    "workflow",
    "innovat",
    "efficienc",
    "collaborat",
    "literacy",
    "educat",
    "trust",
    "constructive",
    "resilien",
    "adapt",
    "adoption stabilizes",
    "responsible",
)

_CHALLENGE_CUES = (
    "not ",
    "no ",
    "however",
    "despite",
    "fail",
    "limit",
    "plateau",
    "saturat",
    "overstat",
    "unverif",
    "contradict",
    "dampen",
    "refute",
    "contain",
    "absent",
    "slow",
    "delay",
    "constrain",
    "halt",
    "cannot",
    "lack",
    "exaggerat",
    "counter",
    "rebut",
    "skeptic",
    "fatigue",
    "churn",
    "error",
    "hallucinat",
    "ban",
    "restrict",
    "bottleneck",
)


def _domain_tokens(text: str) -> set[str]:
    return {
        t
        for t in re.split(r"[^a-zA-Z0-9]+", (text or "").lower())
        if len(t) >= 4 and t not in _DOMAIN_STOPWORDS
    }


def _is_premise_fact(text: str, signal_text: str, confirmed_facts: Optional[List[str]] = None) -> bool:
    """
    Returns True when `text` simply restates verified initial event facts from `signal_text`
    or confirmed facts without introducing a forward challenge/counter-argument.
    Prevents historical premise facts from mistakenly appearing as scenario counter-evidence.
    """
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    if not raw:
        return True
    lower = raw.lower()
    if lower.startswith(("initial signal text reported", "initial event:", "confirmed initial")):
        return True

    tok_text = _domain_tokens(raw)
    if not tok_text:
        return False

    ref_blob = (signal_text or "") + " " + " ".join(confirmed_facts or [])
    tok_ref = _domain_tokens(ref_blob)
    if not tok_ref:
        return False

    overlap = len(tok_text & tok_ref) / len(tok_text)
    has_challenge = any(cue in lower for cue in _CHALLENGE_CUES)

    if not has_challenge and overlap >= 0.45:
        return True
    if overlap >= 0.72 and not any(w in lower for w in ("if ", "unless ", "fail", "contradict", "however", "despite", "overstate")):
        return True
    return False


_BENIGN_COMPETITION_CUES = (
    "startup",
    "startups",
    "mid-sized",
    "democratiz",
    "open-source",
    "open source",
    "lower cost",
    "cost reduction",
    "cost-effective",
    "cheaper",
    "capital inflow",
    "accessible",
    "accessibility",
    "new entrants",
    "developer experimentation",
    "healthy competition",
    "without billion-dollar",
)

_EXPLICIT_HARM_CUES = (
    "failure",
    "outage",
    "breach",
    "exploit",
    "vulnerability",
    "security",
    "safety",
    "harm",
    "loss",
    "losses",
    "bankrupt",
    "layoff",
    "job loss",
    "panic",
    "crash",
    "collapse",
    "freeze",
    "error",
    "hallucinat",
    "unvetted",
    "lockout",
    "strain",
    "overload",
    "fragil",
    "disinformation",
    "misinformation",
    "attack",
    "shortage",
    "insolven",
    "contagion",
    "liability",
    "violation",
    "damage",
)


def _is_benign_structural_shift(text: str) -> bool:
    """
    True when a statement describes competition, cost efficiency, startup entry,
    capital inflows, or open democratization WITHOUT articulating a concrete harm mechanism.
    Prevents competitive market shifts from being misclassified as threats/downsides.
    """
    low = (text or "").lower()
    if not low.strip():
        return False
    has_benign = any(k in low for k in _BENIGN_COMPETITION_CUES)
    has_harm = any(k in low for k in _EXPLICIT_HARM_CUES)
    return has_benign and not has_harm


def _is_unexplained_fragmentation(text: str) -> bool:
    """
    True when a statement raises 'fragmentation' or 'balkanization' as a supposedly
    positive/stabilizing outcome without articulating a concrete resilience benefit.
    """
    low = (text or "").lower()
    has_frag = any(k in low for k in ("fragmentat", "fragmented", "splinter", "balkaniz"))
    if not has_frag:
        return False
    has_benefit = any(
        b in low
        for b in (
            "resilien",
            "redundan",
            "reduces single-point",
            "competition improves",
            "diversif",
            "lower systemic risk",
            "decentralized safety",
        )
    )
    return not has_benefit


def _classify_positive_category(text: str) -> str:
    """Classifies positive/stabilizing items into: Positive Development | Mitigation | Recovery | Uncertainty Reduction."""
    low = (text or "").lower()
    if any(k in low for k in ("benchmark", "disclos", "transparen", "clarif", "base rate", "reproduc", "audit")):
        return "Uncertainty Reduction"
    if any(k in low for k in ("mitigat", "guardrail", "safeguard", "governance", "standard", "contain", "compliance", "verification", "oversight")):
        return "Mitigation"
    if any(k in low for k in ("recover", "restor", "rebound", "normaliz", "stabiliz")):
        return "Recovery"
    return "Positive Development"


def _classify_condition(text: str) -> str:
    """Classifies trigger/invalidation strings into falsification | escalation | mitigation | positive_development | neutral."""
    cond, _ = classify_trigger_and_condition(text)
    return cond


def _classify_trigger_5way(text: str) -> str:
    """Classifies triggers cleanly into Q5's 5 categories: acceleration | escalation | stabilization | reversal | falsification."""
    _, trig = classify_trigger_and_condition(text)
    return trig


def _clip_phrase(text: str, max_len: int = 96) -> str:
    """Shortens text at clean word boundary instead of clipping words mid-token."""
    cleaned = _clean_sentence(str(text or ""))
    if len(cleaned) <= max_len:
        return cleaned
    return textwrap.shorten(cleaned, width=max_len, placeholder="...")


def _extract_event_profile(
    signal_text: str,
    valid_micros: List[Dict[str, Any]],
    meso_activations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Dynamically synthesizes an event-grounded causal, stakeholder, indicator, and scenario
    profile from `signal_text`, extracted entities, and validated Micro outputs.
    Replaces static domain boilerplate tables with dynamic event-specific framing.
    """
    dom_ctx = classify_event_domain(signal_text, valid_micros)
    domain = dom_ctx["domain"]
    subj = dom_ctx.get("subject_label") or "the focal organization"
    stakeholder_map = extract_event_stakeholders(
        signal_text=signal_text,
        valid_micros=valid_micros,
        meso_activations=meso_activations,
        domain_ctx=dom_ctx,
    )

    def _pick_mapped(nid: str, fallback: str) -> str:
        for row in stakeholder_map:
            if row.get("meso_node_id") == nid and row.get("stakeholder"):
                return str(row["stakeholder"])
        return fallback

    inst_stakeholder = _pick_mapped("institutional_analyst", f"Governance & Regulatory Bodies ({subj})")
    skep_stakeholder = _pick_mapped("skeptic_debunker", "Independent Auditors & Verification Analysts")
    patt_stakeholder = _pick_mapped("pattern_seeker", f"Exposed Partners, Customers & Capital Allocators ({subj})")
    ampl_stakeholder = _pick_mapped("amplifier_influencer", "Financial/Tech Media & Retail Market Observers")
    doom_primary = _pick_mapped("doomer", f"Security, Compliance & Operational Risk Teams ({subj})")
    extreme_stakeholders = _dedupe_strs(
        [
            doom_primary,
            patt_stakeholder,
            inst_stakeholder,
        ],
        3,
    )

    # Event-specific thematic descriptors synthesized from signal_text + domain_ctx (zero unrelated defaults!)
    if domain == "corporate_governance":
        inv_word = "independent board inquiry" if dom_ctx.get("has_investigation") else "formal governance review"
        sec_angle = "foreign-intelligence & national-security scrutiny" if dom_ctx.get("has_foreign_intel") else "regulatory & fiduciary compliance scrutiny"
        exec_angle = f"executive continuity at {subj}" if dom_ctx.get("has_executive") else f"leadership oversight at {subj}"

        positive_title = "Positive Trajectory — Clean Inquiry Findings & Stabilized Governance Confidence"
        baseline_summary = (
            f"During the {inv_word} involving {subj}, headline salience remains elevated while customers, partners, "
            f"and capital allocators await formal disclosure without abrupt operational abandonment."
        )
        positive_summary = (
            f"Independent inquiry findings confirming no criminal wrongdoing or operational breach, paired with transparent "
            f"board governance disclosures and {exec_angle}, restore stakeholder confidence quickly."
        )
        positive_short = "Interim board clarifications and intact internal security audits contain early speculative volatility."
        positive_medium = f"Formal conclusion clearing core operations allows enterprise customers and partners of {subj} to resume planned commitments."
        positive_long = "Strengthened board oversight, executive security protocols, and disclosure transparency reinforce long-term governance resilience."
        negative_summary = (
            f"Prolonged evidentiary ambiguity during the {inv_word} combined with persistent media amplification fuels "
            f"executive succession uncertainty, cautious customer due diligence, and {sec_angle}."
        )
        extreme_summary = (
            f"Low-probability downside cascade: substantiated evidence of espionage, compromised IP, or forced executive departure at {subj} "
            f"triggers regulatory intervention and sharp customer/partner retrenchment (Hypothesis)."
        )
        baseline_indicators = [
            f"Cadence of formal {subj} board disclosures vs secondary headline amplification",
            f"Enterprise/cloud customer order retention & multi-vendor hedging trends around {subj}",
        ]
        positive_indicators = [
            f"Publication of official {inv_word} findings clearing operational security and criminal wrongdoing",
            f"Board reaffirmation of executive continuity and strengthened compliance guardrails at {subj}",
        ]
        negative_indicators = [
            f"Extended disclosure delays, scope expansion of {sec_angle}, or executive governance friction",
            f"Rising customer security questionnaires or tentative workload diversification away from {subj}",
        ]
        extreme_indicators = [
            f"Leaked contradictory evidence, regulatory subpoenas, or confirmed compromise triggering executive transition at {subj}",
            "Emergency national-security restrictions or major hyperscaler procurement pauses",
        ]
        baseline_actions = [
            f"Compare official {subj} inquiry status filings against short-term sentiment swings",
            f"Track enterprise customer procurement continuity vs speculative analyst commentary",
        ]
        positive_actions = [
            "Verify formal scope and independence of published board clearing findings",
            f"Monitor normalization of equity volatility and enterprise partner renewals for {subj}",
        ]
        negative_actions = [
            f"Audit widening gap between ongoing {inv_word} ambiguity and customer retention commitments",
            f"Track emerging competing semiconductor or platform contingency trials",
        ]
        extreme_actions = [
            "Monitor official SEC 8-K filings, leaked contradictory findings, or leadership removal announcements",
            f"Watch for binding regulatory or national-security compliance freezes affecting {subj}",
        ]
        reversal_fallback = (
            f"Published independent {inv_word} findings clearing {subj} leadership of wrongdoing or breach reverses negative pressure"
        )
        falsification_fallback = (
            f"Independent board counsel or primary intelligence audit proves initial allegations against {subj} were fabricated, misattributed, or retracted"
        )
    elif domain == "geopolitics_security":
        positive_title = "Positive Trajectory — Verified Security Clearance & De-Escalatory Protocols"
        baseline_summary = (
            f"Counterintelligence and oversight bodies assess allegations tied to {subj} methodically while commercial and allied channels operate under heightened scrutiny."
        )
        positive_summary = (
            f"Verified technical attribution showing intact internal controls, transparent briefings, and diplomatic de-confliction contain spillover around {subj}."
        )
        positive_short = "Primary security audits bound exposure and stabilize immediate partner concern."
        positive_medium = "Codified vetting standards normalize commercial operations without abrupt decoupling."
        positive_long = "Institutionalized counterintelligence hygiene lowers susceptibility to future targeting."
        negative_summary = (
            f"Persistent attribution ambiguity and heightened national-security compliance reviews slow partner approvals and increase hedging around {subj}."
        )
        extreme_summary = (
            f"Low-probability downside cascade: confirmed intelligence compromise or state-secret exfiltration triggers binding sanctions and decoupling across {subj} (Hypothesis)."
        )
        baseline_indicators = [
            f"Verified oversight briefings vs unverified national-security speculation on {subj}",
            "Continuity of cross-border partner clearances and commercial contract renewals",
        ]
        positive_indicators = [
            f"Official security review confirming no compromised secrets or active breach at {subj}",
            "Clear bilateral compliance assurances stabilizing partner operations",
        ]
        negative_indicators = [
            f"Escalating regulatory inquiries or extended partner security reviews around {subj}",
            "Diverging allied disclosure stances increasing commercial compliance overhead",
        ]
        extreme_indicators = [
            f"Confirmed state-sponsored breach, formal indictment, or binding security sanctions against {subj}",
            "Mandatory suspension of sensitive government/enterprise technology contracts",
        ]
        baseline_actions = [
            "Track official oversight statements against secondary media reporting",
            "Monitor partner compliance review timelines for emerging friction",
        ]
        positive_actions = [
            "Verify official perimeter-integrity findings and allied clearance continuity",
            "Confirm restoration of standard enterprise/partner review cadence",
        ]
        negative_actions = [
            "Audit extended regulatory vetting bottlenecks across sensitive partnerships",
            "Track defensive diversification of critical contracts",
        ]
        extreme_actions = [
            "Escalate watch on confirmed intelligence compromise disclosures or emergency restrictions",
            "Monitor coordinated regulatory or contract suspension notices",
        ]
        reversal_fallback = f"Official clearance confirming no actionable security breach or foreign compromise around {subj} reverses escalation"
        falsification_fallback = f"Primary source investigation proves initial foreign-intelligence allegations involving {subj} were unsubstantiated or retracted"
    elif domain == "ai_software":
        positive_title = "Positive Trajectory — Structured Governance & Verified Productivity Integration"
        baseline_summary = (
            f"Broad developer and enterprise experimentation around {subj} coexists with uneven institutional guardrails and gradual workflow integration while reliability debates constrain uniform reliance."
        )
        positive_summary = (
            "Transparent capability benchmarks, audited deployment protocols, and clear institutional policies enable constructive productivity gains while bounding misuse and reliability risk."
        )
        positive_short = "Audits and capability disclosures clarify where human review remains mandatory."
        positive_medium = "Organizations adopt standardized verification workflows that capture efficiency without unchecked error propagation."
        positive_long = "Mature literacy, auditability, and accountability frameworks lower structural fragility as usage normalizes."
        negative_summary = (
            "Elevated public amplification outpaces institutional policy design, driving fragmented bans, unvetted automation errors, and heightened debate over disruption."
        )
        extreme_summary = (
            "Low-probability downside cascade: widespread unverified deployment coincides with acute trust, security, or integrity failures ahead of functional oversight (Hypothesis)."
        )
        baseline_indicators = [
            "Ratio of sustained enterprise/educational workflow integration vs short-term novelty usage",
            "Gap between short-term media salience peaks and 30-day retained workflow usage",
        ]
        positive_indicators = [
            "Publication of formal usage policies, benchmark error-rate audits, and human verification standards",
            "Declining narrative polarization paired with falling share of unverified public claims",
        ]
        negative_indicators = [
            "Rising institutional bans, documented reliability failures, or workforce displacement friction",
            "Widening divergence between headline capability claims and audited production accuracy",
        ]
        extreme_indicators = [
            "Major systemic integrity/security incident triggering coordinated lockouts or emergency moratoria",
            "Repeated unvetted autonomous workflow breaches across critical infrastructure",
        ]
        baseline_actions = [
            "Compare retained active usage metrics against raw media/social attention cycles",
            "Track developer/enterprise retention past initial 30-day experimentation window",
        ]
        positive_actions = [
            "Verify whether primary benchmarks and formal governance policies reduce error incidents",
            "Monitor shrinking gap between attention bursts and audited production adoption",
        ]
        negative_actions = [
            "Track policy whiplash, unvetted deployment incidents, and polarization trends",
            "Identify enterprise workflow stalls driven by unresolved safety/cost audits",
        ]
        extreme_actions = [
            "Audit emerging high-severity failure reports and breakdown of safety oversight channels",
            "Track emergency regulatory lockouts or major vulnerability disclosures",
        ]
        reversal_fallback = "Published technical/institutional refutation or competing audited benchmark reverses initial momentum"
        falsification_fallback = "Primary source audit proves initial release claims or reported performance metrics were inaccurate or retracted"
    elif domain == "trade_supply":
        positive_title = "Positive Trajectory — Procedural Waivers & Smooth Routing Adaptation"
        baseline_summary = (
            f"Initial policy or logistics friction involving {subj} induces selective procurement hedging and elevated sector attention before exemptions and alternative routing stabilize flows."
        )
        positive_summary = (
            "Timely customs guidance, phase-in exemptions, and verified inventory buffers absorb acute bottlenecks while calming precautionary cost pass-throughs."
        )
        positive_short = "Operational buffer data refutes worst-case physical shortage rumors."
        positive_medium = "Temporary hedging premia ease as scheduled exemptions and sourcing clarity take hold."
        positive_long = "Diversified contingency routing strengthens supply resilience on subsequent trade shocks."
        negative_summary = (
            "Prolonged policy ambiguity and synchronized defensive sourcing pass higher friction and compliance costs into downstream sectors."
        )
        extreme_summary = (
            "Low-probability downside cascade: retaliatory restrictions paired with acute transit halts overload interdependent production nodes (Hypothesis)."
        )
        baseline_indicators = [
            "Transit lead-time shifts vs retained procurement reallocation",
            "Inventory buffer burn rates across directly exposed importers",
        ]
        positive_indicators = [
            "Official customs exemptions, phased timelines, or verified buffer capacity data",
            "Stabilizing corridor throughput and narrowing precautionary spot premia",
        ]
        negative_indicators = [
            "Persistent freight/input cost pass-throughs paired with multi-week hedging congestion",
            "Sustained supplier lead-time extensions exceeding baseline seasonal norms",
        ]
        extreme_indicators = [
            "Retaliatory trade restrictions or acute physical corridor halts across major hubs",
            "Simultaneous multi-port clearance stoppages halting downstream production",
        ]
        baseline_actions = [
            "Monitor gap between acute headline alarm and actual customs/lead-time telemetry",
            "Track importer inventory drawdown rates vs baseline buffers",
        ]
        positive_actions = [
            "Check formal exemption schedules and normalized corridor throughput metrics",
            "Verify unwinding of short-term precautionary sourcing surcharges",
        ]
        negative_actions = [
            "Track downstream margin compression and cross-sector cost pass-through",
            "Monitor tier-2 supplier bottlenecks and delivery slippage",
        ]
        extreme_actions = [
            "Escalate watch on retaliatory counter-tariffs or multi-port clearance stoppages",
            "Audit critical manufacturing halt notices across dependent sectors",
        ]
        reversal_fallback = "Formal bilateral tariff suspension, exemption ruling, or rapid port normalization reverses supply friction"
        falsification_fallback = "Customs/port telemetry proves reported restriction or physical halt did not materialize"
    elif domain == "finance_macro":
        positive_title = "Positive Trajectory — Credible Backstops & Confidence Restoration"
        baseline_summary = "Targeted supervisory monitoring and short-term risk repricing absorb acute headline volatility while systemic liquidity channels remain operational."
        positive_summary = "Transparent balance-sheet disclosures and clear supervisory guidance narrow valuation spreads and prevent defensive liquidity hoarding."
        positive_short = "Audited capital and liquidity disclosures stabilize early counterparty concern."
        positive_medium = "Elevated short-term hedging unwinds as funding channels maintain orderly pricing."
        positive_long = "Reinforced supervisory transparency improves market shock absorption capacity."
        negative_summary = "Ambiguous guidance combined with amplified solvency rumors accelerates defensive repricing and tightens secondary credit availability."
        extreme_summary = "Low-probability downside cascade: contagious deposit or margin withdrawals overwhelm standard liquidity backstops (Hypothesis)."
        baseline_indicators = [
            "Funding spread movement vs institutional balance-sheet retention",
            "Daily settlement volume continuity across primary desks",
        ]
        positive_indicators = [
            "Audited solvency confirmations and stabilizing facility utilization declines",
            "Narrowing interbank spreads paired with net deposit stabilization",
        ]
        negative_indicators = [
            "Multi-week tightening across interbank or secondary credit conditions",
            "Persistent deposit outflows or widened duration hedging premia",
        ]
        extreme_indicators = [
            "Rapid withdrawal acceleration, trading halts, or emergency backstop exhaustion",
            "Synchronized counterparty credit freezes across regional institutions",
        ]
        baseline_actions = [
            "Compare funding volatility against verified settlement volume stability",
            "Monitor institutional treasury posture across weekly reserve releases",
        ]
        positive_actions = [
            "Confirm sustained inflow recovery and narrowing risk spreads after disclosures",
            "Verify orderly unwinding of emergency liquidity draws",
        ]
        negative_actions = [
            "Track secondary credit tightening and institutional posture shifts",
            "Audit regional deposit velocity and margin-call clustering",
        ]
        extreme_actions = [
            "Monitor emergency liquidity drawdowns and contagious counterparty stress indicators",
            "Watch for supervisory conservatorship or interbank clearing disruptions",
        ]
        reversal_fallback = "Audited reserve confirmation or coordinated central bank liquidity backstop reverses market stress"
        falsification_fallback = "Primary regulatory filings prove reported balance-sheet impairment did not occur"
    elif domain == "cyber_infra":
        positive_title = "Positive Trajectory — Rapid Root-Cause Containment & Verified Failover"
        baseline_summary = "Localized service disruption triggers emergency failover and security audits while containment patches gradually restore baseline operations."
        positive_summary = "Prompt root-cause disclosure, validated security patches, and clean redundant failover isolate disruption before secondary outages compound."
        positive_short = "Primary forensic telemetry bounds blast radius and curbs rumor escalation."
        positive_medium = "Hardened redundancy protocols prevent recurring single-point outages."
        positive_long = "Institutionalized incident playbooks reduce systemic cyber fragility."
        negative_summary = "Unclear remediation timelines and downstream API/service dependencies extend operational downtime and amplify compliance friction."
        extreme_summary = "Low-probability downside cascade: cascading multi-provider exploitation or uncontained infrastructure outage freezes critical services (Hypothesis)."
        baseline_indicators = [
            "Mean time to service restoration vs persistent failover degradation",
            "Share of affected endpoints restored to verified SLA latency",
        ]
        positive_indicators = [
            "Verified patch rollout and confirmed clean SLA telemetry across dependent services",
            "Completed postmortem root-cause verification with zero repeat regressions",
        ]
        negative_indicators = [
            "Delayed recovery timelines or cascading downstream application failures",
            "Rising secondary vendor SLA breaches and enterprise failover audits",
        ]
        extreme_indicators = [
            "Simultaneous multi-region backbone failure or uncontained exploit propagation",
            "Confirmed data corruption forcing prolonged authentication/payment lockouts",
        ]
        baseline_actions = [
            "Measure service recovery trajectory against downstream ticket/outage reports",
            "Verify failover capacity utilization across alternate routing regions",
        ]
        positive_actions = [
            "Validate patch integrity and post-incident SLA stability metrics",
            "Confirm zero residual error spikes across dependent authentication/API nodes",
        ]
        negative_actions = [
            "Audit secondary vendor dependencies exposed by delayed remediation",
            "Track compliance disclosure notices and enterprise contingency migrations",
        ]
        extreme_actions = [
            "Trigger cross-provider contingency isolation if multi-node failures escalate",
            "Monitor emergency CERT advisories for active lateral exploitation",
        ]
        reversal_fallback = "Verified root-cause patch and redundant failover restoration reverse cascading service disruption"
        falsification_fallback = "Forensic telemetry proves reported systemic outage or compromise was misattributed or localized"
    elif domain == "public_health":
        positive_title = "Positive Trajectory — Empirical Containment & Clear Clinical Guidance"
        baseline_summary = "Early clinical surveillance and public advisories balance precaution with measured behavioral response as primary data matures."
        positive_summary = "Verified epidemiological metrics and transparent health advisories curb alarmist rumors while targeted preparedness keeps clinical impact bounded."
        positive_short = "Confirmed clinical severity and transmission data narrow exaggerated claims."
        positive_medium = "Proportional mitigation protocols protect vulnerable groups without broad disruption."
        positive_long = "Stronger surveillance and public trust lower vulnerability to future health alerts."
        negative_summary = "Contradictory advisories and amplified health anxiety strain regional clinics and deepen public distrust toward guidance."
        extreme_summary = "Low-probability downside cascade: rapid uncontained spread overwhelms regional healthcare capacity alongside severe compliance breakdown (Hypothesis)."
        baseline_indicators = [
            "Verified clinical incidence trends vs public advisory adherence",
            "Diagnostic testing positivity and clinic triage turnaround times",
        ]
        positive_indicators = [
            "Confirming laboratory surveillance data paired with declining narrative alarm",
            "Stable hospital bed utilization alongside consistent official guidance",
        ]
        negative_indicators = [
            "Rising regional healthcare utilization paired with fragmented public messaging",
            "Widening regional compliance gaps driven by conflicting health advisories",
        ]
        extreme_indicators = [
            "Hospital capacity thresholds breached alongside acute trust collapse",
            "Simultaneous severe clinical surge and essential staff absenteeism",
        ]
        baseline_actions = [
            "Cross-check public concern metrics against verified clinical surveillance reports",
            "Track primary testing throughput across affected districts",
        ]
        positive_actions = [
            "Track declining transmission indicators and consistent official guidance compliance",
            "Verify stable ICU and emergency triage capacity buffers",
        ]
        negative_actions = [
            "Identify regional clinical pressure points and rumor-driven care surges",
            "Audit public trust and misinformation spikes affecting preventive uptake",
        ]
        extreme_actions = [
            "Monitor emergency triage strain and essential public-health coordination breakdowns",
            "Escalate watch on regional hospital overload advisories",
        ]
        reversal_fallback = "Confirmed benign severity baseline or effective clinical containment reverses public health alarm"
        falsification_fallback = "Reference laboratory surveillance shows initial outbreak alert stemmed from reporting artifact"
    elif domain == "geopolitics_energy":
        positive_title = "Positive Trajectory — Diplomatic De-Escalation & Buffer Absorption"
        baseline_summary = "Measured diplomatic responses and reserve buffers absorb immediate friction while long-term strategic positioning adjusts gradually."
        positive_summary = "Diplomatic backchannels, targeted carve-outs, and alternative capacity utilization defuse retaliatory spirals early."
        positive_short = "Verified operational buffers and de-escalatory briefings calm volatility."
        positive_medium = "Structured bilateral mechanisms isolate policy disputes from essential civilian flows."
        positive_long = "Diversified strategic infrastructure reduces leverage of future single-corridor shocks."
        negative_summary = "Reciprocal rhetoric and ambiguous policy timelines force costly strategic hedging and deepen cross-sector friction."
        extreme_summary = "Low-probability downside cascade: rapid breakdown of diplomatic channels triggers sustained multilateral escalation (Hypothesis)."
        baseline_indicators = [
            "Concrete policy implementation vs rhetorical escalation gaps",
            "Benchmark energy spot/futures spread vs verified reserve drawdowns",
        ]
        positive_indicators = [
            "Verified diplomatic carve-outs, reserve releases, or stabilizing joint statements",
            "Continued transit flow stability paired with narrowing risk premia",
        ]
        negative_indicators = [
            "Sustained retaliatory announcements paired with persistent capacity friction",
            "Multi-week diversion of critical shipments raising hedging premia",
        ]
        extreme_indicators = [
            "Acute disruption of critical corridors or collapse of de-confliction channels",
            "Synchronized multilateral embargo or physical pipeline/corridor closure",
        ]
        baseline_actions = [
            "Separate verifiable operational policy acts from rhetorical headline spikes",
            "Monitor reserve inventory levels vs spot volatility",
        ]
        positive_actions = [
            "Verify formal carve-out execution and stabilizing corridor utilization",
            "Confirm diplomatic backchannel de-confliction continuity",
        ]
        negative_actions = [
            "Track secondary sector hedging and cross-border coordination delays",
            "Monitor downstream energy cost transmission into industrial demand",
        ]
        extreme_actions = [
            "Maintain high alert for multilateral sanction spirals or physical transit halts",
            "Track emergency strategic reserve coordination announcements",
        ]
        reversal_fallback = "Signed bilateral de-escalation accord or restored physical transit flow reverses geopolitical risk premia"
        falsification_fallback = "Verified diplomatic/operational monitoring proves reported escalation or disruption did not occur"
    else:
        positive_title = f"Positive Trajectory — Transparent Verification & Stabilized {subj} Governance"
        baseline_summary = f"Initial event around {subj} generates localized behavioral shifts, strategic hedging across affected cohorts, and elevated public attention before institutional responses stabilize conditions."
        positive_summary = f"Clear empirical verification and coordinated institutional disclosures absorb early uncertainty around {subj} while bounding adverse spillover."
        positive_short = f"Primary verification and official disclosures narrow polarized claims around {subj}."
        positive_medium = f"Standardized oversight lowers speculative friction across cohorts tied to {subj}."
        positive_long = f"Strengthened institutional accountability reduces fragility during future analogous events."
        negative_summary = f"Ambiguous institutional posture around {subj} paired with high narrative amplification drives fragmented hedging and prolonged operational friction."
        extreme_summary = f"Low-probability downside cascade: breakdown of stabilizing safeguards coincides with compounding secondary shocks across {subj} (Hypothesis)."
        baseline_indicators = [
            f"Gap between conversational attention around {subj} and sustained operational/behavioral shifts",
            f"Cadence of verified official statements vs unconfirmed secondary reporting on {subj}",
        ]
        positive_indicators = [
            f"Verified primary disclosures and clear oversight guidance around {subj} narrowing division",
            f"Declining narrative polarization paired with restored partner/stakeholder confidence",
        ]
        negative_indicators = [
            f"Multi-week rise in narrative polarization paired with defensive stakeholder hedging around {subj}",
            f"Prolonged absence of clarifying primary evidence sustaining speculative pressure",
        ]
        extreme_indicators = [
            f"Acute safeguard or governance failures around {subj} accompanied by compounding secondary disruptions",
            f"Confirmed material breach, forced leadership removal, or binding regulatory intervention",
        ]
        baseline_actions = [
            f"Compare observed behavioral/partner retention against raw attention spikes around {subj}",
            f"Track primary institutional clarifications vs amplified speculation",
        ]
        positive_actions = [
            f"Validate whether official disclosures and empirical evidence lower polarization on {subj}",
            f"Confirm stakeholder normalization and absence of secondary spillovers",
        ]
        negative_actions = [
            f"Monitor rising ambiguity, defensive restructuring, and public/market backlash signals on {subj}",
            f"Audit lagging institutional clarity driving prolonged hedging",
        ]
        extreme_actions = [
            f"Escalate verification on cascading breakdowns if safeguards around {subj} fail",
            "Watch for confirmed material disclosures, sanctions, or executive continuity shocks",
        ]
        reversal_fallback = f"Published official audit or primary clarification refuting adverse assumptions around {subj} reverses momentum"
        falsification_fallback = f"Primary source verification proves initial claims around {subj} were inaccurate, fabricated, or retracted"

    return {
        "domain": domain,
        "domain_ctx": dom_ctx,
        "stakeholder_map": stakeholder_map,
        "inst_stakeholder": inst_stakeholder,
        "skep_stakeholder": skep_stakeholder,
        "patt_stakeholder": patt_stakeholder,
        "ampl_stakeholder": ampl_stakeholder,
        "extreme_stakeholders": extreme_stakeholders,
        "baseline_summary": baseline_summary,
        "positive_title": positive_title,
        "positive_summary": positive_summary,
        "positive_short": positive_short,
        "positive_medium": positive_medium,
        "positive_long": positive_long,
        "negative_summary": negative_summary,
        "extreme_summary": extreme_summary,
        "baseline_indicator": baseline_indicators[0],
        "baseline_indicators": baseline_indicators,
        "positive_indicator": positive_indicators[0],
        "positive_indicators": positive_indicators,
        "negative_indicator": negative_indicators[0],
        "negative_indicators": negative_indicators,
        "extreme_indicator": extreme_indicators[0],
        "extreme_indicators": extreme_indicators,
        "baseline_action": baseline_actions[0],
        "baseline_actions": baseline_actions,
        "positive_action": positive_actions[0],
        "positive_actions": positive_actions,
        "negative_action": negative_actions[0],
        "negative_actions": negative_actions,
        "extreme_action": extreme_actions[0],
        "extreme_actions": extreme_actions,
        "reversal_fallback": reversal_fallback,
        "falsification_fallback": falsification_fallback,
    }


def _text_similarity(a: str, b: str) -> float:
    ta, tb = _domain_tokens(a), _domain_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dedupe_similar(items: List[str], limit: int = 6, sim_thresh: float = 0.68) -> List[str]:
    """Deduplicates strings by both exact case-insensitive match and semantic token overlap."""
    out: List[str] = []
    for raw in items:
        s = _clean_sentence(str(raw or ""))
        if not s:
            continue
        if any(s.lower() == existing.lower() or _text_similarity(s, existing) >= sim_thresh for existing in out):
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _summarize_archetype_slot(
    items: List[Dict[str, Any]],
    fallback_title: str,
    signal_text: str = "",
    domain_ctx: Optional[Dict[str, Any]] = None,
    audit_collector: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merges valid Model A & B slots for one archetype into unified causal inputs with
    strictly separated 1st/2nd/3rd-order consequences, prompt-leak sanitization,
    event-domain relevance filtering, and ungrounded quantitative reframing.
    """
    if not items:
        return {
            "present": False,
            "mean_stance": 0.0,
            "mean_virality": 0.0,
            "mean_credibility": 0.0,
            "mean_uncertainty": 0.85,
            "mean_risk_severity": 0.5,
            "evidence_status": "unsupported",
            "narratives": [],
            "mechanisms": [
                f"{fallback_title} archetype data unavailable; mechanism cannot be derived from missing slot."
            ],
            "actions": [],
            "behaviours": [],
            "consequences": [],
            "consequences_by_order": {"1st_order": [], "2nd_order": [], "3rd_order": [], "non_1st_order": []},
            "triggers": [],
            "invalidations": [],
            "supporting_evidence": [],
            "claims": [],
            "analytical_questions": [],
            "stakeholder_focuses": [],
            "supporting_considerations": [],
            "opposing_considerations": [],
            "title": fallback_title,
        }

    stances = [float(x.get("stance", 0.0)) for x in items]
    virals = [float(x.get("virality", 0.0)) for x in items]
    creds = [float(x.get("credibility", 0.0)) for x in items]
    uncerts = [float(x.get("uncertainty", 0.4)) for x in items]
    sevs = [float(x["risk_severity"]) for x in items if x.get("risk_severity") is not None]

    statuses = [str(x.get("evidence_status") or "inferred") for x in items]
    if "supported" in statuses:
        ev_stat = "supported"
    elif "inferred" in statuses:
        ev_stat = "inferred"
    else:
        ev_stat = "unsupported"

    def _sanitize_and_filter(raw_list: List[str], field_name: str, check_rel: bool = True) -> List[str]:
        out_f: List[str] = []
        for raw_entry in raw_list:
            s = _clean_sentence(str(raw_entry or ""))
            if not s:
                continue
            if domain_ctx is not None:
                s = sanitize_prompt_anchor_leaks(s, domain_ctx)
                if not s:
                    continue
                if check_rel:
                    ok, reason = check_event_relevance(s, domain_ctx, allow_bridged=True)
                    if not ok:
                        if audit_collector is not None:
                            rule_cat = str(reason or "off_domain").split(":")[0]
                            audit_collector["rejected_off_domain"].append(
                                f"[{fallback_title}:{field_name}] filtered under {rule_cat} relevance guard"
                            )
                        continue
            reframed, num_note = reframe_ungrounded_numbers(s, signal_text)
            if num_note and audit_collector is not None:
                audit_collector["withheld_numeric"].append(num_note)
            if reframed:
                out_f.append(reframed)
        return out_f

    narratives = _dedupe_strs(_sanitize_and_filter([str(x.get("primary_narrative") or "") for x in items], "narrative"), 3)
    mechanisms = _dedupe_strs(_sanitize_and_filter([str(x.get("causal_mechanism") or "") for x in items], "mechanism"), 3)
    actions = _dedupe_strs(_sanitize_and_filter([a for x in items for a in (x.get("expected_stakeholder_actions") or [])], "action"), 6)
    behaviours = _dedupe_strs(_sanitize_and_filter([b for x in items for b in (x.get("behavioural_changes") or [])], "behaviour"), 6)

    # Order-aware parsing & prefix normalization across all emergent_consequences
    raw_conseq_entries: List[Tuple[str, str]] = []
    by_order: Dict[str, List[str]] = {"1st_order": [], "2nd_order": [], "3rd_order": [], "non_1st_order": []}
    for x in items:
        horizon = str(x.get("time_horizon") or "short_term_weeks")
        default_order = (
            "1st_order" if horizon in ("immediate_days",)
            else "2nd_order" if horizon in ("short_term_weeks", "medium_term_months")
            else "3rd_order"
        )
        for idx_c, raw_c in enumerate(x.get("emergent_consequences") or []):
            tagged_order, body = _strip_order_prefix(str(raw_c or ""))
            if not body:
                continue
            cleaned_list = _sanitize_and_filter([body], "consequence")
            if not cleaned_list:
                continue
            body_clean = cleaned_list[0]
            target_order = tagged_order or (
                default_order if idx_c == 0 else ("3rd_order" if default_order == "2nd_order" and idx_c >= 1 else default_order)
            )
            by_order.setdefault(target_order, []).append(body_clean)
            raw_conseq_entries.append((target_order, body_clean))

    consequences = _dedupe_similar([b for _, b in raw_conseq_entries], 8, sim_thresh=0.72)
    for k_ord in ("1st_order", "2nd_order", "3rd_order"):
        by_order[k_ord] = _dedupe_similar(by_order[k_ord], 5, sim_thresh=0.72)
    by_order["non_1st_order"] = _dedupe_similar(by_order["2nd_order"] + by_order["3rd_order"], 6, sim_thresh=0.72)

    triggers = _dedupe_strs(_sanitize_and_filter([t for x in items for t in (x.get("trigger_conditions") or [])], "trigger"), 6)
    invalidations = _dedupe_strs(_sanitize_and_filter([inv for x in items for inv in (x.get("invalidation_conditions") or [])], "invalidation"), 6)
    evidence = _dedupe_strs(_sanitize_and_filter([e for x in items for e in (x.get("supporting_evidence") or [])], "evidence", check_rel=False), 4)
    claims = _dedupe_strs(_sanitize_and_filter([c for x in items for c in (x.get("claims") or [])], "claim"), 5)

    analytical_qs = _dedupe_strs(_sanitize_and_filter([str(x.get("analytical_question") or "") for x in items], "analytical_question"), 2)
    stake_focuses = _dedupe_strs(_sanitize_and_filter([str(x.get("stakeholder_focus") or "") for x in items], "stakeholder_focus"), 2)
    supp_consids = _dedupe_strs(_sanitize_and_filter([str(x.get("strongest_supporting_consideration") or "") for x in items], "supporting_consideration"), 3)
    opp_consids = _dedupe_strs(_sanitize_and_filter([str(x.get("strongest_opposing_consideration") or "") for x in items], "opposing_consideration"), 3)

    # Graceful compatibility with older saved simulations lacking emergent_consequences only when claims exist
    if not mechanisms:
        mechanisms = narratives[:1] or [fallback_title]
    if not actions:
        actions = _dedupe_strs(_sanitize_and_filter([k for x in items for k in (x.get("key_arguments") or [])], "key_arg"), 3)
    if not consequences and claims:
        cleaned_claims = [_strip_order_prefix(c)[1] for c in claims[:2]]
        consequences = _dedupe_similar(cleaned_claims, 2)
        by_order["2nd_order"] = consequences[:]
        by_order["non_1st_order"] = consequences[:]

    return {
        "present": True,
        "mean_stance": round(float(np.mean(stances)), 3),
        "mean_virality": round(float(np.mean(virals)), 3),
        "mean_credibility": round(float(np.mean(creds)), 3),
        "mean_uncertainty": round(float(np.mean(uncerts)), 3),
        "mean_risk_severity": round(float(np.mean(sevs)), 3) if sevs else None,
        "evidence_status": ev_stat,
        "narratives": narratives,
        "mechanisms": mechanisms,
        "actions": actions,
        "behaviours": behaviours,
        "consequences": consequences,
        "consequences_by_order": by_order,
        "triggers": triggers,
        "invalidations": invalidations,
        "supporting_evidence": evidence,
        "claims": claims,
        "analytical_questions": analytical_qs,
        "stakeholder_focuses": stake_focuses,
        "supporting_considerations": supp_consids,
        "opposing_considerations": opp_consids,
        "title": items[0].get("archetype_name") or fallback_title,
    }


def _evaluate_weak_link(
    *,
    evidence_status: str,
    uncertainty: float,
    assumption_count: int,
    supporting_evidence: List[str],
    order: str,
    is_missing_link: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Deterministic weak/unsupported causal link detector.
    Explicitly flags links grounded in speculation, stacking >=3 unverified assumptions,
    carrying extreme epistemic uncertainty, missing an intermediate causal mechanism,
    or lacking evidence anchors beyond 1st order.
    """
    reasons: List[str] = []
    if is_missing_link:
        reasons.append("missing grounded intermediate causal mechanism from Micro outputs")
    if evidence_status == "unsupported":
        reasons.append("unsupported/speculative causal premise without primary source verification")
    if assumption_count >= 3:
        reasons.append(f"chains {assumption_count} sequential unproven assumptions")
    if uncertainty >= 0.62:
        reasons.append(f"high epistemic uncertainty ({uncertainty:.2f})")
    if order not in ("1st_order",) and not any(str(s).strip() for s in supporting_evidence):
        reasons.append("lacks explicit supporting evidence anchor")

    if reasons:
        return True, "Weak causal link: " + "; ".join(reasons) + "."
    return False, None


def _make_step(
    *,
    step_id: str,
    order: str,
    time_horizon: str,
    title: str,
    cause: str,
    effect: str,
    mechanism: str,
    affected_stakeholders: List[str],
    dependencies: List[str],
    trigger_conditions: List[str],
    supporting_evidence: List[str],
    evidence_status: str,
    uncertainty: float,
    assumption_count: int,
    source_layer_refs: List[str],
    affected_actor: str = "",
    behavioral_or_institutional_response: str = "",
    invalidation_condition: Optional[str] = None,
    is_missing_link: bool = False,
    missing_link_reason: Optional[str] = None,
) -> CausalStep:
    unc = float(np.clip(uncertainty, 0.0, 1.0))
    ev_stat = evidence_status if evidence_status in ("supported", "inferred", "unsupported") else "inferred"
    is_weak, weak_reason = _evaluate_weak_link(
        evidence_status=ev_stat,
        uncertainty=unc,
        assumption_count=assumption_count,
        supporting_evidence=supporting_evidence,
        order=order,
        is_missing_link=is_missing_link,
    )
    penalty = 0.72 if is_weak else 1.0
    conf = round(float(np.clip((1.0 - unc) * penalty, 0.05, 0.95)), 3)
    stake_clean = _dedupe_strs(affected_stakeholders, 5)
    actor_clean = _clean_sentence(affected_actor) or (stake_clean[0] if stake_clean else "Ecosystem participants")
    resp_clean = _clean_sentence(behavioral_or_institutional_response) or _clean_sentence(effect)
    inv_clean = _clean_sentence(invalidation_condition or "") or None
    miss_clean = _clean_sentence(missing_link_reason or "") or None

    return CausalStep(
        step_id=step_id,
        order=order,
        time_horizon=time_horizon,
        title=_clean_sentence(title),
        cause=_clean_sentence(cause),
        effect=_clean_sentence(effect),
        mechanism=_clean_sentence(mechanism),
        affected_actor=actor_clean,
        behavioral_or_institutional_response=resp_clean,
        affected_stakeholders=stake_clean,
        dependencies=_dedupe_strs(dependencies, 4),
        trigger_conditions=_dedupe_strs(trigger_conditions, 3),
        invalidation_condition=inv_clean,
        supporting_evidence=_dedupe_strs(supporting_evidence, 3),
        evidence_status=ev_stat,
        epistemic_category=(
            "source_confirmed_fact" if ev_stat == "supported" and order == "1st_order"
            else "hypothetical_future_consequence"
        ),
        uncertainty=round(unc, 3),
        confidence=conf,
        assumption_count=assumption_count,
        is_weak_link=is_weak,
        weak_link_reason=weak_reason,
        is_missing_link=is_missing_link,
        missing_link_reason=miss_clean,
        source_layer_refs=_dedupe_strs(source_layer_refs, 5),
    )


def _format_prob_band(weight: float, uncertainty: float, is_failed: bool = False) -> str:
    if is_failed:
        return "N/A (Insufficient Valid Model Output)"
    half_width = 0.07 + 0.11 * float(np.clip(uncertainty, 0.0, 1.0))
    low = max(1, int(round((weight - half_width) * 100)))
    high = min(95, int(round((weight + half_width) * 100)))
    if high <= low:
        high = min(99, low + 5)
    return f"{low}% - {high}% (heuristic conditional band; est. weight {weight * 100:.1f}%)"


def build_forecast_bundle(sim_input: Any, use_ai_summary: Optional[bool] = None) -> Dict[str, Any]:
    """
    Computes the complete Butterfly-Effect causal pathways, 4 future scenarios,
    stakeholder reaction matrix, time-horizon forecast, epistemic breakdown,
    8-question Final Verdict, and single authoritative Forecast Summary strictly
    from completed/partial simulation layers.
    """
    sim = _to_dict(sim_input)
    cfg = sim.get("config") or {}
    iterations = sim.get("iterations") or []
    sim_id = str(sim.get("simulation_id") or "unknown")
    signal_text = str(sim.get("signal_text") or cfg.get("signal") or "")
    event_title = str(sim.get("event_title") or cfg.get("event_title") or _extract_short_signal_anchor(signal_text))
    sim_status = str(sim.get("status") or "completed").lower()
    if use_ai_summary is None:
        import os
        use_ai_summary = os.getenv("BUTTERFLY_AI_SUMMARY_POLISH", "false").lower() == "true"

    all_micros = [m for it in iterations for m in (it.get("micro_interpretations") or [])]
    valid_all = [m for m in all_micros if _is_valid_slot(m)]
    exp_cnt = int(sim.get("expected_interpretations_count") or len(all_micros) or (len(iterations) * 10))
    succ_cnt = int(sim.get("successful_interpretations_count") if sim.get("successful_interpretations_count") is not None else len(valid_all))
    failed_models = sim.get("failed_models") or []
    agr_score = sim.get("model_agreement_score")
    conv_details = sim.get("narrative_convergence_details") or {}

    last_iter = iterations[-1] if iterations else {}
    last_micros = [m for m in (last_iter.get("micro_interpretations") or []) if _is_valid_slot(m)]
    if not last_micros and valid_all:
        last_micros = valid_all[-min(10, len(valid_all)):]
    last_meso = last_iter.get("meso_state") or {}
    last_macro = last_iter.get("macro_state") or {}
    last_fb = last_iter.get("feedback_state") or {}

    signal_anchor = _extract_short_signal_anchor(signal_text)

    # ── Honest Failed Execution Path (0 Valid Outputs) ────────────────────────
    if sim_status == "failed" or len(valid_all) == 0:
        err_bits = [
            f"{f.get('model_slot')} ({f.get('model_name')}): {', '.join(f.get('error_categories') or ['failed'])} — {str(f.get('sample_error') or '')[:120]}"
            for f in failed_models
        ]
        failure_note = (
            f"Simulation '{sim_id}' completed 0/{exp_cnt} valid model interpretations. "
            + ("; ".join(err_bits) if err_bits else "No valid structured archetype interpretations were returned.")
        )
        epistemic_failed = EpistemicBreakdown(
            source_confirmed_facts=[signal_anchor] if signal_anchor else [],
            attributed_claims=[],
            unverified_allegations=[],
            model_interpretations=[],
            simulated_behavioral_outcomes=["Simulation produced 0% active adoption and 0% Meso activation due to model failure."],
            hypothetical_future_consequences=["Forecast generation suspended until at least one valid model slot completes."],
        )
        scenarios_failed = [
            FutureScenario(
                scenario_id=f"scenario_{stype}",
                scenario_type=stype,
                title=f"{stype.capitalize()} Scenario — Unavailable (Failed Run)",
                summary="Insufficient valid Micro interpretations to synthesize causal trajectory.",
                affected_stakeholders=["Model execution pipeline", "Research analyst"],
                trigger_conditions=["Restore healthy local Ollama/cloud LLM inference and rerun simulation"],
                indicators_to_monitor=["Model health check latency", "Structured JSON validity rate"],
                confidence=None,
                uncertainty=1.0,
                conditional_weight=None,
                probability_band="N/A (Simulation Failed)",
                calibration_warning=CALIBRATION_WARNING,
            )
            for stype in ("baseline", "positive", "negative", "extreme")
        ]
        failed_models_cfg = (
            str(cfg.get("model_a") or "Model A"),
            str(cfg.get("model_b") or "Model B"),
        )
        failed_provs_cfg = (
            str(cfg.get("provider_a") or "unknown"),
            str(cfg.get("provider_b") or "unknown"),
        )
        failed_trig = {
            "acceleration": [],
            "escalation": [],
            "stabilization": ["Restore working LLM provider endpoints and rerun verified simulation"],
            "reversal": [],
            "falsification": [],
        }
        failed_gaps = {
            "missing_source_evidence": [failure_note],
            "model_disagreement": ["Indeterminate: 0 valid slots available for cross-model comparison."],
            "unsupported_causal_links": ["All causal links unavailable pending valid Micro execution."],
        }
        failed_sum_pack = synthesize_forecast_summary_and_chain(
            event_title=event_title,
            signal_text=signal_text,
            confirmed_facts=[signal_anchor] if signal_anchor else [],
            unverified_allegations=[],
            q1_immediate="Indeterminate from simulation: 0 valid archetype assessments returned.",
            q2_second="Indeterminate: cross-network spillover cannot be derived without valid Micro inputs.",
            q2_third="Indeterminate: structural consequences unmodeled.",
            q3_downside="Blind spot risk until LLM inference is restored.",
            q4_stabilizing="Indeterminate pending valid institutional execution.",
            stakeholder_reactions=[],
            order_effects={},
            scenarios=[s.model_dump() for s in scenarios_failed],
            leading_scenario=scenarios_failed[0].model_dump(),
            gaps_classified=failed_gaps,
            triggers_classified=failed_trig,
            macro_state=last_macro,
            feedback_state=last_fb,
            model_names=failed_models_cfg,
            provider_names=failed_provs_cfg,
            valid_slots=succ_cnt,
            expected_slots=exp_cnt,
            is_partial=True,
            is_failed=True,
            use_llm=False,
        )
        verdict_failed = FinalVerdict(
            research_question=CORE_RESEARCH_QUESTION,
            headline=f"Insufficient Valid Model Evidence — Forecast Unavailable ({succ_cnt}/{exp_cnt} slots succeeded)",
            executive_summary=(
                f"For initial event '{event_title}', all model inference attempts failed ({succ_cnt}/{exp_cnt} valid). "
                "Per scientific validity safeguards, the platform refuses to substitute deterministic mock outputs or fabricate certainty."
            ),
            forecast_summary=failed_sum_pack["forecast_summary"],
            forecast_summary_meta=failed_sum_pack["forecast_summary_meta"],
            causal_chain_compact=failed_sum_pack["causal_chain_compact"],
            initial_event_date=failed_sum_pack["initial_event_date"],
            most_likely_immediate_outcome="Indeterminate from simulation: 0 valid archetype assessments returned.",
            most_plausible_secondary_cascade="Indeterminate: cross-network spillover cannot be derived without valid Micro inputs.",
            most_dangerous_downside_risk="Blind spot risk: decision-makers lack simulated warning pathways until LLM inference is restored.",
            most_plausible_stabilizing_outcome="Indeterminate pending valid institutional and skeptic model execution.",
            key_triggers_and_turning_points=["Restore working LLM provider endpoints (Ollama/Groq/Gemini) and rerun analysis."],
            triggers_classified=failed_trig,
            evidence_and_information_gaps=[f"[Missing Source Evidence] {failure_note}"],
            evidence_gaps_classified=failed_gaps,
            recommended_monitoring_actions=[
                "Verify Ollama daemon & installed models or valid cloud API keys",
                "Inspect Section 15 / Technical Appendix diagnostics for exact timeout or schema failure codes",
            ],
            monitoring_matrix=[],
            verdict_confidence=None,
            confidence_band="Unavailable / Failed Execution",
            uncertainty=1.0,
            scenario_probabilities={
                "baseline": None,
                "positive": None,
                "negative": None,
                "extreme": None,
                "is_empirical_calibration": False,
                "status": "unavailable_failed_simulation",
            },
            calibration_warning=CALIBRATION_WARNING,
            traceability_refs={
                "simulation_id": sim_id,
                "status": sim_status,
                "valid_slots": succ_cnt,
                "expected_slots": exp_cnt,
                "iterations": len(iterations),
            },
            is_partial_or_failed=True,
            failure_disclosure=failure_note,
        )
        return {
            "research_question": CORE_RESEARCH_QUESTION,
            "causal_pathways": [],
            "order_effects_summary": OrderEffectsSummary(
                first_order=["Unavailable: 0 valid model outputs recorded."],
                second_order=["Unavailable: 0 valid model outputs recorded."],
                third_order_and_systemic=["Unavailable: 0 valid model outputs recorded."],
                unexpected_low_prob_high_impact=["Indeterminate due to model execution failure."],
                prevention_and_reversal_conditions=["Re-run simulation with healthy LLM provider availability."],
            ).model_dump(),
            "stakeholder_reactions": [],
            "scenarios": [s.model_dump() for s in scenarios_failed],
            "time_horizon_forecast": TimeHorizonForecast(
                immediate_days=["Forecast unavailable — 0 valid archetype slots completed."],
                short_term_weeks=["Forecast unavailable — restore model inference and re-run."],
                medium_term_months=["Forecast unavailable."],
                long_term_quarters=["Forecast unavailable."],
                unexpected_tail_risks=["Operational risk of unmonitored event evolution while models fail."],
            ).model_dump(),
            "epistemic_breakdown": epistemic_failed.model_dump(),
            "final_verdict": verdict_failed.model_dump(),
        }

    # ── Valid or Partially Completed Simulation Path ──────────────────────────
    is_partial = sim_status == "partially_completed" or succ_cnt < exp_cnt
    partial_penalty = 0.0 if not is_partial else min(0.30, 0.35 * (1.0 - (succ_cnt / max(1, exp_cnt))))

    audit_collector: Dict[str, List[str]] = {
        "rejected_off_domain": [],
        "withheld_numeric": [],
        "filtered_premise_actions": [],
        "filtered_reassuring_risks": [],
        "filtered_horizon_mismatches": [],
    }
    domain_ctx = classify_event_domain(signal_text, valid_all)

    grouped = _collect_by_archetype(last_micros)
    inst = _summarize_archetype_slot(grouped.get("institutional_analyst", []), "Institutional Analyst", signal_text=signal_text, domain_ctx=domain_ctx, audit_collector=audit_collector)
    skep = _summarize_archetype_slot(grouped.get("skeptic_debunker", []), "Skeptic Debunker", signal_text=signal_text, domain_ctx=domain_ctx, audit_collector=audit_collector)
    patt = _summarize_archetype_slot(grouped.get("pattern_seeker", []), "Pattern Seeker", signal_text=signal_text, domain_ctx=domain_ctx, audit_collector=audit_collector)
    doom = _summarize_archetype_slot(grouped.get("doomer", []), "Doomer", signal_text=signal_text, domain_ctx=domain_ctx, audit_collector=audit_collector)
    ampl = _summarize_archetype_slot(grouped.get("amplifier_influencer", []), "Amplifier Influencer", signal_text=signal_text, domain_ctx=domain_ctx, audit_collector=audit_collector)

    node_acts: Dict[str, float] = dict(last_meso.get("node_activations") or {})
    ext_score = float(last_meso.get("external_influence_score") or 0.0)
    int_score = float(last_meso.get("internal_contagion_score") or 0.0)
    spread_score = float(last_meso.get("effective_spread_potential") or 0.0)

    cum_adopt = float(last_macro.get("cumulative_adoption") or 0.0)
    act_adopt = float(last_macro.get("active_adoption") or 0.0)
    attn_ind = float(last_macro.get("attention_indicator") or 0.0)
    vel = float(last_macro.get("adoption_velocity") or 0.0)
    churn_rate = float(last_macro.get("c_churn") or cfg.get("churn_rate") or 0.08)
    attn_ratio = float(last_macro.get("attention_vs_adoption_ratio") or 1.0)

    pol_series = [float((it.get("feedback_state") or {}).get("polarization_index") or 0.0) for it in iterations]
    polarization = float(last_fb.get("polarization_index") or 0.0)
    pol_start = pol_series[0] if pol_series else polarization
    pol_end = pol_series[-1] if pol_series else polarization
    pol_delta = pol_end - pol_start

    sys_state = dict(last_fb.get("systemic_state") or {})
    inst_dampening = float(sys_state.get("institutional_dampening_index") if sys_state.get("institutional_dampening_index") is not None else 0.50)
    public_pressure = float(sys_state.get("public_pressure") if sys_state.get("public_pressure") is not None else (0.55 * attn_ind + 0.45 * node_acts.get("mainstream_public", act_adopt)))
    market_stress = float(sys_state.get("economic_market_stress") if sys_state.get("economic_market_stress") is not None else (0.45 * act_adopt + 0.30 * spread_score + 0.25 * abs(patt["mean_stance"])))
    escalation_risk = float(sys_state.get("escalation_risk") if sys_state.get("escalation_risk") is not None else min(1.0, 0.40 * polarization + 0.30 * max(0.0, ampl["mean_stance"]) + 0.30 * attn_ind))
    epistemic_fragility = float(sys_state.get("epistemic_fragility") if sys_state.get("epistemic_fragility") is not None else np.mean([inst["mean_uncertainty"], skep["mean_uncertainty"], patt["mean_uncertainty"], doom["mean_uncertainty"], ampl["mean_uncertainty"]]))

    profile = _extract_event_profile(signal_text, valid_all, node_acts)

    # ── 1. Epistemic Breakdown (6 strict categories) ─────────────────────────
    confirmed_facts: List[str] = []
    attributed_claims: List[str] = []
    unverified_allegations: List[str] = []
    model_interpretations: List[str] = []

    def _epi_sanitize(raw_txt: str) -> Optional[str]:
        cleaned = _clean_sentence(raw_txt)
        if not cleaned:
            return None
        cleaned = sanitize_prompt_anchor_leaks(cleaned, domain_ctx)
        ok_rel, _ = check_event_relevance(cleaned, domain_ctx, allow_bridged=True)
        if not ok_rel:
            return None
        reframed, _ = reframe_ungrounded_numbers(cleaned, signal_text)
        return reframed or cleaned

    for m in last_micros:
        ev_stat = str(m.get("evidence_status") or "inferred")
        slot_lbl = f"{m.get('archetype_name') or m.get('archetype')} ({m.get('model_name')})"
        if m.get("primary_narrative"):
            narr_clean = _epi_sanitize(str(m["primary_narrative"]))
            if narr_clean:
                model_interpretations.append(f"[{slot_lbl} | stance {float(m.get('stance', 0.0)):+.2f}] {narr_clean}")
        if ev_stat == "supported":
            confirmed_facts.extend(m.get("supporting_evidence") or [])
            for c in (m.get("claims") or []):
                c_clean = _epi_sanitize(str(c))
                if c_clean:
                    attributed_claims.append(f"[{slot_lbl}] {c_clean}")
        elif ev_stat == "inferred":
            for c in (m.get("claims") or []):
                c_clean = _epi_sanitize(str(c))
                if c_clean:
                    attributed_claims.append(f"[Inferred by {slot_lbl}] {c_clean}")
        else:
            for c in (m.get("claims") or m.get("emergent_consequences") or []):
                c_clean = _epi_sanitize(str(c))
                if c_clean:
                    unverified_allegations.append(f"[Unverified — {slot_lbl}] {c_clean}")

    if not confirmed_facts and signal_anchor:
        confirmed_facts.append(f"Initial signal text reported: \"{signal_anchor}\"")

    simulated_outcomes = [
        f"Meso network equilibrium (Round {last_iter.get('iteration', len(iterations))}): External Influence={ext_score:.1%}, Internal Contagion={int_score:.1%}, Effective Spread={spread_score:.1%}.",
        f"Modified Bass diffusion proxy (Simulated): Active Adoption Proxy A(t)={act_adopt:.1%} (Cumulative Diffusion Proxy N(t)={cum_adopt:.1%}, net velocity dN/dt={vel:+.3f}, behavioral churn c={churn_rate:.1%}).",
        f"Attention vs Adoption Separation (Simulated): Media attention salience registered {attn_ind:.1%} ({attn_ratio:.2f}x simulated active adoption); Polarization Index={polarization:.3f}.",
        f"Systemic Posture Proxy (Simulated): {sys_state.get('institutional_posture', 'Monitored Adjustment')} (Institutional dampening proxy {inst_dampening:.1%}, Escalation risk proxy {escalation_risk:.1%}).",
    ]

    # ── 2. Stakeholder Reactions tied to Meso Nodes ──────────────────────────
    def _clean_items(items: List[str], limit: int = 3) -> List[str]:
        return _dedupe_similar([x for x in items if not _is_premise_fact(x, signal_text, confirmed_facts)], limit, sim_thresh=0.72)

    def _pick_resilience(candidates: List[str], fallback: str) -> str:
        cleaned = _clean_items(candidates, 8)
        preferred = [
            c for c in cleaned
            if _classify_condition(c) in ("mitigation", "positive_development")
            and not _is_unexplained_fragmentation(c)
        ]
        if preferred:
            return preferred[0]
        non_esc = [c for c in cleaned if _classify_condition(c) != "escalation" and not _is_unexplained_fragmentation(c)]
        return non_esc[0] if non_esc else fallback

    def _pick_vulnerability(candidates: List[str], fallback: str) -> str:
        cleaned = _clean_items(candidates, 10)
        preferred = [
            c for c in cleaned
            if not _is_benign_structural_shift(c)
            and not is_reassuring_or_stabilizing(c)
            and (_classify_condition(c) == "escalation" or any(v in c.lower() for v in _VULNERABILITY_CUES))
        ]
        # Avoid selecting purely constructive governance directives as downside vulnerabilities
        filtered = [
            c for c in preferred
            if not any(p in c.lower() for p in ("safety-first sub-market", "documented compliance checklist"))
        ]
        if filtered:
            return filtered[0]
        if preferred:
            return preferred[0]
        return fallback

    inst_clean_actions = [a for a in inst["actions"] if not is_completed_premise_action(a, signal_text, confirmed_facts)] or inst["actions"]

    stakeholder_reactions = [
        StakeholderReaction(
            stakeholder=profile["inst_stakeholder"],
            meso_node_id="institutional_analyst",
            activation_level=round(float(node_acts.get("institutional_analyst", 0.0)), 4),
            expected_reaction=inst["narratives"][0] if inst["narratives"] else "Procedural compliance monitoring and cautious official review.",
            behavioural_shift="; ".join(inst["behaviours"][:2]) or "Cautious verification before issuing binding directives.",
            likely_actions=inst_clean_actions[:3] or ["Publish formal verification timelines", "Audit direct exposure before policy changes"],
            impact_direction="stabilizing",
            evidence_basis=inst["evidence_status"],
            resilience_lever=_pick_resilience(inst_clean_actions, "Publish clear verification standards and phased governance guidance"),
            downside_vulnerability=_pick_vulnerability(
                [c for c in inst["consequences"] if _classify_condition(c) == "escalation"] + doom["consequences"] + ampl["consequences"][1:],
                "Reactive or inconsistent policy directives under public pressure",
            ),
        ),
        StakeholderReaction(
            stakeholder=profile["skep_stakeholder"],
            meso_node_id="skeptic_debunker",
            activation_level=round(float(node_acts.get("skeptic_debunker", 0.0)), 4),
            expected_reaction=skep["narratives"][0] if skep["narratives"] else "Demands primary telemetry and tests limits of extrapolation.",
            behavioural_shift="; ".join(skep["behaviours"][:2]) or "Pushes back against ungrounded claims absent empirical verification.",
            likely_actions=skep["actions"][:3] or ["Publish base-rate reliability audits", "Verify practical constraints"],
            impact_direction="stabilizing",
            evidence_basis=skep["evidence_status"],
            resilience_lever=_pick_resilience(skep["actions"] + inst_clean_actions[:1], "Verify primary telemetry and publish independent technical/governance audits"),
            downside_vulnerability=_pick_vulnerability(
                skep["consequences"] + doom["behaviours"][:1],
                "Critical caveats drowned out when high-arousal hype dominates attention",
            ),
        ),
        StakeholderReaction(
            stakeholder=profile["patt_stakeholder"],
            meso_node_id="pattern_seeker",
            activation_level=round(float(node_acts.get("pattern_seeker", 0.0)), 4),
            expected_reaction=patt["narratives"][0] if patt["narratives"] else "Maps indirect incentive realignments and second-order cross-cohort spillovers.",
            behavioural_shift="; ".join(patt["behaviours"][:2]) or "Reallocates workflows and hedges secondary operational dependencies.",
            likely_actions=patt["actions"][:3] or ["Audit indirect dependencies", "Test contingency alternatives across secondary cohorts"],
            impact_direction="mixed",
            evidence_basis=patt["evidence_status"],
            resilience_lever=_pick_resilience(patt["actions"], "Design modular contingency safeguards and diversify critical dependencies"),
            downside_vulnerability=_pick_vulnerability(
                patt["consequences"][1:] + patt["consequences"][:1] + doom["consequences"],
                "Synchronized premature shifting or lock-in amplifies second-order friction",
            ),
        ),
        StakeholderReaction(
            stakeholder=profile["ampl_stakeholder"],
            meso_node_id="amplifier_influencer",
            activation_level=round(float(node_acts.get("amplifier_influencer", 0.0)), 4),
            expected_reaction=ampl["narratives"][0] if ampl["narratives"] else "Frames breaking urgency and boosts broad conversational attention.",
            behavioural_shift="; ".join(ampl["behaviours"][:2]) or "Accelerates narrative salience faster than formal clarification cycles.",
            likely_actions=ampl["actions"][:3] or ["Broadcast urgent risk frames", "Drive rapid public and market attention surges"],
            impact_direction="polarizing",
            evidence_basis=ampl["evidence_status"],
            resilience_lever="Contextualize uncertainty and label unverified projections clearly during rapid coverage",
            downside_vulnerability=_pick_vulnerability(
                ampl["consequences"] + doom["consequences"][:1],
                "Sensationalist framing outpaces empirical verification and polarizes public response",
            ),
        ),
        StakeholderReaction(
            stakeholder="Mainstream Public & Engaged Subcultures",
            meso_node_id="mainstream_public",
            activation_level=round(float(node_acts.get("mainstream_public", act_adopt)), 4),
            expected_reaction=(
                f"Subcultures activated at {node_acts.get('engaged_subcultures', 0.0):.0%} while Mainstream Public sits at "
                f"{node_acts.get('mainstream_public', 0.0):.0%}; active simulated behavioural adoption proxy is {act_adopt:.1%}."
            ),
            behavioural_shift="; ".join(doom["behaviours"][:1] + ampl["behaviours"][:1]) or "Selective reaction balanced by natural churn and skepticism.",
            likely_actions=_dedupe_strs(ampl["actions"][:2] + inst["behaviours"][:1], 3) or ["Track official disclosures", "Filter public claims vs verified guidance"],
            impact_direction="destabilizing" if escalation_risk >= 0.58 else "mixed",
            evidence_basis="inferred",
            resilience_lever="Adopt empirical literacy norms and wait for primary audits before high-stakes reliance",
            downside_vulnerability=_pick_vulnerability(
                doom["behaviours"] + doom["consequences"],
                "Rapid reactive cycles or rumor confusion increase vulnerability to unverified disruption",
            ),
        ),
    ]

    # ── 3. Competing Multi-Order Causal Pathways & Honest Missing Links ──────
    has_inst_c = bool(inst["consequences"])
    has_skep_c = bool(skep["consequences"])
    has_patt_c = bool(patt["consequences"])
    has_ampl_c = bool(ampl["consequences"])

    # Guard against auto-inventing sudden export controls, bans, or crashes unless corroborated in signal or >=2 slots
    _POLICY_SHOCK_CUES = (
        "export control",
        "export ban",
        "government ban",
        "emergency ban",
        "bilateral ban",
        "tariff",
        "market crash",
        "liquidity freeze",
        "martial",
    )
    sig_and_facts_low = re.sub(r"[-_]+", " ", (signal_text + " " + " ".join(confirmed_facts)).lower())
    all_actions_text = [
        re.sub(r"[-_]+", " ", str(a).lower())
        for m in last_micros
        for a in (m.get("expected_stakeholder_actions") or [])
    ]

    def _is_uncorroborated_shock(text: str) -> bool:
        low = re.sub(r"[-_]+", " ", (text or "").lower())
        for cue in _POLICY_SHOCK_CUES:
            if cue in low and cue not in sig_and_facts_low:
                corroborators = sum(1 for t in all_actions_text if cue in t)
                if corroborators < 2:
                    return True
        return False

    # Filter adverse Doomer items: exclude benign competition, uncorroborated shocks, AND reassuring/exonerating conclusions!
    _PRIORITY_HARM_CUES = (
        "defection", "vulnerab", "breach", "subpoena", "removal", "resign", "loss", "losses",
        "disruption", "downturn", "erosion", "litigation", "boycott", "exfiltrat", "compromis",
        "attrition", "panic", "decline", "drop", "sanction", "delay", "exploit", "outage",
        "margin compression", "failure",
    )

    def _filter_doom_harm(raw_list: List[str], require_harm_cue: bool = False) -> List[str]:
        kept: List[str] = []
        for item in raw_list:
            if is_reassuring_or_stabilizing(item):
                audit_collector["filtered_reassuring_risks"].append(_clip_phrase(item, 88))
                continue
            if _is_benign_structural_shift(item) or _is_uncorroborated_shock(item):
                continue
            if require_harm_cue and not any(h in item.lower() for h in _EXPLICIT_HARM_CUES):
                continue
            kept.append(item)
        # Prioritize concrete adverse/harm mechanisms ahead of passive audit/compliance demands
        kept.sort(key=lambda t: (0 if any(c in t.lower() for c in _PRIORITY_HARM_CUES) else 1))
        return kept

    doom_harm_conseq_2nd_first = _filter_doom_harm(
        doom["consequences_by_order"].get("2nd_order", []) + doom["consequences_by_order"].get("3rd_order", [])
    )
    doom_harm_conseq_3rd_first = _filter_doom_harm(
        doom["consequences_by_order"].get("3rd_order", []) + doom["consequences_by_order"].get("2nd_order", [])
    )
    doom_harm_conseq = _filter_doom_harm(doom["consequences"])
    doom_harm_beh = _filter_doom_harm(doom["behaviours"])
    doom_harm_claims = _filter_doom_harm(doom["claims"], require_harm_cue=True)
    doom_harm_actions = _filter_doom_harm(doom["actions"])
    has_doom_harm = bool(doom_harm_conseq or doom_harm_claims or doom_harm_beh)

    def _pick_first_non_overlap(candidates: List[str], used: List[str]) -> Optional[str]:
        for raw_cand in candidates:
            c = _clean_sentence(raw_cand)
            if not c:
                continue
            if any(_text_similarity(c, u) >= 0.58 for u in used):
                continue
            return c
        return None

    def _pick_distinct_inv(pool: List[str], used: List[str], fallback: str) -> str:
        filtered = [i for i in pool if _classify_condition(i) != "falsification"]
        choice = _pick_first_non_overlap(filtered, used)
        if choice:
            return _clean_sentence(choice)
        return _clean_sentence(fallback)

    def _infer_actor_from_text(text: str, fallback: str) -> str:
        low = (text or "").lower()
        actor_rules = (
            (("government", "regulator", "export control", "compliance", "cfius", "doj", "sec "), "Governments & Regulatory Authorities"),
            (("cloud provider", "cloud platform", "cloud host", "hyperscaler"), "Cloud Infrastructure & Hosting Providers"),
            (("board of directors", "board committee", "audit committee", "governance committee"), "Board of Directors & Governance Committee"),
            (("security breach", "critical infrastructure", "safety compliance", "red-team", "exploit"), "Security, Safety & Critical Infrastructure Operators"),
            (("semiconductor", "gpu", "hardware", "chip"), "Semiconductor Vendors & Hardware Ecosystem"),
            (("enterprise", "corporate", "firm"), "Enterprise Adopters & Corporate Buyers"),
            (("developer", "open-source", "open source", "startup"), "Developers & Startup Ecosystem"),
            (("educat", "university", "school", "student"), "Education & Academic Institutions"),
            (("hospital", "clinic", "health"), "Clinical & Public Health Operators"),
            (("bank", "credit", "liquidity", "investor", "shareholder", "stock"), "Financial Institutions & Capital Allocators"),
        )
        for kws, label in actor_rules:
            if any(k in low for k in kws):
                return label
        return fallback

    def _horizon_aligned_pool(raw_candidates: List[str], target_horizon: str) -> List[str]:
        aligned_out: List[str] = []
        for cand in raw_candidates:
            clean_c, ok_h = enforce_horizon_alignment(cand, target_horizon)
            if not ok_h:
                audit_collector["filtered_horizon_mismatches"].append(
                    f"[{target_horizon}] {_clip_phrase(cand, 76)}"
                )
                continue
            aligned_out.append(clean_c)
        return aligned_out

    # Mitigation / Stabilization triggers vs Escalation triggers
    mitigation_triggers = _dedupe_similar(
        [
            t
            for t in (
                inst["triggers"]
                + skep["triggers"]
                + patt["triggers"]
                + inst["invalidations"]
                + skep["invalidations"]
                + doom["invalidations"]
                + (last_fb.get("preventive_conditions") or [])
            )
            if _classify_condition(t) in ("mitigation", "positive_development")
            and not _is_unexplained_fragmentation(t)
            and not _is_premise_fact(t, signal_text, confirmed_facts)
            and not is_completed_premise_action(t, signal_text, confirmed_facts)
        ],
        4,
    )
    escalation_triggers = _dedupe_similar(
        [
            t
            for t in (doom["triggers"] + ampl["triggers"] + patt["triggers"] + inst["triggers"])
            if _classify_trigger_5way(t) == "escalation"
            and not _is_benign_structural_shift(t)
            and not _is_uncorroborated_shock(t)
            and not is_reassuring_or_stabilizing(t)
            and not _is_premise_fact(t, signal_text, confirmed_facts)
        ],
        4,
    )

    # Pathway A: Institutional Governance & Empirical Verification
    post_trigger_inst_actions: List[str] = []
    for raw_act in inst["actions"]:
        if is_completed_premise_action(raw_act, signal_text, confirmed_facts):
            audit_collector["filtered_premise_actions"].append(_clip_phrase(raw_act, 80))
            continue
        post_trigger_inst_actions.append(raw_act)

    corroborated_inst_actions = [a for a in post_trigger_inst_actions if not _is_uncorroborated_shock(a)]
    p1_s1_eff = (
        corroborated_inst_actions[0]
        if corroborated_inst_actions
        else (
            post_trigger_inst_actions[0] + " [Single-slot policy risk — uncorroborated in initial signal]"
            if post_trigger_inst_actions
            else f"{profile['inst_stakeholder']} establish formal verification timelines and external audit disclosure scope."
        )
    )
    p1_s1_inv = _pick_distinct_inv(
        inst["invalidations"],
        [],
        "Formal statement confirming no policy, procurement, or governance review is initiated",
    )
    p1_s2_inv = _pick_distinct_inv(
        skep["invalidations"] + inst["invalidations"],
        [p1_s1_inv],
        "Confirmed uncontained malfunction or overwhelming consensus bypassing technical audits",
    )
    p1_s3_inv = _pick_distinct_inv(
        inst["invalidations"][1:] + skep["invalidations"],
        [p1_s1_inv, p1_s2_inv],
        "Competing fragmentation or deregulation halts multi-quarter standardization",
    )

    skep_stab_cons = [
        c for c in (skep["consequences_by_order"].get("2nd_order", []) + skep["consequences"])
        if _classify_condition(c) != "escalation" and not _is_unexplained_fragmentation(c)
    ]
    if has_skep_c:
        p1_s2_core = _pick_first_non_overlap(skep_stab_cons, [p1_s1_eff]) or profile["positive_short"]
        p1_s2_eff = f"{p1_s2_core} (simulated churn c={churn_rate:.0%} moderates unverified imitation)"
        p1_s2_missing = False
        p1_s2_miss_reason = None
    else:
        p1_s2_core = "Insufficient grounded skeptic consequence data — intermediate empirical verification step lacks direct Micro support."
        p1_s2_eff = p1_s2_core
        p1_s2_missing = True
        p1_s2_miss_reason = "Skeptic Debunker produced no grounded emergent consequences bridging initial review to verified bounds."

    inst_3rd_candidates = _horizon_aligned_pool(
        [
            c
            for c in (
                inst["consequences_by_order"].get("3rd_order", [])
                + inst["consequences_by_order"].get("2nd_order", [])
                + inst["behaviours"]
            )
            if not _is_unexplained_fragmentation(c) and not _is_uncorroborated_shock(c)
        ],
        "medium_term_months",
    )
    p1_s3_choice = _pick_first_non_overlap(inst_3rd_candidates, [p1_s1_eff, p1_s2_core])
    if has_inst_c and p1_s3_choice:
        inst_cons_1 = p1_s3_choice
        p1_s3_mech = inst["mechanisms"][1] if len(inst["mechanisms"]) > 1 else inst["mechanisms"][0]
        p1_s3_missing = False
        p1_s3_miss_reason = None
    else:
        inst_cons_1 = (
            "Insufficient grounded 3rd-order governance consequence — transition from initial audit to long-term formal standards lacks distinct Micro support."
        )
        p1_s3_mech = "Missing distinct 3rd-order institutional mechanism beyond initial guidance and empirical review."
        p1_s3_missing = True
        p1_s3_miss_reason = "No distinct 3rd-order institutional consequence separating Step 3 from Steps 1-2 was supplied."

    p1_s4_inv = _clean_sentence("Renewed unverified claims bypass audit gates or institutional coordination breaks down")

    pa_steps = [
        _make_step(
            step_id="p1_s1",
            order="1st_order",
            time_horizon="immediate_days",
            title="Initial Event Triggers Formal Evaluation & Baseline Audits",
            cause=signal_anchor,
            effect=p1_s1_eff,
            mechanism=inst["mechanisms"][0],
            affected_actor=_infer_actor_from_text(p1_s1_eff, profile["inst_stakeholder"]),
            behavioral_or_institutional_response=inst["behaviours"][0] if inst["behaviours"] else p1_s1_eff,
            affected_stakeholders=[profile["inst_stakeholder"], profile["skep_stakeholder"]],
            dependencies=[],
            trigger_conditions=[t for t in inst["triggers"] if _classify_condition(t) not in ("falsification", "escalation")][:2] or [profile["baseline_indicator"]],
            invalidation_condition=p1_s1_inv,
            supporting_evidence=inst["supporting_evidence"][:2] or [signal_anchor],
            evidence_status="supported" if (inst["evidence_status"] == "supported" and corroborated_inst_actions) else "inferred",
            uncertainty=max(0.18, min(0.55, inst["mean_uncertainty"] + partial_penalty)),
            assumption_count=1,
            source_layer_refs=["Micro:InstitutionalAnalyst", "Micro:SkepticDebunker"],
        ),
        _make_step(
            step_id="p1_s2",
            order="2nd_order",
            time_horizon="short_term_weeks",
            title="Empirical Verification & Churn Bound Unchecked Claims",
            cause=f"Following initial review ({_clip_phrase(p1_s1_eff)}), technical teams audit reliability boundaries before scaling.",
            effect=p1_s2_eff,
            mechanism=skep["mechanisms"][0],
            affected_actor=_infer_actor_from_text(p1_s2_eff, profile["skep_stakeholder"]),
            behavioral_or_institutional_response=skep["actions"][0] if skep["actions"] else "Run reproducible independent verification audits",
            affected_stakeholders=[profile["skep_stakeholder"], "Mainstream public", profile["patt_stakeholder"]],
            dependencies=["p1_s1"],
            trigger_conditions=mitigation_triggers[:2] or [profile["positive_indicator"]],
            invalidation_condition=p1_s2_inv,
            supporting_evidence=skep["supporting_evidence"][:2] or inst["supporting_evidence"][:1] or [signal_anchor],
            evidence_status="inferred" if (has_skep_c and not p1_s2_missing) else "unsupported",
            uncertainty=max(0.26, min(0.58, skep["mean_uncertainty"] + 0.08 + partial_penalty)) if has_skep_c else 0.72,
            assumption_count=2,
            is_missing_link=p1_s2_missing,
            missing_link_reason=p1_s2_miss_reason,
            source_layer_refs=["Meso:skeptic_debunker", f"Macro:churn={churn_rate:.2f}"],
        ),
        _make_step(
            step_id="p1_s3",
            order="3rd_order",
            time_horizon="medium_term_months",
            title="Codified Governance Standards & Structured Adaptation",
            cause=f"Verified boundaries from Step 2 ({_clip_phrase(p1_s2_core)}) inform standardized institutional policies.",
            effect=inst_cons_1,
            mechanism=p1_s3_mech,
            affected_actor=_infer_actor_from_text(inst_cons_1, profile["inst_stakeholder"]),
            behavioral_or_institutional_response=inst["behaviours"][1] if len(inst["behaviours"]) > 1 else "Formalize evaluation standards into oversight and partner workflows",
            affected_stakeholders=[profile["patt_stakeholder"], profile["inst_stakeholder"], "Exposed partners & investors"],
            dependencies=["p1_s2"],
            trigger_conditions=mitigation_triggers[:1] or ["Clear governance guidelines and operational safeguards published"],
            invalidation_condition=p1_s3_inv,
            supporting_evidence=inst["claims"][:1] or [signal_anchor],
            evidence_status="inferred" if (has_inst_c and not p1_s3_missing) else "unsupported",
            uncertainty=max(0.35, min(0.58, inst["mean_uncertainty"] + 0.16 + partial_penalty)) if not p1_s3_missing else 0.70,
            assumption_count=2 if not p1_s3_missing else 3,
            is_missing_link=p1_s3_missing,
            missing_link_reason=p1_s3_miss_reason,
            source_layer_refs=["Feedback:institutional_posture"],
        ),
        _make_step(
            step_id="p1_s4",
            order="feedback_loop",
            time_horizon="long_term_quarters",
            title="Stabilizing Feedback Sustains Bounded Equilibrium",
            cause=f"Standardized governance ({_clip_phrase(inst_cons_1)}) dampens speculative hype cycles.",
            effect=f"Simulated active adoption settles near {act_adopt:.1%} equilibrium while institutional dampening ({inst_dampening:.1%}) contains renewed hype spikes.",
            mechanism="Corrective oversight feedback rewards verified practices and limits unvetted propagation across cycles.",
            affected_actor="Ecosystem Governance & Oversight Architecture",
            behavioral_or_institutional_response="Maintain periodic verification re-audits and churn-aware risk planning",
            affected_stakeholders=[profile["inst_stakeholder"], "All tracked ecosystem cohorts"],
            dependencies=["p1_s3"],
            trigger_conditions=mitigation_triggers[1:2] or ["Sustained governance oversight across multiple quarters"],
            invalidation_condition=p1_s4_inv,
            supporting_evidence=skep["claims"][:1] or inst["supporting_evidence"][:1],
            evidence_status="inferred" if not p1_s3_missing else "unsupported",
            uncertainty=max(0.46, 0.35 + partial_penalty),
            assumption_count=3,
            is_missing_link=p1_s3_missing,
            missing_link_reason=("Depends on incomplete 3rd-order link in Step p1_s3." if p1_s3_missing else None),
            source_layer_refs=["Feedback:institutional_dampening_index", "Macro:active_adoption"],
        ),
    ]

    # Pathway B: Strategic Adaptation & Multi-Order Ecosystem Spillover
    patt_1st_raw = [a for a in (patt["consequences_by_order"].get("1st_order", []) + patt["actions"]) if not is_completed_premise_action(a, signal_text, confirmed_facts)]
    p2_s1_eff = _clean_sentence(patt_1st_raw[0]) if patt_1st_raw else f"{profile['patt_stakeholder']} audit immediate incentive and workflow shifts."
    p2_s1_inv = _pick_distinct_inv(
        patt["invalidations"],
        [],
        "Clear institutional consensus or technical hurdles remove immediate incentive to alter workflows",
    )

    patt_2nd_pool = patt["consequences_by_order"].get("2nd_order", []) + patt["consequences_by_order"].get("3rd_order", [])
    p2_s2_choice = _pick_first_non_overlap(patt_2nd_pool, [p2_s1_eff])
    if has_patt_c and p2_s2_choice:
        patt_cons_1 = p2_s2_choice
        p2_s2_missing = False
        p2_s2_miss_reason = None
    else:
        patt_cons_1 = "Insufficient grounded secondary spillover consequence — intermediate causal mechanism between early actor moves and ecosystem diffusion is unverified."
        p2_s2_missing = True
        p2_s2_miss_reason = "Pattern Seeker provided no grounded 2nd-order consequence distinct from initial action."

    p2_s2_inv = _pick_distinct_inv(
        patt["invalidations"][1:] + skep["invalidations"],
        [p2_s1_inv],
        "Peer imitation stalls if integration friction or verification costs exceed observed gains",
    )

    # Strict 3rd-order pool: ONLY 3rd_order + 2nd_order + behaviours (never 1st_order!) aligned to medium_term_months
    patt_3rd_pool = _horizon_aligned_pool(
        patt["consequences_by_order"].get("3rd_order", [])
        + patt["consequences_by_order"].get("2nd_order", [])
        + patt["behaviours"],
        "medium_term_months",
    )
    p2_s3_choice = _pick_first_non_overlap(patt_3rd_pool, [p2_s1_eff, patt_cons_1])
    if has_patt_c and p2_s3_choice:
        patt_cons_2 = p2_s3_choice
        p2_s3_missing = False
        p2_s3_miss_reason = None
    else:
        patt_cons_2 = "Insufficient grounded 3rd-order structural effect — downstream transition from Step 2 requires empirical validation."
        p2_s3_missing = True
        p2_s3_miss_reason = "No distinct grounded 3rd-order consequence separating Step 3 from Step 2 was returned by Micro models."

    p2_s3_inv = _pick_distinct_inv(
        patt["invalidations"][2:] + inst["invalidations"],
        [p2_s1_inv, p2_s2_inv],
        "Incumbent moat or high switching costs halt multi-month migration",
    )

    pb_steps = [
        _make_step(
            step_id="p2_s1",
            order="1st_order",
            time_horizon="immediate_days",
            title="Strategic Actors Reconfigure Immediate Workflows & Positioning",
            cause=signal_anchor,
            effect=p2_s1_eff,
            mechanism=patt["mechanisms"][0],
            affected_actor=_infer_actor_from_text(p2_s1_eff, profile["patt_stakeholder"]),
            behavioral_or_institutional_response=patt["actions"][0] if patt["actions"] else p2_s1_eff,
            affected_stakeholders=[profile["patt_stakeholder"], "Early adopters"],
            dependencies=[],
            trigger_conditions=[t for t in patt["triggers"][:2] if _classify_condition(t) != "falsification"] or [profile["baseline_indicator"]],
            invalidation_condition=p2_s1_inv,
            supporting_evidence=patt["supporting_evidence"][:2] or [signal_anchor],
            evidence_status=patt["evidence_status"],
            uncertainty=max(0.26, min(0.58, patt["mean_uncertainty"] + partial_penalty)),
            assumption_count=1,
            source_layer_refs=["Micro:PatternSeeker", "Meso:engaged_subcultures"],
        ),
        _make_step(
            step_id="p2_s2",
            order="2nd_order",
            time_horizon="short_term_weeks",
            title="Peer Contagion Propagates Second-Order Ecosystem Shifts",
            cause=f"Early strategic moves ({_clip_phrase(p2_s1_eff)}) alter competitive cost/benefit benchmarks for peer organizations.",
            effect=patt_cons_1,
            mechanism=f"Internal peer contagion proxy ({int_score:.1%}) transmits incentive pressure across competing teams before standards finalize.",
            affected_actor=_infer_actor_from_text(patt_cons_1, "Competing organizations & ecosystem partners"),
            behavioral_or_institutional_response=patt["behaviours"][0] if patt["behaviours"] else "Re-evaluate exposure, contingency vendors, or governance architecture",
            affected_stakeholders=[profile["patt_stakeholder"], "Peer organizations", "Secondary stakeholders"],
            dependencies=["p2_s1"],
            trigger_conditions=[t for t in patt["triggers"][:1] if _classify_condition(t) != "falsification"] or ["Peer hedging accelerates across engaged networks"],
            invalidation_condition=p2_s2_inv,
            supporting_evidence=patt["claims"][:1] or patt["supporting_evidence"][:1],
            evidence_status="inferred" if (has_patt_c and not p2_s2_missing) else "unsupported",
            uncertainty=max(0.40, min(0.58, patt["mean_uncertainty"] + 0.10 + partial_penalty)) if not p2_s2_missing else 0.72,
            assumption_count=2,
            is_missing_link=p2_s2_missing,
            missing_link_reason=p2_s2_miss_reason,
            source_layer_refs=["Meso:internal_contagion_score", "Macro:q_internal"],
        ),
        _make_step(
            step_id="p2_s3",
            order="3rd_order",
            time_horizon="medium_term_months",
            title="Downstream Institutional & Structural Realignment",
            cause=f"Sustained 2nd-order shifts from Step 2 ({_clip_phrase(patt_cons_1)}) reshape market structure and evaluation norms.",
            effect=patt_cons_2,
            mechanism=(patt["mechanisms"][1] if len(patt["mechanisms"]) > 1 else patt["mechanisms"][0]) if not p2_s3_missing else "Missing distinct 3rd-order causal mechanism between peer imitation and macro restructuring.",
            affected_actor=_infer_actor_from_text(patt_cons_2, "Downstream enterprise adopters & ecosystem coordinators"),
            behavioral_or_institutional_response=patt["behaviours"][1] if len(patt["behaviours"]) > 1 else "Embed revised workflows and diversified vendor dependencies",
            affected_stakeholders=["Downstream institutions", "Partners & customers", profile["inst_stakeholder"]],
            dependencies=["p2_s2"],
            trigger_conditions=["Adaptation pressures persist beyond initial short-term reassessment"],
            invalidation_condition=p2_s3_inv,
            supporting_evidence=patt["claims"][1:2] or patt["claims"][:1],
            evidence_status="inferred" if (has_patt_c and not p2_s3_missing) else "unsupported",
            uncertainty=max(0.54, patt["mean_uncertainty"] + 0.18 + partial_penalty) if not p2_s3_missing else 0.76,
            assumption_count=3,
            is_missing_link=p2_s3_missing,
            missing_link_reason=p2_s3_miss_reason,
            source_layer_refs=["Feedback:economic_market_stress"],
        ),
        _make_step(
            step_id="p2_s4",
            order="systemic_long_term",
            time_horizon="long_term_quarters",
            title="Long-Term Structural Trajectory (Provisional Diffusion Projection)",
            cause=f"Cumulative realignment ({_clip_phrase(patt_cons_2)}) interacts with multi-quarter ecosystem incentives.",
            effect=f"Provisional long-run cumulative simulated diffusion reaches {cum_adopt:.1%}; exact structural form remains unverified beyond 3rd-order Micro scope.",
            mechanism="Long-term Bass accumulation without direct longitudinal empirical Micro trace — flagged as missing causal link beyond Step 3.",
            affected_actor="Broad market & institutional ecosystem",
            behavioral_or_institutional_response="Multi-quarter operating norms adapt only if validated risk/reliability shifts persist",
            affected_stakeholders=[profile["patt_stakeholder"], profile["inst_stakeholder"], "Broader public"],
            dependencies=["p2_s3"],
            trigger_conditions=["Sustained institutional and strategic adjustment across multi-quarter horizon"],
            invalidation_condition="Superior competing developments or resolution of uncertainty reverse trend",
            supporting_evidence=[],
            evidence_status="inferred" if not p2_s3_missing else "unsupported",
            uncertainty=0.68,
            assumption_count=4,
            is_missing_link=True,
            missing_link_reason="No direct multi-quarter causal mechanism supplied by Micro analysts beyond Step 3; projection rests on Bass diffusion continuity.",
            source_layer_refs=["Macro:cumulative_adoption"],
        ),
    ]

    # Pathway C: Viral Amplification & Polarization Feedback (Direction-Accurate)
    if pol_delta >= 0.04:
        pol_title = "Rising Polarization Widens Narrative & Policy Divergence"
        pol_cause = f"Competing reactions to media pressure increased simulation polarization from {pol_start:.3f} to {pol_end:.3f}."
        pol_effect = f"Polarization rises from {pol_start:.3f} to {pol_end:.3f} (Delta {pol_delta:+.3f}), hindering shared interpretation across stakeholders."
        pol_mech = "Diverging cohort incentives reinforce selective attention and reduce trust in uniform briefings."
    elif pol_delta <= -0.04:
        pol_title = "Narrative Polarization Narrowed Across Successive Iterations"
        pol_cause = f"Cross-archetype dialogue and evidence auditing narrowed simulation polarization from {pol_start:.3f} to {pol_end:.3f}."
        pol_effect = f"Polarization falls from {pol_start:.3f} to {pol_end:.3f} (Delta {pol_delta:+.3f}), reflecting convergent framing while verification gaps persist."
        pol_mech = "Iterative feedback dampens extreme disagreement when actors share comparable baseline premises, though unverified claims still require caution."
    else:
        pol_title = "Persistent Narrative Division Sustains Fragile Equilibrium"
        pol_cause = f"Competing media and skeptic frames remained balanced across rounds ({pol_start:.3f} -> {pol_end:.3f})."
        pol_effect = f"Polarization stays stable near {pol_end:.3f} (Delta {pol_delta:+.3f}), leaving future narrative trajectory sensitive to triggering events."
        pol_mech = "Rough parity between amplifying hype and institutional caution maintains unresolved public debate."

    ampl_2nd_pool = [
        c for c in (ampl["consequences_by_order"].get("2nd_order", []) + ampl["consequences_by_order"].get("3rd_order", []))
        if not is_reassuring_or_stabilizing(c)
    ]
    ampl_cons_1 = _clean_sentence(
        _pick_first_non_overlap(ampl_2nd_pool, [f"attention {attn_ind:.1%}"])
        or "Elevated media attention compresses institutional verification and communication windows."
    )
    ampl_3rd_pool = _horizon_aligned_pool(
        [
            c for c in (ampl["consequences_by_order"].get("3rd_order", []) + ampl["consequences_by_order"].get("2nd_order", []) + ampl["behaviours"])
            if not is_reassuring_or_stabilizing(c)
        ],
        "medium_term_months",
    )
    ampl_cons_2_choice = _pick_first_non_overlap(ampl_3rd_pool, [ampl_cons_1])
    ampl_cons_2 = _clean_sentence(
        ampl_cons_2_choice
        or "Visible public and partner pressure induces reactive messaging shifts ahead of slower technical audits."
    )
    p3_s1_inv = _pick_distinct_inv(
        ampl["invalidations"],
        [],
        "Competing news cycle displacement paired with measured empirical metrics",
    )
    p3_s2_inv = _pick_distinct_inv(
        ampl["invalidations"][1:] + skep["invalidations"],
        [p3_s1_inv],
        "Early sentiment spike dissipates once reproducible evaluations contradict viral claims",
    )
    p3_s3_inv = _pick_distinct_inv(
        skep["invalidations"] + inst["invalidations"],
        [p3_s1_inv, p3_s2_inv],
        "Clarifying empirical disclosures defuse public pressure cleanly",
    )
    p3_s4_inv = _pick_distinct_inv(
        inst["invalidations"][1:] + skep["invalidations"][1:],
        [p3_s1_inv, p3_s2_inv, p3_s3_inv],
        "Independent consensus narrows framing gap across cohorts",
    )

    pc_steps = [
        _make_step(
            step_id="p3_s1",
            order="1st_order",
            time_horizon="immediate_days",
            title="High-Arousal Media Framing Accelerates Conversational Salience",
            cause=signal_anchor,
            effect=f"Simulated media attention salience surges to {attn_ind:.1%} ({attn_ratio:.2f}x simulated active adoption).",
            mechanism=ampl["mechanisms"][0],
            affected_actor=profile["ampl_stakeholder"],
            behavioral_or_institutional_response=ampl["actions"][0] if ampl["actions"] else "Amplify headline salience across social & financial feeds",
            affected_stakeholders=[profile["ampl_stakeholder"], "Mainstream public", "Engaged subcultures"],
            dependencies=[],
            trigger_conditions=[t for t in ampl["triggers"][:2] if _classify_condition(t) != "falsification"] or ["High-arousal coverage boosts cross-network sharing"],
            invalidation_condition=p3_s1_inv,
            supporting_evidence=ampl["supporting_evidence"][:2] or [signal_anchor],
            evidence_status=ampl["evidence_status"],
            uncertainty=max(0.30, ampl["mean_uncertainty"] + partial_penalty),
            assumption_count=1,
            source_layer_refs=["Micro:AmplifierInfluencer", "Meso:external_media_broadcast"],
        ),
        _make_step(
            step_id="p3_s2",
            order="2nd_order",
            time_horizon="short_term_weeks",
            title="Rapid Salience Induces Short-Horizon Behavioural Swings",
            cause=f"Sustained conversational salience ({attn_ind:.1%} attention) spills from engaged subcultures into mainstream cohorts.",
            effect=ampl_cons_1,
            mechanism=f"Public pressure proxy ({public_pressure:.0%}) accelerates visible sentiment reactions ahead of formal verification.",
            affected_actor=_infer_actor_from_text(ampl_cons_1, "Mainstream users & market observers"),
            behavioral_or_institutional_response=ampl["behaviours"][0] if ampl["behaviours"] else "Rapid sentiment spikes paired with reactionary volatility",
            affected_stakeholders=["Mainstream observers", "Institutional communicators", "Policy makers"],
            dependencies=["p3_s1"],
            trigger_conditions=["Simulated attention indicator > 55% combined with speculative extrapolations"],
            invalidation_condition=p3_s2_inv,
            supporting_evidence=ampl["claims"][:1] or ampl["supporting_evidence"][:1],
            evidence_status="inferred" if has_ampl_c else "unsupported",
            uncertainty=max(0.46, ampl["mean_uncertainty"] + 0.10 + partial_penalty),
            assumption_count=2,
            is_missing_link=not has_ampl_c,
            missing_link_reason=("Amplifier Influencer provided no grounded 2nd-order consequence." if not has_ampl_c else None),
            source_layer_refs=["Meso:mainstream_public", "Feedback:amplification_warning"],
        ),
        _make_step(
            step_id="p3_s3",
            order="3rd_order",
            time_horizon="medium_term_months",
            title="Reactive Messaging & Institutional Adjustment Under Attention Pressure",
            cause=f"Heightened behavioral swings ({_clip_phrase(ampl_cons_1)}) compel visible responses from decision-makers.",
            effect=ampl_cons_2,
            mechanism="Short-term reputational pressure competes with measured technical evaluation across institutions.",
            affected_actor=_infer_actor_from_text(ampl_cons_2, profile["inst_stakeholder"]),
            behavioral_or_institutional_response="Issue clarifying advisories or reactive posture shifts to address public scrutiny",
            affected_stakeholders=[profile["inst_stakeholder"], "Regulated operators", "Exposed partners"],
            dependencies=["p3_s2"],
            trigger_conditions=["Sustained public debate paired with contradictory preliminary metrics"],
            invalidation_condition=p3_s3_inv,
            supporting_evidence=[],
            evidence_status="unsupported" if not (has_ampl_c and ampl_cons_2_choice) or ampl["evidence_status"] == "unsupported" else "inferred",
            uncertainty=max(0.64, ampl["mean_uncertainty"] + 0.22 + partial_penalty),
            assumption_count=3,
            is_missing_link=not bool(ampl_cons_2_choice),
            missing_link_reason=("No distinct horizon-aligned 3rd-order amplification consequence supplied beyond initial surge." if not bool(ampl_cons_2_choice) else None),
            source_layer_refs=["Feedback:public_pressure"],
        ),
        _make_step(
            step_id="p3_s4",
            order="feedback_loop",
            time_horizon="medium_term_months",
            title=pol_title,
            cause=pol_cause,
            effect=pol_effect,
            mechanism=pol_mech,
            affected_actor="Tracked Narrative & Public Networks",
            behavioral_or_institutional_response="Convergence or polarization across competing stakeholder groups shifts feedback pressure",
            affected_stakeholders=[profile["inst_stakeholder"], profile["ampl_stakeholder"], "Mainstream public"],
            dependencies=["p3_s3"],
            trigger_conditions=["Continued mismatch between high-velocity claims and empirical audits"],
            invalidation_condition=p3_s4_inv,
            supporting_evidence=[],
            evidence_status="inferred",
            uncertainty=0.64,
            assumption_count=3,
            source_layer_refs=["Feedback:polarization_index"],
        ),
    ]

    # Pathway D: Adverse Escalation & Tail-Risk Cascade (Harm-Filtered & Reassuring-Excluded!)
    p4_s1_candidates = [a for a in doom_harm_actions if not is_completed_premise_action(a, signal_text, confirmed_facts)]
    p4_s1_eff = (
        p4_s1_candidates[0]
        if p4_s1_candidates
        else "Safety and risk teams stress-test exposed workflows for failure or misuse vectors (Provisional Hypothesis)."
    )
    doom_2nd_pool = doom_harm_conseq_2nd_first or doom_harm_conseq
    p4_s2_choice = _pick_first_non_overlap(doom_2nd_pool + doom_harm_beh + doom_harm_claims, [p4_s1_eff])
    if p4_s2_choice:
        doom_cons_1 = p4_s2_choice
        p4_s2_missing = False
        p4_s2_miss_reason = None
    else:
        doom_cons_1 = (
            "Insufficient grounded harm mechanism returned by Doomer slots (benign structural shifts and reassuring "
            "rationalizations excluded from adverse path); provisional tail exposure hinges on substantiated breach or governance failure."
        )
        p4_s2_missing = True
        p4_s2_miss_reason = "Doomer archetype output lacked a concrete harm mechanism distinct from benign growth or reassuring exoneration."

    # Strictly horizon-aligned 3rd-order adverse pool (NO 1st_order 7-day dips AND NO reassuring outcomes!)
    doom_3rd_pool = _horizon_aligned_pool(
        doom_harm_conseq_3rd_first + doom_harm_beh + doom_harm_claims,
        "medium_term_months",
    )
    doom_2_choice = _pick_first_non_overlap(
        doom_3rd_pool,
        [p4_s1_eff, doom_cons_1],
    )
    if doom_2_choice:
        doom_cons_2 = doom_2_choice
        p4_s3_missing = False
        p4_s3_miss_reason = None
    else:
        doom_cons_2 = "Insufficient grounded 3rd-order adverse failure cascade beyond provisional vulnerability stress-test (Tail Hypothesis)."
        p4_s3_missing = True
        p4_s3_miss_reason = "No concrete horizon-aligned 3rd-order harm cascade mechanism was provided beyond Step 2."

    p4_s1_inv = _pick_distinct_inv(
        doom["invalidations"],
        [],
        "Verified audits and transparent oversight controls prevent cascading operational harm",
    )
    p4_s2_inv = _pick_distinct_inv(
        doom["invalidations"][1:] + inst["invalidations"],
        [p4_s1_inv],
        "Independent security containment and contingency continuity prevent coupled failure propagation",
    )
    p4_s3_inv = _pick_distinct_inv(
        doom["invalidations"][2:] + inst["invalidations"][1:],
        [p4_s1_inv, p4_s2_inv],
        "Coordinated incident reporting and compliance hardening contain multi-sector disruption",
    )
    p4_s4_inv = _clean_sentence("Coordinated governance and security standards restore verified confidence")

    pd_steps = [
        _make_step(
            step_id="p4_s1",
            order="1st_order",
            time_horizon="immediate_days",
            title="Acute Stress Tests Exposed Safeguards & Vulnerabilities",
            cause=signal_anchor,
            effect=p4_s1_eff,
            mechanism=doom["mechanisms"][0],
            affected_actor=_infer_actor_from_text(p4_s1_eff, profile["extreme_stakeholders"][0] if profile["extreme_stakeholders"] else "Risk & Security Teams"),
            behavioral_or_institutional_response="Evaluate immediate governance, security, or operational failure surface",
            affected_stakeholders=profile["extreme_stakeholders"][:2],
            dependencies=[],
            trigger_conditions=[t for t in escalation_triggers[:2] if not is_reassuring_or_stabilizing(t)] or ["Substantiated adverse evidence bypasses initial verification controls"],
            invalidation_condition=p4_s1_inv,
            supporting_evidence=doom["supporting_evidence"][:1] or [signal_anchor],
            evidence_status=doom["evidence_status"],
            uncertainty=max(0.52, doom["mean_uncertainty"] + partial_penalty),
            assumption_count=2,
            source_layer_refs=["Micro:Doomer", "Feedback:escalation_risk"],
        ),
        _make_step(
            step_id="p4_s2",
            order="2nd_order",
            time_horizon="short_term_weeks",
            title="Safeguard Gaps Amplify Compounding Failure Modes (Hypothesis)",
            cause=f"If adverse findings or governance gaps ({_clip_phrase(p4_s1_eff)}) compound without credible verification, coupled exposure surfaces.",
            effect=doom_cons_1,
            mechanism="Non-linear operational or security stress outpaces safeguards only if independent audit checkpoints fail (Hypothesis).",
            affected_actor=_infer_actor_from_text(doom_cons_1, profile["extreme_stakeholders"][1] if len(profile["extreme_stakeholders"]) > 1 else "Exposed operators"),
            behavioral_or_institutional_response="Emergency reviews, formal disclosures, or heightened partner compliance freezes across exposed operations",
            affected_stakeholders=profile["extreme_stakeholders"],
            dependencies=["p4_s1"],
            trigger_conditions=escalation_triggers[1:3] or [profile["extreme_indicator"]],
            invalidation_condition=p4_s2_inv,
            supporting_evidence=[],
            evidence_status="unsupported",
            uncertainty=max(0.70, doom["mean_uncertainty"] + 0.08 + partial_penalty),
            assumption_count=3,
            is_missing_link=p4_s2_missing,
            missing_link_reason=p4_s2_miss_reason,
            source_layer_refs=["Feedback:epistemic_fragility"],
        ),
        _make_step(
            step_id="p4_s3",
            order="3rd_order",
            time_horizon="medium_term_months",
            title="Cross-Sector Harm & Forced Containment Interventions (Tail Hypothesis)",
            cause=f"Sustained unmitigated adverse findings ({_clip_phrase(doom_cons_1)}) spill into coupled institutional dependencies.",
            effect=doom_cons_2,
            mechanism="Compounding adverse incidents force costly emergency compliance interventions or partner retrenchment when verification fails (Worst-Case Hypothesis).",
            affected_actor=_infer_actor_from_text(doom_cons_2, profile["inst_stakeholder"]),
            behavioral_or_institutional_response="Enforce emergency compliance halts or executive/contractual remediation pending resolution",
            affected_stakeholders=profile["extreme_stakeholders"] + ["Mainstream public"],
            dependencies=["p4_s2"],
            trigger_conditions=["Prolonged unmitigated failures over multiple months without functioning oversight"],
            invalidation_condition=p4_s3_inv,
            supporting_evidence=[],
            evidence_status="unsupported",
            uncertainty=0.78,
            assumption_count=4,
            is_missing_link=p4_s3_missing,
            missing_link_reason=p4_s3_miss_reason,
            source_layer_refs=["Macro:active_adoption", "Feedback:escalation_risk"],
        ),
        _make_step(
            step_id="p4_s4",
            order="systemic_long_term",
            time_horizon="long_term_quarters",
            title="Long-Term Trust Erosion & Defensive Fragmentation (Tail Hypothesis)",
            cause=f"Persistent multi-sector strain ({_clip_phrase(doom_cons_2)}) exhausts cooperative remediation mechanisms.",
            effect="Multi-quarter trust impairment and defensive silos if repeated unmitigated failures remain unresolved (Tail Hypothesis; unverified).",
            mechanism="Path-dependent mistrust locks in defensive compliance bottlenecks when cooperative standards break down.",
            affected_actor="Broad Institutional & Trust Architecture",
            behavioral_or_institutional_response="Entrench isolated risk silos and slow strategic commitments across exposed sectors",
            affected_stakeholders=["Broad institutional architecture", "All tracked stakeholder cohorts"],
            dependencies=["p4_s3"],
            trigger_conditions=[
                f"Systemic breakdown of oversight controls paired with recurring severe incidents affecting "
                f"{profile['extreme_stakeholders'][0] if profile.get('extreme_stakeholders') else 'exposed institutions'}"
            ],
            invalidation_condition=p4_s4_inv,
            supporting_evidence=[],
            evidence_status="unsupported",
            uncertainty=0.84,
            assumption_count=5,
            is_missing_link=True,
            missing_link_reason="Long-term systemic tail projection relies on stacked hypothetical failures lacking direct Micro evidence.",
            source_layer_refs=["Feedback:escalation_risk"],
        ),
    ]

    def _build_pathway(pid: str, name: str, driver: str, summary: str, steps: List[CausalStep], invs: List[str]) -> CausalPathway:
        wk_cnt = sum(1 for s in steps if s.is_weak_link)
        has_unsup = any(s.evidence_status == "unsupported" for s in steps)
        ev_avg = float(np.mean([1.0 if s.evidence_status == "supported" else 0.65 if s.evidence_status == "inferred" else 0.25 for s in steps]))
        plaus = float(np.mean([s.confidence for s in steps])) * (0.88 if has_unsup else 1.0)
        clean_invs = [i for i in invs if _classify_condition(i) != "falsification" and not is_completed_premise_action(i, signal_text, confirmed_facts)]
        return CausalPathway(
            pathway_id=pid,
            name=name,
            archetype_driver=driver,
            narrative_summary=_clean_sentence(summary),
            overall_plausibility=round(float(np.clip(plaus, 0.05, 0.95)), 3),
            evidence_grounding_score=round(float(np.clip(ev_avg, 0.05, 0.95)), 3),
            weak_link_count=wk_cnt,
            has_unsupported_jump=has_unsup,
            invalidation_conditions=_dedupe_similar(clean_invs, 4),
            steps=steps,
        )

    causal_pathways = [
        _build_pathway(
            "pathway_institutional_adaptation",
            "Governance Verification & Managed Adaptation",
            "Institutional Analyst + Skeptic Debunker",
            "Assumes verified guidance, empirical auditing, and natural churn contain acute escalation into structured adaptation.",
            pa_steps,
            inst["invalidations"] + skep["invalidations"],
        ),
        _build_pathway(
            "pathway_second_order_spillover",
            "Strategic Reconfiguration & Second-Order Spillover",
            "Pattern Seeker + Peer Network Contagion",
            "Traces how early actor repositioning and peer imitation propagate 2nd- and 3rd-order incentive shifts across the ecosystem.",
            pb_steps,
            patt["invalidations"] + ["Clear standardized guidelines resolving strategic ambiguity"],
        ),
        _build_pathway(
            "pathway_viral_polarization",
            "Media Amplification & Polarization Dynamics",
            "Amplifier Influencer + External Media Broadcast",
            "Captures how high conversational attention interacts with institutional response and polarization trends across iterations.",
            pc_steps,
            ampl["invalidations"] + ["Attention decay via competing news cycles and verified briefings"],
        ),
        _build_pathway(
            "pathway_tail_cascade",
            "Adverse Escalation & Systemic Cascade Risk",
            "Doomer Tail-Risk Assessment",
            "Models low-probability, high-impact harm cascades should verification and governance safeguards fail under pressure (Hypothesis).",
            pd_steps,
            doom["invalidations"] + post_trigger_inst_actions[:1],
        ),
    ]

    # Collect hypothetical future consequences & preventive conditions across steps
    hypothetical_consequences = _dedupe_strs(
        [
            f"[{pw.name} — {st.order.replace('_', ' ').title()}] {st.effect}"
            for pw in causal_pathways
            for st in pw.steps
            if st.order != "1st_order"
        ],
        8,
    )

    epistemic_breakdown = EpistemicBreakdown(
        source_confirmed_facts=_dedupe_similar(confirmed_facts, 6),
        attributed_claims=_dedupe_similar(attributed_claims, 6),
        unverified_allegations=_dedupe_similar(unverified_allegations, 6) or ["None isolated; speculative downside scenarios remain labeled as hypotheses."],
        model_interpretations=_dedupe_strs(model_interpretations, 8),
        simulated_behavioral_outcomes=simulated_outcomes,
        hypothetical_future_consequences=hypothetical_consequences,
    )

    # ── 4. Order Effects Summary (Strict non-overlapping 1st / 2nd / 3rd + reversal filter) ─
    raw_prev_pool = (
        (last_fb.get("preventive_conditions") or [])
        + mitigation_triggers
        + [inv for inv in (inst["invalidations"] + skep["invalidations"] + doom["invalidations"]) if _classify_condition(inv) != "falsification"]
    )
    prevention_pool = _clean_items(raw_prev_pool, 5)
    order_1_list = _dedupe_similar([s.effect for pw in causal_pathways[:3] for s in pw.steps if s.order == "1st_order"], 4, sim_thresh=0.60)
    order_2_raw = [s.effect for pw in causal_pathways for s in pw.steps if s.order == "2nd_order"]
    order_2_list = _dedupe_similar([e for e in order_2_raw if not any(_text_similarity(e, o1) >= 0.58 for o1 in order_1_list)], 4, sim_thresh=0.60)
    order_3_raw = [s.effect for pw in causal_pathways[:3] for s in pw.steps if s.order in ("3rd_order", "systemic_long_term", "feedback_loop")]
    order_3_list = _dedupe_similar([e for e in order_3_raw if not any(_text_similarity(e, prev) >= 0.58 for prev in (order_1_list + order_2_list))], 5, sim_thresh=0.60)

    order_effects = OrderEffectsSummary(
        first_order=order_1_list,
        second_order=order_2_list or order_2_raw[:2],
        third_order_and_systemic=order_3_list or order_3_raw[:2],
        unexpected_low_prob_high_impact=_dedupe_similar([s.effect for s in pd_steps if s.order in ("2nd_order", "3rd_order", "systemic_long_term")], 3),
        prevention_and_reversal_conditions=prevention_pool or [
            "Verified empirical clarification contradicting exaggerated second-order assumptions",
            "Rapid deployment of transparent governance standards and verification safeguards",
        ],
    )

    # ── 5. Four Distinct Future Scenarios & Heuristic Weights ────────────────
    w_base_raw = 0.85 + 0.90 * (1.0 - abs(escalation_risk - 0.42)) + 0.45 * spread_score
    w_pos_raw = 0.38 + 1.25 * inst_dampening + 0.45 * (1.0 - escalation_risk) - 0.15 * polarization
    w_neg_raw = 0.35 + 1.15 * escalation_risk + 0.55 * market_stress + 0.25 * polarization
    w_ext_raw = 0.12 + 0.70 * max(0.0, escalation_risk - 0.30) + 0.35 * epistemic_fragility + 0.20 * max(0.0, doom["mean_stance"] - 0.5)

    raw_vec = np.array([max(0.08, w_base_raw), max(0.08, w_pos_raw), max(0.08, w_neg_raw), max(0.04, w_ext_raw)], dtype=float)
    norm_weights = raw_vec / np.sum(raw_vec)
    rounded_w = [round(float(x), 4) for x in norm_weights]
    max_idx = int(np.argmax(rounded_w))
    diff = round(1.0 - sum(rounded_w), 4)
    rounded_w[max_idx] = round(rounded_w[max_idx] + diff, 4)

    # Genuine scenario-specific counter-evidence & falsification filter (no initial premise facts!)
    def _filter_counter(candidates: List[str], fallbacks: List[str], limit: int = 3) -> List[str]:
        valid = [c for c in _dedupe_strs(candidates, 8) if not _is_premise_fact(c, signal_text, confirmed_facts)]
        if len(valid) < 2:
            valid = _dedupe_strs(valid + fallbacks, limit)
        return valid[:limit]

    # Positive triggers: strictly stabilization/mitigation/positive_development, no fragmentation or degradation leaks
    positive_triggers = _clean_items(
        [
            t for t in mitigation_triggers
            if _classify_trigger_5way(t) == "stabilization"
            and not _is_unexplained_fragmentation(t)
            and not any(bad in t.lower() for bad in ("widening divergence", "degrades", "inefficienc", "failures", "bans"))
        ],
        3,
    )
    for pos_fb in (
        profile["positive_indicator"],
        profile["positive_indicators"][1] if len(profile["positive_indicators"]) > 1 else "Coordinated governance guidance and verified disclosures stabilize partner confidence",
    ):
        if len(positive_triggers) < 2 and not any(_text_similarity(pos_fb, pt) >= 0.72 for pt in positive_triggers):
            positive_triggers.append(pos_fb)

    # Baseline triggers: acceleration or measured stabilization distinct from primary positive trigger
    base_triggers = _clean_items(
        [
            t for t in (patt["triggers"] + inst["triggers"] + skep["triggers"])
            if _classify_trigger_5way(t) in ("acceleration", "stabilization")
            and _classify_condition(t) not in ("falsification", "escalation")
            and not is_completed_premise_action(t, signal_text, confirmed_facts)
            and not any(_text_similarity(t, pt) >= 0.72 for pt in positive_triggers[:1])
        ]
        or [profile["baseline_indicator"]],
        3,
    )
    for base_fb in (
        profile["baseline_indicator"],
        "Gradual institutional adjustment proceeds without acute breakdown",
    ):
        if len(base_triggers) < 2 and not any(_text_similarity(base_fb, bt) >= 0.72 for bt in base_triggers):
            base_triggers.append(base_fb)

    # Extreme tail triggers: strict harm/breakdown escalation only, excluding benign market dynamics & reassuring cues
    ext_trigger_candidates = [
        t for t in (doom["triggers"] + escalation_triggers)
        if _classify_trigger_5way(t) == "escalation"
        and not _is_benign_structural_shift(t)
        and not _is_uncorroborated_shock(t)
        and not is_reassuring_or_stabilizing(t)
    ]
    ext_triggers = _clean_items(ext_trigger_candidates, 3)
    for ext_fb in (
        profile["extreme_indicator"],
        profile["extreme_indicators"][1] if len(profile["extreme_indicators"]) > 1 else "Simultaneous failure of stabilization safeguards and cascading secondary harm events",
    ):
        if len(ext_triggers) < 2 and not any(_text_similarity(ext_fb, et) >= 0.72 for et in ext_triggers):
            ext_triggers.append(ext_fb if ext_trigger_candidates else f"[Weak Tail Trigger — Insufficient Adverse Evidence] {ext_fb}")

    # Negative triggers: prioritize friction/polarization/bottlenecks distinct from extreme tail trigger when available
    _NEG_FRICTION_CUES = _ESCALATION_TRIGGER_CUES + _VULNERABILITY_CUES + (
        "bottleneck", "shortage", "delay", "polariz", "conflict", "backlog", "friction", "compliance", "ban",
        "pullback", "valuation", "volatility", "disrupt", "erosion", "drawdown", "repricing", "ambiguity",
    )
    neg_primary_candidates = [
        t for t in (ampl["triggers"] + patt["triggers"] + inst["triggers"])
        if not _is_benign_structural_shift(t)
        and not _is_uncorroborated_shock(t)
        and not is_reassuring_or_stabilizing(t)
        and _classify_condition(t) not in ("falsification", "mitigation", "positive_development")
        and (_classify_trigger_5way(t) == "escalation" or any(cue in t.lower() for cue in _NEG_FRICTION_CUES))
        and not any(_text_similarity(t, et) >= 0.72 for et in ext_triggers[:1])
    ]
    neg_fallback_esc = [
        t for t in doom["triggers"]
        if not _is_benign_structural_shift(t)
        and not _is_uncorroborated_shock(t)
        and not is_reassuring_or_stabilizing(t)
        and _classify_condition(t) not in ("falsification", "mitigation", "positive_development")
        and not any(_text_similarity(t, et) >= 0.72 for et in ext_triggers[:1])
    ]
    neg_triggers = _clean_items(neg_primary_candidates or neg_fallback_esc, 3)
    for neg_fb in (
        profile["negative_indicator"],
        profile["negative_indicators"][1] if len(profile["negative_indicators"]) > 1 else "Unresolved verification bottlenecks paired with sustained amplification pressure",
    ):
        if len(neg_triggers) < 2 and not any(_text_similarity(neg_fb, nt) >= 0.72 for nt in neg_triggers):
            neg_triggers.append(neg_fb)

    # Event-specific actor & shift descriptors replacing generic boilerplate
    inst_actor_lbl = _clip_phrase(pa_steps[0].affected_actor or profile["inst_stakeholder"], 70)
    patt_actor_lbl = _clip_phrase(pb_steps[2].affected_actor or profile["patt_stakeholder"], 70)
    tail_actor_lbl = _clip_phrase(pd_steps[2].affected_actor or "Exposed enterprise & safety-critical sectors", 70)

    def _model_hypothesis_claims(claims: List[str], limit: int = 2) -> List[str]:
        out_c: List[str] = []
        for raw_c in claims:
            c = _clean_sentence(raw_c)
            if not c or _is_premise_fact(c, signal_text, confirmed_facts):
                continue
            out_c.append(f"[Attributed Model Hypothesis — Not Independent Evidence] {c}")
            if len(out_c) >= limit:
                break
        return out_c

    neg_short_1 = _clean_sentence(ampl_cons_1) if not is_reassuring_or_stabilizing(ampl_cons_1) else f"Heightened scrutiny and partner due diligence widen verification friction across {patt_actor_lbl}."
    neg_med_list = [
        _clean_sentence(item)
        for item in (patt_cons_2, ampl_cons_2)
        if item and not is_reassuring_or_stabilizing(item) and enforce_horizon_alignment(item, "medium_term_months")[1]
    ]
    if not neg_med_list:
        neg_med_list = [f"Extended governance ambiguity pressures risk hedging and slower commitment cadence across {patt_actor_lbl}."]

    base_conf = float(np.clip(0.76 - 0.35 * epistemic_fragility - partial_penalty, 0.20, 0.85))
    scenarios = [
        FutureScenario(
            scenario_id="scenario_baseline",
            scenario_type="baseline",
            title="Baseline Trajectory — Contained Spillover & Gradual Adaptation",
            summary=profile["baseline_summary"],
            linked_pathway_id="pathway_second_order_spillover",
            affected_stakeholders=[profile["patt_stakeholder"], profile["inst_stakeholder"], "Engaged subcultures", "Mainstream public"],
            short_term_consequences=[
                _clean_sentence(p1_s1_eff),
                f"Simulated media attention salience ({attn_ind:.1%}) initially outpaces retained simulated active adoption ({act_adopt:.1%}).",
            ],
            medium_term_consequences=[
                _clean_sentence(patt_cons_1),
                f"Incremental workflow and oversight adjustments among {patt_actor_lbl} absorb early adaptation friction.",
            ],
            long_term_consequences=[
                f"Normalized baseline operational procedures solidify across {inst_actor_lbl} and peer ecosystems without acute breakdown.",
            ],
            trigger_conditions=base_triggers,
            supporting_evidence=_dedupe_strs(
                [f"[Confirmed / Slot Anchor] {e}" for e in inst["supporting_evidence"][:1]]
                + _model_hypothesis_claims(patt["claims"][:1] + skep["claims"][:1], 1)
                + [
                    f"[Simulated Layer Telemetry] Equilibrium active simulated adoption sits at {act_adopt:.1%} with institutional dampening proxy {inst_dampening:.2f}."
                ],
                3,
            ),
            counter_evidence=_filter_counter(
                [c for c in (doom_harm_conseq[:1] + ampl["consequences"][:1]) if "Insufficient grounded" not in c and not _is_benign_structural_shift(c)],
                [
                    "Faster-than-expected compounding operational failures or backlash could shift dynamics into the Negative regime",
                    "Prompt independent verification consensus and effective safeguards could accelerate resolution into the Positive trajectory",
                ],
                2,
            ),
            indicators_to_monitor=profile["baseline_indicators"][:2],
            confidence=round(base_conf, 3),
            uncertainty=round(1.0 - base_conf, 3),
            conditional_weight=rounded_w[0],
            probability_band=_format_prob_band(rounded_w[0], 1.0 - base_conf),
            calibration_warning=CALIBRATION_WARNING,
        ),
        FutureScenario(
            scenario_id="scenario_positive",
            scenario_type="positive",
            title=profile["positive_title"],
            summary=profile["positive_summary"],
            linked_pathway_id="pathway_institutional_adaptation",
            affected_stakeholders=[profile["inst_stakeholder"], profile["skep_stakeholder"], profile["patt_stakeholder"], "Mainstream public"],
            short_term_consequences=[
                _clean_sentence(p1_s2_core if has_skep_c else profile["positive_short"]),
                "Transparent independent evaluations narrow speculative dispersion across competing claims.",
            ],
            medium_term_consequences=[
                profile["positive_medium"],
            ],
            long_term_consequences=[
                profile["positive_long"],
            ],
            trigger_conditions=positive_triggers,
            supporting_evidence=_dedupe_strs(
                [
                    f"[Simulated Layer Telemetry] Institutional dampening proxy measured {inst_dampening:.2f} with simulated churn c={churn_rate:.0%} capping runaway imitation.",
                ]
                + _model_hypothesis_claims(skep["claims"][:1] + inst["claims"][:1], 2),
                3,
            ),
            counter_evidence=_filter_counter(
                [c for c in (ampl["consequences"][:1] + doom_harm_claims[:1]) if "Insufficient grounded" not in c and not _is_benign_structural_shift(c)],
                [
                    "Delayed institutional coordination or conflicting standards could stall smooth verification rollout",
                    "Competitive pressure to move rapidly can incentivize bypassing independent audits before validation concludes",
                ],
                2,
            ),
            indicators_to_monitor=profile["positive_indicators"][:2],
            confidence=round(float(np.clip(base_conf * 0.92, 0.18, 0.82)), 3),
            uncertainty=round(1.0 - float(np.clip(base_conf * 0.92, 0.18, 0.82)), 3),
            conditional_weight=rounded_w[1],
            probability_band=_format_prob_band(rounded_w[1], epistemic_fragility + 0.1),
            calibration_warning=CALIBRATION_WARNING,
        ),
        FutureScenario(
            scenario_id="scenario_negative",
            scenario_type="negative",
            title="Negative Trajectory — Compounding Second-Order Disruption & Polarization",
            summary=profile["negative_summary"],
            linked_pathway_id="pathway_viral_polarization",
            affected_stakeholders=["Exposed operational sectors", "Mainstream public", profile["ampl_stakeholder"], profile["inst_stakeholder"]],
            short_term_consequences=[
                neg_short_1,
                f"Reactive posture shifts widen verification friction and operational delays across {patt_actor_lbl}.",
            ],
            medium_term_consequences=neg_med_list,
            long_term_consequences=[
                f"Persistent erosion of stakeholder confidence and elevated recurring compliance and coordination overhead across {patt_actor_lbl}.",
            ],
            trigger_conditions=neg_triggers,
            supporting_evidence=_dedupe_strs(
                [
                    f"[Simulated Layer Telemetry] Simulated media attention salience reaches {attn_ind:.1%} while escalation risk proxy stands at {escalation_risk:.1%}.",
                ]
                + _model_hypothesis_claims(patt["claims"][:1] + ampl["claims"][:1], 2),
                3,
            ),
            counter_evidence=_filter_counter(
                [c for c in (skep["consequences"][:1] + inst["consequences"][:1]) if "Insufficient grounded" not in c],
                [
                    f"Simulated institutional dampening proxy ({inst_dampening:.2f}) and independent technical audits may contain early overreaction",
                    f"Simulated behavioral churn (Bass churn c={churn_rate:.0%}) limits prolonged salience surges absent new triggering shocks",
                ],
                2,
            ),
            indicators_to_monitor=profile["negative_indicators"][:2],
            confidence=round(float(np.clip(base_conf * 0.85, 0.15, 0.78)), 3),
            uncertainty=round(1.0 - float(np.clip(base_conf * 0.85, 0.15, 0.78)), 3),
            conditional_weight=rounded_w[2],
            probability_band=_format_prob_band(rounded_w[2], epistemic_fragility + 0.15),
            calibration_warning=CALIBRATION_WARNING,
        ),
        FutureScenario(
            scenario_id="scenario_extreme",
            scenario_type="extreme",
            title="Extreme Tail-Risk Trajectory — Safeguard Breakdown & Systemic Dislocation",
            summary=profile["extreme_summary"],
            linked_pathway_id="pathway_tail_cascade",
            affected_stakeholders=profile["extreme_stakeholders"],
            short_term_consequences=[
                _clean_sentence(doom_cons_1),
            ],
            medium_term_consequences=[
                _clean_sentence(doom_cons_2),
            ],
            long_term_consequences=[
                f"Prolonged trust erosion and emergency oversight fragmentation across {tail_actor_lbl} (Tail Hypothesis; unverified).",
            ],
            trigger_conditions=ext_triggers,
            supporting_evidence=[
                f"[Attributed Tail Hypothesis — Weakly Supported] Relies on stacked unverified links flagged weak ({sum(1 for s in pd_steps if s.is_weak_link)} steps in Pathway D).",
            ],
            counter_evidence=_filter_counter(
                [c for c in (skep["claims"][:1] + inst["claims"][:1]) if not _is_premise_fact(c, signal_text, confirmed_facts)],
                [
                    f"Simulation estimates non-zero institutional dampening proxy ({inst_dampening:.2f}) and bounded escalation risk proxy ({escalation_risk:.1%})",
                    "Requires multiple unverified cascading failures occurring simultaneously without corrective governance intervention",
                ],
                2,
            ),
            indicators_to_monitor=profile["extreme_indicators"][:2],
            confidence=round(float(np.clip(0.38 - partial_penalty * 0.5, 0.10, 0.50)), 3),
            uncertainty=round(float(np.clip(0.75 + 0.15 * epistemic_fragility, 0.65, 0.95)), 3),
            conditional_weight=rounded_w[3],
            probability_band=_format_prob_band(rounded_w[3], 0.80),
            calibration_warning=CALIBRATION_WARNING,
        ),
    ]

    leading_scen = max(scenarios, key=lambda s: s.conditional_weight or 0.0)

    # Scenario diversity / distinctness diagnostics (Priority 8)
    scen_texts = [
        f"{s.title} {s.summary} {' '.join(s.short_term_consequences + s.medium_term_consequences + s.long_term_consequences + s.trigger_conditions + s.indicators_to_monitor)}"
        for s in scenarios
    ]
    pair_sims: List[float] = []
    scen_warnings: List[str] = []
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            sim_ij = _text_similarity(scen_texts[i], scen_texts[j])
            pair_sims.append(sim_ij)
            if sim_ij >= 0.68:
                scen_warnings.append(
                    f"High lexical overlap ({sim_ij:.2f}) between {scenarios[i].scenario_type} and {scenarios[j].scenario_type} scenarios."
                )
    mean_scen_sim = float(np.mean(pair_sims)) if pair_sims else 0.0
    max_scen_sim = float(np.max(pair_sims)) if pair_sims else 0.0
    scenario_diversity_score = round(float(np.clip(1.0 - mean_scen_sim, 0.0, 1.0)), 4)

    # ── 6. Time-Horizon Forecast ─────────────────────────────────────────────
    med_aligned = _horizon_aligned_pool(
        [s.effect for pw in causal_pathways for s in pw.steps if s.time_horizon == "medium_term_months"],
        "medium_term_months",
    ) or [profile["positive_medium"], profile["negative_summary"]]
    long_aligned = _horizon_aligned_pool(
        [s.effect for pw in causal_pathways for s in pw.steps if s.time_horizon == "long_term_quarters"],
        "long_term_quarters",
    ) or [profile["positive_long"]]

    time_horizon_forecast = TimeHorizonForecast(
        immediate_days=_dedupe_strs([s.effect for pw in causal_pathways for s in pw.steps if s.time_horizon == "immediate_days"], 4),
        short_term_weeks=_dedupe_strs([s.effect for pw in causal_pathways for s in pw.steps if s.time_horizon == "short_term_weeks"], 4),
        medium_term_months=_dedupe_strs(med_aligned, 4),
        long_term_quarters=_dedupe_strs(long_aligned, 4),
        unexpected_tail_risks=order_effects.unexpected_low_prob_high_impact,
    )

    # ── 7. Structured 8-Question Final Verdict (Phase 6.3 Strict Audit) ───────
    weak_links_total = sum(pw.weak_link_count for pw in causal_pathways)
    mean_pw_plaus = round(float(np.mean([pw.overall_plausibility for pw in causal_pathways])), 3)
    mean_pw_ground = round(float(np.mean([pw.evidence_grounding_score for pw in causal_pathways])), 3)

    verdict_conf_raw = float(
        np.clip(
            0.78
            - 0.28 * epistemic_fragility
            - 0.04 * max(0, weak_links_total - 3)
            - partial_penalty,
            0.18,
            0.84,
        )
    )
    verdict_conf = round(verdict_conf_raw, 3)
    verdict_unc = round(1.0 - verdict_conf, 3)

    if verdict_conf >= 0.68 and not is_partial:
        conf_band = "Moderate-High (Grounded Heuristic Forecast)"
    elif verdict_conf >= 0.48:
        conf_band = "Moderate (Mixed Evidence & Heuristic Paths)"
    else:
        conf_band = "Low / Cautious (High Uncertainty or Partial Execution)"

    agr_details = sim.get("model_agreement_details") or {}
    audit = last_fb.get("claim_verification_audit") or {}
    unsupported_steps = [st.step_id for pw in causal_pathways for st in pw.steps if st.evidence_status == "unsupported"]
    missing_link_steps = [st.step_id for pw in causal_pathways for st in pw.steps if st.is_missing_link]

    # Q6: Three-way classified Evidence & Information Gaps
    missing_source_gaps: List[str] = []
    if not cfg.get("source_url"):
        missing_source_gaps.append(
            "No external primary source URL was supplied; baseline grounding reflects prompt text alone without verified external corroboration."
        )
    if is_partial:
        missing_source_gaps.append(
            f"Partial model execution: {succ_cnt}/{exp_cnt} Micro slots completed; missing slots reduce ensemble evidence breadth."
        )
    if (sim.get("diagnostics") or {}).get("is_mock_only"):
        missing_source_gaps.append(
            "Executed via Deterministic Mock provider (offline validation mode) rather than live neural weights."
        )
    if audit_collector["withheld_numeric"]:
        missing_source_gaps.append(
            f"{len(audit_collector['withheld_numeric'])} ungrounded quantitative market/sales figure(s) emitted by Micro models lacked empirical baseline in the signal and were reframed as conditional mechanisms pending primary disclosures."
        )
    if attributed_claims:
        missing_source_gaps.append(
            f"{len(attributed_claims)} claim(s) across Micro slots remain [Attributed Model Hypothesis] and require external primary verification."
        )
    if int(audit.get("unsupported", 0)) > 0:
        missing_source_gaps.append(
            f"Feedback claim audit flagged {audit.get('unsupported', 0)}/{audit.get('total_interpretations', 0)} slot(s) with unsupported forward hypotheses."
        )

    model_disagreement_gaps: List[str] = []
    if agr_details.get("stance_without_reasoning_warning"):
        model_disagreement_gaps.append(
            "Cross-model audit flagged high stance agreement despite divergent causal reasoning between Model A and Model B."
        )
    if agr_details.get("causal_semantic_similarity") is not None and float(agr_details.get("causal_semantic_similarity")) < 0.35:
        model_disagreement_gaps.append(
            f"Low inter-model causal reasoning similarity ({float(agr_details.get('causal_semantic_similarity')):.2f}) indicates Model A and Model B invoke distinct explanatory mechanisms."
        )
    if conv_details.get("archetype_collapse_warning"):
        model_disagreement_gaps.append(
            "True archetype collapse detected: all five archetypes converged on similar stances AND overlapping reasoning tokens."
        )
    elif conv_details.get("legitimate_consensus_detected"):
        model_disagreement_gaps.append(
            f"Note on Convergence: Directional stances aligned across archetypes while causal reasoning remained differentiated (diff={conv_details.get('reasoning_differentiation_score', 0.0):.2f}) — legitimate consensus rather than collapse."
        )
    elif agr_score is not None:
        model_disagreement_gaps.append(
            f"Overall Model A vs Model B agreement scored {float(agr_score):.2f}; divergence across archetype lenses defines a hypothesis envelope rather than single-point consensus."
        )

    unsupported_link_gaps: List[str] = []
    if unsupported_steps:
        unsupported_link_gaps.append(
            f"{len(unsupported_steps)} causal step(s) carry 'unsupported' evidence status (tail or multi-hop hypotheses: {', '.join(unsupported_steps[:5])})."
        )
    if missing_link_steps:
        unsupported_link_gaps.append(
            f"{len(missing_link_steps)} step(s) disclose missing intermediate Micro causal mechanisms ({', '.join(missing_link_steps[:5])})."
        )
    if weak_links_total > 0:
        unsupported_link_gaps.append(
            f"{weak_links_total} causal link(s) stack >=3 assumptions or high epistemic uncertainty (>=0.62) and remain unverified hypotheses."
        )
    if audit_collector["rejected_off_domain"]:
        unsupported_link_gaps.append(
            f"{len(audit_collector['rejected_off_domain'])} off-domain / unbridged Micro item(s) excluded from causal synthesis by event-grounding validation."
        )

    evidence_gaps_classified: Dict[str, List[str]] = {
        "missing_source_evidence": _dedupe_strs(missing_source_gaps, 4) or ["Primary source verification beyond initial prompt remains unconfirmed."],
        "model_disagreement": _dedupe_strs(model_disagreement_gaps, 4) or ["Cross-model reasoning divergence remains within normal exploratory hypothesis bounds."],
        "unsupported_causal_links": _dedupe_strs(unsupported_link_gaps, 4) or ["No multi-hop causal links flagged unsupported beyond immediate scope."],
    }
    gaps: List[str] = _dedupe_strs(
        [f"[Missing Source Evidence] {g}" for g in evidence_gaps_classified["missing_source_evidence"][:2]]
        + [f"[Model Disagreement] {g}" for g in evidence_gaps_classified["model_disagreement"][:2]]
        + [f"[Unsupported Causal Link] {g}" for g in evidence_gaps_classified["unsupported_causal_links"][:2]],
        6,
    )

    # Q5: Five-way classified Key Triggers & Turning Points (negation-aware + domain-grounded fallbacks)
    all_trigger_pool = (
        base_triggers
        + positive_triggers
        + neg_triggers
        + ext_triggers
        + [
            t
            for g in (inst, skep, patt, ampl, doom)
            for t in (g["triggers"] + g["invalidations"])
        ]
        + list(last_fb.get("preventive_conditions") or [])
    )
    triggers_classified: Dict[str, List[str]] = {
        "acceleration": [],
        "escalation": [],
        "stabilization": [],
        "reversal": [],
        "falsification": [],
    }
    for raw_t in all_trigger_pool:
        t_clean = _clean_sentence(raw_t)
        if not t_clean:
            continue
        cat = _classify_trigger_5way(t_clean)
        # Exclude pure premise facts for forward categories, but keep genuine falsification/reversal invalidations
        if cat in ("acceleration", "escalation", "stabilization") and _is_premise_fact(t_clean, signal_text, confirmed_facts):
            continue
        if cat == "acceleration" and is_completed_premise_action(t_clean, signal_text, confirmed_facts):
            continue
        if cat == "stabilization" and _is_unexplained_fragmentation(t_clean):
            continue
        if cat == "escalation" and (_is_benign_structural_shift(t_clean) or _is_uncorroborated_shock(t_clean) or is_reassuring_or_stabilizing(t_clean)):
            continue
        triggers_classified[cat].append(t_clean)

    for k_cat in triggers_classified:
        triggers_classified[k_cat] = _dedupe_similar(triggers_classified[k_cat], 4, sim_thresh=0.72)

    if not triggers_classified["acceleration"]:
        triggers_classified["acceleration"] = _dedupe_strs(base_triggers[:2], 2)
    if not triggers_classified["stabilization"]:
        triggers_classified["stabilization"] = _dedupe_strs(positive_triggers[:2], 2)
    if not triggers_classified["escalation"]:
        triggers_classified["escalation"] = _dedupe_strs(neg_triggers[:2], 2)
    if not triggers_classified["reversal"]:
        triggers_classified["reversal"] = [profile["reversal_fallback"]]
    if not triggers_classified["falsification"]:
        triggers_classified["falsification"] = [profile["falsification_fallback"]]

    _CAT_LABELS = {
        "acceleration": "Acceleration",
        "escalation": "Escalation",
        "stabilization": "Stabilization",
        "reversal": "Reversal",
        "falsification": "Falsification",
    }
    key_triggers = _dedupe_strs(
        [
            f"[{_CAT_LABELS[cat]}] {triggers_classified[cat][0]}"
            for cat in ("acceleration", "escalation", "stabilization", "reversal", "falsification")
            if triggers_classified[cat]
        ]
        + [
            f"[{_CAT_LABELS[cat]}] {triggers_classified[cat][1]}"
            for cat in ("acceleration", "escalation", "stabilization")
            if len(triggers_classified[cat]) > 1
        ],
        8,
    )

    # Q7: Cohesive 1-to-1 Non-Repeating Monitoring Matrix
    def _supports_contradicts(sc_type: str, trig: str) -> Tuple[str, str]:
        trig_short = _clip_phrase(trig, 80)
        if sc_type == "baseline":
            return (
                f"Observed evidence aligns with '{trig_short}' while operations remain within normal bounds",
                "Sudden surge in substantiated escalations or rapid formal closure resolving all ambiguity",
            )
        if sc_type == "positive":
            return (
                f"Verified milestone '{trig_short}' alongside declining uncertainty and stabilizing partner confidence",
                "Delayed oversight findings, unresolved compliance gaps, or substantiated adverse disclosures",
            )
        if sc_type == "negative":
            return (
                f"Observed friction around '{trig_short}' paired with stakeholder hedging or extended review timelines",
                "Independent verification audits, clearing disclosures, or fade-out of partner contingency pressure",
            )
        return (
            f"Compounding severe developments matching '{trig_short}' coinciding with oversight or control breakdown",
            "Active governance backstops, verified perimeter clearance, or absence of cascading downstream disruption",
        )

    actions_list_by_type = {
        "baseline": profile["baseline_actions"],
        "positive": profile["positive_actions"],
        "negative": profile["negative_actions"],
        "extreme": profile["extreme_actions"],
    }
    secondary_trigger_hints_by_type = {
        "baseline": profile["baseline_indicators"],
        "positive": profile["positive_indicators"],
        "negative": profile["negative_indicators"],
        "extreme": profile["extreme_indicators"],
    }
    monitoring_matrix: List[Dict[str, str]] = []
    max_inds = max((len(s.indicators_to_monitor) for s in scenarios), default=0)
    seen_inds: set[str] = set()
    used_thresholds: set[str] = set()
    used_actions: set[str] = set()
    for ind_idx in range(max_inds):
        for sc in scenarios:
            if ind_idx < len(sc.indicators_to_monitor):
                ind = _clean_sentence(sc.indicators_to_monitor[ind_idx])
                key = ind.lower()
                if not ind or key in seen_inds:
                    continue
                seen_inds.add(key)

                # Select a non-repeating 1-to-1 trigger threshold without naive modulo duplicates
                cand_trigs = list(sc.trigger_conditions or []) + [
                    f"Sustained shift verified via {hint}"
                    for hint in (secondary_trigger_hints_by_type.get(sc.scenario_type) or [])
                ]
                trig = ""
                for ct in cand_trigs:
                    ct_clean = _clean_sentence(ct)
                    if ct_clean and ct_clean.lower() not in used_thresholds:
                        trig = ct_clean
                        used_thresholds.add(ct_clean.lower())
                        break
                if not trig:
                    trig = f"Verified trajectory inflection measured on: {_clip_phrase(ind, 72)}"
                    used_thresholds.add(trig.lower())

                act_choices = actions_list_by_type.get(sc.scenario_type) or [
                    "Verify empirical telemetry against scenario assumptions"
                ]
                act = ""
                for idx_offset in range(len(act_choices)):
                    cand_a = _clean_sentence(act_choices[(ind_idx + idx_offset) % len(act_choices)])
                    if cand_a and cand_a.lower() not in used_actions:
                        act = cand_a
                        used_actions.add(cand_a.lower())
                        break
                if not act:
                    act = _clean_sentence(act_choices[min(ind_idx, len(act_choices) - 1)])

                supp_mov, contr_mov = _supports_contradicts(sc.scenario_type, trig)
                monitoring_matrix.append({
                    "indicator": ind,
                    "measurable_variable": ind,
                    "scenario_type": sc.scenario_type,
                    "trigger_threshold": trig,
                    "supports_scenario_movement": supp_mov,
                    "contradicts_scenario_movement": contr_mov,
                    "analyst_action": f"{act} (Threshold: {trig})",
                })
                if len(monitoring_matrix) >= 8:
                    break
        if len(monitoring_matrix) >= 8:
            break

    monitoring_actions = [
        f"[{row['scenario_type'].upper()}] Track {row['measurable_variable']} — {row['analyst_action']}"
        for row in monitoring_matrix[:6]
    ] or [
        "Track empirical behavioral retention vs media salience while monitoring independent institutional audits"
    ]

    scenario_probs_summary = {
        s.scenario_type: {
            "title": s.title,
            "conditional_weight": s.conditional_weight,
            "probability_band": s.probability_band,
        }
        for s in scenarios
    }
    scenario_probs_summary["leading_scenario"] = leading_scen.scenario_type
    scenario_probs_summary["is_empirical_calibration"] = False
    scenario_probs_summary["calibration_note"] = (
        "Heuristic weights sum to 1.0000 for structured comparative prioritization only; NOT statistically calibrated probabilities."
    )

    failure_disclosure = None
    if is_partial:
        failed_summary = ", ".join(f"{f.get('model_slot')} ({f.get('model_name')})" for f in failed_models) or "some model slots"
        failure_disclosure = f"Partial completion ({succ_cnt}/{exp_cnt} valid slots). Failures in {failed_summary} lowered verdict confidence by {partial_penalty:.0%}."

    # Q1: Immediate Outcome with explicit mechanism, no premise repetition or auto-invented bans
    q1_eff = _clean_sentence(p1_s1_eff)
    q1_eff_punct = q1_eff if q1_eff.endswith((".", "!", "?")) else f"{q1_eff}."
    q1_mech = _clean_sentence(pa_steps[0].mechanism)
    q1_mech_punct = q1_mech if q1_mech.endswith((".", "!", "?")) else f"{q1_mech}."
    if not corroborated_inst_actions or is_partial:
        q1_insuff = " [Insufficiency Disclosure: Limited corroborated immediate institutional action in signal or Micro ensemble.]"
    elif inst["evidence_status"] != "supported":
        q1_insuff = " [Insufficiency Disclosure: Immediate institutional action is model-inferred; primary source evidence does not yet directly corroborate formal action.]"
    else:
        q1_insuff = ""
    q1_text = (
        f"Immediate reaction (days): {q1_eff_punct} Causal mechanism: {q1_mech_punct} "
        f"Concurrently, simulated broadcast salience lifts simulated media attention to {attn_ind:.1%} "
        f"vs {act_adopt:.1%} simulated active adoption.{q1_insuff}"
    )

    # Q2: Strictly separated 1st -> causal transition -> 2nd -> causal transition -> 3rd order
    q2_first = _clean_sentence(pb_steps[0].effect)
    q2_second = _clean_sentence(patt_cons_1)
    q2_third = _clean_sentence(patt_cons_2)
    q2_t12 = _clip_phrase(pb_steps[1].mechanism, 110)
    q2_t23 = _clip_phrase(pb_steps[2].mechanism, 110)
    if _text_similarity(q2_first, q2_second) >= 0.58 and not p2_s2_missing:
        q2_second = f"Secondary actor repositioning across {patt_actor_lbl} distinct from immediate event response."
    if _text_similarity(q2_second, q2_third) >= 0.58 and not p2_s3_missing:
        q2_third = "Insufficient non-overlapping 3rd-order consequence; requires empirical tracking beyond Step 2."
    q2_text = (
        f"1st-order: {q2_first} --(via {q2_t12})--> "
        f"2nd-order: {q2_second} --(via {q2_t23})--> "
        f"3rd-order: {q2_third}"
    )

    # Q3: Identifiable Harm Mechanisms Only (Benign structural competition & reassuring outcomes excluded)
    sev_note = (
        f"Doomer risk severity={doom['mean_risk_severity']:.2f}/1.00"
        if doom.get("mean_risk_severity") is not None
        else "Harm severity evaluated separately from directional stance"
    )
    q3_harm = _clean_sentence(pd_steps[1].effect)
    q3_mech = _clip_phrase(pd_steps[1].mechanism, 120)
    q3_text = (
        f"Downside harm risk ({rounded_w[3] * 100:.0f}% extreme tail / {rounded_w[2] * 100:.0f}% negative heuristic weight; "
        f"escalation risk proxy {escalation_risk:.1%}; {sev_note}): {q3_harm} "
        f"Harm mechanism: {q3_mech} (Note: benign competition, lower costs, startup entry, and exonerating clearings are excluded from downside risks.)"
    )

    # Q4: Categorized Stabilizing Outcome with clean punctuation & anti-fragmentation guard
    pos_trig_clean = _clean_sentence(positive_triggers[0]).rstrip(".")
    raw_q4_eff = _clean_sentence(pa_steps[1].effect).rstrip(".")
    if "insufficient grounded" in raw_q4_eff.lower():
        raw_q4_eff = (
            f"Model-inferred stabilizing mechanism: {_clean_sentence(profile['positive_short']).rstrip('.')} "
            f"(direct Skeptic Micro follow-up unavailable in record)"
        )
    pos_cat = _classify_positive_category(f"{raw_q4_eff} {pos_trig_clean}")
    q4_eff = raw_q4_eff
    q4_text = (
        f"[{pos_cat}] Stabilizing outcome ({rounded_w[1] * 100:.0f}% positive heuristic weight): "
        f"{q4_eff}. Triggered if {pos_trig_clean}."
    )

    exec_secondary_actor = (
        patt_actor_lbl
        if patt_actor_lbl.lower() != inst_actor_lbl.lower()
        else (profile["patt_stakeholder"] if profile["patt_stakeholder"].lower() != inst_actor_lbl.lower() else "cross-sector adopters")
    )

    # Single authoritative Phase 6.4 Forecast Summary & 5-stage compact causal chain
    scen_dicts = [sc.model_dump() for sc in scenarios]
    lead_scen_dict = leading_scen.model_dump()
    stake_dicts = [sr.model_dump() for sr in stakeholder_reactions]
    order_eff_dict = order_effects.model_dump()
    sum_pack = synthesize_forecast_summary_and_chain(
        event_title=event_title,
        signal_text=signal_text,
        confirmed_facts=epistemic_breakdown.source_confirmed_facts,
        unverified_allegations=epistemic_breakdown.unverified_allegations,
        q1_immediate=q1_eff,
        q2_second=q2_second,
        q2_third=q2_third,
        q3_downside=q3_harm,
        q4_stabilizing=q4_eff,
        stakeholder_reactions=stake_dicts,
        order_effects=order_eff_dict,
        scenarios=scen_dicts,
        leading_scenario=lead_scen_dict,
        gaps_classified=evidence_gaps_classified,
        triggers_classified=triggers_classified,
        macro_state=last_macro,
        feedback_state=last_fb,
        domain_ctx=profile["domain_ctx"],
        model_names=(
            str(cfg.get("model_a") or (sim.get("diagnostics") or {}).get("selected_models", {}).get("model_a", {}).get("model") or "Model A"),
            str(cfg.get("model_b") or (sim.get("diagnostics") or {}).get("selected_models", {}).get("model_b", {}).get("model") or "Model B"),
        ),
        provider_names=(
            str(cfg.get("provider_a") or (sim.get("diagnostics") or {}).get("selected_models", {}).get("model_a", {}).get("provider") or "unknown"),
            str(cfg.get("provider_b") or (sim.get("diagnostics") or {}).get("selected_models", {}).get("model_b", {}).get("provider") or "unknown"),
        ),
        valid_slots=succ_cnt,
        expected_slots=exp_cnt,
        is_partial=is_partial,
        is_failed=False,
        use_llm=bool(use_ai_summary),
    )

    final_verdict = FinalVerdict(
        research_question=CORE_RESEARCH_QUESTION,
        headline=(
            f"Most Plausible Trajectory: {leading_scen.title} "
            f"({leading_scen.conditional_weight * 100:.0f}% heuristic weight)"
        ),
        executive_summary=(
            f"Evaluating '{event_title}' across {len(iterations)} coupled iterations ({succ_cnt}/{exp_cnt} valid slots) "
            f"yields simulated media attention salience of {attn_ind:.1%} vs {act_adopt:.1%} retained simulated active adoption "
            f"(Bass Diffusion proxy, cumulative simulated diffusion {cum_adopt:.1%}). "
            f"Leading heuristic trajectory: '{leading_scen.title}', driven primarily through {inst_actor_lbl} and {exec_secondary_actor}."
        ),
        forecast_summary=sum_pack["forecast_summary"],
        forecast_summary_meta=sum_pack["forecast_summary_meta"],
        causal_chain_compact=sum_pack["causal_chain_compact"],
        initial_event_date=sum_pack["initial_event_date"],
        most_likely_immediate_outcome=q1_text,
        most_plausible_secondary_cascade=q2_text,
        most_dangerous_downside_risk=q3_text,
        most_plausible_stabilizing_outcome=q4_text,
        key_triggers_and_turning_points=key_triggers,
        triggers_classified=triggers_classified,
        evidence_and_information_gaps=gaps,
        evidence_gaps_classified=evidence_gaps_classified,
        recommended_monitoring_actions=monitoring_actions,
        monitoring_matrix=monitoring_matrix,
        verdict_confidence=verdict_conf,
        confidence_band=conf_band,
        uncertainty=verdict_unc,
        scenario_probabilities=scenario_probs_summary,
        calibration_warning=CALIBRATION_WARNING,
        traceability_refs={
            "simulation_id": sim_id,
            "status": sim_status,
            "event_domain": profile["domain"],
            "secondary_themes": profile["domain_ctx"].get("secondary_themes", []),
            "event_entities": profile["domain_ctx"].get("entities", []),
            "stakeholders_mapped": [s["stakeholder"] for s in profile["stakeholder_map"]],
            "iterations_completed": len(iterations),
            "valid_slots": succ_cnt,
            "expected_slots": exp_cnt,
            "model_agreement_score": agr_score,
            "causal_semantic_similarity": agr_details.get("causal_semantic_similarity"),
            "agreement_stance_without_reasoning_warning": bool(agr_details.get("stance_without_reasoning_warning", False)),
            "convergence_direction": conv_details.get("direction"),
            "archetype_collapse_warning": bool(conv_details.get("archetype_collapse_warning", False)),
            "reasoning_differentiation_score": conv_details.get("reasoning_differentiation_score"),
            "legitimate_consensus_detected": bool(conv_details.get("legitimate_consensus_detected", False)),
            "inter_archetype_reasoning_similarity": conv_details.get("inter_archetype_reasoning_similarity"),
            "final_active_adoption": round(act_adopt, 4),
            "final_cumulative_adoption": round(cum_adopt, 4),
            "final_attention_indicator": round(attn_ind, 4),
            "final_polarization_index": round(polarization, 4),
            "polarization_delta": round(pol_delta, 4),
            "weak_links_detected": weak_links_total,
            "mean_pathway_plausibility": mean_pw_plaus,
            "mean_pathway_grounding": mean_pw_ground,
            "scenario_diversity_score": scenario_diversity_score,
            "scenario_max_pairwise_similarity": round(max_scen_sim, 4),
            "scenario_diversity_warnings": scen_warnings,
            "leading_scenario": leading_scen.scenario_type,
            "monitoring_matrix": monitoring_matrix,
            "relevance_audit": {
                "rejected_off_domain_count": len(audit_collector["rejected_off_domain"]),
                "withheld_numeric_count": len(audit_collector["withheld_numeric"]),
                "filtered_premise_actions_count": len(audit_collector["filtered_premise_actions"]),
                "filtered_reassuring_risks_count": len(audit_collector["filtered_reassuring_risks"]),
                "filtered_horizon_mismatches_count": len(audit_collector["filtered_horizon_mismatches"]),
                "rejected_samples": audit_collector["rejected_off_domain"][:5],
                "withheld_numeric_samples": audit_collector["withheld_numeric"][:5],
            },
        },
        is_partial_or_failed=is_partial,
        failure_disclosure=failure_disclosure,
    )

    return {
        "research_question": CORE_RESEARCH_QUESTION,
        "causal_pathways": [pw.model_dump() for pw in causal_pathways],
        "order_effects_summary": order_effects.model_dump(),
        "stakeholder_reactions": [sr.model_dump() for sr in stakeholder_reactions],
        "scenarios": [sc.model_dump() for sc in scenarios],
        "time_horizon_forecast": time_horizon_forecast.model_dump(),
        "epistemic_breakdown": epistemic_breakdown.model_dump(),
        "final_verdict": final_verdict.model_dump(),
    }


def ensure_forecast_fields(sim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Non-destructively returns a copy/view enriched with causal pathways, 4 scenarios,
    epistemic breakdown, Phase 6.3.1 relevance audit & event-grounded stakeholder mappings,
    and Phase 6.3 FinalVerdict fields if an older archived simulation JSON lacks them.
    Does NOT write to disk or mutate callers in place.
    """
    fv = sim_data.get("final_verdict") if isinstance(sim_data, dict) else None
    trace = fv.get("traceability_refs") if isinstance(fv, dict) else None
    iterations = sim_data.get("iterations") if isinstance(sim_data, dict) else None
    last_meso = (
        iterations[-1].get("meso_state")
        if isinstance(iterations, list) and iterations and isinstance(iterations[-1], dict)
        else None
    )
    has_stake_map = (
        not isinstance(last_meso, dict)
        or isinstance(last_meso.get("event_stakeholder_mapping"), list) and len(last_meso.get("event_stakeholder_mapping") or []) >= 1
    )
    has_v631_verdict = (
        isinstance(fv, dict)
        and isinstance(fv.get("triggers_classified"), dict)
        and bool(fv.get("triggers_classified"))
        and isinstance(fv.get("evidence_gaps_classified"), dict)
        and bool(fv.get("evidence_gaps_classified"))
        and isinstance(fv.get("monitoring_matrix"), list)
        and (len(fv.get("monitoring_matrix") or []) >= 1 or bool(fv.get("is_partial_or_failed")))
        and isinstance(trace, dict)
        and (isinstance(trace.get("relevance_audit"), dict) or bool(fv.get("is_partial_or_failed")))
    )
    has_v64_summary = (
        isinstance(fv, dict)
        and isinstance(fv.get("forecast_summary"), str)
        and bool(fv.get("forecast_summary").strip())
        and isinstance(fv.get("causal_chain_compact"), dict)
        and bool(fv.get("causal_chain_compact"))
    )
    if (
        isinstance(sim_data, dict)
        and has_v631_verdict
        and has_v64_summary
        and has_stake_map
        and isinstance(sim_data.get("scenarios"), list)
        and len(sim_data.get("scenarios") or []) >= 4
    ):
        return sim_data

    bundle = build_forecast_bundle(sim_data, use_ai_summary=False)
    enriched = dict(sim_data)
    enriched.update(bundle)

    # Non-destructively enrich last iteration meso_state with event_stakeholder_mapping if missing
    if isinstance(iterations, list) and iterations:
        it_copy = [dict(it) if isinstance(it, dict) else it for it in iterations]
        last_it = it_copy[-1]
        if isinstance(last_it, dict) and isinstance(last_it.get("meso_state"), dict):
            ms_copy = dict(last_it["meso_state"])
            if not ms_copy.get("event_stakeholder_mapping"):
                sig = str(enriched.get("signal_text") or (enriched.get("config") or {}).get("signal") or "")
                valid_m = [
                    m for m in (last_it.get("micro_interpretations") or [])
                    if _is_valid_slot(m)
                ]
                ms_copy["event_stakeholder_mapping"] = extract_event_stakeholders(
                    signal_text=sig,
                    valid_micros=valid_m,
                    meso_activations=ms_copy.get("node_activations") or {},
                )
            last_it["meso_state"] = ms_copy
            enriched["iterations"] = it_copy

    return enriched
