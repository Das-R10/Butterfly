"""
src/core/event_grounding.py

Shared Event-Domain Classification, Entity & Stakeholder Extraction, Causal Relevance
Validation, Premise-Duplication Guards, Exoneration Filtering, and Numerical
Grounding Utilities.

Ensures forecasts, scenarios, stakeholders, causal pathways, and monitoring indicators
stay strictly anchored to the user's initial event rather than leaking unrelated
domain templates (customs/ports/freight/educators/open-source/benchmarks) or ungrounded
quantitative projections.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ── Word-Boundary Keyword Helpers ─────────────────────────────────────────────

_REGEX_CACHE: Dict[str, re.Pattern[str]] = {}


def has_kw(text: str, kw: str) -> bool:
    """Case-insensitive word-boundary match preventing substring collisions ('port' != 'reports')."""
    if not text or not kw:
        return False
    key = kw.lower().strip()
    if not key:
        return False
    pat = _REGEX_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", re.IGNORECASE)
        _REGEX_CACHE[key] = pat
    return bool(pat.search(text))


def any_kw(text: str, kws: Tuple[str, ...]) -> bool:
    return any(has_kw(text, k) for k in kws)


def count_kw_hits(text: str, kws: Tuple[str, ...]) -> int:
    return sum(1 for k in kws if has_kw(text, k))


# ── Domain Cue Dictionaries (Word-Boundary Exactness) ─────────────────────────

DOMAIN_CUES: Dict[str, Tuple[str, ...]] = {
    "corporate_governance": (
        "board",
        "independent board",
        "board investigation",
        "independent investigation",
        "investigation",
        "inquiry",
        "ceo",
        "chief executive",
        "cfo",
        "executive",
        "leadership",
        "director",
        "directors",
        "shareholder",
        "shareholders",
        "investor",
        "investors",
        "governance",
        "corporate governance",
        "whistleblower",
        "misconduct",
        "alleging",
        "allegation",
        "allegations",
        "alleges",
        "audit committee",
        "resignation",
        "succession",
        "fiduciary",
        "compliance review",
        "sec",
        "doj",
        "cfius",
    ),
    "geopolitics_security": (
        "foreign intelligence",
        "intelligence agency",
        "intelligence agencies",
        "intelligence operation",
        "espionage",
        "counterintelligence",
        "national security",
        "state secrets",
        "classified",
        "targeted",
        "targeting",
        "cyber espionage",
        "geopolitical",
    ),
    "ai_software": (
        "ai",
        "artificial intelligence",
        "chatgpt",
        "openai",
        "deepseek",
        "r1",
        "llm",
        "llms",
        "chatbot",
        "generative ai",
        "conversational ai",
        "reasoning model",
        "foundation model",
        "language model",
        "machine learning",
        "neural network",
        "open-source model",
        "open-weight",
        "open weight",
        "inference cost",
        "training cost",
        "software",
        "developer",
        "developers",
    ),
    "trade_supply": (
        "tariff",
        "tariffs",
        "customs",
        "customs duty",
        "import duty",
        "import duties",
        "battery imports",
        "port",
        "ports",
        "seaport",
        "port congestion",
        "port closure",
        "shipping",
        "freight",
        "cargo",
        "dockworker",
        "dockworkers",
        "container",
        "containers",
        "strait",
        "supply chain",
        "logistics",
        "inventory",
        "inventories",
        "export ban",
        "export control",
        "export controls",
        "embargo",
    ),
    "finance_macro": (
        "central bank",
        "federal reserve",
        "fed",
        "liquidity",
        "credit",
        "deposit",
        "deposits",
        "bank run",
        "inflation",
        "interest rate",
        "interest rates",
        "equity",
        "currency",
        "debt",
        "bond",
        "bonds",
        "banking",
        "solvency",
        "recession",
    ),
    "cyber_infra": (
        "cyber",
        "cybersecurity",
        "outage",
        "dns",
        "routing outage",
        "ransomware",
        "data breach",
        "security breach",
        "cloud outage",
        "vulnerability",
        "zero-day",
        "telecom",
        "power grid",
        "grid outage",
        "hack",
        "hacked",
        "failover",
        "payment apis",
        "authentication portals",
        "incident",
    ),
    "public_health": (
        "public health",
        "health",
        "virus",
        "viral",
        "vaccine",
        "hospital",
        "hospitals",
        "disease",
        "outbreak",
        "pandemic",
        "epidemic",
        "pathogen",
        "clinic",
        "clinics",
        "respiratory",
        "clinical",
        "fda",
        "patients",
        "admissions",
    ),
    "geopolitics_energy": (
        "oil",
        "gas",
        "crude",
        "opec",
        "pipeline",
        "lng",
        "energy",
        "sanction",
        "sanctions",
        "treaty",
        "embargo",
        "diplomatic",
        "military",
        "border",
        "alliance",
        "minister",
    ),
}

_STOP_ENTITY_WORDS = {
    "On", "In", "At", "By", "For", "To", "From", "With", "And", "Or", "The", "A", "An",
    "This", "That", "These", "Those", "It", "He", "She", "They", "We", "You", "Some",
    "Any", "All", "Many", "Several", "Such", "First", "Second", "Third", "Given",
    "Reports", "Report", "Initial", "However", "Despite", "Global", "Employment",
    "Consumer", "Enterprise", "Startup", "Open", "Market", "Public", "Users",
    "Since", "When", "Where", "While", "After", "Before", "Following", "Its", "Their",
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "Short", "Medium", "Long", "Immediate", "Phase", "Round", "Note",
    "Analyze", "Forecast", "Explore", "Distinguish", "Evaluate", "Assess", "Examine",
    "Investigate", "Consider", "Identify", "Simulate", "Determine", "Compare", "Explain",
    "Describe", "Discuss", "Provide", "Verify", "Track", "Monitor", "Assume", "Do",
}

_SIGNAL_STOPWORDS = {
    "that", "this", "with", "from", "into", "about", "have", "were", "been",
    "they", "their", "there", "which", "would", "could", "should", "will",
    "more", "than", "also", "when", "what", "where", "while", "after", "before",
    "between", "through", "such", "many", "some", "other", "over", "under",
    "publicly", "initial", "signal", "event", "reported", "reports", "alleging",
    "triggered", "leading", "increased", "widespread", "tool", "system", "impact",
}


def _looks_like_org(ent: str, signal_text: str = "") -> bool:
    if not ent:
        return False
    up = ent.upper()
    if up in {"CEO", "CFO", "CTO", "COO", "CIO", "US", "UK", "EU", "UN", "AI", "IT", "ML"}:
        return False
    if len(ent) >= 2 and ent == up and any(ch.isalpha() for ch in ent) and "-" not in ent:
        return True
    if re.search(r"[a-z][A-Z]", ent):
        return True
    if re.search(
        r"\b(?:Inc|Corp|Corporation|Ltd|LLC|Group|Holdings|Labs|Bank|Financial|Motors|Systems|Technologies|Foundation|University|Ministry|Agency)\b",
        ent,
    ):
        return True
    if signal_text and re.search(
        rf"\b{re.escape(ent)}(?:'s|’s)?\s+(?:CEO|CFO|CTO|COO|Chair|President|Founder|Board)\b",
        signal_text,
    ):
        return True
    return False


def extract_event_entities(signal_text: str) -> Dict[str, Any]:
    """
    Dynamically extracts named entities, companies, leaders, acronyms, and signal
    tokens strictly from the input event text.
    """
    sig = re.sub(r"\s+", " ", (signal_text or "").strip())
    if not sig:
        return {
            "proper_entities": [],
            "org_entity": "",
            "person_entity": "",
            "subject_label": "the subject organization",
            "signal_tokens": set(),
            "first_sentence": "",
        }

    first_sentence = re.split(r"(?<=[.!?])\s+", sig, maxsplit=1)[0].strip()

    # Match capitalized multi-word names & acronyms (e.g. "Jensen Huang", "NVIDIA", "OpenAI", "DeepSeek", "DNS", "R1")
    cand_matches = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9&\-]*(?:\s+[A-Z][A-Za-z0-9&\-]*){0,3}|[A-Z]{2,}(?:-\d+)?)\b",
        sig,
    )
    entities: List[str] = []
    seen_low: set[str] = set()
    _ROLE_SPLIT_RE = re.compile(r"\b(?:CEO|CFO|CTO|COO|CIO|Chairman|Chairwoman|Chair|President|Founder)\b")
    for raw in cand_matches:
        sub_spans = [seg.strip() for seg in _ROLE_SPLIT_RE.split(raw) if seg.strip()]
        if not sub_spans:
            sub_spans = [raw]
        for sub in sub_spans:
            parts = [p for p in sub.split() if p not in _STOP_ENTITY_WORDS]
            clean = " ".join(parts).strip(" ,.;:()[]{}\"'")
            clean = re.sub(r"(?:'s|’s)$", "", clean).strip()
            if not clean or len(clean) < 2:
                continue
            if clean.upper() in {"CEO", "CFO", "CTO", "COO", "CIO", "AI", "US", "UK", "EU", "UN"} and clean.upper() in entities:
                continue
            low = clean.lower()
            if low in seen_low:
                continue
            seen_low.add(low)
            entities.append(clean)

    # Prefer concrete orgs / people cleanly
    priority_entities = [e for e in entities if e.upper() not in {"CEO", "CFO", "CTO", "US", "UK", "EU", "AI", "IT", "ML"}]
    org_entity = next((e for e in priority_entities if _looks_like_org(e, sig)), "")
    person_entity = next(
        (e for e in priority_entities if e != org_entity and not _looks_like_org(e, sig) and "-" not in e),
        "",
    )
    if org_entity and person_entity:
        subject_label = f"{org_entity} / {person_entity}"
    else:
        chosen = (priority_entities or entities)[:4]
        if len(chosen) >= 2:
            subject_label = f"{chosen[0]} / {chosen[1]}"
        elif chosen:
            subject_label = chosen[0]
        else:
            subject_label = "the focal organization and stakeholders"

    signal_tokens = {
        t
        for t in re.split(r"[^a-zA-Z0-9]+", sig.lower())
        if len(t) >= 4 and t not in _SIGNAL_STOPWORDS
    }

    return {
        "proper_entities": entities[:8],
        "org_entity": org_entity or (priority_entities[0] if priority_entities else ""),
        "person_entity": person_entity,
        "subject_label": subject_label,
        "signal_tokens": signal_tokens,
        "first_sentence": first_sentence,
    }


def classify_event_domain(
    signal_text: str,
    valid_micros: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Classifies the event domain primarily from `signal_text` with light Micro tag support.
    Uses strict word boundaries so substrings like 'port' inside 'reports' NEVER collide.
    """
    sig = (signal_text or "").strip()
    micro_corpus = ""
    if valid_micros:
        tags_and_narr: List[str] = []
        for m in valid_micros[:12]:
            tags_and_narr.extend(str(t) for t in (m.get("narrative_tags") or []))
        micro_corpus = " ".join(tags_and_narr)

    sig_counts: Dict[str, int] = {}
    scores: Dict[str, float] = {}
    for dom, cues in DOMAIN_CUES.items():
        s_cnt = count_kw_hits(sig, cues)
        m_cnt = count_kw_hits(micro_corpus, cues)
        sig_counts[dom] = s_cnt
        # Anchor heavily on primary signal_text so Micro off-domain drift cannot hijack domain
        scores[dom] = (3.0 * s_cnt) + (0.45 * m_cnt)

    # Explicit disambiguations:
    # 1. Strong corporate governance / leadership investigation outranks collateral hardware/macro terms
    has_exec_or_board = any_kw(
        sig,
        ("board", "independent board", "board investigation", "independent investigation", "ceo", "cfo", "executive", "director", "shareholder", "whistleblower"),
    )
    has_explicit_trade = any_kw(
        sig,
        ("tariff", "tariffs", "customs", "shipping", "freight", "cargo", "dockworker", "dockworkers", "container", "strait", "port", "ports", "seaport", "import duty", "battery imports", "supply chain"),
    )
    if has_exec_or_board and not has_explicit_trade:
        scores["corporate_governance"] += 3.5
        # Do not let incidental security words classify a corporate board investigation as a pure infrastructure outage
        if not any_kw(sig, ("dns", "outage", "ransomware", "data breach", "power grid", "telecom", "hacked")):
            scores["cyber_infra"] = max(0.0, scores["cyber_infra"] - 2.0)

    # 2. Disambiguate AI compute/chip discussions from physical logistics unless trade words present
    if scores["ai_software"] >= 3.0 and not has_explicit_trade:
        scores["ai_software"] += 3.0

    # 3. Handle pure AI vs pure security/intelligence events
    best_domain = max(scores, key=lambda k: scores[k])
    if best_domain == "geopolitics_security" and sig_counts.get("corporate_governance", 0) >= 2:
        best_domain = "corporate_governance"

    # Maintain backward compatibility with canonical domain names expected across test suites:
    # ("ai_software", "cyber_infra", "public_health", "trade_supply", "finance_macro", "geopolitics_energy", "corporate_governance", "geopolitics_security", "general")
    if scores[best_domain] < 2.5 and sig_counts.get(best_domain, 0) == 0:
        primary_domain = "general"
    else:
        primary_domain = best_domain

    secondary_themes = [
        dom
        for dom, sc in sorted(scores.items(), key=lambda kv: -kv[1])
        if dom != primary_domain and (sig_counts.get(dom, 0) >= 1 or (sc >= 3.0 and dom != "ai_software"))
    ]

    entity_info = extract_event_entities(sig)

    return {
        "domain": primary_domain,
        "secondary_themes": secondary_themes[:3],
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "signal_counts": sig_counts,
        "entities": entity_info["proper_entities"],
        "org_entity": entity_info["org_entity"],
        "person_entity": entity_info["person_entity"],
        "subject_label": entity_info["subject_label"],
        "signal_tokens": entity_info["signal_tokens"],
        "has_ai_signal": sig_counts.get("ai_software", 0) >= 1,
        "has_semiconductor": any_kw(sig, ("nvidia", "semiconductor", "gpu", "gpus", "chip", "chips", "amd", "intel", "tsmc", "wafer", "accelerator", "foundry")),
        "has_investigation": any_kw(sig, ("investigation", "inquiry", "probe", "audit", "alleging", "allegation", "allegations")),
        "has_foreign_intel": any_kw(sig, ("foreign intelligence", "intelligence agency", "intelligence agencies", "espionage", "counterintelligence", "national security")),
        "has_executive": any_kw(sig, ("ceo", "cfo", "executive", "chief executive", "jensen huang", "founder", "leadership")),
        "has_education": any_kw(sig, ("educator", "educators", "university", "universities", "school", "schools", "student", "students", "classroom", "homework")),
        "has_open_source": any_kw(sig, ("open-source", "open source", "open-weight", "open weight")),
        "has_explicit_trade": has_explicit_trade,
    }


# ── Off-Domain Contamination Fingerprints & Multi-Step Causal Bridge Validator ─

_LOGISTICS_TRADE_CONTAMINATION = (
    "transit lead-time",
    "transit lead time",
    "customs exemption",
    "customs exemptions",
    "customs guidance",
    "freight/input",
    "freight cost",
    "multi-port",
    "clearance stoppage",
    "clearance stoppages",
    "retaliatory counter-tariff",
    "counter-tariff",
    "counter-tariffs",
    "retaliatory tariff",
    "procedural waiver",
    "procedural waivers",
    "smooth routing",
    "pull forward orders",
    "buffer hoarding compresses margins",
    "parallel supplier architecture",
    "corridor throughput",
    "corridor halt",
    "physical corridor",
    "dockworker",
    "container backlog",
    "maritime shipping",
    "import duty",
    "bill of lading",
    "trade, customs",
    "trade, customs & regulatory",
    "supply auditors, trade analysts",
    "supply-chain, procurement & market strategists",
    "interdependent trade hubs",
    "logistics & working-capital networks",
    "precautionary cost pass-through",
    "defensive liquidity freeze",
    "bilateral stabilization accords",
)

_EDUCATION_CONTAMINATION = (
    "educator",
    "educators",
    "classroom",
    "k-12",
    "homework",
    "curriculum",
    "plagiarism",
    "syllabus",
)

_OPEN_SOURCE_PROMPT_LEAK = (
    "open-source ecosystems",
    "open-source ecosystem",
    "open-source builders",
    "open-source application builders",
    "open source builders",
    "open source ecosystems",
    "open-source developer",
    "open-source developers",
    "open source developer",
    "open source developers",
)

_CAUSAL_BRIDGE_VERBS = (
    "triggers",
    "triggering",
    "leads to",
    "leading to",
    "causes",
    "causing",
    "resulting in",
    "forces",
    "forcing",
    "prompts",
    "prompting",
    "cascades into",
    "escalates into",
    "spills into",
    "because",
    "->",
    "-->",
)


def _has_explicit_causal_bridge(text: str, domain_ctx: Dict[str, Any]) -> bool:
    """
    Returns True ONLY if `text` connects an anchor concept/entity of the initial event
    to an indirect domain consequence via an explicit multi-step causal mechanism.
    Prevents bare imported jargon while permitting real indirect cross-domain cascades.
    """
    low = (text or "").lower()
    words = [w for w in re.split(r"\s+", low) if w]
    if len(words) < 9:
        return False

    has_connector = any(v in low for v in _CAUSAL_BRIDGE_VERBS)
    if not has_connector:
        return False

    anchor_terms: set[str] = set(domain_ctx.get("signal_tokens") or set())
    for ent in domain_ctx.get("entities") or []:
        for part in re.split(r"[^a-zA-Z0-9]+", ent.lower()):
            if len(part) >= 3:
                anchor_terms.add(part)

    has_anchor = any(has_kw(low, t) for t in anchor_terms)
    return has_anchor


def sanitize_prompt_anchor_leaks(
    text: str,
    domain_ctx: Dict[str, Any],
) -> str:
    """
    Cleanly strips stale static prompt-anchor lists (e.g., ', educators' in
    Institutional Analyst prompt or 'Open-source ecosystems, ' when unrelated to the
    initial event) while keeping real event-relevant references untouched.
    """
    s = str(text or "").strip()
    if not s:
        return ""
    domain = domain_ctx.get("domain", "general")
    allow_edu = domain == "ai_software" or bool(domain_ctx.get("has_education"))
    if not allow_edu:
        s = re.sub(r",\s*educators\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\beducators\s*,\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*\band\s+educators\b", "", s, flags=re.IGNORECASE)

    allow_oss = domain == "ai_software" or bool(domain_ctx.get("has_open_source"))
    if not allow_oss and not _has_explicit_causal_bridge(s, domain_ctx):
        s = re.sub(
            r",\s*open[-\s]?source\s+(?:ecosystems?|builders?|application\s+builders?|developers?)\b",
            "",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"\bopen[-\s]?source\s+(?:ecosystems?|builders?|application\s+builders?|developers?)\s*(?:,|and)\s+",
            "",
            s,
            flags=re.IGNORECASE,
        )
    s = re.sub(r"\s{2,}", " ", s).strip(" ,")
    return s


def check_event_relevance(
    text: str,
    domain_ctx: Dict[str, Any],
    *,
    allow_bridged: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validates whether a generated scenario, indicator, stakeholder, or causal text
    is causally relevant to the initial event domain.
    Returns (is_relevant, rejection_reason).
    """
    raw = str(text or "").strip()
    if not raw:
        return False, "empty_text"
    low = raw.lower()
    domain = domain_ctx.get("domain", "general")
    sec_themes = set(domain_ctx.get("secondary_themes") or [])

    # 1. Logistics / Trade / Customs / Port boilerplate on non-trade events
    if domain != "trade_supply" and not domain_ctx.get("has_explicit_trade"):
        for phrase in _LOGISTICS_TRADE_CONTAMINATION:
            if phrase in low:
                if allow_bridged and _has_explicit_causal_bridge(raw, domain_ctx):
                    return True, None
                return False, f"off_domain_trade_logistics:{phrase}"

    # 2. Unrelated classroom/educators leakage on non-education / non-consumer-AI events
    allow_edu = domain == "ai_software" or bool(domain_ctx.get("has_education"))
    if not allow_edu:
        for edu_kw in _EDUCATION_CONTAMINATION:
            if has_kw(low, edu_kw):
                if allow_bridged and _has_explicit_causal_bridge(raw, domain_ctx):
                    return True, None
                return False, f"off_domain_education:{edu_kw}"

    # 3. Unrelated open-source ecosystem prompt-anchor leak on non-software events
    allow_oss = domain == "ai_software" or bool(domain_ctx.get("has_open_source"))
    if not allow_oss:
        for oss_kw in _OPEN_SOURCE_PROMPT_LEAK:
            if oss_kw in low:
                if allow_bridged and _has_explicit_causal_bridge(raw, domain_ctx):
                    return True, None
                return False, f"off_domain_open_source:{oss_kw}"

    # 4. Hardcoded AI benchmark reversal fallback on non-AI/non-tech events
    if domain not in ("ai_software", "cyber_infra") and "competing benchmark reverses" in low:
        return False, "off_domain_ai_benchmark_fallback"

    return True, None


# ── Completed Premise / Initial-Trigger Duplication Guard ─────────────────────

_INITIATION_VERBS = (
    "initiat", "launch", "open", "start", "conven", "trigger", "announc", "releas", "declar", "begin",
)
_PREMISE_DISRUPTION_OUTCOME_CUES = (
    "finding", "findings", "conclud", "report", "audit", "disclos", "subpoena", "indict",
    "remov", "resign", "sanction", "customer", "shareholder", "investor", "partner",
    "timeline", "within", "expand", "scope", "delay", "suspend", "guidance", "transparency",
    "clarif", "statement", "competitor",
)


def is_completed_premise_action(
    text: str,
    signal_text: str,
    confirmed_facts: Optional[List[str]] = None,
) -> bool:
    """
    Detects when a predicted 1st-order 'action' or 'effect' merely repeats an action
    that ALREADY occurred/triggered in `signal_text` (e.g., predicting 'Board initiates
    formal inquiry protocol' when signal states '...have triggered an independent board
    investigation'). Genuine post-trigger developments (e.g., releasing findings, audits,
    customer due diligence) return False.
    """
    low = re.sub(r"\s+", " ", (text or "").lower()).strip()
    sig_low = re.sub(r"\s+", " ", ((signal_text or "") + " " + " ".join(confirmed_facts or [])).lower()).strip()
    if not low or not sig_low:
        return False

    # Synonymic group normalization for already-triggered investigations/releases/outages
    has_start_verb = any(v in low for v in _INITIATION_VERBS)
    if not has_start_verb:
        return False

    # Case 1: Board investigation already triggered in signal
    if (
        ("board investigation" in sig_low or "independent investigation" in sig_low or "independent board" in sig_low)
        and any(has_kw(low, w) for w in ("board", "committee", "company"))
        and any(has_kw(low, w) for w in ("investigation", "inquiry", "probe", "review"))
    ):
        # If it introduces substantive post-initiation specifics (e.g. findings, audits of access logs, timeline, scope expansion), keep it!
        has_post_trigger_detail = any(c in low for c in _PREMISE_DISRUPTION_OUTCOME_CUES)
        # "protocol" alone ("Board initiates formal inquiry protocol") is NOT a new post-trigger development
        if not has_post_trigger_detail:
            return True

    # Case 2: Generic initial release/announcement repeated as a future prediction
    if (
        ("released" in sig_low or "announced" in sig_low or "launched" in sig_low)
        and any(w in low for w in ("releases ", "announces ", "launches "))
    ):
        sig_toks = {t for t in re.split(r"[^a-z0-9]+", sig_low) if len(t) >= 4}
        txt_toks = {t for t in re.split(r"[^a-z0-9]+", low) if len(t) >= 4}
        if sig_toks and txt_toks and (len(sig_toks & txt_toks) / len(txt_toks)) >= 0.50:
            if not any(c in low for c in ("guidance", "follow-up", "patch", "findings", "audit", "policy", "standard")):
                return True

    return False


# ── Exonerating / Reassuring Stabilization Filter (Exclude From Pathway D/Q3) ─

_REASSURING_CUES = (
    "confirm no ",
    "confirms no ",
    "confirming no ",
    "confirmed no ",
    "concludes with no ",
    "concludes that there is no ",
    "findings clear",
    "clears leadership",
    "clears executive",
    "clears ceo",
    "investigation clears",
    "no operational security breach",
    "no criminal activity",
    "no data breach",
    "no security breach",
    "no wrongdoing",
    "no foreign intelligence",
    "no actionable findings",
    "no compromised secrets",
    "lack of concrete breach data",
    "rapid market rationalization",
    "rapid return to normalcy",
    "rapid return to baseline",
    "quickly restores confidence",
    "restores investor confidence",
    "exonerat",
    "definitively cleared",
    "definitively clearing",
)

_REASSURING_NEGATIVE_OVERRIDES = (
    "fails to",
    "failed to",
    "contradicting",
    "leaked documents",
    "despite confirmation",
    "even if no",
    "absent clear findings",
    "prolonged uncertainty",
    "if trust erodes",
)


def is_reassuring_or_stabilizing(text: str) -> bool:
    """
    True when a statement describes exoneration, no wrongdoing/no breach confirmation,
    or rapid rationalization/recovery rather than an adverse failure or downside risk.
    Must be excluded from Pathway D, Negative/Extreme consequences, and Verdict Q3.
    """
    low = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not low:
        return False
    if any(ov in low for ov in _REASSURING_NEGATIVE_OVERRIDES):
        return False
    return any(cue in low for cue in _REASSURING_CUES)


# ── Unsupported Quantitative Forecast Grounding & Reframing ───────────────────

_TELEMETRY_SAFE_CONTEXT = (
    "simulated",
    "heuristic weight",
    "probability band",
    "adoption proxy",
    "diffusion proxy",
    "c=",
    "a(t)",
    "n(t)",
    "diff=",
    "active adoption",
    "cumulative adoption",
    "active_adoption",
    "cumulative_adoption",
    "agreement score",
    "cross-model agreement",
    "polarization index",
    "escalation risk proxy",
    "institutional dampening",
    "attention indicator",
    "confidence index",
    "verdict_confidence",
    "epistemic fragility",
)

_MARKET_METRIC_CUES = (
    "stock",
    "share",
    "shares",
    "price",
    "valuation",
    "revenue",
    "sales",
    "gpu",
    "gpus",
    "market cap",
    "capex",
    "margin",
    "margins",
    "earnings",
    "drop",
    "decline",
    "decrease",
    "fall",
    "fell",
    "plunge",
    "crash",
    "surge",
    "gain",
    "growth",
    "gdp",
    "demand",
    "volume",
    "cut",
    "loss",
    "losses",
    "defection",
    "defections",
    "customer",
    "customers",
    "client",
    "clients",
    "attrition",
    "retention",
    "contract",
    "contracts",
    "procurement",
    "order",
    "orders",
    "investment",
    "spending",
    "budget",
    "budgets",
    "market",
    "churn",
    "adoption",
)

_PCT_OR_DOLLAR_RE = re.compile(
    r"(?:(?:[+-]?\d+(?:\.\d+)?%?\s*(?:-|to|–)\s*)?[+-]?\d+(?:\.\d+)?\s*%|\$\s*\d+(?:\.\d+)?\s*(?:b|m|t|billion|million|trillion|bn|mn)\b)",
    re.IGNORECASE,
)


def reframe_ungrounded_numbers(
    text: str,
    signal_text: str = "",
) -> Tuple[str, Optional[str]]:
    """
    Detects ungrounded quantitative market/sales/stock percentage or dollar projections
    (such as '10-15% stock price decline', '-5% to -10%', '10-15% drop in NVIDIA GPU sales')
    that are NOT present in `signal_text` and NOT labeled simulation telemetry proxies.

    Rewrites them cleanly into qualitative conditional mechanisms and records what
    empirical evidence would be needed to bound actual magnitude.
    Returns (cleaned_text, audit_summary_or_None).
    """
    raw = str(text or "").strip()
    if not raw:
        return "", None
    low = raw.lower()

    # Never rewrite internal simulation telemetry lines (Bass N(t)/A(t)/churn/heuristic weight)
    if any(tok in low for tok in _TELEMETRY_SAFE_CONTEXT):
        return raw, None

    matches = list(_PCT_OR_DOLLAR_RE.finditer(raw))
    if not matches:
        return raw, None

    # Allow numbers explicitly present in the source signal
    sig_low = (signal_text or "").lower()
    ungrounded = [m for m in matches if m.group(0).strip().lower() not in sig_low]
    if not ungrounded:
        return raw, None

    has_market_ctx = any(has_kw(low, mk) for mk in _MARKET_METRIC_CUES)
    if not has_market_ctx:
        return raw, None

    cleaned = raw
    # Pattern variants:
    #   "stock price (10-15%)" -> "stock price"
    #   "decline of 5-10%" / "decline by 10-15%" -> "decline"
    #   "10-15% stock price decline" / "-5% to -10% drop in sales" -> "potential stock price decline" / "potential drop in sales"
    cleaned = re.sub(
        r"\(\s*" + _PCT_OR_DOLLAR_RE.pattern + r"\s*\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:of|by)\s+" + _PCT_OR_DOLLAR_RE.pattern,
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:an?\s+)?"
        + _PCT_OR_DOLLAR_RE.pattern
        + r"\s+(?=(?:[a-z]+\s+){0,2}(?:drop|decline|decrease|fall|plunge|loss|cut|dip|compression|reduction|slump|selloff|sell-off|increase|gain|rise|surge)\b)",
        "potential ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        _PCT_OR_DOLLAR_RE.pattern,
        "an unverified magnitude",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    # Capitalize first letter cleanly if numeric prefix led sentence
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    evidence_note = "(quantitative magnitude ungrounded in signal — depends on primary filings, customer order telemetry, and board findings)"
    if "magnitude" not in cleaned.lower():
        if cleaned.endswith("."):
            cleaned = cleaned[:-1].rstrip() + f" {evidence_note}."
        else:
            cleaned = cleaned + f" {evidence_note}"

    sanitized_preview = _PCT_OR_DOLLAR_RE.sub("[magnitude withheld]", raw[:90])
    audit_note = (
        f"Quantitative market/operating figure withheld from '{sanitized_preview}': magnitude lacks empirical baseline in signal "
        "(requires SEC disclosures, customer order telemetry, and verified institutional flow data)."
    )
    return cleaned, audit_note


# ── Horizon Normalization & Mismatch Guard ────────────────────────────────────

_LEADING_HORIZON_SUBTAG_RE = re.compile(
    r"^(?:short[-\s]?term|immediate|near[-\s]?term|medium[-\s]?term|long[-\s]?term)?\s*"
    r"(?:\(\s*(?:within\s+)?\d+\s*(?:hours?|days?|weeks?|months?|quarters?|years?)\s*\)|\d+[-\s]?(?:hour|day|week|month))?"
    r"\s*(?:[:\-–]+\s*)?",
    re.IGNORECASE,
)

_SHORT_DAYS_RE = re.compile(
    r"(?:\b(?:48\s*hours?|7\s*days?|within\s+\d+\s*hours?|within\s+[1-9]\s*days?|immediate(?:ly)?)\b|"
    r"\btemporary\b.*\bshort[-\s]?term\b|"
    r"\bshort[-\s]?term\s+(?:speculative|trading|volatility|dip|drop|pullback|selling|reaction)\b)",
    re.IGNORECASE,
)
_MEDIUM_MONTHS_RE = re.compile(r"\b(?:30\s*days?|\d+[-\s]?months?|quarterly|long[-\s]?term|multi[-\s]?month)\b", re.IGNORECASE)


def extract_horizon_and_strip_subtag(text: str) -> Tuple[Optional[str], str]:
    """
    Detects inner leading time qualifiers like 'Short-term (7 days) – ...' or '30 days - ...'
    and returns (normalized_order_or_None, clean_body).
    """
    s = str(text or "").strip()
    if not s:
        return None, ""
    order_hint: Optional[str] = None
    low_head = s[:48].lower()
    if re.search(r"^(?:short[-\s]?term|immediate|\(?\s*(?:7|14)\s*days?\b|\(?\s*48\s*hours?)", low_head):
        order_hint = "1st_order"
    elif re.search(r"^(?:30[-\s]?day|medium[-\s]?term)", low_head):
        order_hint = "2nd_order"
    elif re.search(r"^(?:long[-\s]?term|\d+[-\s]?months?|quarters?)", low_head):
        order_hint = "3rd_order"

    # Strip only when it actually looks like a temporal prefix tag (not genuine sentence prose)
    m = re.match(
        r"^(?:(?:short|near|medium|long)[-\s]?term\s*(?:\([^)]+\))?\s*[:\-–]\s*|"
        r"\(\s*(?:within\s+)?\d+\s*(?:hours?|days?|weeks?|months?)\s*\)\s*[:\-–]\s*|"
        r"\d+[-\s]?(?:hour|day|week|month)\s*[:\-–]\s*)",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        body = s[m.end():].strip()
        if body:
            if body[0].islower():
                body = body[0].upper() + body[1:]
            return order_hint, body
    return order_hint, s


def enforce_horizon_alignment(text: str, target_horizon: str) -> Tuple[str, bool]:
    """
    Ensures short-term 'within 7 days' / '48 hours' wording is not embedded inside
    medium_term_months / long_term_quarters slots.
    Returns (cleaned_text, is_aligned).
    """
    order_hint, body = extract_horizon_and_strip_subtag(text)
    if target_horizon in ("medium_term_months", "long_term_quarters"):
        if order_hint == "1st_order":
            return body, False
        if _SHORT_DAYS_RE.search(body) and not _MEDIUM_MONTHS_RE.search(body):
            return body, False
    return body, True


# ── Negation-Aware Trigger & Invalidation Semantic Classification ─────────────

_FALSIF_EXACT = (
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
    "proven to be fabricated",
    "based on misinformation",
)

_CONTRADICTION_OR_ESCALATION_STRONG = (
    "contradicting",
    "contradict",
    "leaked documents",
    "executive removal",
    "leadership removal",
    "forced removal",
    "resignation",
    "stepping down",
    "ousted",
    "indict",
    "indictment",
    "subpoena",
    "severe operational disruption",
    "ip theft",
    "intellectual property theft",
    "formal confirmation of espionage",
    "operational sabotage",
    "impose sanctions",
    "export control",
    "export controls",
    "export ban",
    "retaliat",
    "crackdown",
    "emergency moratorium",
    "compromised intellectual property",
    "confirmed data compromise",
    "confirmed espionage",
    "state secrets extracted",
    "active compromise of",
    "critical failure",
    "systemic outage",
    "breached",
)

_EXONERATING_STABILIZATION_CUES = (
    "confirming no ",
    "confirms no ",
    "confirmed no ",
    "confirm no ",
    "confirmation of no ",
    "finds no ",
    "found no ",
    "finding no ",
    "concludes with no ",
    "concludes that there is no ",
    "no criminal activity",
    "no data breach",
    "no security breach",
    "no operational security",
    "no wrongdoing",
    "no actionable findings",
    "no evidence of",
    "no compromised",
    "exonerat",
    "clears ",
    "clearing ",
    "dismisses allegations",
    "full operational transparency",
    "confirms no leadership changes",
    "remains ceo",
    "remains in control",
)


def classify_trigger_and_condition(text: str) -> Tuple[str, str]:
    """
    Unified negation-aware classifier returning:
      (condition_5way_legacy, trigger_5way)
    where:
      condition_5way_legacy in {'falsification', 'escalation', 'mitigation', 'positive_development', 'neutral'}
      trigger_5way in {'acceleration', 'escalation', 'stabilization', 'reversal', 'falsification'}
    """
    low = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not low:
        return "neutral", "acceleration"

    # 1. Explicit premise falsification ("never occurred", "fabricated", etc.)
    if any(c in low for c in _FALSIF_EXACT):
        return "falsification", "falsification"

    # 2. Explicit escalation contradictions, removals, leaks, sanctions, or sabotage
    #    CHECK THIS BEFORE negation checks because "leaked documents contradicting the 'no compromise' narrative" is ESCALATION!
    if any(c in low for c in _CONTRADICTION_OR_ESCALATION_STRONG):
        # Exception: if sentence explicitly states "investigation concludes ... no evidence of espionage/compromise/sabotage"
        has_exon_neg = any(e in low for e in _EXONERATING_STABILIZATION_CUES)
        has_contra = any(x in low for x in ("contradicting", "leaked", "removal", "resignation", "indict", "sabotage", "impose sanctions"))
        if not (has_exon_neg and not has_contra):
            return "escalation", "escalation"

    # 3. Exonerating / clearing / no-breach / no-wrongdoing findings -> Stabilization / Mitigation
    if any(e in low for e in _EXONERATING_STABILIZATION_CUES):
        return "mitigation", "stabilization"

    # 3b. Explicit absence of crisis ("without acute breakdown", "without disruption") -> stabilization, NOT escalation
    if re.search(
        r"\bwithout\s+(?:any\s+|acute\s+|abrupt\s+|severe\s+|systemic\s+|major\s+)?(?:breakdown|disruption|decoupling|crisis|collapse|failure|failures|breach|incident)\b",
        low,
    ):
        return "mitigation", "stabilization"

    # 4. Reversal cues (rollback, retract, supersede, displace, rescind, pause, debunked, cost/performance anomaly or degradation reversal)
    if any(
        k in low
        for k in (
            "rollback", "retract", "cancel", "suspend", "unwind", "abandoned",
            "supersed", "displace", "news cycle displacement", "pullback", "reverse",
            "rescind", "debunked", "is an anomaly", "inaccurate", "not reproducible",
            "fails to maintain performance", "fails to maintain parity",
        )
    ):
        return "neutral", "reversal"

    # 5. Absence / failure of mitigation -> escalation if threat cues exist
    _ABSENCE_GUARDS = (
        "lack of", "absence of", "without any", "without functioning", "ineffective",
        "remain negligible", "fail to meet", "fails to", "lags behind", "lagging behind",
    )
    _ESC_BASE = (
        "harm", "incident", "failure", "fail ", "failed", "collapse", "crisis",
        "misinformation", "disinformation", "deepfake", "fraud", "outcry", "backlash",
        "unrest", "volatility", "displacement", "layoff", "job loss", "loss ", "losses",
        "breach", "outage", "erosion", "unacceptable risk", "breakdown", "ban ", "bans",
        "banning", "prohibit", "shortage", "spike in", "spikes", "manipulation",
        "sabotage", "lawsuit", "sanction", "disruption", "negative findings",
        "evidence of espionage", "public accusation", "damaging intelligence",
        "classified documents", "code compromise", "data exfiltration", "data compromise",
        "protocol gap", "protocol gaps", "security gap", "security gaps", "control gap",
        "control gaps", "deficienc", "vulnerabilit", "degrad", "inefficienc",
        "hidden cost", "widening divergence",
    )
    _MITIG = (
        "waiver", "exemption", "audit", "verification", "verify", "guideline", "standard",
        "guardrail", "safeguard", "clarif", "compliance", "buffer", "patch", "backstop",
        "stabiliz", "mitigat", "contain", "transparen", "oversight", "framework", "protocol",
        "governance", "calibration", "cooperation", "coordinat", "continuity", "neutraliz",
    )
    _POS = (
        "productiv", "benefi", "workflow", "innovat", "efficienc", "collaborat",
        "literacy", "trust", "constructive", "resilien", "adapt", "responsible",
        "equilibrium", "consensus", "standardiz",
    )

    if any(g in low for g in _ABSENCE_GUARDS):
        if any(e in low for e in _ESC_BASE):
            return "escalation", "escalation"
        return "neutral", "stabilization"

    # 6. General escalation triggers (harm, verified breach, outage, compliance failure)
    #    Exclude genuine reduction/neutralization statements unless overt disruption/removal occurs
    has_esc = any(e in low for e in _ESC_BASE)
    has_reduc = any(
        p in low
        for p in (
            "reduction in ", "reduce ", "prevent ", "decreas", "resolve ", "curb ",
            "avert ", "neutraliz", "defus", "absorb ", "offset ", "narrowing", "declining",
        )
    )
    if has_esc and not has_reduc:
        return "escalation", "escalation"

    # 7. Stabilization / Mitigation / Positive dev
    if any(m in low for m in _MITIG):
        return "mitigation", "stabilization"
    if any(p in low for p in _POS):
        return "positive_development", "stabilization"

    # 8. Acceleration vs Stabilization fallback
    if any(
        k in low
        for k in (
            "accelerat", "adoption", "surge", "viral", "inflow", "startup", "breakthrough",
            "cost", "pricing", "deployment", "scale", "commercial", "preliminary findings",
            "statement", "deadline", "hours", "days", "public interest", "media coverage",
        )
    ):
        return "neutral", "acceleration"
    return "neutral", "stabilization"


# ── Event-Specific Stakeholder Extraction & Meso Relevance Mapping ────────────

def extract_event_stakeholders(
    signal_text: str,
    valid_micros: Optional[List[Dict[str, Any]]] = None,
    meso_activations: Optional[Dict[str, float]] = None,
    domain_ctx: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Dynamically derives event-specific stakeholder cohorts from `signal_text` and
    valid Micro outputs (excluding contaminated off-domain labels), mapping each cohort
    onto canonical Meso channels with relevance_weight and causal_role.
    """
    ctx = domain_ctx or classify_event_domain(signal_text, valid_micros)
    domain = ctx.get("domain", "general")
    entities = ctx.get("entities") or []
    lead_entity = ctx.get("org_entity") or next(
        (e for e in entities if _looks_like_org(e, signal_text)),
        next((e for e in entities if e.upper() not in {"CEO", "CFO", "CTO", "US", "AI"}), ""),
    )
    person_or_sec = ctx.get("person_entity") or next(
        (e for e in entities if e != lead_entity and e.upper() not in {"CEO", "CFO", "CTO", "US", "AI"}),
        "",
    )

    acts = meso_activations or {}
    corpus_micro = " ".join(
        " ".join([
            str(m.get("stakeholder_focus") or ""),
            str(m.get("primary_narrative") or ""),
            " ".join(m.get("expected_stakeholder_actions") or []),
            " ".join(m.get("emergent_consequences") or []),
        ])
        for m in (valid_micros or [])
    ).lower()

    specs: List[Dict[str, Any]] = []

    if domain == "corporate_governance" or (ctx.get("has_investigation") and ctx.get("has_executive")):
        org_lbl = lead_entity or "Subject Company"
        exec_suffix = f" ({person_or_sec})" if person_or_sec else ""
        sig_and_micro = ((signal_text or "") + " " + corpus_micro).lower()
        is_semi_or_cloud = any(
            has_kw(sig_and_micro, k)
            for k in ("nvidia", "semiconductor", "chip", "chips", "gpu", "gpus", "cloud", "hyperscaler", "data center")
        )
        cust_label = (
            "Cloud Hyperscalers, Enterprise Customers & Strategic Partners"
            if is_semi_or_cloud
            else "Major Enterprise Customers, Clients & Strategic Partners"
        )
        comp_label = (
            "Competing Semiconductor Vendors & Hardware Alternatives"
            if is_semi_or_cloud
            else "Direct Industry Competitors & Substitute Providers"
        )
        comp_role = (
            "Tests enterprise customer willingness to diversify hardware roadmaps if governance uncertainty persists."
            if is_semi_or_cloud
            else "Tests customer willingness to diversify vendor relationships if governance uncertainty persists."
        )
        specs = [
            {
                "stakeholder": f"{org_lbl} Board of Directors & Independent Governance Committee",
                "meso_node_id": "institutional_analyst",
                "keywords": ("board", "governance", "inquiry", "investigation", "audit", "committee", "director"),
                "signal_boost": 0.92,
                "causal_role": "Controls formal inquiry scope, evidentiary findings release, executive oversight, and disclosure credibility.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": f"{org_lbl} Executive Leadership{exec_suffix} & Internal Teams",
                "meso_node_id": "skeptic_debunker",
                "keywords": ("ceo", "executive", "employees", "engineers", "operational", "continuity", "succession", "cfo"),
                "signal_boost": 0.88,
                "causal_role": "Manages day-to-day execution continuity, internal security verification, employee retention, and succession contingency.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Institutional Investors, Shareholders & Equity Analysts",
                "meso_node_id": "pattern_seeker",
                "keywords": ("investor", "investors", "shareholder", "stock", "valuation", "capital", "analyst", "downgrade"),
                "signal_boost": 0.85,
                "causal_role": "Reprices governance uncertainty premium and transmits sentiment shifts across equity & capital allocation.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": cust_label,
                "meso_node_id": "engaged_subcultures",
                "keywords": ("cloud", "hyperscaler", "enterprise", "customer", "partner", "procurement", "multivendor", "aws", "azure", "gcp"),
                "signal_boost": 0.78,
                "causal_role": "Conducts security due diligence on critical dependencies and weighs multi-vendor contingency options.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": comp_label,
                "meso_node_id": "pattern_seeker",
                "keywords": ("competitor", "amd", "intel", "semiconductor", "accelerator", "alternative", "market share"),
                "signal_boost": 0.72,
                "causal_role": comp_role,
                "impact_direction": "opportunistic",
            },
            {
                "stakeholder": "National-Security Agencies & Regulatory Authorities",
                "meso_node_id": "doomer",
                "keywords": ("regulator", "cfius", "commerce", "intelligence", "national security", "export", "sanction", "doj", "sec", "international"),
                "signal_boost": 0.84,
                "causal_role": "Assesses foreign-intelligence exposure, IP compliance controls, and bilateral commercial/regulatory implications.",
                "impact_direction": "polarizing",
            },
            {
                "stakeholder": "Financial/Tech Media & Retail Market Observers",
                "meso_node_id": "amplifier_influencer",
                "keywords": ("media", "retail", "social", "coverage", "headline", "speculat", "public"),
                "signal_boost": 0.70,
                "causal_role": "Amplifies unverified intelligence headlines ahead of formal inquiry disclosures, driving attention spikes.",
                "impact_direction": "polarizing",
            },
        ]
    elif domain == "ai_software":
        subj = lead_entity or "AI Platform"
        specs = [
            {
                "stakeholder": f"{subj} Governance, Platform Trust & Policy Regulators",
                "meso_node_id": "institutional_analyst",
                "keywords": ("governance", "policy", "regulator", "compliance", "guideline", "audit"),
                "signal_boost": 0.85,
                "causal_role": "Establishes safety, compliance, acceptable-use, and auditing guardrails.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Independent Auditors, AI Engineers & Reliability Skeptics",
                "meso_node_id": "skeptic_debunker",
                "keywords": ("benchmark", "audit", "engineer", "skeptic", "cost", "verification", "reliability"),
                "signal_boost": 0.82,
                "causal_role": "Tests reproducibility, cost claims, and reliability limits against real-world workloads.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Enterprise Buyers, Developers & Open-Source Ecosystem Builders",
                "meso_node_id": "pattern_seeker",
                "keywords": ("developer", "enterprise", "open-source", "startup", "application", "cloud"),
                "signal_boost": 0.84,
                "causal_role": "Restructures software tooling, capex, and competitive workflows around new capabilities.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": "Safety Risk Teams, Critical Systems & Vulnerable Workers",
                "meso_node_id": "doomer",
                "keywords": ("security", "exploit", "misuse", "risk", "safety", "workforce", "displacement"),
                "signal_boost": 0.74,
                "causal_role": "Monitors misuse surfaces, unvetted deployment failures, and transition disruption.",
                "impact_direction": "destabilizing",
            },
            {
                "stakeholder": "Tech Media, Creators & Mainstream Digital Public",
                "meso_node_id": "amplifier_influencer",
                "keywords": ("media", "viral", "public", "social", "attention", "educator"),
                "signal_boost": 0.76,
                "causal_role": "Accelerates public awareness and experimentation well ahead of enterprise governance validation.",
                "impact_direction": "polarizing",
            },
        ]
    elif domain == "trade_supply":
        specs = [
            {
                "stakeholder": "Trade, Customs & Regulatory Institutions",
                "meso_node_id": "institutional_analyst",
                "keywords": ("tariff", "customs", "regulator", "waiver", "policy"),
                "signal_boost": 0.88,
                "causal_role": "Issues tariff schedules, customs exemptions, and bilateral compliance directives.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Supply Auditors, Trade Analysts & Cost Skeptics",
                "meso_node_id": "skeptic_debunker",
                "keywords": ("inventory", "buffer", "audit", "cost", "baseline"),
                "signal_boost": 0.78,
                "causal_role": "Assesses physical buffer stock capacity and tests actual vs rumored shortage timelines.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Supply-Chain, Procurement & Logistics Operators",
                "meso_node_id": "pattern_seeker",
                "keywords": ("procurement", "shipping", "freight", "supplier", "manufacturing"),
                "signal_boost": 0.86,
                "causal_role": "Reallocates sourcing routes, forward inventory orders, and freight contracts.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": "Exposed Manufacturers & Downstream Retailers",
                "meso_node_id": "doomer",
                "keywords": ("shortage", "margin", "stoppage", "halt", "disrupt"),
                "signal_boost": 0.76,
                "causal_role": "Faces working-capital strain and input cost pass-throughs if transit friction persists.",
                "impact_direction": "destabilizing",
            },
            {
                "stakeholder": "Commodity/Trade Press & Market Observers",
                "meso_node_id": "amplifier_influencer",
                "keywords": ("media", "market", "panic", "headline"),
                "signal_boost": 0.72,
                "causal_role": "Amplifies short-term scarcity headlines and precautionary hedging cycles.",
                "impact_direction": "polarizing",
            },
        ]
    elif domain == "cyber_infra":
        specs = [
            {
                "stakeholder": "Cyber Incident Response & Infrastructure Regulators",
                "meso_node_id": "institutional_analyst",
                "keywords": ("cert", "regulator", "compliance", "disclosure", "patch"),
                "signal_boost": 0.86,
                "causal_role": "Coordinates mandatory disclosures, patch verification, and regulatory reporting.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Forensic Security Engineers & Reliability Auditors",
                "meso_node_id": "skeptic_debunker",
                "keywords": ("forensic", "failover", "dns", "routing", "sla", "telemetry"),
                "signal_boost": 0.84,
                "causal_role": "Isolates root cause, verifies SLA restoration, and audits blast radius bounds.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Enterprise CIOs, Cloud Operators & Payment Networks",
                "meso_node_id": "pattern_seeker",
                "keywords": ("cloud", "api", "payment", "authentication", "vendor", "redundancy"),
                "signal_boost": 0.86,
                "causal_role": "Executes multi-cloud failover and reassesses single-provider routing dependencies.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": "Dependent End Users & Security/Fraud Operations",
                "meso_node_id": "doomer",
                "keywords": ("exploit", "downtime", "corrupt", "lockout", "outage"),
                "signal_boost": 0.78,
                "causal_role": "Monitors secondary exploit risk and cascading authentication/settlement freezes.",
                "impact_direction": "destabilizing",
            },
            {
                "stakeholder": "Tech/Security Press & Status-Monitoring Public",
                "meso_node_id": "amplifier_influencer",
                "keywords": ("media", "outage", "status", "social"),
                "signal_boost": 0.72,
                "causal_role": "Broadcasts outage severity faster than root-cause engineering postmortems publish.",
                "impact_direction": "polarizing",
            },
        ]
    elif domain == "public_health":
        specs = [
            {
                "stakeholder": "Public Health Authorities & Epidemiological Agencies",
                "meso_node_id": "institutional_analyst",
                "keywords": ("health", "agency", "ministry", "surveillance", "guideline"),
                "signal_boost": 0.88,
                "causal_role": "Issues clinical advisories, testing protocols, and coordinated hospital readiness guidance.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Independent Clinicians & Epidemiological Auditors",
                "meso_node_id": "skeptic_debunker",
                "keywords": ("clinical", "data", "lab", "severity", "hospital"),
                "signal_boost": 0.82,
                "causal_role": "Validates transmission rates and severity baselines against alarmist claims.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Regional Hospital Networks & Healthcare Operators",
                "meso_node_id": "pattern_seeker",
                "keywords": ("hospital", "clinic", "triage", "admission", "capacity"),
                "signal_boost": 0.86,
                "causal_role": "Reallocates clinical surge capacity, staffing, and diagnostic screening flows.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": "Vulnerable Populations & Essential Workplaces",
                "meso_node_id": "doomer",
                "keywords": ("overload", "shortage", "vulnerable", "outbreak"),
                "signal_boost": 0.76,
                "causal_role": "Experiences acute health exposure or access bottlenecks if clinical capacity strains.",
                "impact_direction": "destabilizing",
            },
            {
                "stakeholder": "News Media & Community Health Networks",
                "meso_node_id": "amplifier_influencer",
                "keywords": ("media", "rumor", "public", "anxiety"),
                "signal_boost": 0.72,
                "causal_role": "Shapes public precaution compliance or anxiety waves ahead of confirmed lab metrics.",
                "impact_direction": "polarizing",
            },
        ]
    elif domain == "finance_macro":
        specs = [
            {
                "stakeholder": "Central Banks & Financial Supervisory Authorities",
                "meso_node_id": "institutional_analyst",
                "keywords": ("central bank", "fed", "regulator", "liquidity", "supervis"),
                "signal_boost": 0.88,
                "causal_role": "Deploys liquidity backstops, forward guidance, and prudential stress inspections.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Balance-Sheet Auditors & Fixed-Income Risk Desks",
                "meso_node_id": "skeptic_debunker",
                "keywords": ("balance", "reserve", "solvency", "capital", "audit"),
                "signal_boost": 0.82,
                "causal_role": "Separates mark-to-market noise from true solvency and capital buffer adequacy.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Institutional Capital Allocators & Corporate Treasurers",
                "meso_node_id": "pattern_seeker",
                "keywords": ("credit", "allocation", "treasurer", "bond", "yield", "rebalanc"),
                "signal_boost": 0.84,
                "causal_role": "Reprices duration risk, collateral quality, and secondary credit availability.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": "Exposed Lenders & Leveraged Market Participants",
                "meso_node_id": "doomer",
                "keywords": ("withdrawal", "deposit", "default", "contagion", "freeze"),
                "signal_boost": 0.78,
                "causal_role": "Vulnerable to funding squeeze if confidence loss triggers synchronized withdrawals.",
                "impact_direction": "destabilizing",
            },
            {
                "stakeholder": "Financial Press & Retail Market Commentators",
                "meso_node_id": "amplifier_influencer",
                "keywords": ("media", "retail", "sell", "volatility"),
                "signal_boost": 0.74,
                "causal_role": "Accelerates rapid sentiment repricing ahead of verified reserve data.",
                "impact_direction": "polarizing",
            },
        ]
    else:
        subj = ctx.get("subject_label") or "Focal Institution"
        specs = [
            {
                "stakeholder": f"Oversight, Governance & Regulatory Bodies ({subj})",
                "meso_node_id": "institutional_analyst",
                "keywords": ("regulator", "board", "governance", "official", "oversight", "agency"),
                "signal_boost": 0.84,
                "causal_role": "Coordinates formal review, procedural verification, and official guidance.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": "Independent Auditors & Technical Verification Analysts",
                "meso_node_id": "skeptic_debunker",
                "keywords": ("audit", "evidence", "verify", "analyst", "skeptic"),
                "signal_boost": 0.78,
                "causal_role": "Evaluates primary evidence against speculative extrapolation.",
                "impact_direction": "stabilizing",
            },
            {
                "stakeholder": f"Directly Exposed Ecosystem Partners & Capital Allocators ({subj})",
                "meso_node_id": "pattern_seeker",
                "keywords": ("investor", "partner", "customer", "competitor", "capital"),
                "signal_boost": 0.80,
                "causal_role": "Recalibrates operational dependencies, contracts, and strategic hedging.",
                "impact_direction": "mixed",
            },
            {
                "stakeholder": "Security, Operational Risk & Affected Workforce Teams",
                "meso_node_id": "doomer",
                "keywords": ("security", "risk", "employee", "vulnerability", "disrupt"),
                "signal_boost": 0.74,
                "causal_role": "Monitors failure surfaces, operational continuity, and compliance exposure.",
                "impact_direction": "destabilizing",
            },
            {
                "stakeholder": "Media Channels & Engaged Public Observers",
                "meso_node_id": "amplifier_influencer",
                "keywords": ("media", "public", "social", "headline"),
                "signal_boost": 0.70,
                "causal_role": "Drives public salience and conversational momentum across broadcast/social ties.",
                "impact_direction": "polarizing",
            },
        ]

    # Compute relevance weights combining signal salience, Micro mention share, and Meso node activation
    out: List[Dict[str, Any]] = []
    sig_low = (signal_text or "").lower()
    for sp in specs:
        kws = sp["keywords"]
        sig_hits = sum(1 for k in kws if k in sig_low)
        micro_hits = sum(1 for k in kws if k in corpus_micro)
        sig_score = min(1.0, sp["signal_boost"] * 0.75 + 0.12 * sig_hits)
        micro_score = min(1.0, micro_hits / max(1.0, len(kws) * 0.65))
        nid = sp["meso_node_id"]
        node_act = float(acts.get(nid, 0.25))
        rel_weight = round(min(1.0, max(0.15, 0.50 * sig_score + 0.30 * micro_score + 0.20 * min(1.0, node_act * 1.5))), 4)
        out.append({
            "stakeholder": sp["stakeholder"],
            "meso_node_id": nid,
            "relevance_weight": rel_weight,
            "meso_node_activation": round(node_act, 4),
            "weighted_influence_score": round(rel_weight * node_act, 4),
            "causal_role": sp["causal_role"],
            "impact_direction": sp["impact_direction"],
        })

    out.sort(key=lambda r: (-r["relevance_weight"], -r["meso_node_activation"]))
    return out
