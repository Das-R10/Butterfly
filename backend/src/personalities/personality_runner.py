"""
src/personalities/personality_runner.py  — v2.0 (council model)

Changes from v1:
  - All 16 archetypes run on every signal. Domain pool filtering removed.
  - Each archetype receives a domain context tag ("primary domain: X")
    but is NOT restricted to it. Cross-domain outputs are intentional.
  - MBTI population reaction report (from L2A) is prepended to every
    archetype prompt so analysts know how the population is already reacting.
  - Ambiguity Analyst runs after all 16 archetypes via ambiguity_analyst.py.
  - run_all_models() now accepts mbti_report and domain_context as params.

LLM calling logic, ProgressTracker, and model definitions unchanged from v1.
Temperature 0.3 locked.
"""

import json
import time
import threading
from datetime import datetime
from loguru import logger
from crewai import Agent, Task, Crew, Process, LLM

from src.personalities.prompts import PERSONALITIES, PERSONALITY_ORDER

TEMPERATURE = 0.3   # locked — do not change

MODELS = {
    "mistral": {
        "model_string": "ollama/mistral:7b",
        "display_name": "Mistral 7B",
        "extra_body": {},
    },
    "qwen3.5": {
        "model_string": "ollama/qwen3.5:4b",
        "display_name": "Qwen 3.5 4B",
        "extra_body": {"think": False},
    },
}

TEST_SIGNALS = {
    "ukraine_nato": (
        "Breaking: Ukrainian parliament votes to formally apply for NATO membership. "
        "Russia's foreign ministry calls it a 'red line crossed' and summons the US "
        "ambassador. NATO Secretary General says the alliance will 'consider the application "
        "through established processes.' Markets in Eastern Europe drop 3% in early trading."
    ),
    "ukraine_prewar": (
        "Multiple posts on r/geopolitics from late January 2022 note unusual US military "
        "logistics activity near the Polish border — large food supply orders, unusual vehicle "
        "concentrations visible in satellite imagery. Ukrainian government officials have gone "
        "notably quiet. Pentagon says it is 'monitoring the situation'."
    ),
    "iran_war_hormuz": (
        "Day 16 of US-Israel war with Iran. Oil hits $105/barrel as Strait of Hormuz "
        "remains blocked — 20% of world oil supply cut off. Iran fired 500+ ballistic "
        "missiles and 2000+ drones at US bases across the Middle East. Trump says Iran "
        "wants a deal but terms are not good enough yet. Israel running critically low "
        "on missile interceptors. Dubai airport suspended after drone incident. "
        "13 US service members killed so far. Trump: war ends when I feel it in my bones."
    ),
    "tariff_war_2025": (
        "Trump announces 145% tariffs on all Chinese imports effective immediately. "
        "China retaliates with 125% tariffs on US goods and restricts rare earth exports. "
        "Dow drops 4% in early trading. Apple, Nike warn of major supply chain disruption. "
        "Treasury Secretary says this is a negotiating tactic not a permanent measure. "
        "Beijing calls it economic aggression and summons US ambassador."
    ),
}

PERSONALITY_NAMES = [PERSONALITIES[k]["name"] for k in PERSONALITY_ORDER]


# ── Domain classifier ─────────────────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "geopolitical": [
        "war", "military", "troops", "nato", "missile", "nuclear", "border",
        "invasion", "sanctions", "conflict", "iran", "russia", "china", "ukraine",
        "taiwan", "attack", "diplomatic", "foreign", "president", "prime minister",
    ],
    "economic": [
        "tariff", "trade", "market", "gdp", "inflation", "recession", "fed",
        "rate", "dollar", "oil", "supply chain", "stock", "economic", "bank",
        "financial", "commodity", "export", "import", "currency",
    ],
    "technology": [
        "ai", "algorithm", "software", "tech", "startup", "launch", "product",
        "platform", "data", "cyber", "hack", "chip", "semiconductor", "model",
        "digital", "automation", "robot",
    ],
    "cultural": [
        "celebrity", "film", "music", "culture", "religion", "community",
        "social media", "viral", "trend", "artist", "sport", "entertainment",
        "scandal", "influencer",
    ],
    "political": [
        "election", "vote", "party", "congress", "parliament", "legislation",
        "policy", "government", "law", "court", "supreme", "reform", "bill",
    ],
    "scientific": [
        "study", "research", "vaccine", "climate", "cancer", "drug", "fda",
        "trial", "discovery", "science", "health", "medical", "virus", "outbreak",
    ],
    "corporate": [
        "ceo", "earnings", "company", "layoff", "merger", "acquisition",
        "ipo", "revenue", "profit", "startup", "executive", "board",
    ],
}


def classify_domain(signal: str) -> str:
    """
    Classify the primary domain of a signal using keyword matching.
    Returns the domain with the most keyword hits.
    Falls back to 'general' if no clear domain detected.
    """
    signal_lower = signal.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in signal_lower)

    if max(scores.values()) == 0:
        return "general"

    return max(scores, key=scores.get)


# ── Live progress spinner (unchanged from v1) ─────────────────────────────────

class ProgressTracker:
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, model_display: str, total: int):
        self.model_display = model_display
        self.total = total
        self.current = 0
        self.current_name = "starting..."
        self.start_time = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def update(self, index: int, name: str):
        self.current = index
        self.current_name = name

    def stop(self):
        self._stop.set()
        self._thread.join()
        print("\r" + " " * 80 + "\r", end="", flush=True)

    def _run(self):
        spin_i = 0
        while not self._stop.is_set():
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            done = self.current
            remaining = self.total - done
            bar_done = "█" * done
            bar_left = "░" * remaining
            spinner = self.SPINNER[spin_i % len(self.SPINNER)]
            spin_i += 1
            line = (
                f"\r  {spinner} {self.model_display}  "
                f"[{bar_done}{bar_left}] {done}/{self.total}  "
                f"elapsed: {time_str}  "
                f"current: {self.current_name[:28]:<28}"
            )
            print(line, end="", flush=True)
            time.sleep(0.2)


# ── Domain context injection ──────────────────────────────────────────────────

def _build_domain_context_block(domain: str) -> str:
    """
    Build the domain context tag prepended to every archetype task.
    Tells the archetype which domain is primary — but does NOT restrict them.
    Cross-domain insights are explicitly invited.
    """
    return (
        f"[DOMAIN CONTEXT: Primary domain classification is '{domain}'. "
        f"Analyze through your own lens — if you see angles in other domains "
        f"(economic, geopolitical, technological, humanitarian, etc.) that are "
        f"significant, include them. Cross-domain insights are valuable.]"
    )


def _build_mbti_context_block(mbti_report: dict) -> str:
    """
    Build the MBTI context block prepended to every archetype task.
    Summarises the population reaction so analysts know the emotional environment.
    """
    if not mbti_report:
        return ""

    report_text = mbti_report.get("report_text", "")
    if not report_text:
        return ""

    # Truncate to keep prompts manageable — key stats + cluster summaries only
    lines = report_text.split("\n")
    # Keep first 25 lines — enough to capture dominant emotion, velocity, and clusters
    summary_lines = [l for l in lines[:30] if l.strip()]
    truncated = "\n".join(summary_lines[:25])

    return (
        f"\n[POPULATION CONTEXT from L2A MBTI Layer — how the public is already reacting:]\n"
        f"{truncated}\n"
        f"[END POPULATION CONTEXT — factor this into your analysis.]\n"
    )


# ── Core runner (LLM logic unchanged from v1) ─────────────────────────────────

def build_llm(model_key: str) -> LLM:
    cfg = MODELS[model_key]
    kwargs = dict(
        model=cfg["model_string"],
        base_url="http://localhost:11434",
        temperature=TEMPERATURE,
    )
    if cfg.get("extra_body"):
        kwargs["extra_body"] = cfg["extra_body"]
    return LLM(**kwargs)


def run_model(
    model_key: str,
    signal: str,
    domain_context: str = "",
    mbti_context: str = "",
) -> list[dict]:
    """
    Run all 16 archetypes for one model.
    All 16 always run — no domain filtering.
    Domain context tag and MBTI context prepended to every task.
    """
    cfg   = MODELS[model_key]
    total = len(PERSONALITY_ORDER)

    print(f"\n  ── {cfg['display_name']} ──────────────────────────────────────")
    if domain_context:
        print(f"  Domain: {domain_context[:60]}...")

    llm    = build_llm(model_key)
    agents = []
    tasks  = []

    for key in PERSONALITY_ORDER:
        p = PERSONALITIES[key]

        agent = Agent(
            role=p["name"],
            goal=(
                f"Analyze the signal as a {p['name']} and respond authentically "
                f"to how this archetype would interpret it. 3-5 sentences max. "
                f"Stay fully in character. Do not break the roleplay under any circumstances."
            ),
            backstory=p["system_prompt"],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # Build task description with domain context + MBTI context prepended
        task_desc = ""
        if domain_context:
            task_desc += domain_context + "\n\n"
        if mbti_context:
            task_desc += mbti_context + "\n\n"
        task_desc += (
            f"SIGNAL: {signal}\n\n"
            f"Respond as a {p['name']} would — authentic to your worldview, "
            f"3-5 sentences, direct and specific. No AI disclaimers. "
            f"Stay fully in character. Do not break the roleplay under any circumstances."
        )

        task = Task(
            description=task_desc,
            expected_output=f"A {p['name']} perspective in 3-5 sentences.",
            agent=agent,
        )
        agents.append(agent)
        tasks.append((key, task))

    # Start live progress tracker
    tracker = ProgressTracker(cfg["display_name"], total)
    tracker.start()

    try:
        outputs    = []
        run_start  = time.time()

        for i, (key, task) in enumerate(tasks):
            p = PERSONALITIES[key]
            tracker.update(i, p["name"] + "...")

            single_crew = Crew(
                agents=[agents[i]],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            single_crew.kickoff()

            raw = ""
            if hasattr(task, "output") and task.output:
                raw = task.output.raw if hasattr(task.output, "raw") else str(task.output)

            if not raw:
                raw = "[No output generated]"

            elapsed = round(time.time() - run_start, 1)
            tracker.update(i + 1, f"done ({elapsed}s)")

            outputs.append({
                "model":              model_key,
                "model_display":      cfg["display_name"],
                "personality":        key,
                "personality_name":   p["name"],
                "response":           raw.strip(),
                "emotional_range":    p.get("emotional_range", ""),
                "virality":           p.get("virality", ""),
                "position_in_debate": i + 1,
                "domain_context":     domain_context,
                "timestamp":          datetime.utcnow().isoformat() + "Z",
            })

    finally:
        tracker.stop()

    total_time = round(time.time() - tracker.start_time, 1)
    print(f"  ✓ {cfg['display_name']} complete — "
          f"{len(outputs)}/16 archetypes — {total_time}s total")

    for i, o in enumerate(outputs):
        status = "✓" if o["response"] != "[No output generated]" else "✗"
        chars  = len(o["response"])
        print(f"    {status} [{i+1:>2}/16] {o['personality_name']:<28} {chars} chars")

    return outputs


# ── Behavioral sensitivity scorer ─────────────────────────────────────────────

def compute_behavioral_sensitivity(
    outputs_with_mbti: list[dict],
    outputs_without_mbti: list[dict],
) -> dict[str, float]:
    """
    Measure how much the MBTI population context shifted each archetype's output.
    Compares semantic similarity between outputs with and without MBTI context.

    Returns dict of personality_key → sensitivity_score (0.0 = unchanged, 1.0 = completely different)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Index by personality
        with_mbti    = {o["personality"]: o["response"] for o in outputs_with_mbti}
        without_mbti = {o["personality"]: o["response"] for o in outputs_without_mbti}

        sensitivity = {}
        for key in with_mbti:
            if key not in without_mbti:
                continue
            text_a = with_mbti[key]
            text_b = without_mbti[key]
            embs   = model.encode([text_a, text_b], normalize_embeddings=True)
            sim    = float(np.dot(embs[0], embs[1]))
            # Convert similarity to change score (1 - similarity)
            sensitivity[key] = round(1 - sim, 4)

        return sensitivity

    except ImportError:
        logger.debug("sentence-transformers not available — skipping sensitivity scoring")
        return {}


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_models(
    signal: str,
    signal_id: str = None,
    mbti_report: dict = None,
    dynamic_buckets: list = None,
    domain_context: str = None,
    run_ambiguity: bool = True,
) -> dict:
    """
    Run all 16 archetypes across all models.

    Args:
        signal:          The enriched signal text
        signal_id:       Optional signal identifier
        mbti_report:     Output from run_mbti_layer() — prepended to every archetype prompt
        dynamic_buckets: L0 dynamic buckets — passed to Ambiguity Analyst
        domain_context:  Primary domain classification string.
                         If None, auto-classified from signal keywords.
        run_ambiguity:   Whether to run the Ambiguity Analyst after archetypes.

    Returns:
        Result dict with outputs, clusters (empty until L4), ambiguity_report,
        domain, and metadata.
    """
    if not signal_id:
        signal_id = f"signal_{int(time.time())}"

    # Classify domain if not provided
    if domain_context is None:
        domain_context = classify_domain(signal)
        logger.info(f"Auto-classified domain: {domain_context}")
    else:
        logger.info(f"Domain provided: {domain_context}")

    # Build context blocks
    domain_block = _build_domain_context_block(domain_context)
    mbti_block   = _build_mbti_context_block(mbti_report) if mbti_report else ""

    if mbti_block:
        logger.info("MBTI population context injected into archetype prompts")

    logger.info(f"Signal ({len(signal)} chars): {signal[:80]}...")
    logger.info(f"Running {len(PERSONALITIES)} archetypes × {len(MODELS)} models")

    all_outputs    = []
    total_models   = len(MODELS)

    for mi, model_key in enumerate(MODELS):
        cfg = MODELS[model_key]
        print(f"\n  Model {mi+1}/{total_models}: {cfg['display_name']}")
        try:
            outputs = run_model(
                model_key=model_key,
                signal=signal,
                domain_context=domain_block,
                mbti_context=mbti_block,
            )
            all_outputs.extend(outputs)
        except Exception as e:
            logger.error(f"Model {model_key} failed: {e}")
            print(f"  ✗ {cfg['display_name']} failed: {e}")

    result = {
        "signal_id":      signal_id,
        "signal_text":    signal,
        "domain":         domain_context,
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "models_used":    list(MODELS.keys()),
        "total_outputs":  len(all_outputs),
        "outputs":        all_outputs,
        "clusters":       [],    # populated by L4 dynamic_bucket_scorer
        "mbti_report":    mbti_report,
        "ambiguity_report": None,
    }

    # Run Ambiguity Analyst on all outputs
    if run_ambiguity and all_outputs:
        try:
            from src.personalities.ambiguity_analyst import run_ambiguity_analyst
            ambiguity_report = run_ambiguity_analyst(
                signal=signal,
                outputs=all_outputs,
                primary_domain=domain_context,
                dynamic_buckets=dynamic_buckets or [],
            )
            result["ambiguity_report"] = ambiguity_report
            # Outputs are annotated in-place with cross_domain flags by ambiguity_analyst
            result["outputs"] = all_outputs
        except Exception as e:
            logger.error(f"Ambiguity Analyst failed: {e}")

    return result


def save_result(result: dict, path: str = None) -> str:
    import os
    if not path:
        path = f"data/outputs/{result['signal_id']}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved to {path}")
    return path
