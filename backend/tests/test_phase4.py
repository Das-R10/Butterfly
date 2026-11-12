"""
tests/test_phase4.py

Phase 4 validation — Council Model.

Tests the full L0→L5 pipeline with:
  - Dynamic bucket generation (L0)
  - MBTI population reaction layer (L2A)
  - All-16 archetypes with domain context (L2B)
  - Ambiguity Analyst post-pass (L2B)
  - Dynamic multi-tag scoring (L4)
  - Convergence gap calculation (L4)
  - Debate Layer — three critics (L5)

Run from project root:
    python tests/test_phase4.py                         # full pipeline, ukraine_nato
    python tests/test_phase4.py tariff_war_2025         # different signal
    python tests/test_phase4.py mbti                    # MBTI layer only (fast check)
    python tests/test_phase4.py buckets                 # bucket generation only (fastest)
    python tests/test_phase4.py pure                    # pure mode (no MBTI)
    python tests/test_phase4.py focused "your signal"   # live input

Phase 4 is DONE when ALL of:
  [1] Dynamic buckets are generated and signal-specific (not hardcoded)
  [2] MBTI report is produced before analyst run (16 types, population weights)
  [3] All 16 archetypes produce outputs with domain_context field set
  [4] Ambiguity Analyst report is attached with cross_domain_insights field
  [5] Convergence gap is calculated (Validated / Amplified / Latent)
  [6] Critics reference specific archetype names in their critique
  [7] Volatility score is a float between 0 and 1
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.personalities.personality_runner import TEST_SIGNALS


# ── Individual layer tests ─────────────────────────────────────────────────────

def test_buckets(signal_text: str = None) -> list[dict]:
    """[1] Test dynamic bucket generation from signal text."""
    from src.ensemble.dynamic_bucket_scorer import generate_dynamic_buckets

    signal = signal_text or TEST_SIGNALS["ukraine_nato"]
    print(f"\n  [TEST] Dynamic bucket generation")
    print(f"  Signal: {signal[:80]}...\n")

    buckets = generate_dynamic_buckets(signal)

    print(f"  Generated {len(buckets)} buckets:")
    for b in buckets:
        print(f"    • {b['label']:<30} (key: {b['key']}, color: {b['color']})")

    # Validation
    assert len(buckets) >= 2, f"Expected ≥2 buckets, got {len(buckets)}"
    assert all("key" in b and "label" in b and "reference" in b for b in buckets), \
        "Buckets missing required fields"

    print(f"\n  ✓ PASS: {len(buckets)} signal-specific buckets generated")
    return buckets


def test_mbti(signal_text: str = None, buckets: list = None) -> dict:
    """[2] Test MBTI population reaction layer."""
    from src.personalities.mbti_layer import run_mbti_layer
    from src.ensemble.dynamic_bucket_scorer import generate_dynamic_buckets

    signal = signal_text or TEST_SIGNALS["ukraine_nato"]

    if buckets is None:
        buckets = generate_dynamic_buckets(signal)

    print(f"\n  [TEST] MBTI population reaction layer")
    print(f"  Signal: {signal[:80]}...\n")

    mbti_report = run_mbti_layer(
        signal=signal,
        world_state_brief="",
        dynamic_buckets=buckets,
    )

    reactions = mbti_report.get("reactions", {})
    print(f"\n  Results:")
    print(f"    Types run:        {len(reactions)}/16")
    print(f"    Dominant emotion: {mbti_report.get('dominant_emotion', '?')}")
    print(f"    Sharing velocity: {mbti_report.get('sharing_velocity', 0):.0%}")

    pop_buckets = mbti_report.get("population_buckets", {})
    if pop_buckets:
        top_bucket = list(pop_buckets.items())[0]
        print(f"    Top pop bucket:   {top_bucket[0]} ({top_bucket[1]:.0%})")

    # Show cluster summary
    clusters = mbti_report.get("reaction_clusters", {})
    print(f"\n  Reaction clusters:")
    for name, data in clusters.items():
        readable = name.replace("_", " ").title()
        print(f"    {readable:<35} {data['population_label']}")

    # Validation
    assert len(reactions) == 16, f"Expected 16 MBTI types, got {len(reactions)}"
    assert mbti_report.get("dominant_emotion"), "No dominant emotion computed"
    assert mbti_report.get("report_text"), "No report text generated"
    assert "population_buckets" in mbti_report, "No population bucket alignment"

    print(f"\n  ✓ PASS: 16 MBTI types ran, population report generated")
    return mbti_report


def test_archetypes(signal_text: str = None, mbti_report: dict = None,
                    buckets: list = None) -> dict:
    """[3+4] Test archetype layer — all 16 run, Ambiguity Analyst attached."""
    from src.personalities.personality_runner import run_all_models
    from src.ensemble.dynamic_bucket_scorer import generate_dynamic_buckets
    from src.personalities.mbti_layer import run_mbti_layer

    signal = signal_text or TEST_SIGNALS["ukraine_nato"]

    if buckets is None:
        buckets = generate_dynamic_buckets(signal)
    if mbti_report is None:
        mbti_report = run_mbti_layer(signal=signal, dynamic_buckets=buckets)

    print(f"\n  [TEST] Archetype layer (all 16) + Ambiguity Analyst")
    print(f"  Signal: {signal[:80]}...\n")

    result = run_all_models(
        signal=signal,
        signal_id="test_phase4",
        mbti_report=mbti_report,
        dynamic_buckets=buckets,
        domain_context=None,
        run_ambiguity=True,
    )

    outputs         = result.get("outputs", [])
    ambiguity       = result.get("ambiguity_report", {})
    personalities   = set(o["personality"] for o in outputs)
    domain_contexts = [o.get("domain_context", "") for o in outputs]

    print(f"\n  Results:")
    print(f"    Total outputs:    {len(outputs)}")
    print(f"    Unique archetypes:{len(personalities)}/16")
    print(f"    Domain set:       {result.get('domain', '?')}")
    print(f"    MBTI injected:    {'yes' if any('MBTI' in dc or 'POPULATION' in dc for dc in domain_contexts) else 'no'}")

    if ambiguity:
        cross = ambiguity.get("cross_domain_insights", [])
        challenge = ambiguity.get("domain_challenge", {})
        missing = ambiguity.get("missing_context", [])
        print(f"\n  Ambiguity Analyst:")
        print(f"    Cross-domain insights: {len(cross)}")
        print(f"    Domain challenge:      {challenge.get('challenge', False)}")
        print(f"    Missing context items: {len(missing)}")
        print(f"    Has significant flags: {ambiguity.get('has_significant_findings', False)}")
        if cross:
            top = cross[0]
            print(f"    Top insight: [{top.get('archetype')}] → {top.get('secondary_domain')} ({top.get('significance', 0):.0%})")

    # Validation
    assert len(outputs) >= 16, f"Expected ≥16 outputs, got {len(outputs)}"
    assert len(personalities) == 16, f"Expected 16 unique archetypes, got {len(personalities)}"
    assert all(o.get("domain_context") for o in outputs), \
        "Some outputs missing domain_context field"
    assert ambiguity is not None, "Ambiguity report not attached"
    assert "cross_domain_insights" in ambiguity, "Ambiguity report missing cross_domain_insights"

    print(f"\n  ✓ PASS: 16 archetypes ran, domain context set, Ambiguity Analyst ran")
    return result


def test_scoring_and_gap(result: dict, mbti_report: dict = None) -> dict:
    """[4+5] Test dynamic scoring and convergence gap."""
    from src.ensemble.dynamic_bucket_scorer import score_and_map_dynamic
    from src.personalities.mbti_layer import compute_convergence_gap

    print(f"\n  [TEST] Dynamic scoring + convergence gap")

    result = score_and_map_dynamic(result)
    clusters = result.get("clusters", [])

    print(f"\n  Probability map ({len(clusters)} buckets):")
    for c in clusters[:6]:
        bar = "█" * int(c["probability"] * 20) + "░" * (20 - int(c["probability"] * 20))
        mt  = f" [multi-tag: {c.get('multi_tag_outputs', 0)}]" if c.get("multi_tag_outputs") else ""
        print(f"    {bar} {c['probability_pct']:>6}  {c['label'][:35]}{mt}")

    # Convergence gap
    if mbti_report and clusters:
        gap = compute_convergence_gap(
            population_buckets=mbti_report.get("population_buckets", {}),
            analyst_probability_map=clusters,
        )
        result["convergence_gap"] = gap
        print(f"\n  Convergence gap:")
        print(f"    Status:   {gap.get('gap_status', '?')}")
        print(f"    Score:    {gap.get('gap_score', 0):.3f}")
        print(f"    Meaning:  {gap.get('meaning', '')}")
        print(f"    Action:   {gap.get('action', '')}")

    # Validation
    assert clusters, "No clusters produced by dynamic scoring"
    assert clusters[0].get("probability") > 0, "Dominant cluster has zero probability"
    assert result.get("scoring_method") == "dynamic_multi_tag", \
        f"Wrong scoring method: {result.get('scoring_method')}"

    print(f"\n  ✓ PASS: Dynamic scoring produced {len(clusters)} clusters")
    return result


def test_debate(result: dict, mbti_report: dict = None) -> dict:
    """[6+7] Test debate layer — critics + volatility."""
    from src.ensemble.debate_layer import run_debate_layer

    print(f"\n  [TEST] Debate Layer — three critics")

    clusters = result.get("clusters", [])
    outputs  = result.get("outputs", [])
    signal   = result.get("signal_text", "")

    assert clusters, "No clusters to debate"

    debate = run_debate_layer(
        signal=signal,
        probability_map=clusters,
        outputs=outputs,
        mbti_report=mbti_report,
        narrative_drift_score=0.0,
    )

    print(f"\n  Debate results:")
    print(f"    Volatility score: {debate.get('volatility_score', '?'):.3f}")
    print(f"    High-Octane:      {debate.get('high_octane_narrative', False)}")
    print(f"    Outlier:          {debate.get('outlier_archetype_name', '?')}")

    fact_check   = debate.get("fact_check_critique", "")
    cynic        = debate.get("cynic_critique", "")
    chaos_monkey = debate.get("chaos_monkey_critique", "")

    print(f"\n  Fact-Checker ({len(fact_check)} chars):")
    print(f"    {fact_check[:150]}...")
    print(f"\n  Cynic ({len(cynic)} chars):")
    print(f"    {cynic[:150]}...")
    print(f"\n  Chaos Monkey ({len(chaos_monkey)} chars):")
    print(f"    {chaos_monkey[:150]}...")

    # Check critics reference archetype names
    all_crit_text = (fact_check + cynic + chaos_monkey).lower()
    archetype_names = [
        "pattern seeker", "doomer", "skeptic", "institutional analyst",
        "cynical realist", "historian", "economist",
    ]
    referenced = [n for n in archetype_names if n in all_crit_text]

    print(f"\n  Archetype references in critique: {referenced}")

    # Validation
    assert isinstance(debate.get("volatility_score"), float), \
        "Volatility score must be a float"
    assert 0.0 <= debate["volatility_score"] <= 1.0, \
        f"Volatility score out of range: {debate['volatility_score']}"
    assert debate.get("adjusted_probability_map"), "No adjusted probability map"
    assert len(debate.get("fact_check_critique", "")) > 50, "Fact-Checker too short"

    print(f"\n  ✓ PASS: Debate Layer complete, volatility={debate['volatility_score']:.3f}")
    return debate


# ── Full pipeline test ────────────────────────────────────────────────────────

def test_full_pipeline(signal_key: str = "ukraine_nato", mode: str = "support"):
    """Run the complete L0→L5 pipeline and validate all checkpoints."""
    from src.ingestion.focused_pipeline import run_focused_signal

    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    signal = TEST_SIGNALS.get(signal_key, TEST_SIGNALS["ukraine_nato"])

    print(f"\n{'═'*65}")
    print(f"  PHASE 4 VALIDATION — FULL COUNCIL MODEL PIPELINE")
    print(f"  Signal: '{signal_key}' | Mode: {mode.upper()}")
    print(f"{'═'*65}")

    result = run_focused_signal(
        user_input=signal,
        mode=mode,
        run_debate=True,
        lock_prediction=False,
    )

    # ── Checkpoint summary ────────────────────────────────────────────────────
    outputs     = result.get("outputs", [])
    clusters    = result.get("clusters", [])
    ambiguity   = result.get("ambiguity_report", {})
    mbti        = result.get("mbti_report", {})
    debate      = result.get("debate_result", {})
    conv_gap    = result.get("convergence_gap", {})
    buckets     = result.get("dynamic_buckets", [])

    print(f"\n{'═'*65}")
    print(f"  PHASE 4 CHECKPOINT RESULTS")
    print(f"{'═'*65}\n")

    checks = [
        ("[1] Dynamic buckets generated",
         len(buckets) >= 2,
         f"{len(buckets)} buckets: {[b['label'] for b in buckets]}"),

        ("[2] MBTI report produced",
         mbti is not None and len(mbti.get("reactions", {})) == 16,
         f"16 types, dominant={mbti.get('dominant_emotion', '?') if mbti else 'N/A'}"),

        ("[3] All 16 archetypes ran",
         len(set(o["personality"] for o in outputs)) == 16,
         f"{len(set(o['personality'] for o in outputs))}/16 unique archetypes"),

        ("[4] Domain context injected",
         all(o.get("domain_context") for o in outputs),
         f"domain={result.get('domain', '?')}"),

        ("[4] Ambiguity Analyst ran",
         ambiguity is not None and "cross_domain_insights" in ambiguity,
         f"{len(ambiguity.get('cross_domain_insights', []))} cross-domain insights" if ambiguity else "missing"),

        ("[4] Convergence gap calculated",
         bool(conv_gap.get("gap_status")),
         conv_gap.get("gap_status", "not calculated")),

        ("[6] Critics ran (debate layer)",
         bool(debate.get("fact_check_critique")),
         f"volatility={debate.get('volatility_score', '?')}"),

        ("[7] Volatility is valid float",
         isinstance(debate.get("volatility_score"), float) and 0 <= debate.get("volatility_score", -1) <= 1,
         str(debate.get("volatility_score", "missing"))),
    ]

    all_pass = True
    for label, passed, detail in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  {label}")
        print(f"           {detail}\n")

    print(f"{'═'*65}")
    if all_pass:
        print(f"  ✅ PHASE 4 COMPLETE — all checkpoints passed")
        print(f"     Ready to proceed to Phase 5 (Pipeline Orchestration)")
    else:
        print(f"  ❌ PHASE 4 INCOMPLETE — fix failing checkpoints before Phase 5")
    print(f"{'═'*65}\n")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    mode_arg   = sys.argv[1] if len(sys.argv) > 1 else "ukraine_nato"
    extra_args = sys.argv[2:] if len(sys.argv) > 2 else []

    if mode_arg == "buckets":
        signal = " ".join(extra_args) if extra_args else None
        test_buckets(signal)

    elif mode_arg == "mbti":
        signal = " ".join(extra_args) if extra_args else None
        buckets = test_buckets(signal)
        test_mbti(signal, buckets)

    elif mode_arg == "pure":
        signal_key = extra_args[0] if extra_args else "ukraine_nato"
        test_full_pipeline(signal_key, mode="pure")

    elif mode_arg == "focused":
        if not extra_args:
            print('\n  Usage: python tests/test_phase4.py focused "your signal"\n')
            sys.exit(1)
        user_input = " ".join(extra_args)
        from src.ingestion.focused_pipeline import run_focused_signal
        run_focused_signal(user_input, mode="support", run_debate=True)

    elif mode_arg in TEST_SIGNALS:
        test_full_pipeline(mode_arg, mode="support")

    else:
        # Default: full pipeline on ukraine_nato
        test_full_pipeline("ukraine_nato", mode="support")
