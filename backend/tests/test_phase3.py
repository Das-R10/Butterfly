"""
tests/test_phase3.py

Phase 3 test — validates semantic scoring pipeline.

similarity_scorer and narrative_summarizer have been removed.
scoring is now handled by dynamic_bucket_scorer (live analysis)
and semantic_scorer is kept for Brier Score validation only.

Run from project root:
    python tests/test_phase3.py                      # ukraine_nato (default)
    python tests/test_phase3.py iran_war_hormuz
    python tests/test_phase3.py tariff_war_2025
    python tests/test_phase3.py compare              # dynamic vs semantic (Brier) comparison
    python tests/test_phase3.py focused "your text"  # live signal

Done when:
    - Doomer outputs land in a conflict-type bucket not a historical one
    - Qwen Pattern Seeker lands in conflict/escalation not diplomatic
    - Cross-model agreement count is >= Phase 1 results
    - Dynamic buckets are signal-specific (not hardcoded labels)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.personalities.personality_runner import run_all_models, save_result, TEST_SIGNALS
from src.ensemble.semantic_scorer import score_and_map_semantic
from src.ensemble.dynamic_bucket_scorer import score_and_map_dynamic


# ── Inline summary printer (replaces deleted narrative_summarizer) ────────────

def print_summary(result: dict):
    """Print a readable probability map summary to stdout."""
    clusters = result.get("clusters", [])
    method   = result.get("scoring_method", "unknown")
    buckets  = result.get("dynamic_buckets", [])

    print(f"\n  {'═'*60}")
    print(f"  SCORING SUMMARY  [{method}]")
    print(f"  {'═'*60}")

    if buckets:
        print(f"  Dynamic buckets generated: {[b['label'] for b in buckets]}")
        print()

    if not clusters:
        print("  No clusters produced.")
        return

    print(f"  {'Narrative':<38} {'Prob':>6}  {'Conf':<8} {'Cross-model'}")
    print(f"  {'-'*65}")
    for c in clusters:
        bar_len = int(c.get("probability", 0) * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        label   = c.get("label", c.get("bucket", "?"))[:36]
        pct     = c.get("probability_pct", "?")
        conf    = c.get("confidence", "?")
        cm      = c.get("cross_model_count", 0)
        print(f"  {label:<38} {pct:>6}  {conf:<8} {cm} personalities")

    dominant = clusters[0] if clusters else {}
    print(f"\n  Dominant: '{dominant.get('label', '?')}' at {dominant.get('probability_pct', '?')}")
    print(f"  {'═'*60}\n")


def write_text_report(result: dict, path: str):
    """Write a plain text report of the probability map to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    clusters = result.get("clusters", [])
    method   = result.get("scoring_method", "unknown")

    lines = [
        f"ButterflyAI — Scoring Report",
        f"Signal ID: {result.get('signal_id', '?')}",
        f"Method: {method}",
        f"Timestamp: {result.get('timestamp', '?')}",
        "",
        "Probability Map:",
        "-" * 50,
    ]
    for c in clusters:
        lines.append(
            f"  {c.get('label', c.get('bucket', '?')):<38} "
            f"{c.get('probability_pct', '?'):>6}  "
            f"[{c.get('confidence', '?')}]"
        )
    lines += ["", "Sample responses:"]
    for c in clusters[:3]:
        lines.append(f"\n  [{c.get('label', '?')}]")
        lines.append(f"  {c.get('sample_response', '')[:200]}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {path}")


# ── Test functions ────────────────────────────────────────────────────────────

def run_dynamic(signal_key: str = "ukraine_nato"):
    """Run dynamic scoring (the live scorer — replaces old semantic test)."""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    if signal_key not in TEST_SIGNALS:
        print(f"\n  Unknown signal: '{signal_key}'")
        print(f"  Available: {list(TEST_SIGNALS.keys())}\n")
        return

    signal    = TEST_SIGNALS[signal_key]
    signal_id = f"phase3_{signal_key}"

    print(f"\n  Butterfly Algorithm — Phase 3 Dynamic Scoring")
    print(f"  Signal : '{signal_key}'")
    print(f"  Text   : {signal[:100]}...\n")
    print("  Step 1: Running personalities through both models...\n")

    result = run_all_models(signal=signal, signal_id=signal_id)

    print(f"\n  Step 2: Dynamic scoring {result['total_outputs']} outputs...")
    result = score_and_map_dynamic(result)

    print("  Step 3: Saving files...")
    json_path = f"data/outputs/{signal_id}.json"
    txt_path  = f"data/outputs/{signal_id}_report.txt"

    save_result(result, json_path)
    write_text_report(result, txt_path)

    print(f"\n  Files saved:")
    print(f"    {json_path}")
    print(f"    {txt_path}\n")

    print_summary(result)

    outputs = result.get("outputs", [])
    buckets = result.get("dynamic_buckets", [])
    print(f"  PHASE 3 SANITY CHECK")
    print(f"  Personalities  : {len(set(o['personality'] for o in outputs))}/16")
    print(f"  Models         : {len(set(o['model'] for o in outputs))}/2")
    print(f"  Total outputs  : {len(outputs)}/32")
    print(f"  Scoring method : {result.get('scoring_method', 'unknown')}")
    print(f"  Dynamic buckets: {[b['label'] for b in buckets]}\n")

    return result


def run_compare(signal_key: str = "ukraine_nato"):
    """
    Compare dynamic_bucket_scorer (live analysis) vs semantic_scorer (Brier/fixed).
    Shows where the two scorers disagree — useful for calibrating the dynamic system.
    The fixed-bucket scorer is ground truth for validation; dynamic is ground truth for analysis.
    """
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    if signal_key not in TEST_SIGNALS:
        signal_key = "ukraine_nato"

    signal    = TEST_SIGNALS[signal_key]
    signal_id = f"phase3_compare_{signal_key}"

    print(f"\n  Phase 3 — Dynamic vs Fixed-Bucket Comparison")
    print(f"  Signal: '{signal_key}'\n")

    result_base = run_all_models(signal=signal, signal_id=signal_id)

    import copy
    result_dynamic = copy.deepcopy(result_base)
    result_fixed   = copy.deepcopy(result_base)

    print("\n  Scoring with dynamic buckets...")
    result_dynamic = score_and_map_dynamic(result_dynamic)

    print("  Scoring with fixed buckets (Brier/semantic)...")
    result_fixed = score_and_map_semantic(result_fixed)

    # Per-personality comparison
    print("\n" + "=" * 72)
    print("  DYNAMIC vs FIXED — PER-PERSONALITY BUCKET ASSIGNMENT")
    print("=" * 72)
    print(f"  {'Personality':<28} {'Dynamic':<24} {'Fixed':<24} {'Match'}")
    print("  " + "-" * 70)

    outputs_dyn = {(o["personality"], o["model"]): o for o in result_dynamic["outputs"]}
    outputs_fix = {(o["personality"], o["model"]): o for o in result_fixed["outputs"]}

    mismatches = []
    matches    = []

    from src.personalities.prompts import PERSONALITY_ORDER
    for p_key in PERSONALITY_ORDER:
        for model_key in ["mistral", "qwen3.5"]:
            key    = (p_key, model_key)
            dyn_o  = outputs_dyn.get(key)
            fix_o  = outputs_fix.get(key)
            if not dyn_o or not fix_o:
                continue

            dyn_bucket = dyn_o.get("bucket", "?")
            fix_bucket = fix_o.get("bucket", "?")

            # Semantic match: do the two buckets mean the same thing?
            # e.g. "military_escalation" (dynamic) ≈ "conflict_escalation" (fixed)
            CONFLICT_TERMS = {"conflict", "escalat", "war", "attack", "militar", "strike"}
            ECON_TERMS     = {"economic", "market", "crash", "sanction", "trade", "financial"}
            DIPL_TERMS     = {"diplomatic", "peace", "resolut", "ceasefire", "negot"}
            CONSPI_TERMS   = {"conspiracy", "manipulat", "staged", "false_flag"}

            def bucket_family(b: str) -> str:
                b = b.lower()
                if any(t in b for t in CONFLICT_TERMS): return "conflict"
                if any(t in b for t in ECON_TERMS):     return "economic"
                if any(t in b for t in DIPL_TERMS):     return "diplomatic"
                if any(t in b for t in CONSPI_TERMS):   return "conspiracy"
                return "other"

            dyn_fam = bucket_family(dyn_bucket)
            fix_fam = bucket_family(fix_bucket)
            semantic_match = dyn_fam == fix_fam
            match_str = "✓" if semantic_match else "✗ DIFF"

            p_name  = dyn_o["personality_name"]
            m_short = "Mistral" if "Mistral" in dyn_o.get("model_display", "") else "Qwen"
            label   = f"{p_name} ({m_short})"

            # Shorten bucket labels for display
            def short(b: str) -> str:
                b = b.lower()
                if "conflict" in b or "escalat" in b or "war" in b: return "Conflict/Escalation"
                if "economic" in b or "market"  in b: return "Economic"
                if "diplomatic" in b or "peace" in b: return "Diplomatic"
                if "conspiracy" in b:                  return "Conspiracy"
                if "routine" in b or "no_impact" in b: return "Routine/NoImpact"
                if "historical" in b:                  return "Historical"
                return b[:20]

            print(f"  {label:<28} {short(dyn_bucket):<24} {short(fix_bucket):<24} {match_str}")

            if not semantic_match:
                mismatches.append((p_name, m_short, dyn_bucket, fix_bucket,
                                   dyn_o.get("response", "")[:120]))
            else:
                matches.append((p_name, dyn_bucket))

    total_pairs = len(matches) + len(mismatches)
    print(f"\n  Semantic agreement: {len(matches)}/{total_pairs} outputs")
    print(f"  Differences:        {len(mismatches)}/{total_pairs} outputs\n")

    if mismatches:
        print("  CASES WHERE DYNAMIC DIFFERS FROM FIXED:")
        print("  (Dynamic is signal-specific; fixed is for Brier Score)\n")
        for p_name, model, dyn, fix, preview in mismatches[:5]:
            print(f"  {p_name} ({model})")
            print(f"    Dynamic → {dyn}")
            print(f"    Fixed   → {fix}")
            print(f"    Preview : {preview[:100]}...")
            print()

    # Dominant narrative comparison
    dyn_dom = result_dynamic.get("dominant_narrative", {})
    fix_dom = result_fixed.get("dominant_narrative", {})

    print("=" * 72)
    print("  DOMINANT NARRATIVE COMPARISON")
    print("=" * 72)
    print(f"  Dynamic : {dyn_dom.get('label', '?')} — {dyn_dom.get('probability_pct', '?')} "
          f"[{dyn_dom.get('confidence', '?')}] · {dyn_dom.get('cross_model_count', 0)} cross-model")
    print(f"  Fixed   : {fix_dom.get('label', '?')} — {fix_dom.get('probability_pct', '?')} "
          f"[{fix_dom.get('confidence', '?')}] · {fix_dom.get('cross_model_count', 0)} cross-model")

    dyn_buckets = result_dynamic.get("dynamic_buckets", [])
    print(f"\n  Dynamic buckets for this signal: {[b['label'] for b in dyn_buckets]}")
    print()

    return result_dynamic, result_fixed


def run_focused_dynamic(user_input: str):
    """Run focused pipeline with dynamic scoring."""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    from src.ingestion.focused_pipeline import build_focused_signal

    print(f"\n  Phase 3 — Focused Dynamic Analysis")
    print(f"  Input: '{user_input[:100]}'\n")

    signal_dict = build_focused_signal(user_input)
    signal_text = signal_dict["signal_text"]
    signal_id   = f"phase3_focused_{signal_dict['signal_id']}"
    dyn_buckets = signal_dict.get("dynamic_buckets", [])

    result = run_all_models(
        signal=signal_text,
        signal_id=signal_id,
    )

    print(f"\n  Dynamic scoring {result['total_outputs']} outputs...")
    # Pass the already-generated buckets into the scorer if available
    if dyn_buckets:
        result["dynamic_buckets"] = dyn_buckets
    result = score_and_map_dynamic(result)

    json_path = f"data/outputs/{signal_id}.json"
    txt_path  = f"data/outputs/{signal_id}_report.txt"

    save_result(result, json_path)
    write_text_report(result, txt_path)

    print_summary(result)
    return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ukraine_nato"

    if mode == "compare":
        signal_key = sys.argv[2] if len(sys.argv) > 2 else "ukraine_nato"
        run_compare(signal_key)
    elif mode == "focused":
        user_input = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        if not user_input:
            print('\n  Usage: python tests/test_phase3.py focused "your signal"\n')
            sys.exit(1)
        run_focused_dynamic(user_input)
    elif mode in TEST_SIGNALS:
        run_dynamic(mode)
    else:
        print(f"\n  Unknown mode: '{mode}'")
        print("  Usage:")
        print("    python tests/test_phase3.py ukraine_nato")
        print("    python tests/test_phase3.py compare ukraine_nato")
        print('    python tests/test_phase3.py focused "Iran fires missiles"')