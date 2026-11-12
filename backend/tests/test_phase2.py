"""
tests/test_phase2.py

Run modes:
  python tests/test_phase2.py focused "your signal text"   ← PRIMARY: focused single input
  python tests/test_phase2.py ingest                       ← background auto-ingest check
  python tests/test_phase2.py run N                        ← run signal N from ingest

The focused mode is the main production flow:
  user input → keywords → related articles → synthesized signal → verdict
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.personalities.personality_runner import run_all_models, save_result
from src.ensemble.similarity_scorer import score_and_map, write_text_report
from src.ensemble.narrative_summarizer import write_summary, print_summary


def run_focused(user_input: str):
    """
    PRIMARY MODE — user provides signal, system finds related
    articles, synthesizes, runs full personality engine, gives verdict.
    """
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    from src.ingestion.focused_pipeline import build_focused_signal

    print(f"\n  Butterfly Algorithm — Focused Signal Analysis")
    print(f"  Input: '{user_input[:100]}'\n")
    print("  Step 1: Building focused signal from your input + related articles...\n")

    signal = build_focused_signal(user_input)

    print(f"  Step 2: Running {signal['total_related']} related articles through personality engine...\n")

    result = run_all_models(
        signal=signal["signal_text"],
        signal_id=f"focused_{signal['signal_id']}",
    )

    result["user_input"]       = signal["user_input"]
    result["topic"]            = signal.get("topic", "")
    result["queries_used"]     = signal.get("queries_used", [])
    result["related_articles"] = signal.get("related_articles", [])

    print(f"\n  Step 3: Scoring {result['total_outputs']} outputs...")
    result = score_and_map(result)

    signal_id = result["signal_id"]
    json_path    = f"data/outputs/{signal_id}.json"
    txt_path     = f"data/outputs/{signal_id}_report.txt"
    summary_path = f"data/outputs/{signal_id}_summary.txt"

    save_result(result, json_path)
    write_text_report(result, txt_path)
    write_summary(result, summary_path)

    print(f"\n  Files saved:")
    print(f"    {json_path}")
    print(f"    {txt_path}")
    print(f"    {summary_path}\n")

    print_summary(result)

    outputs = result.get("outputs", [])
    print(f"\n  SANITY CHECK")
    print(f"  Personalities : {len(set(o['personality'] for o in outputs))}/16")
    print(f"  Models        : {len(set(o['model'] for o in outputs))}/2")
    print(f"  Total outputs : {len(outputs)}/32")
    print(f"  Related found : {signal['total_related']} articles\n")

    if len(outputs) == 32:
        print("  FOCUSED PIPELINE COMPLETE\n")

    return result


def ingest_only():
    """Background ingest — check what's available today."""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    from src.ingestion.ingest import run_ingestion, print_signals

    print("\n  Butterfly Algorithm — Background Ingestion Check\n")
    signals = run_ingestion(max_signals=10)
    print_signals(signals)
    return signals


def run_top_signal(signals=None, pick=0):
    """Run a specific signal from ingest through the engine."""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    if signals is None:
        from src.ingestion.ingest import run_ingestion
        signals = run_ingestion(max_signals=10)

    if not signals:
        print("  No signals found.")
        return

    signal = signals[pick]
    signal_text = signal["signal_text"]
    signal_id   = f"phase2_{signal['signal_id']}"

    print(f"\n  Running signal through personality engine...")
    print(f"  Source: {signal.get('source_type')} | {signal.get('source', '')}")
    print(f"  Score:  {signal['score']:.2f}")
    print(f"  Text:   {signal_text[:120]}...\n")

    result = run_all_models(signal=signal_text, signal_id=signal_id)
    result = score_and_map(result)

    json_path    = f"data/outputs/{signal_id}.json"
    txt_path     = f"data/outputs/{signal_id}_report.txt"
    summary_path = f"data/outputs/{signal_id}_summary.txt"

    save_result(result, json_path)
    write_text_report(result, txt_path)
    write_summary(result, summary_path)

    print(f"\n  Files saved:")
    print(f"    {json_path}")
    print(f"    {txt_path}")
    print(f"    {summary_path}\n")

    print_summary(result)

    outputs = result.get("outputs", [])
    print(f"\n  PHASE 2 SANITY CHECK")
    print(f"  Personalities : {len(set(o['personality'] for o in outputs))}/16")
    print(f"  Models        : {len(set(o['model'] for o in outputs))}/2")
    print(f"  Total outputs : {len(outputs)}/32\n")

    if len(outputs) == 32:
        print("  PHASE 2 COMPLETE\n")

    return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "focused":
        # Primary mode — user provides signal text
        user_input = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        if not user_input:
            print("\n  Usage: python tests/test_phase2.py focused \"your signal text here\"")
            print("  Example: python tests/test_phase2.py focused \"Iran fires missiles at Diego Garcia\"")
            sys.exit(1)
        run_focused(user_input)

    elif mode == "ingest":
        ingest_only()

    elif mode == "run":
        pick = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
        signals = ingest_only()
        if signals:
            run_top_signal(signals, pick=pick)

    else:
        print("\n  Butterfly Algorithm — Phase 2")
        print("  Usage:")
        print('    python tests/test_phase2.py focused "Iran fires missiles at Diego Garcia"')
        print("    python tests/test_phase2.py ingest")
        print("    python tests/test_phase2.py run 3")
        print()