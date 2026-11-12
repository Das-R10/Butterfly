"""
tests/test_phase1.py

Run from project root:
  python tests/test_phase1.py                    # ukraine_nato (default)
  python tests/test_phase1.py iran_war_hormuz    # Iran war signal
  python tests/test_phase1.py tariff_war_2025    # US-China tariffs
  python tests/test_phase1.py ukraine_prewar     # Ukraine pre-war
  python tests/test_phase1.py both               # ukraine_nato + ukraine_prewar

Output files per run:
  data/outputs/<id>.json           <- raw data, all 20 outputs
  data/outputs/<id>_report.txt     <- full untruncated responses
  data/outputs/<id>_summary.txt    <- verdict + narrative summaries + mutation chain
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.personalities.personality_runner import run_all_models, save_result, TEST_SIGNALS
from src.ensemble.similarity_scorer import score_and_map, print_probability_report, write_text_report
from src.ensemble.narrative_summarizer import write_summary, print_summary


def run_test(signal_key: str = "ukraine_nato"):
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    if signal_key not in TEST_SIGNALS:
        print(f"\n  Unknown signal: '{signal_key}'")
        print(f"  Available: {list(TEST_SIGNALS.keys())}\n")
        return

    signal = TEST_SIGNALS[signal_key]
    signal_id = f"phase1_{signal_key}"

    print(f"\n  Butterfly Algorithm — Phase 1")
    print(f"  Signal : '{signal_key}'")
    print(f"  Text   : {signal[:100]}...\n")
    print("  Step 1: Running personalities through both models...\n")

    result = run_all_models(signal=signal, signal_id=signal_id)

    print(f"\n  Step 2: Scoring {result['total_outputs']} outputs...")
    result = score_and_map(result)

    print("\n  Step 3: Saving files...")

    json_path = f"data/outputs/{signal_id}.json"
    save_result(result, json_path)

    txt_path = f"data/outputs/{signal_id}_report.txt"
    write_text_report(result, txt_path)

    summary_path = f"data/outputs/{signal_id}_summary.txt"
    write_summary(result, summary_path)

    print(f"\n  Files saved:")
    print(f"    JSON    -> {json_path}")
    print(f"    Report  -> {txt_path}   (full untruncated responses)")
    print(f"    Summary -> {summary_path}  (verdict + narrative summaries)\n")

    # Print the summary to terminal
    print_summary(result)

    # Sanity check
    outputs = result.get("outputs", [])
    personalities_seen = set(o["personality"] for o in outputs)
    models_seen = set(o["model"] for o in outputs)

    print(f"\n  SANITY CHECK")
    print(f"  Personalities : {len(personalities_seen)}/10")
    print(f"  Models        : {len(models_seen)}/2")
    print(f"  Total outputs : {len(outputs)}/20")

    if len(personalities_seen) == 16 and len(models_seen) == 2 and len(outputs) == 32:
        print(f"\n  PHASE 1 COMPLETE — open _summary.txt for the full narrative breakdown\n")
    else:
        print("\n  Some outputs missing — check Ollama is running.\n")

    return result


def run_both():
    for key in ["ukraine_nato", "ukraine_prewar"]:
        print(f"\n{'='*60}\n  Running: {key}\n{'='*60}")
        run_test(key)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "ukraine_nato"
    if arg == "both":
        run_both()
    else:
        run_test(arg)