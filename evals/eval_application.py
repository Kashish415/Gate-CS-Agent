"""Application eval: Safety metrics + operational health from SQLite.
Safety: ToxicityMetric and BiasMetric on published questions.
Operational: acceptance rate, confidence, retry rate from daily_log.db.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig
from deepeval.metrics import BiasMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

from evals.judge import GroqJudge

DATA_DIR = Path("data")
DB_PATH = Path("observability/daily_log.db")

# -- Safety metrics --

judge = GroqJudge()
toxicity = ToxicityMetric(threshold=0.9, model=judge)
bias = BiasMetric(threshold=0.9, model=judge)


def _load_safety_cases():
    cache_path = DATA_DIR / "verified_cache.json"
    if not cache_path.exists():
        print("No verified_cache.json found.")
        return []

    questions = json.loads(cache_path.read_text(encoding="utf-8"))
    cases = []
    for q in questions:
        full_text = q["question"]
        if q.get("options"):
            full_text += "\nOptions: " + " | ".join(q["options"])
        full_text += f"\nAnswer: {q['answer']}\nExplanation: {q['explanation']}"
        cases.append(LLMTestCase(input="GATE CS question", actual_output=full_text))
    return cases


def _operational_metrics():
    if not DB_PATH.exists():
        print("No daily_log.db found — run the pipeline first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM daily_log").fetchone()[0]
    if total == 0:
        print("No rows in daily_log.")
        conn.close()
        return

    published = cursor.execute("SELECT COUNT(*) FROM daily_log WHERE published = 1").fetchone()[0]
    avg_confidence = cursor.execute(
        "SELECT AVG(confidence) FROM daily_log WHERE published = 1 AND confidence IS NOT NULL"
    ).fetchone()[0]
    retried = cursor.execute("SELECT COUNT(*) FROM daily_log WHERE retry_count > 0").fetchone()[0]
    exhausted = cursor.execute(
        "SELECT COUNT(*) FROM daily_log WHERE failure_reason = 'retry_exhausted'"
    ).fetchone()[0]
    avg_latency = cursor.execute(
        "SELECT AVG(latency_ms) FROM daily_log WHERE published = 1"
    ).fetchone()[0]
    agreed = cursor.execute(
        "SELECT COUNT(*) FROM daily_log WHERE agreement = 1"
    ).fetchone()[0]
    conn.close()

    acceptance_rate = published / total * 100
    retry_rate = retried / total * 100
    fallback_rate = exhausted / total * 100
    agreement_rate = agreed / total * 100

    print("\n--- Operational Metrics ---")
    print(f"Total slots logged:    {total}")
    print(f"Acceptance rate:       {acceptance_rate:.1f}% (target: >=70%)")
    print(f"Avg confidence:        {avg_confidence or 0:.1f}/5 (target: >=3.5)")
    print(f"Retry rate:            {retry_rate:.1f}% (target: <=40%)")
    print(f"Fallback rate:         {fallback_rate:.1f}% (target: <=15%)")
    print(f"Avg latency (pub'd):   {avg_latency or 0:.0f}ms (target: <=5000ms)")
    print(f"Agreement rate:        {agreement_rate:.1f}% (target: >=80%)")

    passed = all([
        acceptance_rate >= 70,
        (avg_confidence or 0) >= 3.5,
        retry_rate <= 40,
        fallback_rate <= 15,
        (avg_latency or 0) <= 5000,
        agreement_rate >= 80,
    ])
    print(f"\nOperational health: {'PASS' if passed else 'FAIL'}")


def main():
    # Safety evals
    cases = _load_safety_cases()
    if cases:
        print(f"\nEvaluating {len(cases)} questions for safety in batches with sleep...")
        batch_size = 2
        delay = 5
        metrics = [toxicity, bias]
        cache_cfg = CacheConfig(write_cache=False)
        for i in range(0, len(cases), batch_size):
            batch = cases[i : i + batch_size]
            print(f"\n--- Safety Batch {i // batch_size + 1} / {(len(cases) + batch_size - 1) // batch_size} ({len(batch)} cases) ---")
            evaluate(batch, metrics, cache_config=cache_cfg)
            if i + batch_size < len(cases):
                print(f"Sleeping {delay}s to respect rate limits...")
                time.sleep(delay)

    # Operational metrics
    _operational_metrics()


if __name__ == "__main__":
    main()
