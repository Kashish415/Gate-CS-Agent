import json
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.metrics import BiasMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

from evals.judge import GroqJudge

DATA_DIR = Path("data")
DB_PATH = Path("observability/daily_log.db")

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
    total = conn.execute("SELECT COUNT(*) FROM daily_log").fetchone()[0]
    if total == 0:
        print("No rows in daily_log.")
        conn.close()
        return

    published = conn.execute("SELECT COUNT(*) FROM daily_log WHERE published = 1").fetchone()[0]
    avg_confidence = conn.execute(
        "SELECT AVG(confidence) FROM daily_log WHERE published = 1 AND confidence IS NOT NULL"
    ).fetchone()[0]
    retried = conn.execute("SELECT COUNT(*) FROM daily_log WHERE retry_count > 0").fetchone()[0]
    exhausted = conn.execute(
        "SELECT COUNT(*) FROM daily_log WHERE failure_reason = 'retry_exhausted'"
    ).fetchone()[0]
    avg_latency = conn.execute(
        "SELECT AVG(latency_ms) FROM daily_log WHERE published = 1"
    ).fetchone()[0]
    agreed = conn.execute("SELECT COUNT(*) FROM daily_log WHERE agreement = 1").fetchone()[0]
    conn.close()

    print("\n--- Operational Metrics ---")
    print(f"Total slots:      {total}")
    print(f"Acceptance rate:  {published / total * 100:.1f}%")
    print(f"Avg confidence:   {avg_confidence or 0:.1f}/5")
    print(f"Retry rate:       {retried / total * 100:.1f}%")
    print(f"Fallback rate:    {exhausted / total * 100:.1f}%")
    print(f"Avg latency:      {avg_latency or 0:.0f}ms")
    print(f"Agreement rate:   {agreed / total * 100:.1f}%")


def main():
    load_dotenv()

    cases = _load_safety_cases()
    if cases:
        print(f"\nEvaluating {len(cases)} questions for safety...")
        batch_size = 2
        metrics = [toxicity, bias]
        cache_cfg = CacheConfig(write_cache=False)
        for i in range(0, len(cases), batch_size):
            batch = cases[i : i + batch_size]
            print(f"\n--- Safety Batch {i // batch_size + 1} ({len(batch)} cases) ---")
            evaluate(batch, metrics, cache_config=cache_cfg)
            if i + batch_size < len(cases):
                time.sleep(10)

    _operational_metrics()


if __name__ == "__main__":
    main()
