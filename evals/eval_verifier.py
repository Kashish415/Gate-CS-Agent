"""Component eval: Verifier in isolation.
Feeds golden questions to the verifier and checks if it gets the right answer.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from langchain_groq import ChatGroq

from evals.judge import GroqJudge
from src.config import Settings
from src.domain import QuestionType, SlotSpec
from src.pipeline import _build_ver_chain, _invoke_with_retry

DATA_DIR = Path("data")

# -- Metrics --

judge = GroqJudge()

verifier_accuracy = GEval(
    name="Verifier Accuracy",
    criteria="Compare the verifier answer in actual_output to the expected correct answer in expected_output.",
    evaluation_steps=[
        "Extract verifier answer from actual_output and expected answer from expected_output.",
        "Check match: For MCQ, option letters must match; for MSQ, sets of letters must match; for NAT, numbers must be within 1% tolerance.",
        "Assign 1.0 if answers match, and 0.0 if answers do not match."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=judge, threshold=0.8,
)

confidence_calibration = GEval(
    name="Confidence Calibration",
    criteria="Evaluate whether the verifier's confidence score (1-5) is justified by answer correctness.",
    evaluation_steps=[
        "Check if verifier answer matches expected answer.",
        "If verifier answer is wrong AND verifier confidence is high (4 or 5), assign 0.0 (poor calibration).",
        "If verifier answer is correct AND verifier confidence is high (4 or 5), assign 1.0 (good calibration).",
        "If verifier confidence is low (1-3), assign 1.0 (appropriate caution)."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=judge, threshold=0.7,
)


async def _verify_test_cases():
    settings = Settings()
    golden = json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))

    model = ChatGroq(
        model=settings.verifier_model, api_key=settings.groq_api_key,
        max_tokens=settings.verifier_max_tokens, temperature=0.3, max_retries=0,
    )

    test_cases = []
    for i, item in enumerate(golden[:10]):
        slot = SlotSpec(
            slot_index=i, subject=item["subject"], subtopic=item["subtopic"],
            question_type=QuestionType(item["question_type"]), marks=2,
        )
        chain = _build_ver_chain(model, slot, item["question"], item.get("options"))
        try:
            payload = await _invoke_with_retry(chain)
            output = json.dumps(payload.model_dump(), indent=2)
            expected = json.dumps({"expected_answer": item["answer"]})
            test_cases.append(LLMTestCase(
                input=f"Verify: {item['question'][:200]}",
                actual_output=output,
                expected_output=expected,
            ))
        except Exception as err:
            print(f"[Slot {i}] Verification failed: {err}")
        time.sleep(2)

    return test_cases


def main():
    test_cases = asyncio.run(_verify_test_cases())
    if not test_cases:
        print("No test cases generated!")
        return
    print(f"\nEvaluating {len(test_cases)} verifier outputs in batches with sleep...")

    batch_size = 2
    delay = 5
    metrics = [verifier_accuracy, confidence_calibration]
    cache_cfg = CacheConfig(write_cache=False)
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i : i + batch_size]
        print(f"\n--- Batch {i // batch_size + 1} / {(len(test_cases) + batch_size - 1) // batch_size} ({len(batch)} cases) ---")
        evaluate(batch, metrics, cache_config=cache_cfg)
        if i + batch_size < len(test_cases):
            print(f"Sleeping {delay}s to respect rate limits...")
            time.sleep(delay)


if __name__ == "__main__":
    main()
