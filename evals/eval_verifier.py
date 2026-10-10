import asyncio
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_groq import ChatGroq

from evals.judge import GroqJudge
from src.config import VERIFIER_MODEL
from src.domain import QuestionType, SlotSpec
from src.pipeline import _build_ver_chain

DATA_DIR = Path("data")
judge = GroqJudge()

verifier_accuracy = GEval(
    name="Verifier Accuracy",
    criteria="Compare the verifier answer to the known correct answer.",
    evaluation_steps=[
        "Extract verifier answer from actual_output and expected answer from expected_output.",
        "For MCQ, letters must match. For MSQ, sets must match. For NAT, numbers within 1%.",
        "Score 1.0 if match, 0.0 if mismatch."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=judge, threshold=0.8,
)

confidence_calibration = GEval(
    name="Confidence Calibration",
    criteria="Is the verifier's confidence score justified by answer correctness?",
    evaluation_steps=[
        "Check if verifier answer matches expected answer.",
        "If wrong AND confidence is 4-5, score 0.0 (overconfident).",
        "If correct AND confidence is 4-5, score 1.0 (well calibrated).",
        "If confidence is 1-3, score 1.0 (appropriate caution)."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=judge, threshold=0.7,
)


async def _verify_test_cases():
    load_dotenv()
    golden = json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))

    model = ChatGroq(
        model=VERIFIER_MODEL, api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3, max_retries=3,
    )

    test_cases = []
    for i, item in enumerate(golden[:10]):
        slot = SlotSpec(
            slot_index=i, subject=item["subject"], subtopic=item["subtopic"],
            question_type=QuestionType(item["question_type"]), marks=2,
        )
        chain = _build_ver_chain(model, slot, item["question"], item.get("options"))
        try:
            payload = await chain.ainvoke({})
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
    print(f"\nEvaluating {len(test_cases)} verifier outputs in batches...")

    batch_size = 2
    metrics = [verifier_accuracy, confidence_calibration]
    cache_cfg = CacheConfig(write_cache=False)
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i : i + batch_size]
        print(f"\n--- Batch {i // batch_size + 1} ({len(batch)} cases) ---")
        evaluate(batch, metrics, cache_config=cache_cfg)
        if i + batch_size < len(test_cases):
            time.sleep(5)


if __name__ == "__main__":
    main()
