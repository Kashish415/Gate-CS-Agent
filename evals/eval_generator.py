"""Component eval: Generator in isolation.
Generates questions for golden dataset slots and evaluates with GEval metrics.
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
from src.pipeline import _build_gen_chain, _invoke_with_retry, GROQ_RATE_LIMIT_DELAY

DATA_DIR = Path("data")

# -- Metrics --

judge = GroqJudge()

topic_adherence = GEval(
    name="Topic Adherence",
    criteria="Does this GATE CS question genuinely test knowledge of the specified subject and subtopic?",
    evaluation_steps=[
        "Check the requested subject and subtopic in the input prompt.",
        "Check if the generated question directly tests technical concepts within that subject and subtopic.",
        "Assign 1.0 for genuine topic alignment, and 0.0 for unrelated/wrong topic questions."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.7,
)

answer_correctness = GEval(
    name="Answer Correctness",
    criteria="Solve the generated question independently to verify if the provided answer key is correct.",
    evaluation_steps=[
        "Solve the question independently step by step.",
        "Compare your solution with the marked answer.",
        "Assign 1.0 if the provided answer is correct, and 0.0 if the provided answer is incorrect."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.8,
)

spec_compliance = GEval(
    name="Spec Compliance",
    criteria="Check whether the generated question strictly follows format and structural constraints.",
    evaluation_steps=[
        "For MCQ: check that options list has exactly 4 distinct strings and answer is a single letter (A-D).",
        "For MSQ: check that options list has exactly 4 distinct strings and answer is a list of letters (A-D).",
        "For NAT: check that options is null/empty and answer is a numeric integer or float.",
        "Assign 1.0 if format rules are fully satisfied, and 0.0 if any format rule is violated."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.9,
)


async def _generate_test_cases():
    settings = Settings()
    golden = json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))
    examples = json.loads((DATA_DIR / "examples.json").read_text(encoding="utf-8"))

    model = ChatGroq(
        model=settings.generator_model, api_key=settings.groq_api_key,
        max_tokens=settings.generator_max_tokens, temperature=0.3, max_retries=0,
    )

    test_cases = []
    for i, item in enumerate(golden[:10]):
        slot = SlotSpec(
            slot_index=i, subject=item["subject"], subtopic=item["subtopic"],
            question_type=QuestionType(item["question_type"]), marks=2,
        )
        slot_examples = examples.get(f"{slot.subject} / {slot.subtopic}", [])[:3]
        chain = _build_gen_chain(model, slot, slot_examples)
        try:
            payload = await _invoke_with_retry(chain)
            output = json.dumps(payload.model_dump(), indent=2)
            test_cases.append(LLMTestCase(
                input=f"Generate a {item['question_type']} question about {item['subject']} / {item['subtopic']}",
                actual_output=output,
            ))
        except Exception as err:
            print(f"[Slot {i}] Generation failed: {err}")
        time.sleep(2)

    return test_cases


def main():
    test_cases = asyncio.run(_generate_test_cases())
    if not test_cases:
        print("No test cases generated!")
        return
    print(f"\nEvaluating {len(test_cases)} generated questions in batches with sleep...")

    batch_size = 2
    delay = 5
    metrics = [topic_adherence, answer_correctness, spec_compliance]
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
