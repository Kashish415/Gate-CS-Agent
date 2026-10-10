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
from src.config import GENERATOR_MODEL
from src.domain import QuestionType, SlotSpec
from src.pipeline import _build_gen_chain, answers_match

DATA_DIR = Path("data")
judge = GroqJudge()

topic_adherence = GEval(
    name="Topic Adherence",
    criteria="Does this GATE CS question test knowledge of the specified subject and subtopic?",
    evaluation_steps=[
        "Check the requested subject and subtopic in the input.",
        "Check if the generated question tests concepts within that topic.",
        "Score 1.0 for correct topic, 0.0 for wrong topic."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.7,
)

answer_correctness = GEval(
    name="Answer Correctness",
    criteria="Solve the generated question independently to verify the answer key.",
    evaluation_steps=[
        "Solve the question step by step.",
        "Compare your solution with the marked answer.",
        "Score 1.0 if the answer is correct, 0.0 if incorrect."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.8,
)


async def _generate_test_cases():
    load_dotenv()
    golden = json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))
    examples = json.loads((DATA_DIR / "examples.json").read_text(encoding="utf-8"))

    model = ChatGroq(
        model=GENERATOR_MODEL, api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3, max_retries=3,
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
            payload = await chain.ainvoke({})
            output = json.dumps(payload.model_dump(), indent=2)
            test_cases.append(LLMTestCase(
                input=f"Generate {item['question_type']} about {item['subject']} / {item['subtopic']}",
                actual_output=output,
            ))
        except Exception as err:
            print(f"[Slot {i}] Generation failed: {err}")
        time.sleep(5)

    return test_cases


def main():
    test_cases = asyncio.run(_generate_test_cases())
    if not test_cases:
        print("No test cases generated!")
        return
    print(f"\nEvaluating {len(test_cases)} generated questions in batches...")

    batch_size = 2
    metrics = [topic_adherence, answer_correctness]
    cache_cfg = CacheConfig(write_cache=False)
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i : i + batch_size]
        print(f"\n--- Batch {i // batch_size + 1} ({len(batch)} cases) ---")
        evaluate(batch, metrics, cache_config=cache_cfg)
        if i + batch_size < len(test_cases):
            time.sleep(10)


if __name__ == "__main__":
    main()
