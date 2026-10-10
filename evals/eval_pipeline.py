import json
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

from evals.judge import GroqJudge

DATA_DIR = Path("data")
judge = GroqJudge()

spec_faithfulness = GEval(
    name="Specification Faithfulness",
    criteria="Does the generated question match the requested subject, subtopic, and question type?",
    evaluation_steps=[
        "Check if the question tests the requested subject and subtopic.",
        "Check format: MCQ has 4 options + single letter answer, MSQ has 4 options + list of letters, NAT has no options + numeric answer.",
        "Score 1.0 if all requirements met, 0.0 if any violated."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.8,
)

agreement_quality = GEval(
    name="Agreement Quality",
    criteria="Is the agreed answer between generator and verifier technically correct?",
    evaluation_steps=[
        "Extract the generator answer and verifier_answer from actual_output.",
        "Solve the question independently.",
        "Score 1.0 if the agreed answer is correct, 0.0 if wrong."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.8,
)

question_quality = GEval(
    name="Question Quality",
    criteria="Rate the technical accuracy, clarity, and quality of the GATE CS question.",
    evaluation_steps=[
        "Check technical accuracy: is the question logically sound?",
        "Check quality: MCQ/MSQ options should be plausible distractors. NAT should have a clear numeric target.",
        "Check that the question text does not leak the answer.",
        "Score 0.0 to 1.0 (1.0 = excellent GATE question)."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.6,
)


def _load_test_cases():
    cache_path = DATA_DIR / "verified_cache.json"
    if not cache_path.exists():
        print("No verified_cache.json found — run the pipeline first.")
        return []

    questions = json.loads(cache_path.read_text(encoding="utf-8"))
    test_cases = []
    for q in questions:
        slot = q["slot"]
        question_output = {
            "question": q["question"],
            "options": q.get("options"),
            "answer": q["answer"],
            "explanation": q["explanation"],
        }
        context = (
            f"Generate {slot['question_type']} about {slot['subject']} / {slot['subtopic']}. "
            f"Verifier answer: {q['verifier']['answer']}, confidence: {q['verifier']['confidence']}"
        )
        test_cases.append(LLMTestCase(
            input=context,
            actual_output=json.dumps(question_output, indent=2),
        ))
    return test_cases


def main():
    load_dotenv()
    test_cases = _load_test_cases()
    if not test_cases:
        return
    print(f"\nEvaluating {len(test_cases)} pipeline outputs in batches...")

    batch_size = 2
    metrics = [spec_faithfulness, agreement_quality, question_quality]
    cache_cfg = CacheConfig(write_cache=False)
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i : i + batch_size]
        print(f"\n--- Batch {i // batch_size + 1} ({len(batch)} cases) ---")
        evaluate(batch, metrics, cache_config=cache_cfg)
        if i + batch_size < len(test_cases):
            time.sleep(10)


if __name__ == "__main__":
    main()
