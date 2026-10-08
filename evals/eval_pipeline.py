"""Pipeline eval: End-to-end generate -> verify -> resolve quality.
Uses verified_cache.json (questions that passed the full pipeline).
"""
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

from evals.judge import GroqJudge

DATA_DIR = Path("data")

# -- Metrics (analogous to RAG triad) --

judge = GroqJudge()

spec_faithfulness = GEval(
    name="Specification Faithfulness",
    criteria="The pipeline was asked to generate a question for a specific subject, subtopic, and question type. Check if the generated output matches all requirements.",
    evaluation_steps=[
        "Check if the question tests knowledge of the subject and subtopic requested in the input.",
        "Check format compliance: MCQ has 4 options and single letter answer (A-D); MSQ has 4 options and list of answer letters (A-D); NAT has options=null and a numeric answer.",
        "For MSQ, having 1 or more correct answers in the list is valid.",
        "If all requirements are met, score 1.0. If format or topic is wrong, score 0.0."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.8,
)

agreement_quality = GEval(
    name="Agreement Quality",
    criteria="Evaluate whether the agreed answer between generator and verifier is technically correct.",
    evaluation_steps=[
        "Check if generator answer and verifier_answer in actual_output match.",
        "Solve the question independently to determine the true correct answer.",
        "If the agreed answer is correct, assign a score of 1.0. If the agreed answer is wrong, assign a score of 0.0."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.8,
)

question_quality = GEval(
    name="Question Quality",
    criteria="Rate the technical accuracy, clarity, and quality of the GATE CS question.",
    evaluation_steps=[
        "Check technical accuracy: Is the question mathematically and logically sound?",
        "Check format-appropriate quality: For MCQ/MSQ, options must be plausible distractors. For NAT questions, options=null is EXPECTED and correct, and the target must be a clear numeric value.",
        "Check that the question text does not leak the answer.",
        "Assign a final quality score between 0.0 and 1.0 (1.0 = excellent GATE question, 0.0 = invalid/broken question)."
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge, threshold=0.6,
)


def _load_test_cases():
    cache_path = DATA_DIR / "verified_cache.json"
    if not cache_path.exists():
        print("No verified_cache.json found — run the pipeline first to generate questions.")
        return []

    questions = json.loads(cache_path.read_text(encoding="utf-8"))
    test_cases = []
    for q in questions:
        slot = q["slot"]
        input_text = f"Generate {slot['question_type']} about {slot['subject']} / {slot['subtopic']}"

        output = {
            "question": q["question"],
            "options": q.get("options"),
            "answer": q["answer"],
            "explanation": q["explanation"],
            "verifier_answer": q["verifier"]["answer"],
            "verifier_confidence": q["verifier"]["confidence"],
        }

        test_cases.append(LLMTestCase(
            input=input_text,
            actual_output=json.dumps(output, indent=2),
        ))

    return test_cases


def main():
    test_cases = _load_test_cases()
    if not test_cases:
        return
    print(f"\nEvaluating {len(test_cases)} pipeline outputs in batches with sleep...")

    batch_size = 2
    delay = 5
    metrics = [spec_faithfulness, agreement_quality, question_quality]
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
