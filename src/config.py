import random
from pathlib import Path
from .domain import Difficulty, QuestionType, SlotSpec

SLOT_TEMPLATE = (
    (QuestionType.MCQ, 1, Difficulty.EASY),
    (QuestionType.MSQ, 2, Difficulty.MEDIUM),
    (QuestionType.NAT, 2, Difficulty.HARD),
)

SYLLABUS_PATH = Path("data/syllabus.json")
EXAMPLES_PATH = Path("data/examples.json")
VERIFIED_CACHE_PATH = Path("data/verified_cache.json")
DB_PATH = Path("observability/daily_log.db")

GENERATOR_MODEL = "openai/gpt-oss-120b"
VERIFIER_MODEL = "openai/gpt-oss-20b"


def pick_daily_slots(day, syllabus, recent_usage):
    rng = random.Random(day.isoformat())
    subject_keys = list(syllabus.keys())
    rng.shuffle(subject_keys)

    slots = []
    for i, ((qtype, marks, difficulty), subject) in enumerate(
        zip(SLOT_TEMPLATE, subject_keys, strict=False)
    ):
        subtopics = syllabus[subject]
        available = [s for s in subtopics if (subject, s) not in recent_usage]
        subtopic = rng.choice(available or subtopics)
        slots.append(SlotSpec(
            slot_index=i, subject=subject, subtopic=subtopic,
            question_type=qtype, marks=marks, difficulty=difficulty,
        ))
    return slots
