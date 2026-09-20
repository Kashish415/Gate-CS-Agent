import random
from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain import Difficulty, QuestionType, SlotSpec

SLOT_TEMPLATE: tuple[tuple[QuestionType, int, Difficulty], ...] = (
	(QuestionType.MCQ, 1, Difficulty.EASY),
	(QuestionType.MSQ, 2, Difficulty.MEDIUM),
	(QuestionType.NAT, 2, Difficulty.HARD),
)

SYLLABUS_PATH = Path("data/syllabus.json")
EXAMPLES_PATH = Path("data/examples.json")
VERIFIED_CACHE_PATH = Path("data/verified_cache.json")
DB_PATH = Path("observability/daily_log.db")

MAX_RETRIES_PER_SLOT = 2
NAT_ABSOLUTE_TOLERANCE = 0.01
NAT_RELATIVE_TOLERANCE = 0.01
MIN_VERIFIER_CONFIDENCE = 3
TELEGRAM_PUBLISH_RETRIES = 3
GENERATOR_MAX_TOKENS = 2048
VERIFIER_MAX_TOKENS = 1500
GENERATOR_MODEL = "openai/gpt-oss-120b"
VERIFIER_MODEL = "openai/gpt-oss-20b"


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env")
	groq_api_key: str
	telegram_bot_token: str
	telegram_channel_id: str
	langsmith_api_key: str
	langsmith_project: str = "gate-cs-agent"
	langsmith_endpoint: str = "https://api.smith.langchain.com"
	langsmith_tracing: bool = True
	max_retries_per_slot: int = MAX_RETRIES_PER_SLOT
	nat_absolute_tolerance: float = NAT_ABSOLUTE_TOLERANCE
	nat_relative_tolerance: float = NAT_RELATIVE_TOLERANCE
	min_verifier_confidence: int = MIN_VERIFIER_CONFIDENCE
	telegram_publish_retries: int = TELEGRAM_PUBLISH_RETRIES
	generator_max_tokens: int = GENERATOR_MAX_TOKENS
	verifier_max_tokens: int = VERIFIER_MAX_TOKENS


def pick_daily_slots(
	day: date,
	syllabus: dict[str, list[str]],
	recent_usage: set[tuple[str, str]],
) -> list[SlotSpec]:
	if len(syllabus) < len(SLOT_TEMPLATE):
		raise RuntimeError(
			f"Cannot pick {len(SLOT_TEMPLATE)} distinct subjects from syllabus; "
			f"only {len(syllabus)} available."
		)

	rng = random.Random(day.isoformat())
	subject_keys = list(syllabus.keys())
	rng.shuffle(subject_keys)

	slots: list[SlotSpec] = []

	for slot_index, ((qtype, marks, difficulty), subject) in enumerate(
		zip(SLOT_TEMPLATE, subject_keys, strict=False)
	):
		subtopics = syllabus[subject]
		available = [s for s in subtopics if (subject, s) not in recent_usage]
		subtopic = rng.choice(available or subtopics)
		slots.append(
			SlotSpec(
				slot_index=slot_index,
				subject=subject,
				subtopic=subtopic,
				question_type=qtype,
				marks=marks,
				difficulty=difficulty,
			)
		)
	return slots
