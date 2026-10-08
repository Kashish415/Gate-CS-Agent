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


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", extra="ignore")
	groq_api_key: str
	telegram_bot_token: str
	telegram_channel_id: str
	langsmith_api_key: str
	langsmith_project: str = "gate-cs-agent"
	langsmith_endpoint: str = "https://api.smith.langchain.com"
	langsmith_tracing: bool = True
	generator_model: str = "openai/gpt-oss-120b"
	verifier_model: str = "openai/gpt-oss-20b"
	max_retries_per_slot: int = 2
	min_verifier_confidence: int = 3
	telegram_publish_retries: int = 3
	generator_max_tokens: int = 2048
	verifier_max_tokens: int = 1500


def pick_daily_slots(day, syllabus, recent_usage):
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
