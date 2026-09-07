from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from src.domain.enums import QuestionType
from src.domain.models import SlotSpec

load_dotenv()

SLOT_TEMPLATE: tuple[tuple[QuestionType, int], ...] = (
    (QuestionType.MCQ, 1),
    (QuestionType.MCQ, 1),
    (QuestionType.MCQ, 2),
    (QuestionType.MSQ, 2),
    (QuestionType.NAT, 2),
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


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    telegram_bot_token: str
    telegram_channel_id: str
    langsmith_api_key: str
    langsmith_project: str
    langsmith_tracing: bool
    max_retries_per_slot: int
    nat_absolute_tolerance: float
    nat_relative_tolerance: float
    min_verifier_confidence: int
    telegram_publish_retries: int


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def load_settings() -> Settings:
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_channel_id = os.getenv("TELEGRAM_CHANNEL_ID", "")
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "gate-cs-agent")
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"

    missing = [
        name
        for name, value in (
            ("GROQ_API_KEY", groq_api_key),
            ("TELEGRAM_BOT_TOKEN", telegram_bot_token),
            ("TELEGRAM_CHANNEL_ID", telegram_channel_id),
            ("LANGSMITH_API_KEY", langsmith_api_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        groq_api_key=groq_api_key,
        telegram_bot_token=telegram_bot_token,
        telegram_channel_id=telegram_channel_id,
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
        langsmith_tracing=langsmith_tracing,
        max_retries_per_slot=_env_int("MAX_RETRIES_PER_SLOT", MAX_RETRIES_PER_SLOT),
        nat_absolute_tolerance=_env_float("NAT_ABSOLUTE_TOLERANCE", NAT_ABSOLUTE_TOLERANCE),
        nat_relative_tolerance=_env_float("NAT_RELATIVE_TOLERANCE", NAT_RELATIVE_TOLERANCE),
        min_verifier_confidence=_env_int("MIN_VERIFIER_CONFIDENCE", MIN_VERIFIER_CONFIDENCE),
        telegram_publish_retries=_env_int(
            "TELEGRAM_PUBLISH_RETRIES", TELEGRAM_PUBLISH_RETRIES
        ),
    )


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
    for slot_index, ((qtype, marks), subject) in enumerate(
        zip(SLOT_TEMPLATE, subject_keys)
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
            )
        )
    return slots
