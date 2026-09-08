import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, cast

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.comparators.registry import build_comparators
from src.config import (
	EXAMPLES_PATH,
	SYLLABUS_PATH,
	VERIFIED_CACHE_PATH,
	load_settings,
	pick_daily_slots,
)
from src.graph.state import PipelineState
from src.graph.workflow import build_graph
from src.observability import db
from src.observability.cache import VerifiedCache
from src.publishing.client import BotAPIClient
from src.validators.registry import VALIDATORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
	with path.open(encoding="utf-8") as file:
		return json.load(file)


async def run() -> None:
	settings = load_settings()
	today = date.today()
	db.init_db()
	if db.today_already_posted(today):
		logger.info("Daily run already completed for %s", today.isoformat())
		return

	syllabus = cast(dict[str, list[str]], _load_json(SYLLABUS_PATH))
	examples = cast(dict[str, list[dict[str, object]]], _load_json(EXAMPLES_PATH))
	recent_usage = db.recent_subtopic_usage()
	slots = pick_daily_slots(today, syllabus, recent_usage)

	generator_model = ChatGroq(
		model="llama-3.3-70b-versatile",
		api_key=settings.groq_api_key,
	)
	verifier_model = ChatGoogleGenerativeAI(
		model="gemini-2.5-flash",
		api_key=settings.google_api_key,
	)
	telegram = BotAPIClient(
		token=settings.telegram_bot_token,
		retries=settings.telegram_publish_retries,
	)
	cache = VerifiedCache(VERIFIED_CACHE_PATH)

	def examples_lookup(subject: str, subtopic: str) -> list[dict[str, object]]:
		return examples.get(f"{subject} / {subtopic}", [])

	graph = build_graph(
		generator_model=generator_model,
		verifier_model=verifier_model,
		telegram=telegram,
		cache=cache,
		db=db,
		examples_lookup=examples_lookup,
		validators=VALIDATORS,
		comparators=build_comparators(settings),
		chat_id=settings.telegram_channel_id,
		max_retries=settings.max_retries_per_slot,
		min_confidence=settings.min_verifier_confidence,
	)
	initial_state: PipelineState = {
		"slots": slots,
		"generated": {},
		"validation": {},
		"verifier": {},
		"retries": {},
		"resolved": {},
		"publish_results": {},
	}
	await graph.ainvoke(initial_state)


def main() -> None:
	try:
		asyncio.run(run())
	except Exception:
		logger.exception("Daily GATE-CS pipeline failed")


if __name__ == "__main__":
	main()
