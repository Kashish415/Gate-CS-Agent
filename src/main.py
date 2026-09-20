import asyncio
import json
import logging
from datetime import date

import langsmith

from langchain_groq import ChatGroq

from .comparators import build_comparators
from .config import (
	DB_PATH,
	EXAMPLES_PATH,
	GENERATOR_MODEL,
	SLOT_TEMPLATE,
	SYLLABUS_PATH,
	VERIFIED_CACHE_PATH,
	VERIFIER_MODEL,
	Settings,
	pick_daily_slots,
)
from .graph import PipelineState, pipeline
from .observability import SQLiteDatabase, VerifiedCache
from .publishing import BotAPIClient
from .validators import VALIDATORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run() -> None:
	settings = Settings()
	today = date.today()
	db = SQLiteDatabase(DB_PATH)

	if db.today_already_posted(today, len(SLOT_TEMPLATE)):
		logger.info("Daily run already completed for %s", today.isoformat())
		return

	syllabus: dict[str, list[str]] = json.loads(
		SYLLABUS_PATH.read_text(encoding="utf-8")
	)
	examples: dict[str, list[dict[str, object]]] = json.loads(
		EXAMPLES_PATH.read_text(encoding="utf-8")
	)
	recent_usage = db.recent_subtopic_usage()
	slots = pick_daily_slots(today, syllabus, recent_usage)

	generator_model = ChatGroq(
		model=GENERATOR_MODEL,
		api_key=settings.groq_api_key,
		max_tokens=settings.generator_max_tokens,
		temperature=0.3,
		max_retries=0,
	)
	verifier_model = ChatGroq(
		model=VERIFIER_MODEL,
		api_key=settings.groq_api_key,
		max_tokens=settings.verifier_max_tokens,
		temperature=0.3,
		max_retries=0,
	)
	telegram = BotAPIClient(
		token=settings.telegram_bot_token,
		retries=settings.telegram_publish_retries,
	)
	cache = VerifiedCache(VERIFIED_CACHE_PATH)

	def examples_lookup(subject: str, subtopic: str) -> list[dict[str, object]]:
		return examples.get(f"{subject} / {subtopic}", [])[:3]

	initial_state: PipelineState = {
		"slots": slots,
		"generated": {},
		"validation": {},
		"verifier": {},
		"retries": {},
		"resolved": {},
		"publish_results": {},
	}
	config = {
		"configurable": {
			"generator_model": generator_model,
			"verifier_model": verifier_model,
			"telegram": telegram,
			"cache": cache,
			"db": db,
			"examples_lookup": examples_lookup,
			"validators": VALIDATORS,
			"comparators": build_comparators(settings),
			"chat_id": settings.telegram_channel_id,
			"max_retries": settings.max_retries_per_slot,
			"min_confidence": settings.min_verifier_confidence,
		}
	}
	ls_client = langsmith.Client(
		api_key=settings.langsmith_api_key,
		api_url=settings.langsmith_endpoint,
	)
	with langsmith.tracing_context(
		project_name=settings.langsmith_project,
		enabled=settings.langsmith_tracing,
		client=ls_client,
	):
		await pipeline.ainvoke(initial_state, config=config)


def main() -> None:
	try:
		asyncio.run(run())
	except Exception:
		logger.exception("Daily GATE-CS pipeline failed")


if __name__ == "__main__":
	main()
