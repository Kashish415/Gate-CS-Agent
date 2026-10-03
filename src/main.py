import asyncio
import json
import logging
import os
from datetime import date

from langchain_groq import ChatGroq

from .config import DB_PATH, EXAMPLES_PATH, SLOT_TEMPLATE, SYLLABUS_PATH, VERIFIED_CACHE_PATH, Settings, pick_daily_slots
from .pipeline import pipeline
from .storage import Cache, Database
from .telegram import TelegramBot

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run():
    settings = Settings()
    today = date.today()
    db = Database(DB_PATH)

    if db.today_already_posted(today, len(SLOT_TEMPLATE)):
        logger.info("Already posted for %s", today)
        return

    os.environ.update({
        "LANGSMITH_API_KEY": settings.langsmith_api_key,
        "LANGSMITH_TRACING": str(settings.langsmith_tracing).lower(),
        "LANGSMITH_PROJECT": settings.langsmith_project,
        "LANGSMITH_ENDPOINT": settings.langsmith_endpoint,
    })

    syllabus = json.loads(SYLLABUS_PATH.read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    slots = pick_daily_slots(today, syllabus, db.recent_subtopic_usage())

    generator = ChatGroq(
        model=settings.generator_model, api_key=settings.groq_api_key,
        max_tokens=settings.generator_max_tokens, temperature=0.3, max_retries=0,
    )
    verifier = ChatGroq(
        model=settings.verifier_model, api_key=settings.groq_api_key,
        max_tokens=settings.verifier_max_tokens, temperature=0.3, max_retries=0,
    )

    await pipeline.ainvoke(
        {"slots": slots, "generated": {}, "validation": {}, "verifier": {},
         "retries": {}, "resolved": {}, "publish_results": {}},
        config={"configurable": {
            "generator_model": generator,
            "verifier_model": verifier,
            "telegram": TelegramBot(settings.telegram_bot_token, settings.telegram_publish_retries),
            "cache": Cache(VERIFIED_CACHE_PATH),
            "db": db,
            "examples": examples,
            "chat_id": settings.telegram_channel_id,
            "max_retries": settings.max_retries_per_slot,
            "min_confidence": settings.min_verifier_confidence,
        }},
    )


def main():
    try:
        asyncio.run(run())
    except Exception:
        logger.exception("Pipeline failed")


if __name__ == "__main__":
    main()
