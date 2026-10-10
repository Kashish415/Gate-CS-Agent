import asyncio
import json
import logging
import os
from datetime import date

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from .config import (
    DB_PATH, EXAMPLES_PATH, GENERATOR_MODEL, VERIFIER_MODEL,
    SLOT_TEMPLATE, SYLLABUS_PATH, VERIFIED_CACHE_PATH, pick_daily_slots,
)
from .pipeline import pipeline
from .storage import Cache, Database
from .telegram import TelegramBot

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run():
    load_dotenv()
    today = date.today()
    db = Database(DB_PATH)

    if db.today_already_posted(today, len(SLOT_TEMPLATE)):
        logger.info("Already posted for %s", today)
        return

    syllabus = json.loads(SYLLABUS_PATH.read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    slots = pick_daily_slots(today, syllabus, db.recent_subtopic_usage())

    api_key = os.getenv("GROQ_API_KEY")
    generator = ChatGroq(model=GENERATOR_MODEL, api_key=api_key, temperature=0.3, max_retries=3)
    verifier = ChatGroq(model=VERIFIER_MODEL, api_key=api_key, temperature=0.3, max_retries=3)

    await pipeline.ainvoke(
        {"slots": slots, "generated": {}, "validation": {}, "verifier": {},
         "retries": {}, "resolved": {}, "publish_results": {}},
        config={"configurable": {
            "generator_model": generator,
            "verifier_model": verifier,
            "telegram": TelegramBot(os.getenv("TELEGRAM_BOT_TOKEN")),
            "cache": Cache(VERIFIED_CACHE_PATH),
            "db": db,
            "examples": examples,
            "chat_id": os.getenv("TELEGRAM_CHANNEL_ID"),
            "max_retries": 2,
            "min_confidence": 3,
        }},
    )


def main():
    try:
        asyncio.run(run())
    except Exception:
        logger.exception("Pipeline failed")


if __name__ == "__main__":
    main()
