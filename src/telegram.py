import asyncio
import logging

from telegram import Bot
from telegram.error import TelegramError
from telegram.helpers import escape_markdown

from .domain import QuestionType

logger = logging.getLogger(__name__)


def _escape_v2(text):
    parts = text.split("||")
    return "||".join(escape_markdown(p, version=2) for p in parts)


class TelegramBot:
    def __init__(self, token, retries=3):
        self._token = token
        self._retries = retries

    async def _send(self, chat_id, action):
        async with Bot(token=self._token) as bot:
            for attempt in range(self._retries):
                try:
                    await action(bot)
                    return True
                except TelegramError as exc:
                    logger.error("[Telegram] Attempt %s/%s: %s", attempt + 1, self._retries, exc.message)
                    if attempt + 1 < self._retries:
                        await asyncio.sleep(2 ** attempt)
            return False

    async def publish(self, chat_id, question):
        qtype = question.slot.question_type

        if qtype == QuestionType.MCQ:
            correct_id = ord(question.answer.upper()) - ord("A")
            explanation = question.explanation[:200] if question.explanation else None
            return await self._send(chat_id, lambda bot: bot.send_poll(
                chat_id=chat_id, question=question.question, options=list(question.options),
                type="quiz", correct_option_id=correct_id, explanation=explanation,
                is_anonymous=True,
            ))

        if qtype == QuestionType.MSQ:
            ok = await self._send(chat_id, lambda bot: bot.send_poll(
                chat_id=chat_id, question=question.question, options=list(question.options),
                type="regular", allows_multiple_answers=True, is_anonymous=True,
            ))
            if ok:
                answer_text = ", ".join(question.answer)
                ok = await self._send(chat_id, lambda bot: bot.send_message(
                    chat_id=chat_id, text=_escape_v2(f"Click to view the answer:\n||{answer_text}||"),
                    parse_mode="MarkdownV2",
                ))
            return ok

        # NAT
        for text in [question.question, f"Click to view the answer:\n||{question.answer}||"]:
            if not await self._send(chat_id, lambda bot: bot.send_message(
                chat_id=chat_id, text=_escape_v2(text), parse_mode="MarkdownV2",
            )):
                return False
        return True

    async def send_skipped(self, chat_id, slot):
        text = f"Slot {slot.slot_index + 1} skipped: no verified {slot.question_type.value} question available today."
        return await self._send(chat_id, lambda bot: bot.send_message(
            chat_id=chat_id, text=_escape_v2(text), parse_mode="MarkdownV2",
        ))
