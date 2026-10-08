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


def _truncate_explanation(text: str | None, max_len: int = 200) -> str | None:
    if not text:
        return None
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class TelegramBot:
    def __init__(self, token, retries=3):
        self._token = token
        self._retries = retries

    async def _send(self, action):
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

    async def send_poll(self, chat_id, question, options, correct_id, explanation=None,
                        poll_type="quiz", multiple=False):
        return await self._send(lambda bot: bot.send_poll(
            chat_id=chat_id, question=question, options=list(options),
            type=poll_type, correct_option_id=correct_id,
            explanation=_truncate_explanation(explanation),
            is_anonymous=True, allows_multiple_answers=multiple,
        ))

    async def send_text(self, chat_id, text):
        return await self._send(lambda bot: bot.send_message(
            chat_id=chat_id, text=text, parse_mode="MarkdownV2",
        ))

    async def publish(self, chat_id, question):
        qtype = question.slot.question_type

        if qtype == QuestionType.MCQ:
            correct_id = ord(question.answer.upper()) - ord("A")
            return await self.send_poll(
                chat_id, question.question, question.options,
                correct_id, question.explanation,
            )

        if qtype == QuestionType.MSQ:
            ok = await self.send_poll(
                chat_id, question.question, question.options,
                None, poll_type="regular", multiple=True,
            )
            if ok:
                answer_text = ", ".join(question.answer)
                ok = await self.send_text(chat_id, _escape_v2(f"Click to view the answer:\n||{answer_text}||"))
            return ok

        # NAT
        for text in [question.question, f"Click to view the answer:\n||{question.answer}||"]:
            if not await self.send_text(chat_id, _escape_v2(text)):
                return False
        return True

    async def send_skipped(self, chat_id, slot):
        text = f"Slot {slot.slot_index + 1} skipped: no new verified {slot.question_type.value} question was available today."
        return await self.send_text(chat_id, _escape_v2(text))
