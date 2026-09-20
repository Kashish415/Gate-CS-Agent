import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Literal

from telegram import Bot
from telegram.error import TelegramError

from .domain import QuestionType, SlotSpec, VerifiedQuestion

logger = logging.getLogger(__name__)

_MARKDOWN_V2_SPECIAL = frozenset("_*[]()~`>#+-=|{}.!\\")


def _escape_markdown_v2(text: str) -> str:
	# preserve Telegram spoiler syntax (||text||) while escaping everything else
	parts = text.split("||")
	escaped_parts = []
	for part in parts:
		escaped = "".join(
			f"\\{char}" if char in _MARKDOWN_V2_SPECIAL else char
			for char in part
		)
		escaped_parts.append(escaped)
	return "||".join(escaped_parts)


def format_mcq(q: VerifiedQuestion) -> tuple[str, tuple[str, ...], int, str]:
	if q.slot.question_type is not QuestionType.MCQ or q.options is None:
		raise ValueError("format_mcq requires an MCQ with options")
	if not isinstance(q.answer, str) or not q.answer.isalpha():
		raise ValueError("MCQ answer must be an option label")
	correct_option_id = ord(q.answer.upper()) - ord("A")
	return q.question, tuple(q.options), correct_option_id, q.explanation[:200]


def format_msq(q: VerifiedQuestion) -> tuple[str, tuple[str, ...]]:
	if q.slot.question_type is not QuestionType.MSQ or q.options is None:
		raise ValueError("format_msq requires an MSQ with options")
	return q.question, tuple(q.options)


def format_msq_answer(q: VerifiedQuestion) -> str:
	if not isinstance(q.answer, list):
		raise ValueError("MSQ answer must be a list of option labels")
	answer = ", ".join(q.answer)
	return _escape_markdown_v2(f"Click to view the answer:\n||{answer}||")


def format_nat(q: VerifiedQuestion) -> list[str]:
	return [
		_escape_markdown_v2(q.question),
		_escape_markdown_v2(f"Click to view the answer:\n||{q.answer}||"),
	]


def format_skipped_slot(slot: SlotSpec) -> str:
	return _escape_markdown_v2(
		f"Slot {slot.slot_index + 1} skipped: no new verified {slot.question_type.value} "
		"question was available today."
	)


class BotAPIClient:
	def __init__(self, token: str, retries: int = 3) -> None:
		self._token = token
		self._retries = retries

	async def send_poll(
		self,
		chat_id: str,
		question: str,
		options: tuple[str, ...],
		correct_option_id: int | None,
		explanation: str | None = None,
		poll_type: Literal["quiz", "regular"] = "quiz",
		allow_multiple_answers: bool = False,
	) -> bool:
		async def send(bot: Bot) -> None:
			await bot.send_poll(
				chat_id=chat_id,
				question=question,
				options=list(options),
				type=poll_type,
				correct_option_id=correct_option_id,
				explanation=explanation[:200] if explanation else None,
				is_anonymous=True,
				allows_multiple_answers=allow_multiple_answers,
			)

		return await self._run(send)

	async def send_text(self, chat_id: str, text: str) -> bool:
		async def send(bot: Bot) -> None:
			await bot.send_message(
				chat_id=chat_id,
				text=text,
				parse_mode="MarkdownV2",
			)

		return await self._run(send)

	async def _run(self, operation: Callable[[Bot], Awaitable[None]]) -> bool:
		for attempt in range(self._retries):
			try:
				async with Bot(token=self._token) as bot:
					await operation(bot)
				return True
			except TelegramError as exc:
				logger.error(
					"[Telegram] Attempt %s/%s failed: %s: %s",
					attempt + 1,
					self._retries,
					type(exc).__name__,
					exc.message,
				)
				if attempt + 1 < self._retries:
					await asyncio.sleep(2**attempt)
		return False


async def _publish_mcq(telegram: BotAPIClient, chat_id: str, q: VerifiedQuestion) -> bool:
	question_text, options, correct_id, explanation = format_mcq(q)
	return await telegram.send_poll(chat_id, question_text, options, correct_id, explanation)


async def _publish_msq(telegram: BotAPIClient, chat_id: str, q: VerifiedQuestion) -> bool:
	question_text, options = format_msq(q)
	published = await telegram.send_poll(
		chat_id, question_text, options, None, poll_type="regular", allow_multiple_answers=True,
	)
	if published:
		published = await telegram.send_text(chat_id, format_msq_answer(q))
	return published


async def _publish_nat(telegram: BotAPIClient, chat_id: str, q: VerifiedQuestion) -> bool:
	messages = format_nat(q)
	results = [await telegram.send_text(chat_id, message) for message in messages]
	return all(results)


PUBLISHERS: dict[QuestionType, Callable[..., Awaitable[bool]]] = {
	QuestionType.MCQ: _publish_mcq,
	QuestionType.MSQ: _publish_msq,
	QuestionType.NAT: _publish_nat,
}
