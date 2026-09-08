import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

from telegram import Bot
from telegram.error import TelegramError


class TelegramClient(Protocol):
	async def send_poll(
		self,
		chat_id: str,
		question: str,
		options: tuple[str, ...],
		correct_option_id: int | None,
		explanation: str | None = None,
		poll_type: Literal["quiz", "regular"] = "quiz",
		allow_multiple_answers: bool = False,
	) -> bool: ...

	async def send_text(self, chat_id: str, text: str) -> bool: ...


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
			except TelegramError:
				if attempt + 1 < self._retries:
					await asyncio.sleep(2**attempt)
		return False
