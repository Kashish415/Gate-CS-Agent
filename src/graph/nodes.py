import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from time import perf_counter
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel

from src.comparators.base import AnswerComparator
from src.domain.enums import FailureReason, QuestionType
from src.domain.models import (
	DailyLogRow,
	GeneratedQuestion,
	GeneratedQuestionPayload,
	SlotSpec,
	ValidationOutcome,
	VerifierResult,
	VerifiedQuestion,
)
from src.graph.state import PipelineState
from src.llm.prompts import build_generator_chain, build_verifier_chain
from src.publishing.client import TelegramClient
from src.publishing.formatter import format_mcq, format_msq, format_msq_or_nat
from src.validators.base import QuestionValidator
class ExamplesLookup(Protocol):
	def __call__(self, subject: str, subtopic: str) -> list[dict[str, object]]: ...


class CacheStore(Protocol):
	def get(self, slot: SlotSpec) -> VerifiedQuestion | None: ...

	def get_excluded_questions(self) -> set[str]: ...

	def save(self, question: VerifiedQuestion) -> None: ...


class Database(Protocol):
	def log_slot(self, row: DailyLogRow) -> None: ...

	def published_question_texts(self) -> set[str]: ...


def _unresolved_slots(state: PipelineState) -> Sequence[SlotSpec]:
	return [
		slot
		for slot in state["slots"]
		if state["resolved"].get(slot.slot_index) is None
	]


def _log_row(
	slot: SlotSpec,
	generated: GeneratedQuestion | None,
	verifier: VerifierResult | None,
	retries: int,
	published: bool,
	failure_reason: FailureReason | None,
	latency_ms: int,
) -> DailyLogRow:
	return DailyLogRow(
		date=date.today(),
		slot_index=slot.slot_index,
		subject=slot.subject,
		subtopic=slot.subtopic,
		question_type=slot.question_type,
		marks=slot.marks,
		question_text=generated.question if generated else None,
		generator_answer=generated.answer if generated else None,
		verifier_answer=verifier.answer if verifier else None,
		agreement=verifier is not None and failure_reason is None,
		confidence=verifier.confidence if verifier else None,
		retry_count=retries,
		latency_ms=latency_ms,
		published=published,
		failure_reason=failure_reason,
	)


def generate_node(
	generator_model: BaseChatModel,
	examples_lookup: ExamplesLookup,
) -> Callable[[PipelineState], Awaitable[PipelineState]]:
	async def node(state: PipelineState) -> PipelineState:
		slots = [
			slot
			for slot in _unresolved_slots(state)
			if state["generated"].get(slot.slot_index) is None
		]
		semaphore = asyncio.Semaphore(2)

		async def generate(slot: SlotSpec) -> tuple[int, GeneratedQuestion | None]:
			async with semaphore:
				chain = build_generator_chain(
					generator_model,
					slot,
					examples_lookup(slot.subject, slot.subtopic),
				)
				try:
					payload = await chain.ainvoke({})
				except Exception as error:
					print(f"[Generator Error] Slot {slot.slot_index}: {error}")
					return slot.slot_index, None
				if not isinstance(payload, GeneratedQuestionPayload):
					return slot.slot_index, None
				return slot.slot_index, GeneratedQuestion(slot=slot, **payload.model_dump())

		results = await asyncio.gather(*(generate(slot) for slot in slots))
		for slot_index, question in results:
			if question is not None:
				state["generated"][slot_index] = question
		return state

	return node


def validate_node(
	validators: dict[QuestionType, QuestionValidator],
) -> Callable[[PipelineState], Awaitable[PipelineState]]:
	async def node(state: PipelineState) -> PipelineState:
		for slot in _unresolved_slots(state):
			slot_index = slot.slot_index
			question = state["generated"].get(slot_index)
			if question is None or slot_index in state["validation"]:
				continue
			state["validation"][slot_index] = validators[slot.question_type].validate(question)
		return state

	return node


def verify_node(
	verifier_model: BaseChatModel,
) -> Callable[[PipelineState], Awaitable[PipelineState]]:
	async def node(state: PipelineState) -> PipelineState:
		slots = [
			slot
			for slot in _unresolved_slots(state)
			if state["validation"].get(slot.slot_index, ValidationOutcome(passed=False)).passed
			and state["verifier"].get(slot.slot_index) is None
		]

		verifier_semaphore = asyncio.Semaphore(2)

		async def verify(slot: SlotSpec) -> tuple[int, VerifierResult | None]:
			async with verifier_semaphore:
				question = state["generated"][slot.slot_index]
				if question is None:
					return slot.slot_index, None
				chain = build_verifier_chain(
					verifier_model,
					slot,
					question.question,
					question.options,
				)
				try:
					result = await chain.ainvoke({})
				except Exception as e:
					print(f"[Verifier Error] Slot {slot.slot_index} failed: {e}")
					return slot.slot_index, None
				return slot.slot_index, result if isinstance(result, VerifierResult) else None

		results = await asyncio.gather(*(verify(slot) for slot in slots))
		for slot_index, result in results:
			state["verifier"][slot_index] = result
		return state

	return node


def resolve_node(
	comparators: dict[QuestionType, AnswerComparator],
	min_confidence: int,
	max_retries: int,
) -> Callable[[PipelineState], Awaitable[PipelineState]]:
	async def node(state: PipelineState) -> PipelineState:
		for slot in _unresolved_slots(state):
			slot_index = slot.slot_index
			generated = state["generated"].get(slot_index)
			validation = state["validation"].get(slot_index)
			verifier = state["verifier"].get(slot_index)

			failure: FailureReason | None = None
			if generated is None:
				failure = FailureReason.SCHEMA_INVALID
			elif validation is None or not validation.passed:
				failure = (validation.reason if validation else None) or FailureReason.SCHEMA_INVALID
			elif verifier is None:
				failure = FailureReason.VERIFIER_DISAGREES
			elif verifier.ambiguous:
				failure = FailureReason.FLAGGED_AMBIGUOUS
			elif verifier.confidence < min_confidence:
				failure = FailureReason.LOW_CONFIDENCE
			elif not comparators[slot.question_type].matches(
				generated.answer,
				verifier.answer,
			):
				failure = FailureReason.VERIFIER_DISAGREES

			if failure is None:
				state["resolved"][slot_index] = VerifiedQuestion(
					**generated.model_dump(),
					verifier=verifier,
					published=False,
					cached_on=date.today(),
				)
				continue

			state["retries"][slot_index] = state["retries"].get(slot_index, 0) + 1
			if state["retries"][slot_index] < max_retries:
				state["generated"][slot_index] = None
				state["validation"].pop(slot_index, None)
				state["verifier"][slot_index] = None
		return state

	return node


def fallback_node(
	cache: CacheStore,
	db: Database,
	max_retries: int,
) -> Callable[[PipelineState], Awaitable[PipelineState]]:
	async def node(state: PipelineState) -> PipelineState:
		for slot in _unresolved_slots(state):
			slot_index = slot.slot_index
			if state["retries"].get(slot_index, 0) < max_retries:
				continue
			started = perf_counter()
			cached = cache.get(slot, db.published_question_texts())
			if cached is None:
				db.log_slot(
					_log_row(
						slot,
						state["generated"].get(slot_index),
						state["verifier"].get(slot_index),
						state["retries"].get(slot_index, 0),
						False,
						FailureReason.RETRY_EXHAUSTED,
						int((perf_counter() - started) * 1000),
					)
				)
				continue
			state["resolved"][slot_index] = cached
		return state

	return node


def publish_node(
	telegram: TelegramClient,
	cache: CacheStore,
	db: Database,
	chat_id: str,
) -> Callable[[PipelineState], Awaitable[PipelineState]]:
	async def node(state: PipelineState) -> PipelineState:
		for slot in state["slots"]:
			slot_index = slot.slot_index
			question = state["resolved"].get(slot_index)
			if question is None or slot_index in state["publish_results"]:
				continue
			started = perf_counter()
			if slot.question_type is QuestionType.MCQ:
				question_text, options, correct_id, explanation = format_mcq(question)
				published = await telegram.send_poll(
					chat_id,
					question_text,
					options,
					correct_id,
					explanation,
				)
			elif slot.question_type is QuestionType.MSQ:
				question_text, options = format_msq(question)
				published = await telegram.send_poll(
					chat_id,
					question_text,
					options,
					None,
					poll_type="regular",
					allow_multiple_answers=True,
				)
				if published:
					published = await telegram.send_text(
						chat_id,
						format_msq_or_nat(question)[1],
					)
			else:
				messages = format_msq_or_nat(question)
				results = [
					await telegram.send_text(chat_id, message) for message in messages
				]
				published = all(results)
			state["publish_results"][slot_index] = published
			if published:
				cache.save(question)
			db.log_slot(
				_log_row(
					slot,
					question,
					question.verifier,
					state["retries"].get(slot_index, 0),
					published,
					None if published else FailureReason.PUBLISH_FAILED,
					int((perf_counter() - started) * 1000),
				)
			)
		return state

	return node
