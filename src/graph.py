import asyncio
import logging
from collections.abc import Sequence
from datetime import date
from time import perf_counter
from typing import Any, Literal, TypedDict

import groq
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .comparators import Comparator
from .domain import (
	DailyLogRow,
	FailureReason,
	GeneratedQuestion,
	QuestionType,
	SlotSpec,
	ValidationOutcome,
	VerifiedQuestion,
	VerifierResult,
)
from .llm import build_generator_chain, build_verifier_chain
from .publishing import PUBLISHERS, format_skipped_slot
from .validators import Validator

logger = logging.getLogger(__name__)

GROQ_RATE_LIMIT_DELAY = 1.0
_RATE_LIMIT_RETRIES = 3


class PipelineState(TypedDict):
	slots: Sequence[SlotSpec]
	generated: dict[int, GeneratedQuestion | None]
	validation: dict[int, ValidationOutcome]
	verifier: dict[int, VerifierResult | None]
	retries: dict[int, int]
	resolved: dict[int, VerifiedQuestion | None]
	publish_results: dict[int, bool]


def _unresolved_slots(state: PipelineState) -> Sequence[SlotSpec]:
	return [
		slot
		for slot in state["slots"]
		if state["resolved"].get(slot.slot_index) is None
	]


async def _invoke_with_rate_limit_retry(chain: Runnable) -> Any:
	for attempt in range(_RATE_LIMIT_RETRIES):
		try:
			return await chain.ainvoke({})
		except groq.RateLimitError as exc:
			if attempt + 1 == _RATE_LIMIT_RETRIES:
				raise
			wait = int(exc.response.headers.get("retry-after", 30))
			logger.warning(
				"[Groq] Rate limited — waiting %ss (attempt %s/%s)",
				wait, attempt + 1, _RATE_LIMIT_RETRIES,
			)
			await asyncio.sleep(wait)


async def generate_node(state: PipelineState, config: RunnableConfig) -> dict:
	model = config["configurable"]["generator_model"]
	examples_lookup = config["configurable"]["examples_lookup"]
	slots = [
		slot
		for slot in _unresolved_slots(state)
		if state["generated"].get(slot.slot_index) is None
	]
	generated = dict(state["generated"])
	for slot in slots:
		chain = build_generator_chain(
			model, slot, examples_lookup(slot.subject, slot.subtopic),
		)
		try:
			payload = await _invoke_with_rate_limit_retry(chain)
			generated[slot.slot_index] = GeneratedQuestion(
				slot=slot, **payload.model_dump()
			)
		except Exception as error:
			logger.error("[Generator] Slot %s: %s", slot.slot_index, error)
		await asyncio.sleep(GROQ_RATE_LIMIT_DELAY)
	return {"generated": generated}


async def validate_node(state: PipelineState, config: RunnableConfig) -> dict:
	validators: dict[QuestionType, Validator] = config["configurable"]["validators"]
	validation = dict(state["validation"])
	for slot in _unresolved_slots(state):
		slot_index = slot.slot_index
		question = state["generated"].get(slot_index)
		if question is None or slot_index in validation:
			continue
		outcome = validators[slot.question_type].validate(question)
		validation[slot_index] = outcome
		logger.info(
			"[Validate Slot %s] %s: %s",
			slot_index,
			outcome.reason if outcome.reason else "PASSED",
			outcome.detail if outcome.detail else "Passed structural checks",
		)
	return {"validation": validation}


async def verify_node(state: PipelineState, config: RunnableConfig) -> dict:
	model = config["configurable"]["verifier_model"]
	slots = [
		slot
		for slot in _unresolved_slots(state)
		if state["validation"].get(slot.slot_index, ValidationOutcome(passed=False)).passed
		and state["verifier"].get(slot.slot_index) is None
	]
	verifier = dict(state["verifier"])
	for slot in slots:
		question = state["generated"].get(slot.slot_index)
		if question is None:
			continue
		chain = build_verifier_chain(model, slot, question.question, question.options)
		try:
			payload = await _invoke_with_rate_limit_retry(chain)
			verifier[slot.slot_index] = VerifierResult(**payload.model_dump())
		except Exception as error:
			logger.error("[Verifier] Slot %s: %s", slot.slot_index, error)
		await asyncio.sleep(GROQ_RATE_LIMIT_DELAY)
	return {"verifier": verifier}


async def resolve_node(state: PipelineState, config: RunnableConfig) -> dict:
	comparators: dict[QuestionType, Comparator] = config["configurable"]["comparators"]
	min_confidence: int = config["configurable"]["min_confidence"]
	max_retries: int = config["configurable"]["max_retries"]

	generated = dict(state["generated"])
	validation = dict(state["validation"])
	verifier = dict(state["verifier"])
	retries = dict(state["retries"])
	resolved = dict(state["resolved"])

	for slot in _unresolved_slots(state):
		slot_index = slot.slot_index
		gen = generated.get(slot_index)
		val = validation.get(slot_index)
		ver = verifier.get(slot_index)

		failure: FailureReason | None = None
		if gen is None:
			failure = FailureReason.SCHEMA_INVALID
		elif val is None or not val.passed:
			failure = (val.reason if val else None) or FailureReason.SCHEMA_INVALID
		elif ver is None:
			failure = FailureReason.VERIFIER_DISAGREES
		elif ver.ambiguous:
			failure = FailureReason.FLAGGED_AMBIGUOUS
		elif ver.confidence < min_confidence:
			failure = FailureReason.LOW_CONFIDENCE
		else:
			if not comparators[slot.question_type].matches(gen.answer, ver.answer):
				failure = FailureReason.VERIFIER_DISAGREES

		if failure is None and gen is not None and ver is not None:
			logger.info("[Resolve Slot %s - %s] SUCCESS", slot_index, slot.question_type.value)
			resolved[slot_index] = VerifiedQuestion(
				**gen.model_dump(),
				verifier=ver,
				published=False,
				cached_on=date.today(),
			)
			continue

		retries[slot_index] = retries.get(slot_index, 0) + 1
		logger.info(
			"[Resolve Slot %s - %s] Rejected (%s, retry %s/%s)",
			slot_index,
			slot.question_type.value,
			failure.value if failure else "unknown",
			retries[slot_index],
			max_retries,
		)
		if retries[slot_index] < max_retries:
			generated[slot_index] = None
			validation.pop(slot_index, None)
			verifier[slot_index] = None

	return {
		"generated": generated,
		"validation": validation,
		"verifier": verifier,
		"retries": retries,
		"resolved": resolved,
	}


async def fallback_node(state: PipelineState, config: RunnableConfig) -> dict:
	cache = config["configurable"]["cache"]
	db = config["configurable"]["db"]
	max_retries: int = config["configurable"]["max_retries"]
	resolved = dict(state["resolved"])
	for slot in _unresolved_slots(state):
		slot_index = slot.slot_index
		if state["retries"].get(slot_index, 0) < max_retries:
			continue
		started = perf_counter()
		excluded_questions = db.published_question_texts()
		excluded_questions.update(
			question.question
			for question in resolved.values()
			if question is not None
		)
		cached = cache.get(slot, excluded_questions)
		if cached is None:
			gen = state["generated"].get(slot_index)
			ver = state["verifier"].get(slot_index)
			db.log_slot(DailyLogRow(
				date=date.today(),
				slot_index=slot.slot_index,
				subject=slot.subject,
				subtopic=slot.subtopic,
				question_type=slot.question_type,
				marks=slot.marks,
				difficulty=slot.difficulty,
				question_text=gen.question if gen else None,
				generator_answer=gen.answer if gen else None,
				verifier_answer=ver.answer if ver else None,
				agreement=False,
				confidence=ver.confidence if ver else None,
				retry_count=state["retries"].get(slot_index, 0),
				latency_ms=int((perf_counter() - started) * 1000),
				published=False,
				failure_reason=FailureReason.RETRY_EXHAUSTED,
			))
			continue
		resolved[slot_index] = cached
	return {"resolved": resolved}


async def publish_node(state: PipelineState, config: RunnableConfig) -> dict:
	telegram = config["configurable"]["telegram"]
	cache = config["configurable"]["cache"]
	db = config["configurable"]["db"]
	chat_id: str = config["configurable"]["chat_id"]
	publish_results = dict(state["publish_results"])
	for slot in state["slots"]:
		slot_index = slot.slot_index
		question = state["resolved"].get(slot_index)
		if slot_index in publish_results:
			continue
		if question is None:
			publish_results[slot_index] = await telegram.send_text(
				chat_id, format_skipped_slot(slot),
			)
			continue
		started = perf_counter()
		publisher = PUBLISHERS[slot.question_type]
		published = await publisher(telegram, chat_id, question)
		publish_results[slot_index] = published
		if published:
			question.published = True
			cache.save(question)
		db.log_slot(DailyLogRow(
			date=date.today(),
			slot_index=slot.slot_index,
			subject=slot.subject,
			subtopic=slot.subtopic,
			question_type=slot.question_type,
			marks=slot.marks,
			difficulty=slot.difficulty,
			question_text=question.question,
			generator_answer=question.answer,
			verifier_answer=question.verifier.answer,
			agreement=published,
			confidence=question.verifier.confidence,
			retry_count=state["retries"].get(slot_index, 0),
			latency_ms=int((perf_counter() - started) * 1000),
			published=published,
			failure_reason=None if published else FailureReason.PUBLISH_FAILED,
		))
	return {"publish_results": publish_results}


def route_after_validate(state: PipelineState) -> Literal["verify", "resolve"]:
	has_valid_question = any(
		state["validation"].get(slot.slot_index, ValidationOutcome(passed=False)).passed
		for slot in state["slots"]
		if state["resolved"].get(slot.slot_index) is None
	)
	return "verify" if has_valid_question else "resolve"


def route_after_resolve(
	state: PipelineState, config: RunnableConfig,
) -> Literal["generate", "fallback", "publish"]:
	max_retries: int = config["configurable"]["max_retries"]
	unresolved = [
		slot
		for slot in state["slots"]
		if state["resolved"].get(slot.slot_index) is None
	]
	if not unresolved:
		return "publish"
	if any(state["retries"].get(slot.slot_index, 0) < max_retries for slot in unresolved):
		return "generate"
	return "fallback"


def build_graph() -> CompiledStateGraph:
	graph = StateGraph(PipelineState)
	graph.add_node("generate", generate_node)
	graph.add_node("validate", validate_node)
	graph.add_node("verify", verify_node)
	graph.add_node("resolve", resolve_node)
	graph.add_node("fallback", fallback_node)
	graph.add_node("publish", publish_node)

	graph.add_edge(START, "generate")
	graph.add_edge("generate", "validate")
	graph.add_conditional_edges(
		"validate",
		route_after_validate,
		{"verify": "verify", "resolve": "resolve"},
	)
	graph.add_edge("verify", "resolve")
	graph.add_conditional_edges(
		"resolve",
		route_after_resolve,
		{"generate": "generate", "fallback": "fallback", "publish": "publish"},
	)
	graph.add_edge("fallback", "publish")
	graph.add_edge("publish", END)
	return graph.compile()


pipeline = build_graph()

if __name__ == "__main__":
	from pathlib import Path

	png_data = pipeline.get_graph().draw_mermaid_png()
	Path("graph.png").write_bytes(png_data)
	print("Graph saved to graph.png")
