from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.comparators.base import AnswerComparator
from src.config import MAX_RETRIES_PER_SLOT
from src.domain.enums import QuestionType
from src.graph.nodes import (
	CacheStore,
	Database,
	ExamplesLookup,
	generate_node,
	fallback_node,
	publish_node,
	resolve_node,
	validate_node,
	verify_node,
)
from src.graph.state import PipelineState
from src.publishing.client import TelegramClient
from src.validators.base import QuestionValidator


def route_after_validate(state: PipelineState) -> Literal["verify", "resolve"]:
	has_valid_question = False
	for slot in state["slots"]:
		if state["resolved"].get(slot.slot_index) is not None:
			continue
		validation = state["validation"].get(slot.slot_index)
		if validation is not None and validation.passed:
			has_valid_question = True
			break
	return "verify" if has_valid_question else "resolve"


def route_after_resolve(
	state: PipelineState,
	max_retries: int = MAX_RETRIES_PER_SLOT,
) -> Literal["generate", "fallback", "publish"]:
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


def build_graph(
	generator_model: BaseChatModel,
	verifier_model: BaseChatModel,
	telegram: TelegramClient,
	cache: CacheStore,
	db: Database,
	examples_lookup: ExamplesLookup,
	validators: dict[QuestionType, QuestionValidator],
	comparators: dict[QuestionType, AnswerComparator],
	chat_id: str,
	max_retries: int = MAX_RETRIES_PER_SLOT,
	min_confidence: int = 3,
) -> CompiledStateGraph:
	graph = StateGraph(PipelineState)
	graph.add_node(
		"generate",
		generate_node(
			generator_model,
			examples_lookup,
		),
	)
	graph.add_node("validate", validate_node(validators))
	graph.add_node("verify", verify_node(verifier_model))
	graph.add_node(
		"resolve",
		resolve_node(
			comparators,
			min_confidence=min_confidence,
			max_retries=max_retries,
		),
	)
	graph.add_node("fallback", fallback_node(cache, db, max_retries))
	graph.add_node("publish", publish_node(telegram, cache, db, chat_id))

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
		lambda state: route_after_resolve(state, max_retries),
		{"generate": "generate", "fallback": "fallback", "publish": "publish"},
	)
	graph.add_edge("fallback", "publish")
	graph.add_edge("publish", END)
	return graph.compile()
