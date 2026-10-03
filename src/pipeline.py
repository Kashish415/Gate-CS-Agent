import asyncio
import json
import logging
from datetime import date
from time import perf_counter
from typing import TypedDict

import groq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from .domain import (
    DailyLogRow, FailureReason, GeneratedQuestion, MCQPayload,
    MCQVerifierPayload, MSQPayload, MSQVerifierPayload, NATPayload,
    NATVerifierPayload, QuestionType, SlotSpec, ValidationOutcome,
    VerifiedQuestion, VerifierResult,
)

logger = logging.getLogger(__name__)

GROQ_RATE_LIMIT_DELAY = 1.0
_RATE_LIMIT_RETRIES = 3

# -- LLM chain builders --

_TYPE_RULES = {
    QuestionType.MCQ: "Provide exactly 4 distinct options (A-D). Answer is a single letter.",
    QuestionType.MSQ: "Provide exactly 4 distinct options (A-D). Answer is a list of correct letters.",
    QuestionType.NAT: "No options. Answer must be a concrete number, never a variable like 'n'.",
}

_GEN_SCHEMAS = {QuestionType.MCQ: MCQPayload, QuestionType.MSQ: MSQPayload, QuestionType.NAT: NATPayload}
_VER_SCHEMAS = {QuestionType.MCQ: MCQVerifierPayload, QuestionType.MSQ: MSQVerifierPayload, QuestionType.NAT: NATVerifierPayload}

_GEN_SYSTEM = """You write GATE-CS style practice questions.
Match the difficulty and trap design shown in the examples. Do not copy any example.
Return a concise explanation of at most two sentences.
Respond in valid JSON matching the requested schema."""

_VER_SYSTEM = """You are an independent GATE-CS examiner solving this question from scratch.
You have not seen any author's answer. Solve it yourself and judge whether the question is unambiguous.
Return your independent answer, brief reasoning (max 2 sentences), confidence 1-5, and ambiguity flag.
Respond in valid JSON matching the requested schema."""


def _build_gen_chain(model, slot, examples):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _GEN_SYSTEM),
        ("user", "Target: {subject} / {subtopic}, type={qtype}, marks={marks}.\n{type_rules}\n\nExamples:\n{examples}\n\nWrite one new question. Keep it concise."),
    ])
    examples_text = "\n\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)
    return prompt.partial(
        subject=slot.subject, subtopic=slot.subtopic, qtype=slot.question_type.value,
        marks=slot.marks, type_rules=_TYPE_RULES[slot.question_type], examples=examples_text,
    ) | model.with_structured_output(_GEN_SCHEMAS[slot.question_type], method="json_schema")


def _build_ver_chain(model, slot, question, options):
    formatted = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options)) if options else "(no options)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", _VER_SYSTEM),
        ("user", "Target: {subject} / {subtopic}, type={qtype}, marks={marks}.\n{type_rules}\n\nQuestion:\n{question}\n\nOptions:\n{options}\n\nReturn your independent answer."),
    ])
    return prompt.partial(
        subject=slot.subject, subtopic=slot.subtopic, qtype=slot.question_type.value,
        marks=slot.marks, type_rules=_TYPE_RULES[slot.question_type],
        question=question, options=formatted,
    ) | model.with_structured_output(_VER_SCHEMAS[slot.question_type], method="json_schema")


# -- Validation (inline, no separate file) --

QUESTION_LIMIT = 300
OPTION_LIMIT = 100


def _validate(question):
    qtype = question.slot.question_type
    opts = question.options

    # shared option checks for MCQ/MSQ
    if qtype in (QuestionType.MCQ, QuestionType.MSQ):
        if len(question.question) > QUESTION_LIMIT:
            return ValidationOutcome(passed=False, reason=FailureReason.SCHEMA_INVALID)
        if opts is None or len(opts) != 4:
            return ValidationOutcome(passed=False, reason=FailureReason.OPTION_COUNT)
        if any(len(o) > OPTION_LIMIT for o in opts):
            return ValidationOutcome(passed=False, reason=FailureReason.SCHEMA_INVALID)
        if len(set(opts)) != 4:
            return ValidationOutcome(passed=False, reason=FailureReason.DUPLICATE_OPTION)

    if qtype == QuestionType.MCQ:
        if question.answer not in "ABCD":
            return ValidationOutcome(passed=False, reason=FailureReason.SCHEMA_INVALID)
        if opts[ord(question.answer) - ord("A")] in question.question:
            return ValidationOutcome(passed=False, reason=FailureReason.ANSWER_LEAK)

    elif qtype == QuestionType.MSQ:
        if not isinstance(question.answer, list) or not question.answer:
            return ValidationOutcome(passed=False, reason=FailureReason.SCHEMA_INVALID)
        valid = {chr(ord("A") + i) for i in range(len(opts))}
        if not all(label in valid for label in question.answer):
            return ValidationOutcome(passed=False, reason=FailureReason.SCHEMA_INVALID)
        selected = {opts[ord(label) - ord("A")] for label in question.answer}
        if any(opt in question.question for opt in selected):
            return ValidationOutcome(passed=False, reason=FailureReason.ANSWER_LEAK)

    else:  # NAT
        if opts is not None:
            return ValidationOutcome(passed=False, reason=FailureReason.SCHEMA_INVALID)
        if len(question.question) > QUESTION_LIMIT:
            return ValidationOutcome(passed=False, reason=FailureReason.SCHEMA_INVALID)
        try:
            float(question.answer)
        except (TypeError, ValueError):
            return ValidationOutcome(passed=False, reason=FailureReason.UNPARSEABLE_NAT)

    return ValidationOutcome(passed=True)


# -- Answer comparison (inline, no separate file) --

def _answers_match(qtype, gen_answer, ver_answer):
    if qtype == QuestionType.MCQ:
        return isinstance(gen_answer, str) and isinstance(ver_answer, str) and gen_answer == ver_answer
    if qtype == QuestionType.MSQ:
        return isinstance(gen_answer, list) and isinstance(ver_answer, list) and set(gen_answer) == set(ver_answer)
    # NAT
    try:
        g, v = float(gen_answer), float(ver_answer)
    except (TypeError, ValueError):
        return False
    diff = abs(g - v)
    if diff <= 0.01:
        return True
    denom = max(abs(g), abs(v))
    return denom > 0 and diff / denom <= 0.01


# -- Pipeline state + helpers --

class PipelineState(TypedDict):
    slots: list[SlotSpec]
    generated: dict[int, GeneratedQuestion | None]
    validation: dict[int, ValidationOutcome]
    verifier: dict[int, VerifierResult | None]
    retries: dict[int, int]
    resolved: dict[int, VerifiedQuestion | None]
    publish_results: dict[int, bool]


def _unresolved(state):
    return [s for s in state["slots"] if state["resolved"].get(s.slot_index) is None]


async def _invoke_with_retry(chain):
    for attempt in range(_RATE_LIMIT_RETRIES):
        try:
            return await chain.ainvoke({})
        except groq.RateLimitError as exc:
            if attempt + 1 == _RATE_LIMIT_RETRIES:
                raise
            wait = int(exc.response.headers.get("retry-after", 30))
            logger.warning("[Groq] Rate limited — waiting %ss (attempt %s/%s)", wait, attempt + 1, _RATE_LIMIT_RETRIES)
            await asyncio.sleep(wait)


def _log_slot(db, slot, state, gen=None, ver=None, published=False, failure=None, latency=0):
    db.log_slot(DailyLogRow(
        date=date.today(), slot_index=slot.slot_index,
        subject=slot.subject, subtopic=slot.subtopic,
        question_type=slot.question_type, marks=slot.marks, difficulty=slot.difficulty,
        question_text=gen.question if gen else None,
        generator_answer=gen.answer if gen else None,
        verifier_answer=ver.answer if ver else None,
        agreement=published, confidence=ver.confidence if ver else None,
        retry_count=state["retries"].get(slot.slot_index, 0),
        latency_ms=latency, published=published, failure_reason=failure,
    ))


# -- Graph nodes --

async def generate_node(state, config):
    model = config["configurable"]["generator_model"]
    examples_data = config["configurable"]["examples"]
    generated = dict(state["generated"])
    for slot in _unresolved(state):
        if generated.get(slot.slot_index) is not None:
            continue
        examples = examples_data.get(f"{slot.subject} / {slot.subtopic}", [])[:3]
        try:
            payload = await _invoke_with_retry(_build_gen_chain(model, slot, examples))
            generated[slot.slot_index] = GeneratedQuestion(slot=slot, **payload.model_dump())
        except Exception as err:
            logger.error("[Generator] Slot %s: %s", slot.slot_index, err)
        await asyncio.sleep(GROQ_RATE_LIMIT_DELAY)
    return {"generated": generated}


async def validate_node(state, config):
    validation = dict(state["validation"])
    for slot in _unresolved(state):
        question = state["generated"].get(slot.slot_index)
        if question is None or slot.slot_index in validation:
            continue
        outcome = _validate(question)
        validation[slot.slot_index] = outcome
        logger.info("[Validate Slot %s] %s", slot.slot_index, outcome.reason or "PASSED")
    return {"validation": validation}


async def verify_node(state, config):
    model = config["configurable"]["verifier_model"]
    verifier = dict(state["verifier"])
    for slot in _unresolved(state):
        val = state["validation"].get(slot.slot_index)
        if not (val and val.passed) or verifier.get(slot.slot_index) is not None:
            continue
        question = state["generated"].get(slot.slot_index)
        if question is None:
            continue
        try:
            payload = await _invoke_with_retry(_build_ver_chain(model, slot, question.question, question.options))
            verifier[slot.slot_index] = VerifierResult(**payload.model_dump())
        except Exception as err:
            logger.error("[Verifier] Slot %s: %s", slot.slot_index, err)
        await asyncio.sleep(GROQ_RATE_LIMIT_DELAY)
    return {"verifier": verifier}


async def resolve_node(state, config):
    min_confidence = config["configurable"]["min_confidence"]
    max_retries = config["configurable"]["max_retries"]

    generated = dict(state["generated"])
    validation = dict(state["validation"])
    verifier = dict(state["verifier"])
    retries = dict(state["retries"])
    resolved = dict(state["resolved"])

    for slot in _unresolved(state):
        idx = slot.slot_index
        gen = generated.get(idx)
        val = validation.get(idx)
        ver = verifier.get(idx)

        failure = None
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
        elif not _answers_match(slot.question_type, gen.answer, ver.answer):
            failure = FailureReason.VERIFIER_DISAGREES

        if failure is None and gen and ver:
            logger.info("[Resolve Slot %s] SUCCESS", idx)
            resolved[idx] = VerifiedQuestion(**gen.model_dump(), verifier=ver, published=False, cached_on=date.today())
            continue

        retries[idx] = retries.get(idx, 0) + 1
        logger.info("[Resolve Slot %s] Rejected (%s, retry %s/%s)", idx,
                    failure.value if failure else "unknown", retries[idx], max_retries)
        if retries[idx] < max_retries:
            generated[idx] = None
            validation.pop(idx, None)
            verifier[idx] = None

    return {"generated": generated, "validation": validation, "verifier": verifier,
            "retries": retries, "resolved": resolved}


async def fallback_node(state, config):
    cache = config["configurable"]["cache"]
    db = config["configurable"]["db"]
    max_retries = config["configurable"]["max_retries"]
    resolved = dict(state["resolved"])
    for slot in _unresolved(state):
        idx = slot.slot_index
        if state["retries"].get(idx, 0) < max_retries:
            continue
        started = perf_counter()
        excluded = db.published_question_texts()
        excluded.update(q.question for q in resolved.values() if q is not None)
        cached = cache.get(slot, excluded)
        if cached is None:
            _log_slot(db, slot, state,
                      gen=state["generated"].get(idx), ver=state["verifier"].get(idx),
                      failure=FailureReason.RETRY_EXHAUSTED,
                      latency=int((perf_counter() - started) * 1000))
            continue
        resolved[idx] = cached
    return {"resolved": resolved}


async def publish_node(state, config):
    telegram = config["configurable"]["telegram"]
    cache = config["configurable"]["cache"]
    db = config["configurable"]["db"]
    chat_id = config["configurable"]["chat_id"]
    publish_results = dict(state["publish_results"])
    for slot in state["slots"]:
        idx = slot.slot_index
        question = state["resolved"].get(idx)
        if idx in publish_results:
            continue
        if question is None:
            publish_results[idx] = await telegram.send_skipped(chat_id, slot)
            continue
        started = perf_counter()
        published = await telegram.publish(chat_id, question)
        publish_results[idx] = published
        if published:
            question.published = True
            cache.save(question)
        _log_slot(db, slot, state,
                  gen=question, ver=question.verifier, published=published,
                  failure=None if published else FailureReason.PUBLISH_FAILED,
                  latency=int((perf_counter() - started) * 1000))
    return {"publish_results": publish_results}


# -- Routing --

def route_after_validate(state):
    for s in state["slots"]:
        if state["resolved"].get(s.slot_index) is None:
            val = state["validation"].get(s.slot_index)
            if val and val.passed:
                return "verify"
    return "resolve"


def route_after_resolve(state, config):
    max_retries = config["configurable"]["max_retries"]
    unresolved = [s for s in state["slots"] if state["resolved"].get(s.slot_index) is None]
    if not unresolved:
        return "publish"
    if any(state["retries"].get(s.slot_index, 0) < max_retries for s in unresolved):
        return "generate"
    return "fallback"


# -- Build graph --

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("publish", publish_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"verify": "verify", "resolve": "resolve"})
    graph.add_edge("verify", "resolve")
    graph.add_conditional_edges("resolve", route_after_resolve, {"generate": "generate", "fallback": "fallback", "publish": "publish"})
    graph.add_edge("fallback", "publish")
    graph.add_edge("publish", END)
    return graph.compile()


pipeline = build_graph()
