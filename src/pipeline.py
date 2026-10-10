import asyncio
import json
import logging
from time import perf_counter
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from .domain import QuestionPayload, QuestionType, SlotSpec, VerifiedQuestion, VerifierPayload

logger = logging.getLogger(__name__)

TYPE_RULES = {
    QuestionType.MCQ: "Provide exactly 4 distinct options (A-D). Answer is a single letter.",
    QuestionType.MSQ: "Provide exactly 4 distinct options (A-D). Answer is a list of correct letters.",
    QuestionType.NAT: "No options. Answer must be a concrete number, never a variable like 'n'.",
}

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
        marks=slot.marks, type_rules=TYPE_RULES[slot.question_type], examples=examples_text,
    ) | model.with_structured_output(QuestionPayload, method="json_schema")


def _build_ver_chain(model, slot, question, options):
    formatted = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options)) if options else "(no options)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", _VER_SYSTEM),
        ("user", "Target: {subject} / {subtopic}, type={qtype}, marks={marks}.\n{type_rules}\n\nQuestion:\n{question}\n\nOptions:\n{options}\n\nReturn your independent answer."),
    ])
    return prompt.partial(
        subject=slot.subject, subtopic=slot.subtopic, qtype=slot.question_type.value,
        marks=slot.marks, type_rules=TYPE_RULES[slot.question_type],
        question=question, options=formatted,
    ) | model.with_structured_output(VerifierPayload, method="json_schema")


def validate(question):
    qtype = question.slot.question_type
    opts = question.options

    if qtype in (QuestionType.MCQ, QuestionType.MSQ):
        if opts is None or len(opts) != 4:
            return "option_count"
        if len(set(opts)) != 4:
            return "duplicate_option"

    if qtype == QuestionType.MCQ:
        if question.answer not in "ABCD":
            return "invalid_answer"
        if opts[ord(question.answer) - ord("A")] in question.question:
            return "answer_leak"

    elif qtype == QuestionType.MSQ:
        if not isinstance(question.answer, list) or not question.answer:
            return "invalid_answer"
        valid_labels = {chr(ord("A") + i) for i in range(len(opts))}
        if not all(label in valid_labels for label in question.answer):
            return "invalid_answer"
        selected = {opts[ord(label) - ord("A")] for label in question.answer}
        if any(opt in question.question for opt in selected):
            return "answer_leak"

    else:
        if opts is not None:
            return "nat_has_options"
        try:
            float(question.answer)
        except (TypeError, ValueError):
            return "unparseable_nat"

    return None


def answers_match(qtype, gen_answer, ver_answer):
    if qtype == QuestionType.MCQ:
        return isinstance(gen_answer, str) and isinstance(ver_answer, str) and gen_answer == ver_answer
    if qtype == QuestionType.MSQ:
        return isinstance(gen_answer, list) and isinstance(ver_answer, list) and set(gen_answer) == set(ver_answer)
    try:
        g, v = float(gen_answer), float(ver_answer)
    except (TypeError, ValueError):
        return False
    return abs(g - v) <= 0.01 * max(1, abs(g), abs(v))


# -- Pipeline state --

class PipelineState(TypedDict):
    slots: list[SlotSpec]
    generated: dict[int, QuestionPayload | None]
    validation: dict[int, str | None]
    verifier: dict[int, VerifierPayload | None]
    retries: dict[int, int]
    resolved: dict[int, VerifiedQuestion | None]
    publish_results: dict[int, bool]


# -- Graph nodes --

async def generate_node(state, config):
    model = config["configurable"]["generator_model"]
    examples_data = config["configurable"]["examples"]
    generated = dict(state["generated"])

    for slot in state["slots"]:
        if state["resolved"].get(slot.slot_index) is not None:
            continue
        if generated.get(slot.slot_index) is not None:
            continue
        examples = examples_data.get(f"{slot.subject} / {slot.subtopic}", [])[:3]
        try:
            payload = await _build_gen_chain(model, slot, examples).ainvoke({})
            payload.slot = slot
            generated[slot.slot_index] = payload
        except Exception as err:
            logger.error("[Generator] Slot %s: %s", slot.slot_index, err)
        await asyncio.sleep(1)

    return {"generated": generated}


async def validate_node(state, config):
    validation = dict(state["validation"])
    for slot in state["slots"]:
        if state["resolved"].get(slot.slot_index) is not None:
            continue
        question = state["generated"].get(slot.slot_index)
        if question is None or slot.slot_index in validation:
            continue
        reason = validate(question)
        validation[slot.slot_index] = reason
        logger.info("[Validate Slot %s] %s", slot.slot_index, reason or "PASSED")
    return {"validation": validation}


async def verify_node(state, config):
    model = config["configurable"]["verifier_model"]
    verifier = dict(state["verifier"])

    for slot in state["slots"]:
        if state["resolved"].get(slot.slot_index) is not None:
            continue
        if state["validation"].get(slot.slot_index) is not None:
            continue
        if verifier.get(slot.slot_index) is not None:
            continue
        question = state["generated"].get(slot.slot_index)
        if question is None:
            continue
        try:
            payload = await _build_ver_chain(model, slot, question.question, question.options).ainvoke({})
            verifier[slot.slot_index] = payload
        except Exception as err:
            logger.error("[Verifier] Slot %s: %s", slot.slot_index, err)
        await asyncio.sleep(1)

    return {"verifier": verifier}


async def resolve_node(state, config):
    max_retries = config["configurable"]["max_retries"]
    min_confidence = config["configurable"]["min_confidence"]

    generated = dict(state["generated"])
    validation = dict(state["validation"])
    verifier = dict(state["verifier"])
    retries = dict(state["retries"])
    resolved = dict(state["resolved"])

    for slot in state["slots"]:
        idx = slot.slot_index
        if resolved.get(idx) is not None:
            continue

        gen = generated.get(idx)
        val = validation.get(idx)
        ver = verifier.get(idx)

        failure = None
        if gen is None:
            failure = "generation_failed"
        elif val is not None:
            failure = val
        elif ver is None:
            failure = "verification_failed"
        elif ver.ambiguous:
            failure = "flagged_ambiguous"
        elif ver.confidence < min_confidence:
            failure = "low_confidence"
        elif not answers_match(slot.question_type, gen.answer, ver.answer):
            failure = "verifier_disagrees"

        if failure is None and gen and ver:
            logger.info("[Resolve Slot %s] SUCCESS", idx)
            resolved[idx] = VerifiedQuestion(
                slot=slot, question=gen.question, options=gen.options,
                answer=gen.answer, explanation=gen.explanation, verifier=ver,
            )
            continue

        retries[idx] = retries.get(idx, 0) + 1
        logger.info("[Resolve Slot %s] Rejected (%s, retry %s/%s)", idx, failure, retries[idx], max_retries)
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

    for slot in state["slots"]:
        idx = slot.slot_index
        if resolved.get(idx) is not None:
            continue
        if state["retries"].get(idx, 0) < max_retries:
            continue
        excluded = db.published_question_texts()
        excluded.update(q.question for q in resolved.values() if q is not None)
        cached = cache.get(slot, excluded)
        if cached is None:
            db.log_slot(slot, state, failure="retry_exhausted")
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
        if idx in publish_results:
            continue
        question = state["resolved"].get(idx)
        if question is None:
            publish_results[idx] = await telegram.send_skipped(chat_id, slot)
            continue
        started = perf_counter()
        published = await telegram.publish(chat_id, question)
        publish_results[idx] = published
        if published:
            question.published = True
            cache.save(question)
        db.log_slot(slot, state, gen=question, ver=question.verifier,
                    published=published, failure=None if published else "publish_failed",
                    latency=int((perf_counter() - started) * 1000))

    return {"publish_results": publish_results}


# -- Routing --

def route_after_validate(state):
    for s in state["slots"]:
        if state["resolved"].get(s.slot_index) is None:
            if state["validation"].get(s.slot_index) is None:
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
