import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.domain.enums import QuestionType
from src.domain.models import GeneratedQuestionPayload, SlotSpec, VerifierResult

_TYPE_RULES: dict[QuestionType, str] = {
    QuestionType.MCQ: (
        "MCQ rules: exactly four options, exactly one correct answer, "
        "answer is the option label as a string."
    ),
    QuestionType.MSQ: (
        "MSQ rules: exactly four options, one or more correct, "
        "answer is a JSON array of correct option labels."
    ),
    QuestionType.NAT: (
        "NAT rules: no options, answer is a single number (int or float), "
        "no extraneous text in the answer field."
    ),
}

_GENERATOR_SYSTEM = """You write GATE-CS style practice questions.
Match the difficulty, distractor style, and trap design shown in the examples.
Do not copy or paraphrase any example question.
Return a concise explanation of at most two sentences.
For MCQ and MSQ, answer only with option labels such as A or [A, C], never option text.
For NAT, answer only with a number."""

_VERIFIER_SYSTEM = """You are an independent GATE-CS examiner solving this question from scratch.
You have not seen any author's answer. Solve it yourself and judge whether the question itself is unambiguous.
Set confidence to 1 if the question is ambiguous or you cannot solve it; 5 only if the question is unambiguous and the answer is certain."""


def _format_examples(examples: list[dict[str, Any]]) -> str:
    return "\n\n".join(json.dumps(ex, ensure_ascii=False, indent=2) for ex in examples)


def _format_options(options: list[str] | None) -> str:
    if not options:
        return "(no options for this question type)"
    return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))


def build_generator_chain(
    model: BaseChatModel,
    slot: SlotSpec,
    examples: list[dict[str, Any]],
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _GENERATOR_SYSTEM),
            (
                "user",
                """Target: subject={subject}, subtopic={subtopic}, type={qtype}, marks={marks}.
{type_rules}

Reference examples:
{examples}

Now write one new question for the target above.
Keep the question and options concise so the complete response stays short.""",
            ),
        ]
    )
    return (
        prompt.partial(
            subject=slot.subject,
            subtopic=slot.subtopic,
            qtype=slot.question_type.value,
            marks=slot.marks,
            type_rules=_TYPE_RULES[slot.question_type],
            examples=_format_examples(examples),
        )
        | model.with_structured_output(GeneratedQuestionPayload, method="json_schema")
    )


def build_verifier_chain(
    model: BaseChatModel,
    slot: SlotSpec,
    question: str,
    options: list[str] | None,
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _VERIFIER_SYSTEM),
            (
                "user",
                """Target: subject={subject}, subtopic={subtopic}, type={qtype}, marks={marks}.
{type_rules}

Question:
{question}

Options:
{options}

Return your independent answer, brief reasoning, confidence, and ambiguity flag.""",
            ),
        ]
    )
    return (
        prompt.partial(
            subject=slot.subject,
            subtopic=slot.subtopic,
            qtype=slot.question_type.value,
            marks=slot.marks,
            type_rules=_TYPE_RULES[slot.question_type],
            question=question,
            options=_format_options(options),
        )
        | model.with_structured_output(VerifierResult, method="json_schema")
    )
