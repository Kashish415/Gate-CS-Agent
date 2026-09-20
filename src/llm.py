import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .domain import (
	MCQPayload,
	MCQVerifierPayload,
	MSQPayload,
	MSQVerifierPayload,
	NATPayload,
	NATVerifierPayload,
	QuestionType,
	SlotSpec,
)

_TYPE_RULES: dict[QuestionType, str] = {
	QuestionType.MCQ: (
		"MCQ rules: options MUST be a list of 4 strings e.g. [\"Option A text\", \"Option B text\", \"Option C text\", \"Option D text\"]. "
		"The answer field MUST be a single option label string (e.g. \"A\")."
	),
	QuestionType.MSQ: (
		"MSQ rules: options MUST be a list of 4 strings e.g. [\"Option A text\", \"Option B text\", \"Option C text\", \"Option D text\"]. "
		"The answer field MUST be a list of string option labels (e.g. [\"A\", \"C\"])."
	),
	QuestionType.NAT: (
		"NAT rules: no options (options=null), answer is a single number (int or float). "
		"The answer must always be a single concrete number — never a variable, expression, or symbolic term like 'n'. "
		"The JSON response must always include all four fields — question, options, answer, explanation — even for NAT questions with no options. Never omit explanation."
	),
}

_GENERATOR_SCHEMAS: dict[QuestionType, type] = {
	QuestionType.MCQ: MCQPayload,
	QuestionType.MSQ: MSQPayload,
	QuestionType.NAT: NATPayload,
}

_VERIFIER_SCHEMAS: dict[QuestionType, type] = {
	QuestionType.MCQ: MCQVerifierPayload,
	QuestionType.MSQ: MSQVerifierPayload,
	QuestionType.NAT: NATVerifierPayload,
}

_GENERATOR_SYSTEM = """You write GATE-CS style practice questions.
Match the difficulty, distractor style, and trap design shown in the examples.
Do not copy or paraphrase any example question.
Return a concise explanation of at most two sentences.
Respond in valid JSON format matching the requested schema."""

_VERIFIER_SYSTEM = """You are an independent GATE-CS examiner solving this question from scratch.
You have not seen any author's answer. Solve it yourself and judge whether the question itself is unambiguous.
Return your independent answer, brief step-by-step reasoning (at most 2 sentences), confidence score from 1 to 5, and ambiguity flag.
Respond in valid JSON format matching the requested schema."""


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
			examples="\n\n".join(
				json.dumps(ex, ensure_ascii=False, indent=2) for ex in examples
			),
		)
		| model.with_structured_output(
			_GENERATOR_SCHEMAS[slot.question_type], method="json_mode"
		)
	)


def build_verifier_chain(
	model: BaseChatModel,
	slot: SlotSpec,
	question: str,
	options: list[str] | None,
) -> Runnable:
	formatted_options = (
		"\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
		if options
		else "(no options for this question type)"
	)
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
			options=formatted_options,
		)
		| model.with_structured_output(
			_VERIFIER_SCHEMAS[slot.question_type], method="json_mode"
		)
	)
