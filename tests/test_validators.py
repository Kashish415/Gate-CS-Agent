from src.domain import (
	Difficulty,
	FailureReason,
	GeneratedQuestion,
	QuestionType,
	SlotSpec,
)
from src.validators import MCQValidator, MSQValidator, NATValidator


def _slot(qtype: QuestionType) -> SlotSpec:
	return SlotSpec(
		slot_index=0,
		subject="Algorithms",
		subtopic="Sorting",
		question_type=qtype,
		marks=1,
		difficulty=Difficulty.EASY,
	)


def _question(
	qtype: QuestionType,
	question: str = "Short question?",
	options: list[str] | None = None,
	answer: str | list[str] | float = "A",
) -> GeneratedQuestion:
	return GeneratedQuestion(
		slot=_slot(qtype),
		question=question,
		options=options,
		answer=answer,
		explanation="Brief explanation.",
	)


# MCQValidator

def test_mcq_valid() -> None:
	q = _question(QuestionType.MCQ, options=["Alpha", "Beta", "Gamma", "Delta"], answer="B")
	assert MCQValidator().validate(q).passed is True


def test_mcq_rejects_wrong_option_count() -> None:
	q = _question(QuestionType.MCQ, options=["A", "B", "C"], answer="A")
	result = MCQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.OPTION_COUNT


def test_mcq_rejects_none_options() -> None:
	q = _question(QuestionType.MCQ, options=None, answer="A")
	result = MCQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.OPTION_COUNT


def test_mcq_rejects_duplicate_options() -> None:
	q = _question(QuestionType.MCQ, options=["Same", "Same", "Gamma", "Delta"], answer="C")
	result = MCQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.DUPLICATE_OPTION


def test_mcq_rejects_answer_leak() -> None:
	q = _question(
		QuestionType.MCQ,
		question="What is Alpha?",
		options=["Alpha", "Beta", "Gamma", "Delta"],
		answer="A",
	)
	result = MCQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.ANSWER_LEAK


def test_mcq_rejects_invalid_answer_label() -> None:
	q = _question(QuestionType.MCQ, options=["A", "B", "C", "D"], answer="E")
	result = MCQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID


def test_mcq_rejects_question_exceeding_length() -> None:
	q = _question(QuestionType.MCQ, question="x" * 301, options=["A", "B", "C", "D"], answer="A")
	result = MCQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID
	assert "300" in (result.detail or "")


def test_mcq_rejects_option_exceeding_length() -> None:
	q = _question(
		QuestionType.MCQ,
		options=["A", "B", "o" * 101, "D"],
		answer="A",
	)
	result = MCQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID
	assert "100" in (result.detail or "")


def test_mcq_accepts_exactly_at_length_limits() -> None:
	q = _question(
		QuestionType.MCQ,
		question="q" * 300,
		options=["o" * 100, "Beta", "Gamma", "Delta"],
		answer="B",
	)
	assert MCQValidator().validate(q).passed is True


# MSQValidator

def test_msq_valid() -> None:
	q = _question(QuestionType.MSQ, options=["Alpha", "Beta", "Gamma", "Delta"], answer=["A", "C"])
	assert MSQValidator().validate(q).passed is True


def test_msq_rejects_wrong_option_count() -> None:
	q = _question(QuestionType.MSQ, options=["A", "B"], answer=["A"])
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.OPTION_COUNT


def test_msq_rejects_duplicate_options() -> None:
	q = _question(QuestionType.MSQ, options=["X", "X", "Y", "Z"], answer=["A"])
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.DUPLICATE_OPTION


def test_msq_rejects_non_list_answer() -> None:
	q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer="A")
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_invalid_answer_labels() -> None:
	q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer=["A", "Z"])
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_answer_leak() -> None:
	q = _question(
		QuestionType.MSQ,
		question="Which is Alpha or Gamma?",
		options=["Alpha", "Beta", "Gamma", "Delta"],
		answer=["A", "C"],
	)
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.ANSWER_LEAK


def test_msq_rejects_question_exceeding_length() -> None:
	q = _question(QuestionType.MSQ, question="x" * 301, options=["A", "B", "C", "D"], answer=["A"])
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_option_exceeding_length() -> None:
	q = _question(QuestionType.MSQ, options=["A", "o" * 101, "C", "D"], answer=["A"])
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_empty_answer_list() -> None:
	q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer=[])
	result = MSQValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID


# NATValidator

def test_nat_valid_integer() -> None:
	q = _question(QuestionType.NAT, options=None, answer=42.0)
	assert NATValidator().validate(q).passed is True


def test_nat_valid_float() -> None:
	q = _question(QuestionType.NAT, options=None, answer=3.14)
	assert NATValidator().validate(q).passed is True


def test_nat_rejects_options_present() -> None:
	q = _question(QuestionType.NAT, options=["A", "B", "C", "D"], answer=5.0)
	result = NATValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.SCHEMA_INVALID


def test_nat_rejects_non_numeric_answer() -> None:
	q = _question(QuestionType.NAT, options=None, answer="not a number")
	result = NATValidator().validate(q)
	assert result.passed is False
	assert result.reason is FailureReason.UNPARSEABLE_NAT
