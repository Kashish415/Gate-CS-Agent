from src.domain import Difficulty, FailureReason, GeneratedQuestion, QuestionType, SlotSpec
from src.pipeline import _validate


def _slot(qtype):
    return SlotSpec(slot_index=0, subject="Algorithms", subtopic="Sorting",
                    question_type=qtype, marks=1, difficulty=Difficulty.EASY)


def _question(qtype, question="Short question?", options=None, answer="A"):
    return GeneratedQuestion(slot=_slot(qtype), question=question,
                             options=options, answer=answer, explanation="Brief explanation.")


# -- MCQ --

def test_mcq_valid():
    q = _question(QuestionType.MCQ, options=["Alpha", "Beta", "Gamma", "Delta"], answer="B")
    assert _validate(q).passed is True


def test_mcq_rejects_wrong_option_count():
    q = _question(QuestionType.MCQ, options=["A", "B", "C"], answer="A")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.OPTION_COUNT


def test_mcq_rejects_none_options():
    q = _question(QuestionType.MCQ, options=None, answer="A")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.OPTION_COUNT


def test_mcq_rejects_duplicate_options():
    q = _question(QuestionType.MCQ, options=["Same", "Same", "Gamma", "Delta"], answer="C")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.DUPLICATE_OPTION


def test_mcq_rejects_answer_leak():
    q = _question(QuestionType.MCQ, question="What is Alpha?",
                  options=["Alpha", "Beta", "Gamma", "Delta"], answer="A")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.ANSWER_LEAK


def test_mcq_rejects_invalid_answer_label():
    q = _question(QuestionType.MCQ, options=["A", "B", "C", "D"], answer="E")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_mcq_rejects_question_exceeding_length():
    q = _question(QuestionType.MCQ, question="x" * 301, options=["A", "B", "C", "D"], answer="A")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_mcq_rejects_option_exceeding_length():
    q = _question(QuestionType.MCQ, options=["A", "B", "o" * 101, "D"], answer="A")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_mcq_accepts_exactly_at_length_limits():
    q = _question(QuestionType.MCQ, question="q" * 300,
                  options=["o" * 100, "Beta", "Gamma", "Delta"], answer="B")
    assert _validate(q).passed is True


# -- MSQ --

def test_msq_valid():
    q = _question(QuestionType.MSQ, options=["Alpha", "Beta", "Gamma", "Delta"], answer=["A", "C"])
    assert _validate(q).passed is True


def test_msq_rejects_wrong_option_count():
    q = _question(QuestionType.MSQ, options=["A", "B"], answer=["A"])
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.OPTION_COUNT


def test_msq_rejects_duplicate_options():
    q = _question(QuestionType.MSQ, options=["X", "X", "Y", "Z"], answer=["A"])
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.DUPLICATE_OPTION


def test_msq_rejects_non_list_answer():
    q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer="A")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_invalid_answer_labels():
    q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer=["A", "Z"])
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_answer_leak():
    q = _question(QuestionType.MSQ, question="Which is Alpha or Gamma?",
                  options=["Alpha", "Beta", "Gamma", "Delta"], answer=["A", "C"])
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.ANSWER_LEAK


def test_msq_rejects_question_exceeding_length():
    q = _question(QuestionType.MSQ, question="x" * 301, options=["A", "B", "C", "D"], answer=["A"])
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_option_exceeding_length():
    q = _question(QuestionType.MSQ, options=["A", "o" * 101, "C", "D"], answer=["A"])
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_msq_rejects_empty_answer_list():
    q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer=[])
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


# -- NAT --

def test_nat_valid_integer():
    q = _question(QuestionType.NAT, options=None, answer=42.0)
    assert _validate(q).passed is True


def test_nat_valid_float():
    q = _question(QuestionType.NAT, options=None, answer=3.14)
    assert _validate(q).passed is True


def test_nat_rejects_options_present():
    q = _question(QuestionType.NAT, options=["A", "B", "C", "D"], answer=5.0)
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.SCHEMA_INVALID


def test_nat_rejects_non_numeric_answer():
    q = _question(QuestionType.NAT, options=None, answer="not a number")
    result = _validate(q)
    assert result.passed is False
    assert result.reason is FailureReason.UNPARSEABLE_NAT
