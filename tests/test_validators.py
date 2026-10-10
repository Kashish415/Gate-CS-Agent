from src.domain import Difficulty, QuestionType, QuestionPayload, SlotSpec
from src.pipeline import validate


def _slot(qtype):
    return SlotSpec(slot_index=0, subject="Algorithms", subtopic="Sorting",
                    question_type=qtype, marks=1, difficulty=Difficulty.EASY)


def _question(qtype, question="Short question?", options=None, answer="A"):
    slot = _slot(qtype)
    return QuestionPayload(slot=slot, question=question,
                           options=options, answer=answer, explanation="Brief explanation.")


# -- MCQ --

def test_mcq_valid():
    q = _question(QuestionType.MCQ, options=["Alpha", "Beta", "Gamma", "Delta"], answer="B")
    assert validate(q) is None


def test_mcq_rejects_wrong_option_count():
    q = _question(QuestionType.MCQ, options=["A", "B", "C"], answer="A")
    assert validate(q) == "option_count"


def test_mcq_rejects_none_options():
    q = _question(QuestionType.MCQ, options=None, answer="A")
    assert validate(q) == "option_count"


def test_mcq_rejects_duplicate_options():
    q = _question(QuestionType.MCQ, options=["Same", "Same", "Gamma", "Delta"], answer="C")
    assert validate(q) == "duplicate_option"


def test_mcq_rejects_answer_leak():
    q = _question(QuestionType.MCQ, question="What is Alpha?",
                  options=["Alpha", "Beta", "Gamma", "Delta"], answer="A")
    assert validate(q) == "answer_leak"


def test_mcq_rejects_invalid_answer_label():
    q = _question(QuestionType.MCQ, options=["A", "B", "C", "D"], answer="E")
    assert validate(q) == "invalid_answer"


# -- MSQ --

def test_msq_valid():
    q = _question(QuestionType.MSQ, options=["Alpha", "Beta", "Gamma", "Delta"], answer=["A", "C"])
    assert validate(q) is None


def test_msq_rejects_wrong_option_count():
    q = _question(QuestionType.MSQ, options=["A", "B"], answer=["A"])
    assert validate(q) == "option_count"


def test_msq_rejects_duplicate_options():
    q = _question(QuestionType.MSQ, options=["X", "X", "Y", "Z"], answer=["A"])
    assert validate(q) == "duplicate_option"


def test_msq_rejects_non_list_answer():
    q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer="A")
    assert validate(q) == "invalid_answer"


def test_msq_rejects_invalid_answer_labels():
    q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer=["A", "Z"])
    assert validate(q) == "invalid_answer"


def test_msq_rejects_answer_leak():
    q = _question(QuestionType.MSQ, question="Which is Alpha or Gamma?",
                  options=["Alpha", "Beta", "Gamma", "Delta"], answer=["A", "C"])
    assert validate(q) == "answer_leak"


def test_msq_rejects_empty_answer_list():
    q = _question(QuestionType.MSQ, options=["A", "B", "C", "D"], answer=[])
    assert validate(q) == "invalid_answer"


# -- NAT --

def test_nat_valid_integer():
    q = _question(QuestionType.NAT, options=None, answer=42.0)
    assert validate(q) is None


def test_nat_valid_float():
    q = _question(QuestionType.NAT, options=None, answer=3.14)
    assert validate(q) is None


def test_nat_rejects_options_present():
    q = _question(QuestionType.NAT, options=["A", "B", "C", "D"], answer=5.0)
    assert validate(q) == "nat_has_options"


def test_nat_rejects_non_numeric_answer():
    q = _question(QuestionType.NAT, options=None, answer="not a number")
    assert validate(q) == "unparseable_nat"
