from src.pipeline import answers_match
from src.domain import QuestionType


# -- MCQ --

def test_mcq_exact_match():
    assert answers_match(QuestionType.MCQ, "A", "A") is True

def test_mcq_different_labels():
    assert answers_match(QuestionType.MCQ, "A", "B") is False

def test_mcq_type_mismatch_list():
    assert answers_match(QuestionType.MCQ, ["A"], "A") is False

def test_mcq_type_mismatch_float():
    assert answers_match(QuestionType.MCQ, 1.0, "A") is False


# -- MSQ --

def test_msq_same_set():
    assert answers_match(QuestionType.MSQ, ["A", "C"], ["C", "A"]) is True

def test_msq_different_sets():
    assert answers_match(QuestionType.MSQ, ["A", "B"], ["A", "C"]) is False

def test_msq_subset_is_not_match():
    assert answers_match(QuestionType.MSQ, ["A"], ["A", "B"]) is False

def test_msq_type_mismatch_string():
    assert answers_match(QuestionType.MSQ, "A", ["A"]) is False

def test_msq_type_mismatch_both_strings():
    assert answers_match(QuestionType.MSQ, "A", "A") is False


# -- NAT --

def test_nat_exact_match():
    assert answers_match(QuestionType.NAT, 3.14, 3.14) is True

def test_nat_within_absolute_tolerance():
    assert answers_match(QuestionType.NAT, 1.005, 1.01) is True

def test_nat_within_relative_tolerance():
    assert answers_match(QuestionType.NAT, 100.0, 100.5) is True

def test_nat_zero_values():
    assert answers_match(QuestionType.NAT, 0.0, 0.005) is True

def test_nat_negative_values():
    assert answers_match(QuestionType.NAT, -5.0, -5.005) is True

def test_nat_string_numbers():
    assert answers_match(QuestionType.NAT, "3.14", "3.14") is True

def test_nat_unparseable():
    assert answers_match(QuestionType.NAT, "abc", "3.14") is False

def test_nat_list_input():
    assert answers_match(QuestionType.NAT, [1, 2], 3.0) is False
