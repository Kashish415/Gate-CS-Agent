import logging

from src.comparators import MCQComparator, MSQComparator, NATComparator


# MCQComparator

def test_mcq_exact_match() -> None:
	assert MCQComparator().matches("A", "A") is True


def test_mcq_different_labels() -> None:
	assert MCQComparator().matches("A", "B") is False


def test_mcq_type_mismatch_list(caplog: logging.LogRecord) -> None:
	with caplog.at_level(logging.WARNING):
		result = MCQComparator().matches(["A"], "A")
	assert result is False
	assert "Type mismatch" in caplog.text


def test_mcq_type_mismatch_float(caplog: logging.LogRecord) -> None:
	with caplog.at_level(logging.WARNING):
		result = MCQComparator().matches(1.0, "A")
	assert result is False
	assert "Type mismatch" in caplog.text


# MSQComparator

def test_msq_same_set() -> None:
	assert MSQComparator().matches(["A", "C"], ["C", "A"]) is True


def test_msq_different_sets() -> None:
	assert MSQComparator().matches(["A", "B"], ["A", "C"]) is False


def test_msq_subset_is_not_match() -> None:
	assert MSQComparator().matches(["A"], ["A", "B"]) is False


def test_msq_type_mismatch_string(caplog: logging.LogRecord) -> None:
	with caplog.at_level(logging.WARNING):
		result = MSQComparator().matches("A", ["A"])
	assert result is False
	assert "Type mismatch" in caplog.text


def test_msq_type_mismatch_both_strings(caplog: logging.LogRecord) -> None:
	with caplog.at_level(logging.WARNING):
		result = MSQComparator().matches("A", "A")
	assert result is False
	assert "Type mismatch" in caplog.text


# NATComparator

def test_nat_exact_match() -> None:
	assert NATComparator().matches(42.0, 42.0) is True


def test_nat_within_absolute_tolerance() -> None:
	comparator = NATComparator(absolute_tolerance=0.01, relative_tolerance=0.0)
	assert comparator.matches(1.0, 1.005) is True
	assert comparator.matches(1.0, 1.02) is False


def test_nat_within_relative_tolerance() -> None:
	comparator = NATComparator(absolute_tolerance=0.0, relative_tolerance=0.01)
	assert comparator.matches(100.0, 100.5) is True
	assert comparator.matches(100.0, 102.0) is False


def test_nat_zero_values() -> None:
	comparator = NATComparator(absolute_tolerance=0.01, relative_tolerance=0.01)
	assert comparator.matches(0.0, 0.0) is True
	assert comparator.matches(0.0, 0.005) is True


def test_nat_negative_values() -> None:
	assert NATComparator().matches(-5.0, -5.0) is True


def test_nat_string_numbers() -> None:
	assert NATComparator().matches("42", "42.0") is True


def test_nat_type_mismatch_unparseable(caplog: logging.LogRecord) -> None:
	with caplog.at_level(logging.WARNING):
		result = NATComparator().matches("abc", 42.0)
	assert result is False
	assert "Unparseable" in caplog.text


def test_nat_type_mismatch_list(caplog: logging.LogRecord) -> None:
	with caplog.at_level(logging.WARNING):
		result = NATComparator().matches(["A"], 42.0)
	assert result is False
	assert "Unparseable" in caplog.text
