import logging
from typing import Protocol

from .config import Settings
from .domain import QuestionType

logger = logging.getLogger(__name__)


class Comparator(Protocol):
	def matches(
		self,
		generator_answer: str | list[str] | float,
		verifier_answer: str | list[str] | float,
	) -> bool: ...


class MCQComparator:
	def matches(
		self,
		generator_answer: str | list[str] | float,
		verifier_answer: str | list[str] | float,
	) -> bool:
		if not isinstance(generator_answer, str) or not isinstance(verifier_answer, str):
			logger.warning(
				"[MCQ Comparator] Type mismatch: generator=%s (%s), verifier=%s (%s)",
				generator_answer, type(generator_answer).__name__,
				verifier_answer, type(verifier_answer).__name__,
			)
			return False
		return generator_answer == verifier_answer


class MSQComparator:
	def matches(
		self,
		generator_answer: str | list[str] | float,
		verifier_answer: str | list[str] | float,
	) -> bool:
		if not isinstance(generator_answer, list) or not isinstance(verifier_answer, list):
			logger.warning(
				"[MSQ Comparator] Type mismatch: generator=%s (%s), verifier=%s (%s)",
				generator_answer, type(generator_answer).__name__,
				verifier_answer, type(verifier_answer).__name__,
			)
			return False
		return set(generator_answer) == set(verifier_answer)


class NATComparator:
	def __init__(
		self,
		absolute_tolerance: float = 0.01,
		relative_tolerance: float = 0.01,
	) -> None:
		self._abs_tol = absolute_tolerance
		self._rel_tol = relative_tolerance

	def matches(
		self,
		generator_answer: str | list[str] | float,
		verifier_answer: str | list[str] | float,
	) -> bool:
		try:
			gen_val = float(generator_answer)
			ver_val = float(verifier_answer)
		except (TypeError, ValueError):
			logger.warning(
				"[NAT Comparator] Unparseable: generator=%r, verifier=%r",
				generator_answer, verifier_answer,
			)
			return False
		diff = abs(gen_val - ver_val)
		if diff <= self._abs_tol:
			return True
		max_val = max(abs(gen_val), abs(ver_val))
		return max_val > 0 and (diff / max_val) <= self._rel_tol


def build_comparators(settings: Settings) -> dict[QuestionType, Comparator]:
	return {
		QuestionType.MCQ: MCQComparator(),
		QuestionType.MSQ: MSQComparator(),
		QuestionType.NAT: NATComparator(
			absolute_tolerance=settings.nat_absolute_tolerance,
			relative_tolerance=settings.nat_relative_tolerance,
		),
	}

