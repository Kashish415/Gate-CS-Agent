from typing import Protocol

from .domain import FailureReason, GeneratedQuestion, QuestionType, ValidationOutcome

TELEGRAM_QUESTION_LIMIT = 300
TELEGRAM_OPTION_LIMIT = 100


class Validator(Protocol):
	def validate(self, question: GeneratedQuestion) -> ValidationOutcome: ...


def _validate_options(q: GeneratedQuestion) -> ValidationOutcome | None:
	if len(q.question) > TELEGRAM_QUESTION_LIMIT:
		return ValidationOutcome(
			passed=False,
			reason=FailureReason.SCHEMA_INVALID,
			detail=f"Question exceeds {TELEGRAM_QUESTION_LIMIT} chars ({len(q.question)})",
		)
	if q.options is None or len(q.options) != 4:
		return ValidationOutcome(
			passed=False,
			reason=FailureReason.OPTION_COUNT,
			detail=f"Requires exactly 4 options, got {0 if q.options is None else len(q.options)}",
		)
	if any(len(opt) > TELEGRAM_OPTION_LIMIT for opt in q.options):
		return ValidationOutcome(
			passed=False,
			reason=FailureReason.SCHEMA_INVALID,
			detail=f"Option exceeds {TELEGRAM_OPTION_LIMIT} chars",
		)
	if len(set(q.options)) != 4:
		return ValidationOutcome(
			passed=False,
			reason=FailureReason.DUPLICATE_OPTION,
			detail="Options contain duplicates",
		)
	return None


class MCQValidator:
	def validate(self, q: GeneratedQuestion) -> ValidationOutcome:
		shared = _validate_options(q)
		if shared is not None:
			return shared
		valid_labels = {chr(65 + i) for i in range(len(q.options))}
		if not isinstance(q.answer, str) or q.answer not in valid_labels:
			return ValidationOutcome(
				passed=False,
				reason=FailureReason.SCHEMA_INVALID,
				detail=f"MCQ answer must be one of {sorted(valid_labels)}",
			)
		answer_text = q.options[ord(q.answer) - 65]
		if answer_text in q.question:
			return ValidationOutcome(
				passed=False,
				reason=FailureReason.ANSWER_LEAK,
				detail="MCQ correct option text appears in question",
			)
		return ValidationOutcome(passed=True)


class MSQValidator:
	def validate(self, q: GeneratedQuestion) -> ValidationOutcome:
		shared = _validate_options(q)
		if shared is not None:
			return shared
		valid_labels = {chr(65 + i) for i in range(len(q.options))}
		if not isinstance(q.answer, list) or not (1 <= len(q.answer) <= 4):
			return ValidationOutcome(
				passed=False,
				reason=FailureReason.SCHEMA_INVALID,
				detail="MSQ answer must be a non-empty list of 1-4 option labels",
			)
		invalid = [a for a in q.answer if not isinstance(a, str) or a not in valid_labels]
		if invalid:
			return ValidationOutcome(
				passed=False,
				reason=FailureReason.SCHEMA_INVALID,
				detail=f"MSQ answer contains invalid labels: {invalid}",
			)
		leaked = [q.options[ord(a) - 65] for a in q.answer if q.options[ord(a) - 65] in q.question]
		if leaked:
			return ValidationOutcome(
				passed=False,
				reason=FailureReason.ANSWER_LEAK,
				detail=f"MSQ correct option text appears in question: {leaked}",
			)
		return ValidationOutcome(passed=True)


class NATValidator:
	def validate(self, q: GeneratedQuestion) -> ValidationOutcome:
		if q.options is not None:
			return ValidationOutcome(
				passed=False,
				reason=FailureReason.SCHEMA_INVALID,
				detail="NAT must not have options",
			)
		try:
			float(q.answer)
		except (TypeError, ValueError):
			return ValidationOutcome(
				passed=False,
				reason=FailureReason.UNPARSEABLE_NAT,
				detail=f"NAT answer is not parseable as a number: {q.answer!r}",
			)
		return ValidationOutcome(passed=True)


VALIDATORS: dict[QuestionType, Validator] = {
	QuestionType.MCQ: MCQValidator(),
	QuestionType.MSQ: MSQValidator(),
	QuestionType.NAT: NATValidator(),
}
