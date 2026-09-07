from src.domain.enums import QuestionType
from src.domain.models import VerifiedQuestion

_MARKDOWN_V2_SPECIAL = frozenset("_*[]()~`>#+-=|{}.!\\")


def _escape_markdown_v2(text: str) -> str:
	return "".join(
		f"\\{character}" if character in _MARKDOWN_V2_SPECIAL else character
		for character in text
	)


def format_mcq(q: VerifiedQuestion) -> tuple[str, tuple[str, ...], int, str]:
	if q.slot.question_type is not QuestionType.MCQ or q.options is None:
		raise ValueError("format_mcq requires an MCQ with options")
	if not isinstance(q.answer, str) or not q.answer.isalpha():
		raise ValueError("MCQ answer must be an option label")
	correct_option_id = ord(q.answer.upper()) - ord("A")
	return q.question, tuple(q.options), correct_option_id, q.explanation[:200]


def format_msq_or_nat(q: VerifiedQuestion) -> list[str]:
	if q.slot.question_type is QuestionType.MSQ:
		if q.options is None or not isinstance(q.answer, list):
			raise ValueError("MSQ answer must be a list of option labels")
		options = "\n".join(
			f"{chr(65 + index)}. {option}"
			for index, option in enumerate(q.options)
		)
		question = f"{q.question}\n{options}"
		answer = ", ".join(q.answer)
	elif q.slot.question_type is QuestionType.NAT:
		question = q.question
		answer = str(q.answer)
	else:
		raise ValueError("format_msq_or_nat requires an MSQ or NAT")
	return [_escape_markdown_v2(question), f"||{_escape_markdown_v2(answer)}||"]
