from src.domain.enums import FailureReason
from src.domain.models import GeneratedQuestion, ValidationOutcome

class MCQValidator:
    def validate(self, q: GeneratedQuestion) -> ValidationOutcome:
        if q.options is None or len(q.options) != 4:
            return ValidationOutcome(
                passed=False,
                reason=FailureReason.OPTION_COUNT,
                detail=f"MCQ requires exactly 4 options, got {0 if q.options is None else len(q.options)}",
            )
        if len(set(q.options)) != 4:
            return ValidationOutcome(
                passed=False,
                reason=FailureReason.DUPLICATE_OPTION,
                detail="MCQ options contain duplicates",
            )
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
