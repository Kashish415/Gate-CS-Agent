from src.domain.enums import FailureReason
from src.domain.models import GeneratedQuestion, ValidationOutcome

class MSQValidator:
    def validate(self, q: GeneratedQuestion) -> ValidationOutcome:
        if q.options is None or len(q.options) != 4:
            return ValidationOutcome(
                passed=False,
                reason=FailureReason.OPTION_COUNT,
                detail=f"MSQ requires exactly 4 options, got {0 if q.options is None else len(q.options)}",
            )
        if len(set(q.options)) != 4:
            return ValidationOutcome(
                passed=False,
                reason=FailureReason.DUPLICATE_OPTION,
                detail="MSQ options contain duplicates",
            )
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
