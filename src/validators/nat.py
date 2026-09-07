from src.domain.enums import FailureReason
from src.domain.models import GeneratedQuestion, ValidationOutcome

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
