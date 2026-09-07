from typing import Protocol

from src.domain.models import GeneratedQuestion, ValidationOutcome

class QuestionValidator(Protocol):
    def validate(self, q: GeneratedQuestion) -> ValidationOutcome: ...
