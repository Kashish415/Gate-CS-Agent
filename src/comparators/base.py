from typing import Protocol

class AnswerComparator(Protocol):
    def matches(
        self,
        generator_answer: str | list[str] | float,
        verifier_answer: str | list[str] | float,
    ) -> bool: ...
