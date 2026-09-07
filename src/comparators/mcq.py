from src.comparators.base import AnswerComparator

class ExactMatchComparator:
    def matches(
        self,
        generator_answer: str | list[str] | float,
        verifier_answer: str | list[str] | float,
    ) -> bool:
        return (
            isinstance(generator_answer, str)
            and isinstance(verifier_answer, str)
            and generator_answer == verifier_answer
        )
