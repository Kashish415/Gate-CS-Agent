from src.comparators.base import AnswerComparator

class SetMatchComparator:
    def matches(
        self,
        generator_answer: str | list[str] | float,
        verifier_answer: str | list[str] | float,
    ) -> bool:
        if not isinstance(generator_answer, list) or not isinstance(verifier_answer, list):
            return False
        return set(generator_answer) == set(verifier_answer)
