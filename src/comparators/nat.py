from src.comparators.base import AnswerComparator

class ToleranceComparator:
    def __init__(self, absolute: float, relative: float) -> None:
        self._absolute = absolute
        self._relative = relative

    def matches(
        self,
        generator_answer: str | list[str] | float,
        verifier_answer: str | list[str] | float,
    ) -> bool:
        try:
            g = float(generator_answer)
            v = float(verifier_answer)
        except (TypeError, ValueError):
            return False
        return abs(g - v) <= max(self._absolute, self._relative * abs(v))
