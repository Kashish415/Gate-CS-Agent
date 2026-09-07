from src.comparators.base import AnswerComparator
from src.comparators.mcq import ExactMatchComparator
from src.comparators.msq import SetMatchComparator
from src.comparators.nat import ToleranceComparator
from src.config import Settings
from src.domain.enums import QuestionType

def build_comparators(settings: Settings) -> dict[QuestionType, AnswerComparator]:
    return {
        QuestionType.MCQ: ExactMatchComparator(),
        QuestionType.MSQ: SetMatchComparator(),
        QuestionType.NAT: ToleranceComparator(
            absolute=settings.nat_absolute_tolerance,
            relative=settings.nat_relative_tolerance,
        ),
    }
