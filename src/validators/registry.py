from src.domain.enums import QuestionType
from src.validators.base import QuestionValidator
from src.validators.mcq import MCQValidator
from src.validators.msq import MSQValidator
from src.validators.nat import NATValidator

VALIDATORS: dict[QuestionType, QuestionValidator] = {
    QuestionType.MCQ: MCQValidator(),
    QuestionType.MSQ: MSQValidator(),
    QuestionType.NAT: NATValidator(),
}
