from enum import Enum

class QuestionType(str, Enum):
    MCQ = "MCQ"
    MSQ = "MSQ"
    NAT = "NAT"

class FailureReason(str, Enum):
    SCHEMA_INVALID = "schema_invalid"
    OPTION_COUNT = "option_count"
    DUPLICATE_OPTION = "duplicate_option"
    ANSWER_LEAK = "answer_leak"
    UNPARSEABLE_NAT = "unparseable_nat"
    VERIFIER_DISAGREES = "verifier_disagrees"
    LOW_CONFIDENCE = "low_confidence"
    FLAGGED_AMBIGUOUS = "flagged_ambiguous"
    RETRY_EXHAUSTED = "retry_exhausted"
    PUBLISH_FAILED = "publish_failed"
