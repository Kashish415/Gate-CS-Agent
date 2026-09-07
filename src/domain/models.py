from datetime import date
from typing import Literal
from pydantic import BaseModel, Field
from src.domain.enums import FailureReason, QuestionType

class SlotSpec(BaseModel):
    slot_index: int
    subject: str
    subtopic: str
    question_type: QuestionType
    marks: Literal[1, 2]

class GeneratedQuestionPayload(BaseModel):
    question: str
    options: list[str] | None = None
    answer: str | list[str] | float
    explanation: str

class GeneratedQuestion(GeneratedQuestionPayload):
    slot: SlotSpec

class VerifierResult(BaseModel):
    answer: str | list[str] | float
    reasoning: str
    confidence: int = Field(ge=1, le=5)
    ambiguous: bool

class ValidationOutcome(BaseModel):
    passed: bool
    reason: FailureReason | None = None
    detail: str | None = None

class VerifiedQuestion(GeneratedQuestion):
    verifier: VerifierResult
    published: bool
    cached_on: date
