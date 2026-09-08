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

class DailyLogRow(BaseModel):
    date: date
    slot_index: int
    subject: str
    subtopic: str
    question_type: QuestionType
    marks: Literal[1, 2]
    question_text: str | None = None
    generator_answer: str | list[str] | float | None = None
    verifier_answer: str | list[str] | float | None = None
    agreement: bool = False
    confidence: int | None = None
    retry_count: int
    latency_ms: int
    published: bool
    failure_reason: FailureReason | None = None

class VerifiedQuestion(GeneratedQuestion):
    verifier: VerifierResult
    published: bool
    cached_on: date
