from datetime import date
from typing import Literal
from pydantic import BaseModel, Field
from __future__ import annotations
from src.domain.enums import QuestionType
from src.domain.enums import FailureReason

class SlotSpec(BaseModel):
    slot_index: int
    subject: str
    subtopic: str
    question_type: QuestionType
    marks: Literal[1, 2]

class GeneratedQuestion(BaseModel):
    slot: SlotSpec
    question: str
    options: list[str] | None = None
    answer: str | list[str] | float
    explanation: str

class VerifierResult(BaseModel):
    answer: str | list[str] | float
    reasoning: str
    confidence: int = Field(ge=1, le=5)
    ambiguous: bool

class ValidationOutcome(BaseModel):
    passed: bool
    reason: "FailureReason | None" = None
    detail: str | None = None

class VerifiedQuestion(BaseModel):
    slot: SlotSpec
    question: str
    options: list[str] | None = None
    answer: str | list[str] | float
    explanation: str
    verifier: VerifierResult
    published: bool
    cached_on: date

ValidationOutcome.model_rebuild()
