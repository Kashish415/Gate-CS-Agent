from datetime import date
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field


class QuestionType(StrEnum):
    MCQ = "MCQ"
    MSQ = "MSQ"
    NAT = "NAT"


class Difficulty(StrEnum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class SlotSpec(BaseModel):
    slot_index: int
    subject: str
    subtopic: str
    question_type: QuestionType
    marks: Literal[1, 2]
    difficulty: Difficulty = Difficulty.MEDIUM


class QuestionPayload(BaseModel):
    question: str = Field(description="The question prompt text")
    options: list[str] | None = Field(default=None, description="List of 4 options for MCQ/MSQ, None for NAT")
    answer: str | list[str] | float = Field(description="Answer: letter for MCQ, list of letters for MSQ, number for NAT")
    explanation: str = Field(description="Brief explanation of why the answer is correct")
    slot: SlotSpec | None = Field(default=None, exclude=True)


class VerifierPayload(BaseModel):
    answer: str | list[str] | float = Field(description="Independent answer")
    reasoning: str = Field(default="", description="Brief step-by-step reasoning")
    confidence: int = Field(default=1, ge=1, le=5, description="Confidence 1-5")
    ambiguous: bool = Field(default=False, description="True if question is ambiguous or flawed")


class VerifiedQuestion(BaseModel):
    slot: SlotSpec
    question: str
    options: list[str] | None = None
    answer: str | list[str] | float
    explanation: str
    verifier: VerifierPayload
    published: bool = False
    cached_on: date = Field(default_factory=date.today)
