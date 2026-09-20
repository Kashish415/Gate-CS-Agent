from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator


class QuestionType(StrEnum):
	MCQ = "MCQ"
	MSQ = "MSQ"
	NAT = "NAT"


class Difficulty(StrEnum):
	EASY = "EASY"
	MEDIUM = "MEDIUM"
	HARD = "HARD"


class FailureReason(StrEnum):
	SCHEMA_INVALID = "schema_invalid"
	OPTION_COUNT = "option_count"
	DUPLICATE_OPTION = "duplicate_option"
	UNPARSEABLE_NAT = "unparseable_nat"
	ANSWER_LEAK = "answer_leak"
	VERIFIER_DISAGREES = "verifier_disagrees"
	FLAGGED_AMBIGUOUS = "flagged_ambiguous"
	LOW_CONFIDENCE = "low_confidence"
	RETRY_EXHAUSTED = "retry_exhausted"
	PUBLISH_FAILED = "publish_failed"


OptionLabel = Literal["A", "B", "C", "D"]


class SlotSpec(BaseModel):
	slot_index: int
	subject: str
	subtopic: str
	question_type: QuestionType
	marks: Literal[1, 2]
	difficulty: Difficulty = Difficulty.MEDIUM


class QuestionPayload(BaseModel):
	question: str = Field(description="The question prompt text")
	explanation: str = Field(description="Brief explanation of why the answer is correct")


def _normalize_options(v: Any) -> Any:
	if isinstance(v, dict):
		return list(v.values())
	return v


class MCQPayload(QuestionPayload):
	options: list[str] = Field(description="List of exactly 4 option text strings corresponding to A, B, C, D")
	answer: OptionLabel = Field(description="Option letter string: A, B, C, or D")

	@field_validator("options", mode="before")
	@classmethod
	def validate_options(cls, v: Any) -> Any:
		return _normalize_options(v)


class MSQPayload(QuestionPayload):
	options: list[str] = Field(description="List of exactly 4 option text strings corresponding to A, B, C, D")
	answer: list[OptionLabel] = Field(description="List of correct option letters e.g. ['A', 'C']")

	@field_validator("options", mode="before")
	@classmethod
	def validate_options(cls, v: Any) -> Any:
		return _normalize_options(v)


class NATPayload(QuestionPayload):
	options: None = None
	answer: float = Field(description="Numeric answer value")


class VerifierPayload(BaseModel):
	reasoning: str = Field(default="", description="Brief step-by-step reasoning")
	confidence: int = Field(default=1, ge=1, le=5, description="Confidence score from 1 to 5")
	ambiguous: bool = Field(
		default=False,
		validation_alias=AliasChoices("ambiguous", "ambiguity"),
		description="True if question is ambiguous or flawed",
	)


class MCQVerifierPayload(VerifierPayload):
	answer: OptionLabel = Field(description="Option letter string: A, B, C, or D")


class MSQVerifierPayload(VerifierPayload):
	answer: list[OptionLabel] = Field(description="List of correct option letters e.g. ['A', 'C']")


class NATVerifierPayload(VerifierPayload):
	answer: float = Field(description="Numeric answer value")


class GeneratedQuestion(QuestionPayload):
	slot: SlotSpec
	options: list[str] | None = None
	answer: str | list[str] | float


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
	difficulty: Difficulty = Difficulty.MEDIUM
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
