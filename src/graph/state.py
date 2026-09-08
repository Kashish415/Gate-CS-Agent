from typing import TypedDict

from src.domain.models import (
	GeneratedQuestion,
	SlotSpec,
	ValidationOutcome,
	VerifierResult,
	VerifiedQuestion,
)


class PipelineState(TypedDict):
	slots: list[SlotSpec]
	generated: dict[int, GeneratedQuestion | None]
	validation: dict[int, ValidationOutcome]
	verifier: dict[int, VerifierResult | None]
	retries: dict[int, int]
	resolved: dict[int, VerifiedQuestion | None]
	publish_results: dict[int, bool]
