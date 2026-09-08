import json
from pathlib import Path

from src.domain.enums import QuestionType
from src.domain.models import SlotSpec, VerifiedQuestion


class VerifiedCache:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get(
        self,
        slot: SlotSpec,
        excluded_questions: set[str] | None = None,
    ) -> VerifiedQuestion | None:
        excluded = excluded_questions or set()
        for question in reversed(self._load()):
            if (
                question.slot.question_type is slot.question_type
                and question.slot.marks == slot.marks
                and question.question not in excluded
            ):
                return question
        return None

    def save(self, question: VerifiedQuestion) -> None:
        questions = [
            cached
            for cached in self._load()
            if not (
                cached.slot.subject == question.slot.subject
                and cached.slot.subtopic == question.slot.subtopic
                and cached.slot.question_type is question.slot.question_type
            )
        ]
        questions.append(question)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([item.model_dump(mode="json") for item in questions], indent=2),
            encoding="utf-8",
        )

    def _load(self) -> list[VerifiedQuestion]:
        if not self._path.exists() or not self._path.read_text(encoding="utf-8").strip():
            return []
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return [VerifiedQuestion.model_validate(item) for item in raw]