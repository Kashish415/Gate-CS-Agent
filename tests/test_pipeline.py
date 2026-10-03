import tempfile
from datetime import date
from pathlib import Path

from src.domain import DailyLogRow, Difficulty, QuestionType, SlotSpec, VerifiedQuestion, VerifierResult
from src.storage import Database
from src.telegram import _escape_v2


def test_sqlite_database_slot_logging():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        db = Database(db_path)
        for i in range(3):
            row = DailyLogRow(
                date=date(2026, 9, 9),
                slot_index=i,
                subject="Engineering Mathematics",
                subtopic="Discrete Mathematics",
                question_type=QuestionType.MCQ,
                marks=1,
                difficulty=Difficulty.EASY,
                question_text=f"Sample question {i}",
                generator_answer="A",
                verifier_answer="A",
                agreement=True,
                confidence=5,
                retry_count=0,
                latency_ms=100,
                published=True,
                failure_reason=None,
            )
            db.log_slot(row)
        assert db.today_already_posted(date(2026, 9, 9), expected_slot_count=3) is True
        assert db.today_already_posted(date(2026, 9, 10), expected_slot_count=3) is False


def test_telegram_markdown_v2_escaping():
    escaped = _escape_v2("What is O(n*log(n))? Choice [A] standard.")
    assert r"O\(n\*log\(n\)\)" in escaped
    assert r"\[A\]" in escaped
