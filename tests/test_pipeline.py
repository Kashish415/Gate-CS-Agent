import tempfile
from datetime import date
from pathlib import Path

from src.domain import Difficulty, QuestionType, SlotSpec
from src.storage import Database
from src.telegram import _escape_v2


def test_sqlite_database_slot_logging():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        db = Database(db_path)
        slot = SlotSpec(
            slot_index=0, subject="Engineering Mathematics",
            subtopic="Discrete Mathematics", question_type=QuestionType.MCQ,
            marks=1, difficulty=Difficulty.EASY,
        )
        state = {"retries": {0: 0}}
        db.log_slot(slot, state, published=True, latency=100)
        assert db.today_already_posted(date.today(), expected_slot_count=1) is True
        assert db.today_already_posted(date(2020, 1, 1), expected_slot_count=1) is False


def test_telegram_markdown_v2_escaping():
    escaped = _escape_v2("What is O(n*log(n))? Choice [A] standard.")
    assert r"O\(n\*log\(n\)\)" in escaped
    assert r"\[A\]" in escaped
