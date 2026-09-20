import tempfile
from datetime import date
from pathlib import Path

from src.domain import DailyLogRow, Difficulty, QuestionType, SlotSpec, VerifiedQuestion, VerifierResult
from src.observability import SQLiteDatabase
from src.publishing import _escape_markdown_v2, format_mcq, format_msq_answer


def test_sqlite_database_slot_logging() -> None:
	with tempfile.TemporaryDirectory() as tmp_dir:
		db_path = Path(tmp_dir) / "test.db"
		db = SQLiteDatabase(db_path)
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


def test_telegram_markdown_v2_escaping() -> None:
	escaped = _escape_markdown_v2("What is O(n*log(n))? Choice [A] standard.")
	assert r"O\(n\*log\(n\)\)" in escaped
	assert r"\[A\]" in escaped


def test_format_msq_answer_spoiler() -> None:
	slot = SlotSpec(
		slot_index=0,
		subject="Algorithms",
		subtopic="Sorting",
		question_type=QuestionType.MSQ,
		marks=2,
		difficulty=Difficulty.MEDIUM,
	)
	question = VerifiedQuestion(
		slot=slot,
		question="Which of the following sorting algorithms have worst-case space complexity of O(1)?",
		options=["Heap Sort", "Merge Sort", "Quick Sort", "Bubble Sort"],
		answer=["A", "D"],
		explanation="Heap sort and Bubble sort use O(1) auxiliary space.",
		verifier=VerifierResult(
			answer=["A", "D"],
			reasoning="Heap sort and bubble sort are in-place.",
			confidence=5,
			ambiguous=False,
		),
		published=False,
		cached_on=date.today(),
	)
	answer_text = format_msq_answer(question)
	assert "||" in answer_text
	assert "A, D" in answer_text


def test_format_mcq() -> None:
	slot = SlotSpec(
		slot_index=0,
		subject="Computer Networks",
		subtopic="DNS",
		question_type=QuestionType.MCQ,
		marks=1,
		difficulty=Difficulty.EASY,
	)
	question = VerifiedQuestion(
		slot=slot,
		question="Which protocol resolves domain names to IP addresses?",
		options=["HTTP", "DNS", "FTP", "SMTP"],
		answer="B",
		explanation="DNS resolves domain names.",
		verifier=VerifierResult(
			answer="B",
			reasoning="DNS is the standard for name resolution.",
			confidence=5,
			ambiguous=False,
		),
		published=False,
		cached_on=date.today(),
	)
	text, options, correct_id, explanation = format_mcq(question)
	assert text == "Which protocol resolves domain names to IP addresses?"
	assert options == ("HTTP", "DNS", "FTP", "SMTP")
	assert correct_id == 1
	assert len(explanation) <= 200
