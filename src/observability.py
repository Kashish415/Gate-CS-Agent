import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

from .domain import DailyLogRow, SlotSpec, VerifiedQuestion

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS daily_log (
	date TEXT NOT NULL,
	slot_index INTEGER NOT NULL,
	subject TEXT NOT NULL,
	subtopic TEXT NOT NULL,
	type TEXT NOT NULL,
	marks INTEGER NOT NULL,
	difficulty TEXT,
	question_text TEXT,
	generator_answer TEXT,
	verifier_answer TEXT,
	agreement INTEGER NOT NULL,
	confidence INTEGER,
	retry_count INTEGER NOT NULL,
	latency_ms INTEGER NOT NULL,
	published INTEGER NOT NULL,
	failure_reason TEXT,
	PRIMARY KEY (date, slot_index)
)
"""


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
	db_path.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(db_path)
	try:
		with connection:
			yield connection
	finally:
		connection.close()


class SQLiteDatabase:

	def __init__(self, db_path: Path) -> None:
		self._db_path = db_path
		self._init_db()

	def _init_db(self) -> None:
		with _connect(self._db_path) as connection:
			connection.execute(_CREATE_TABLE)
			columns = {
				row[1] for row in connection.execute("PRAGMA table_info(daily_log)")
			}
			if "question_text" not in columns:
				connection.execute("ALTER TABLE daily_log ADD COLUMN question_text TEXT")
			if "difficulty" not in columns:
				connection.execute("ALTER TABLE daily_log ADD COLUMN difficulty TEXT")

	def log_slot(self, row: DailyLogRow) -> None:
		values = (
			row.date.isoformat(),
			row.slot_index,
			row.subject,
			row.subtopic,
			row.question_type.value,
			row.marks,
			row.difficulty.value if row.difficulty else None,
			row.question_text,
			json.dumps(row.generator_answer),
			json.dumps(row.verifier_answer),
			int(row.agreement),
			row.confidence,
			row.retry_count,
			row.latency_ms,
			int(row.published),
			row.failure_reason.value if row.failure_reason else None,
		)
		with _connect(self._db_path) as connection:
			connection.execute(
				"""
				INSERT OR IGNORE INTO daily_log (
					date, slot_index, subject, subtopic, type, marks, difficulty, question_text,
					generator_answer, verifier_answer, agreement, confidence,
					retry_count, latency_ms, published, failure_reason
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				values,
			)

	def published_question_texts(self) -> set[str]:
		with _connect(self._db_path) as connection:
			rows = connection.execute(
				"SELECT question_text FROM daily_log WHERE published = 1 AND question_text IS NOT NULL"
			).fetchall()
		return {question_text for (question_text,) in rows}

	def today_already_posted(self, day: date, expected_slot_count: int) -> bool:
		with _connect(self._db_path) as connection:
			row = connection.execute(
				"SELECT COUNT(*) FROM daily_log WHERE date = ?",
				(day.isoformat(),),
			).fetchone()
		return row is not None and row[0] == expected_slot_count

	def recent_subtopic_usage(
		self,
		days: int = 7,
		today: date | None = None,
	) -> set[tuple[str, str]]:
		end = today or date.today()
		start = end - timedelta(days=days - 1)
		with _connect(self._db_path) as connection:
			rows = connection.execute(
				"""
				SELECT subject, subtopic
				FROM daily_log
				WHERE date BETWEEN ? AND ?
				""",
				(start.isoformat(), end.isoformat()),
			).fetchall()
		return {(subject, subtopic) for subject, subtopic in rows}


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
		if not self._path.exists():
			return []
		text = self._path.read_text(encoding="utf-8")
		if not text.strip():
			return []
		try:
			return [VerifiedQuestion.model_validate(item) for item in json.loads(text)]
		except json.JSONDecodeError:
			logger.warning("verified_cache.json is corrupted — treating as empty")
			return []
