import json
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

from src.config import DB_PATH, SLOT_TEMPLATE
from src.domain.models import DailyLogRow

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS daily_log (
	date TEXT NOT NULL,
	slot_index INTEGER NOT NULL,
	subject TEXT NOT NULL,
	subtopic TEXT NOT NULL,
	type TEXT NOT NULL,
	marks INTEGER NOT NULL,
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


def init_db(db_path: Path = DB_PATH) -> None:
	with _connect(db_path) as connection:
		connection.execute(_CREATE_TABLE)
		columns = {
			row[1] for row in connection.execute("PRAGMA table_info(daily_log)")
		}
		if "question_text" not in columns:
			connection.execute("ALTER TABLE daily_log ADD COLUMN question_text TEXT")


def log_slot(row: DailyLogRow, db_path: Path = DB_PATH) -> None:
	values = (
		row.date.isoformat(),
		row.slot_index,
		row.subject,
		row.subtopic,
		row.question_type.value,
		row.marks,
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
	with _connect(db_path) as connection:
		connection.execute(
			"""
			INSERT OR IGNORE INTO daily_log (
				date, slot_index, subject, subtopic, type, marks, question_text,
				generator_answer, verifier_answer, agreement, confidence,
				retry_count, latency_ms, published, failure_reason
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
			values,
		)


def today_already_posted(day: date, db_path: Path = DB_PATH) -> bool:
	with _connect(db_path) as connection:
		row = connection.execute(
			"SELECT COUNT(*) FROM daily_log WHERE date = ?",
			(day.isoformat(),),
		).fetchone()
	return row is not None and row[0] == len(SLOT_TEMPLATE)


def recent_subtopic_usage(
	days: int = 7,
	db_path: Path = DB_PATH,
	today: date | None = None,
) -> set[tuple[str, str]]:
	end = today or date.today()
	start = end - timedelta(days=days - 1)
	with _connect(db_path) as connection:
		rows = connection.execute(
			"""
			SELECT subject, subtopic
			FROM daily_log
			WHERE date BETWEEN ? AND ?
			""",
			(start.isoformat(), end.isoformat()),
		).fetchall()
	return {(subject, subtopic) for subject, subtopic in rows}


def published_question_texts(db_path: Path = DB_PATH) -> set[str]:
	with _connect(db_path) as connection:
		rows = connection.execute(
			"SELECT question_text FROM daily_log WHERE published = 1 AND question_text IS NOT NULL"
		).fetchall()
	return {question_text for (question_text,) in rows}
