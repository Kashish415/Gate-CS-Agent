import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta

from .domain import VerifiedQuestion

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
def _connect(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


class Database:
    def __init__(self, db_path):
        self._path = db_path
        with _connect(self._path) as conn:
            conn.execute(_CREATE_TABLE)

    def log_slot(self, row):
        with _connect(self._path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO daily_log
                (date, slot_index, subject, subtopic, type, marks, difficulty, question_text,
                generator_answer, verifier_answer, agreement, confidence,
                retry_count, latency_ms, published, failure_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row.date.isoformat(), row.slot_index, row.subject, row.subtopic,
                    row.question_type.value, row.marks,
                    row.difficulty.value if row.difficulty else None,
                    row.question_text,
                    json.dumps(row.generator_answer), json.dumps(row.verifier_answer),
                    int(row.agreement), row.confidence, row.retry_count, row.latency_ms,
                    int(row.published),
                    row.failure_reason.value if row.failure_reason else None,
                ),
            )

    def published_question_texts(self):
        with _connect(self._path) as conn:
            rows = conn.execute(
                "SELECT question_text FROM daily_log WHERE published = 1 AND question_text IS NOT NULL"
            ).fetchall()
        return {text for (text,) in rows}

    def today_already_posted(self, day, expected_slot_count):
        with _connect(self._path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM daily_log WHERE date = ?", (day.isoformat(),)
            ).fetchone()
        return row is not None and row[0] == expected_slot_count

    def recent_subtopic_usage(self, days=7, today=None):
        end = today or date.today()
        start = end - timedelta(days=days - 1)
        with _connect(self._path) as conn:
            rows = conn.execute(
                "SELECT subject, subtopic FROM daily_log WHERE date BETWEEN ? AND ?",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return {(subject, subtopic) for subject, subtopic in rows}


class Cache:
    def __init__(self, path):
        self._path = path

    def get(self, slot, excluded_questions=None):
        excluded = excluded_questions or set()
        for question in reversed(self._load()):
            if (question.slot.question_type == slot.question_type
                    and question.slot.marks == slot.marks
                    and question.question not in excluded):
                return question
        return None

    def save(self, question):
        questions = [
            cached for cached in self._load()
            if not (cached.slot.subject == question.slot.subject
                    and cached.slot.subtopic == question.slot.subtopic
                    and cached.slot.question_type == question.slot.question_type)
        ]
        questions.append(question)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([q.model_dump(mode="json") for q in questions], indent=2),
            encoding="utf-8",
        )

    def _load(self):
        if not self._path.exists():
            return []
        text = self._path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        try:
            return [VerifiedQuestion.model_validate(item) for item in json.loads(text)]
        except json.JSONDecodeError:
            logger.warning("verified_cache.json corrupted, treating as empty")
            return []
