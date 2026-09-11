"""
tests/test_intervention.py
Unit tests for core/intervention.py — rate limiter and logger.
TTS is tested in silent mode (no audio hardware needed).
SQLite logger is tested against an in-memory database.
"""

import sys
import os
import sqlite3
import tempfile
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.intervention import TTSEngine, SessionLogger, LogRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_IV_CFG = {
    "intervention": {
        "min_interval_sec": 1.0,  # short for testing
        "tts_rate":   175,
        "tts_volume": 0.9,
        "messages": {
            "tier1": ["Soft message."],
            "tier2": ["Firm message."],
            "tier3": ["Urgent message."],
        },
    },
    "logging": {
        "db_path": "",  # will be set per test
        "log_interval_sec": 0.0,  # log every call in tests
    },
}


def make_log_row(**kwargs) -> LogRow:
    defaults = dict(
        session_id="test-session",
        timestamp=time.time(),
        state="READING",
        attention_raw=0.8,
        attention_filtered=0.8,
        gaze_score=0.9,
        head_score=0.8,
        context_score=0.7,
        blink_score=1.0,
        yaw=0.0, pitch=5.0, roll=0.0,
        gaze_zone="book_zone",
        intervention_tier=0,
        intervention_fired=False,
    )
    defaults.update(kwargs)
    return LogRow(**defaults)


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

class TestTTSRateLimiter:
    """
    Tests the rate-limiting logic without actually speaking.
    We patch _available = True and _last_spoken manually.
    """

    def _make_engine(self):
        cfg = {
            "intervention": {
                "min_interval_sec": 1.0,
                "tts_rate": 175,
                "tts_volume": 0.9,
                "messages": {
                    "tier1": ["Test message."],
                    "tier2": ["Test message 2."],
                    "tier3": ["Test message 3."],
                },
            }
        }
        engine = TTSEngine.__new__(TTSEngine)
        import queue
        engine._rate         = 175
        engine._volume       = 0.9
        engine._messages     = cfg["intervention"]["messages"]
        engine._min_interval = 1.0
        engine._queue        = queue.Queue()
        engine._available    = True  # bypass TTS init
        engine._last_spoken  = 0.0
        engine._stop_event   = threading.Event()
        return engine

    def test_tier0_never_speaks(self):
        engine = self._make_engine()
        assert engine.speak(0) is False

    def test_tier1_queues_message(self):
        engine = self._make_engine()
        result = engine.speak(1)
        assert result is True
        assert not engine._queue.empty()

    def test_rate_limited_second_call(self):
        engine = self._make_engine()
        engine.speak(1)
        # Immediately call again — should be rate-limited
        result = engine.speak(1)
        assert result is False

    def test_allowed_after_interval(self):
        engine = self._make_engine()
        engine._last_spoken = time.monotonic() - 2.0  # 2s ago, interval is 1s
        result = engine.speak(1)
        assert result is True

    def test_unavailable_engine_never_speaks(self):
        engine = self._make_engine()
        engine._available = False
        assert engine.speak(3) is False


# ---------------------------------------------------------------------------
# SQLite logger tests
# ---------------------------------------------------------------------------

class TestSessionLogger:
    def _make_logger(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        cfg = {
            "logging": {
                "db_path": db_path,
                "log_interval_sec": 0.0,
            }
        }
        return SessionLogger(cfg), db_path

    def test_creates_table(self, tmp_path):
        logger, db_path = self._make_logger(tmp_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        logger.close()
        assert ("session_log",) in tables

    def test_logs_row(self, tmp_path):
        logger, db_path = self._make_logger(tmp_path)
        row = make_log_row()
        logged = logger.maybe_log(row)
        logger.close()
        assert logged is True

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM session_log").fetchone()[0]
        conn.close()
        assert count == 1

    def test_row_fields_are_correct(self, tmp_path):
        logger, db_path = self._make_logger(tmp_path)
        row = make_log_row(state="DISTRACTED", attention_filtered=0.3, yaw=30.0)
        logger.maybe_log(row)
        logger.close()

        conn = sqlite3.connect(db_path)
        result = conn.execute(
            "SELECT state, attention_filtered, yaw FROM session_log LIMIT 1"
        ).fetchone()
        conn.close()
        assert result[0] == "DISTRACTED"
        assert abs(result[1] - 0.3) < 0.001
        assert abs(result[2] - 30.0) < 0.001

    def test_rate_gate_suppresses_rapid_calls(self, tmp_path):
        db_path = str(tmp_path / "test2.db")
        cfg = {
            "logging": {
                "db_path": db_path,
                "log_interval_sec": 10.0,  # high interval → only first call logs
            }
        }
        logger = SessionLogger(cfg)
        row = make_log_row()
        first  = logger.maybe_log(row)  # should log
        second = logger.maybe_log(row)  # should be suppressed
        logger.close()
        assert first  is True
        assert second is False

    def test_intervention_fired_flag(self, tmp_path):
        logger, db_path = self._make_logger(tmp_path)
        logger.maybe_log(make_log_row(intervention_fired=True))
        logger.close()

        conn = sqlite3.connect(db_path)
        val = conn.execute(
            "SELECT intervention_fired FROM session_log LIMIT 1"
        ).fetchone()[0]
        conn.close()
        assert val == 1

    def test_multiple_sessions(self, tmp_path):
        logger, db_path = self._make_logger(tmp_path)
        logger.maybe_log(make_log_row(session_id="session-A"))
        time.sleep(0.01)
        logger.maybe_log(make_log_row(session_id="session-B"))
        logger.close()

        conn = sqlite3.connect(db_path)
        sessions = {r[0] for r in conn.execute(
            "SELECT DISTINCT session_id FROM session_log"
        ).fetchall()}
        conn.close()
        assert "session-A" in sessions
        assert "session-B" in sessions
