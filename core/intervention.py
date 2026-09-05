"""
core/intervention.py
TTS intervention engine + SQLite session logger.

TTS:
  - Runs pyttsx3 speech in a dedicated daemon thread to avoid blocking the
    main video loop. Uses a queue to pass speech commands.
  - Rate-limited: enforces a minimum interval between any two TTS calls.
  - Silent-mode fallback if pyttsx3 initialisation fails (e.g. no TTS backend).

SQLite Logger:
  - Writes one row per sample (every log_interval_sec) to data/sessions.db.
  - Schema: session_id, timestamp, state, attention_raw, attention_filtered,
            gaze_score, head_score, context_score, blink_score,
            yaw, pitch, roll, gaze_zone, intervention_tier, intervention_fired.
  - Connection is opened once per session and closed on shutdown.
"""

from __future__ import annotations

import logging
import os
import queue
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTS Engine (daemon thread)
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Thread-safe TTS wrapper.
    pyttsx3.runAndWait() blocks, so we push strings onto a queue
    and let a daemon thread consume them.
    """

    def __init__(self, cfg: dict):
        iv = cfg["intervention"]
        self._rate   = iv["tts_rate"]
        self._volume = iv["tts_volume"]
        self._messages: dict[str, list[str]] = iv["messages"]
        self._min_interval = iv["min_interval_sec"]

        self._queue: queue.Queue[str] = queue.Queue()
        self._available = False
        self._last_spoken: float = 0.0
        self._stop_event = threading.Event()

        # Initialise pyttsx3 in background thread context
        self._thread = threading.Thread(
            target=self._run, name="tts-engine", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        """Daemon thread: initialise engine, then consume messages."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            self._available = True
            logger.info("TTS engine initialised (SAPI5).")
        except Exception as exc:
            logger.warning("TTS unavailable: %s. Running in silent mode.", exc)
            # Drain queue silently so the thread doesn't hang
            while not self._stop_event.is_set():
                try:
                    self._queue.get(timeout=1.0)
                except queue.Empty:
                    pass
            return

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                logger.warning("TTS speak error: %s", exc)

    def speak(self, tier: int) -> bool:
        """
        Queue a TTS message for the given tier (1–3).
        Returns True if the message was queued, False if rate-limited or silent.
        Tier 0 = silent.
        """
        if tier <= 0 or not self._available:
            return False

        now = time.monotonic()
        if now - self._last_spoken < self._min_interval:
            return False  # Rate-limited

        key = f"tier{tier}"
        candidates = self._messages.get(key, [])
        if not candidates:
            return False

        text = random.choice(candidates)
        try:
            self._queue.put_nowait(text)
            self._last_spoken = now
            logger.info("TTS queued [tier%d]: %s", tier, text)
            return True
        except queue.Full:
            return False  # Already speaking

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# SQLite Logger
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    timestamp           REAL    NOT NULL,
    state               TEXT    NOT NULL,
    attention_raw       REAL,
    attention_filtered  REAL,
    gaze_score          REAL,
    head_score          REAL,
    context_score       REAL,
    blink_score         REAL,
    yaw                 REAL,
    pitch               REAL,
    roll                REAL,
    gaze_zone           TEXT,
    intervention_tier   INTEGER,
    intervention_fired  INTEGER   -- 0 or 1
);
"""


@dataclass
class LogRow:
    session_id:         str
    timestamp:          float
    state:              str
    attention_raw:      float
    attention_filtered: float
    gaze_score:         float
    head_score:         float
    context_score:      float
    blink_score:        float
    yaw:                float
    pitch:              float
    roll:               float
    gaze_zone:          str
    intervention_tier:  int
    intervention_fired: bool


class SessionLogger:
    """
    Writes session data to SQLite.
    All writes happen synchronously in the calling thread (fast enough for 2Hz logging).
    """

    def __init__(self, cfg: dict):
        db_path = cfg["logging"]["db_path"]
        self._interval = cfg["logging"]["log_interval_sec"]
        self._last_log  = 0.0

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()
        self._lock = threading.Lock()
        logger.info("SessionLogger: connected to %s", db_path)

    def maybe_log(self, row: LogRow) -> bool:
        """Log row if enough time has passed since last log. Returns True if logged."""
        now = time.monotonic()
        if now - self._last_log < self._interval:
            return False

        insert_sql = """
        INSERT INTO session_log (
            session_id, timestamp, state,
            attention_raw, attention_filtered,
            gaze_score, head_score, context_score, blink_score,
            yaw, pitch, roll, gaze_zone,
            intervention_tier, intervention_fired
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            row.session_id, row.timestamp, row.state,
            row.attention_raw, row.attention_filtered,
            row.gaze_score, row.head_score, row.context_score, row.blink_score,
            row.yaw, row.pitch, row.roll, row.gaze_zone,
            row.intervention_tier, int(row.intervention_fired),
        )
        with self._lock:
            self._conn.execute(insert_sql, values)
            self._conn.commit()
        self._last_log = now
        return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("SessionLogger closed.")
