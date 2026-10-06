"""
Request logging for continual learning (Week 7).

Every /predict call writes one row here: what came in (pixel stats),
what the model said (digit, confidence), how long it took, and whether
it was a cache hit. This table is the single source of truth that the
dashboard (Week 7) and the retraining trigger (Week 8) both read from --
neither of them talks to the model or the cache directly.

SQLite is a blocking, synchronous library (the sqlite3 module does real
disk I/O with no async support). Calling it directly inside an `async def`
endpoint would block the event loop for every other in-flight request,
exactly like the model inference problem in deployment3 -- the fix is the
same one: run the blocking call in a worker thread via anyio.to_thread.run_sync.
"""

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "predictions.db"


def _connect() -> sqlite3.Connection:
    # timeout: wait up to 10s for a lock instead of failing instantly, since
    # the dashboard and retrain script read this same file while the API writes.
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db() -> None:
    """Create the table if it doesn't exist yet. Called once at startup."""
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            digit INTEGER NOT NULL,
            confidence REAL NOT NULL,
            latency_ms REAL NOT NULL,
            pixel_mean REAL,
            pixel_std REAL,
            cache_hit INTEGER NOT NULL,
            source TEXT NOT NULL,
            true_label INTEGER,
            image_png BLOB,
            used_in_retrain INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # HTTP-level view: one row per /predict call INCLUDING failures. The
    # predictions table above only ever sees requests that reached the model,
    # so it can never show an error rate. This table can.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            endpoint TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            latency_ms REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_api_request(*, endpoint: str, status_code: int, latency_ms: float) -> None:
    """Blocking insert; called via anyio.to_thread.run_sync like log_prediction."""
    conn = _connect()
    conn.execute(
        "INSERT INTO api_requests (timestamp, endpoint, status_code, latency_ms) VALUES (?, ?, ?, ?)",
        (time.time(), endpoint, status_code, latency_ms),
    )
    conn.commit()
    conn.close()


def log_prediction(
    *,
    digit: int,
    confidence: float,
    latency_ms: float,
    pixel_mean: float | None,
    pixel_std: float | None,
    cache_hit: bool,
    source: str = "live",
    true_label: int | None = None,
    image_bytes: bytes | None = None,
) -> None:
    """
    Blocking insert -- this function itself is plain sync code with no
    `async`. The caller (main.py) is responsible for running it via
    anyio.to_thread.run_sync so it doesn't block the event loop.

    true_label/image_bytes are only ever non-None for traffic that came
    with a known answer attached (the workload generator, replaying real
    MNIST images). Real live requests have no ground truth, so for them
    both stay NULL -- we deliberately do NOT store raw image bytes for
    traffic we can't use for supervised fine-tuning anyway; keeping them
    would just be wasted storage.
    """
    conn = _connect()
    conn.execute(
        """
        INSERT INTO predictions
            (timestamp, digit, confidence, latency_ms, pixel_mean, pixel_std,
             cache_hit, source, true_label, image_png)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            time.time(), digit, confidence, latency_ms, pixel_mean, pixel_std,
            int(cache_hit), source, true_label, image_bytes,
        ),
    )
    conn.commit()
    conn.close()


def fetch_unused_labeled_rows() -> list[dict]:
    """
    Rows with a true_label AND an image AND not yet consumed by a previous
    retraining cycle. This is what the Week 8 pipeline fine-tunes on.
    """
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT id, true_label, image_png FROM predictions
        WHERE true_label IS NOT NULL AND image_png IS NOT NULL AND used_in_retrain = 0
        """
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def mark_rows_used(ids: list[int]) -> None:
    if not ids:
        return
    conn = _connect()
    conn.executemany(
        "UPDATE predictions SET used_in_retrain = 1 WHERE id = ?",
        [(i,) for i in ids],
    )
    conn.commit()
    conn.close()


def recent_avg_confidence(window_seconds: float) -> float | None:
    """Average confidence over the last `window_seconds`, or None if empty."""
    conn = _connect()
    cutoff = time.time() - window_seconds
    cur = conn.execute(
        "SELECT AVG(confidence) FROM predictions WHERE timestamp >= ?", (cutoff,)
    )
    (avg,) = cur.fetchone()
    conn.close()
    return avg
