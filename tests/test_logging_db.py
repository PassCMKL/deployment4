import time

import logging_db


def _log(**kw):
    base = dict(digit=1, confidence=0.9, latency_ms=5.0, pixel_mean=0.1, pixel_std=0.2, cache_hit=False)
    logging_db.log_prediction(**{**base, **kw})


def test_only_labeled_unused_rows_are_fetched():
    _log()                                              # unlabeled
    _log(true_label=3, image_bytes=b"png")              # labeled
    rows = logging_db.fetch_unused_labeled_rows()
    assert len(rows) == 1 and rows[0]["true_label"] == 3


def test_marked_rows_are_not_fetched_again():
    _log(true_label=3, image_bytes=b"png")
    ids = [r["id"] for r in logging_db.fetch_unused_labeled_rows()]
    logging_db.mark_rows_used(ids)
    assert logging_db.fetch_unused_labeled_rows() == []


def test_recent_avg_confidence_none_when_empty():
    assert logging_db.recent_avg_confidence(3600) is None


def test_recent_avg_confidence_averages_window():
    _log(confidence=0.4)
    _log(confidence=0.6)
    assert abs(logging_db.recent_avg_confidence(3600) - 0.5) < 1e-9


def test_api_request_is_stored():
    logging_db.log_api_request(endpoint="/predict", status_code=503, latency_ms=1.0)
    conn = logging_db._connect()
    assert conn.execute("SELECT status_code FROM api_requests").fetchall() == [(503,)]
    conn.close()
    assert time.time()  # keeps import used
