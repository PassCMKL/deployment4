import sqlite3
from pathlib import Path

import logging_db
import pytest

FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "sample_digit.png"


def _post(client, headers=None):
    with FIXTURE_IMAGE.open("rb") as f:
        return client.post("/predict", files={"file": ("s.png", f, "image/png")}, headers=headers or {})


def _rows(sql):
    conn = sqlite3.connect(logging_db.DB_PATH)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_valid_distribution(client):
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["digit"] <= 9
    assert len(body["probabilities"]) == 10
    assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-3)


def test_non_image_is_400(client):
    response = client.post("/predict", files={"file": ("n.txt", b"x", "text/plain")})
    assert response.status_code == 400


def test_corrupt_image_is_400(client):
    response = client.post("/predict", files={"file": ("b.png", b"\x00\x01nope", "image/png")})
    assert response.status_code == 400


def test_missing_file_is_422(client):
    assert client.post("/predict").status_code == 422


def test_prediction_is_logged_with_pixel_stats(client):
    _post(client)
    (digit, cache_hit, pixel_mean, source), = _rows("SELECT digit, cache_hit, pixel_mean, source FROM predictions")
    assert cache_hit == 0 and pixel_mean is not None and source == "live"


def test_second_identical_request_is_cache_hit_with_null_pixel_stats(client):
    first, second = _post(client), _post(client)
    assert first.json() == second.json()
    rows = _rows("SELECT cache_hit, pixel_mean FROM predictions ORDER BY id")
    assert rows == [(0, rows[0][1]), (1, None)]


def test_labeled_request_stores_label_and_image(client):
    _post(client, {"X-True-Label": "7", "X-Traffic-Source": "synthetic_normal"})
    (label, has_image, source), = _rows("SELECT true_label, image_png IS NOT NULL, source FROM predictions")
    assert (label, has_image, source) == (7, 1, "synthetic_normal")


def test_unlabeled_request_does_not_store_image(client):
    _post(client)
    (has_image,), = _rows("SELECT image_png IS NOT NULL FROM predictions")
    assert has_image == 0


@pytest.mark.parametrize("bad", ["abc", "10", "-1", ""])
def test_bad_true_label_is_400(client, bad):
    assert _post(client, {"X-True-Label": bad}).status_code == 400


def test_every_predict_call_is_logged_as_api_request_including_errors(client):
    _post(client)
    client.post("/predict", files={"file": ("n.txt", b"x", "text/plain")})
    statuses = sorted(r[0] for r in _rows("SELECT status_code FROM api_requests"))
    assert statuses == [200, 400]


def test_reload_flushes_cache(client):
    _post(client)
    body = client.post("/admin/reload-model").json()
    assert body["status"] == "reloaded" and body["cache_entries_flushed"] == 1
    _post(client)
    assert _rows("SELECT cache_hit FROM predictions ORDER BY id") == [(0,), (0,)]
