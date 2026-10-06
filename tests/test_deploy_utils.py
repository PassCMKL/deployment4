import deploy_utils as du
import pytest
import requests


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name, fname in [("ONNX_PATH", "live.onnx"), ("ONNX_PREV_PATH", "prev.onnx"),
                        ("WEIGHTS_PATH", "cur.pt"), ("WEIGHTS_PREV_PATH", "prev.pt"),
                        ("STATE_PATH", "state.json"), ("HISTORY_PATH", "hist.jsonl")]:
        monkeypatch.setattr(du, name, tmp_path / fname)
    monkeypatch.setattr(du, "reload_server", lambda *a, **k: "reloaded")
    return tmp_path


def test_rollback_restores_previous_files(sandbox):
    du.ONNX_PATH.write_bytes(b"BAD")
    du.ONNX_PREV_PATH.write_bytes(b"GOOD")
    du.WEIGHTS_PATH.write_bytes(b"bad-w")
    du.WEIGHTS_PREV_PATH.write_bytes(b"good-w")
    result = du.rollback()
    assert result["rolled_back"] and du.ONNX_PATH.read_bytes() == b"GOOD" and du.WEIGHTS_PATH.read_bytes() == b"good-w"


def test_rollback_without_backup_does_nothing(sandbox):
    du.ONNX_PATH.write_bytes(b"ONLY")
    assert du.rollback()["rolled_back"] is False and du.ONNX_PATH.read_bytes() == b"ONLY"


def test_backup_then_rollback_round_trip(sandbox):
    du.ONNX_PATH.write_bytes(b"v1")
    du.backup_live_model()
    du.ONNX_PATH.write_bytes(b"v2-broken")
    du.rollback()
    assert du.ONNX_PATH.read_bytes() == b"v1"


def test_state_and_history_round_trip(sandbox):
    assert du.read_state()["last_retrain_time"] == 0.0
    du.write_state({"last_retrain_time": 5.0})
    assert du.read_state()["last_retrain_time"] == 5.0
    du.append_history({"outcome": "x"})
    assert "x" in du.HISTORY_PATH.read_text()


def test_smoke_test_accepts_wellformed_answer(monkeypatch):
    class R:
        status_code = 200
        def json(self): return {"probabilities": [0.1] * 10}
    monkeypatch.setattr(du.requests, "post", lambda *a, **k: R())
    assert du.smoke_test_server(b"png")


def test_smoke_test_rejects_failure(monkeypatch):
    def boom(*a, **k): raise requests.ConnectionError()
    monkeypatch.setattr(du.requests, "post", boom)
    assert not du.smoke_test_server(b"png")


def test_reload_server_reports_unreachable(monkeypatch):
    def boom(*a, **k): raise requests.ConnectionError()
    monkeypatch.setattr(du.requests, "post", boom)
    assert du.reload_server() == "unreachable"
