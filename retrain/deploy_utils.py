"""
File swapping, server reload, smoke test, rollback and the audit log.
No torch here, so it is cheap to import and test.

Deploying a model is more than overwriting a file:
  1. the candidate is written to a SEPARATE path and validated first
  2. the live model + its weights are backed up
  3. the candidate replaces the live file atomically (os.replace)
  4. the running server is told to reload (it loaded the old model at startup)
  5. a smoke test confirms the server answers with the new model
  6. if 4 or 5 fail, the backup is restored: automatic rollback
"""

import json
import os
import shutil
import time
from pathlib import Path

import requests

RETRAIN_DIR = Path(__file__).resolve().parent
ROOT = RETRAIN_DIR.parent
BACKEND_DIR = ROOT / "backend"
MODELS_DIR = ROOT / "models"

ONNX_PATH = BACKEND_DIR / "model_fp32.onnx"
ONNX_PREV_PATH = BACKEND_DIR / "model_fp32_prev.onnx"
ONNX_CANDIDATE_PATH = BACKEND_DIR / "model_candidate.onnx"
WEIGHTS_PATH = MODELS_DIR / "current_deployed.pt"
WEIGHTS_PREV_PATH = MODELS_DIR / "previous_deployed.pt"
BASELINE_WEIGHTS_PATH = MODELS_DIR / "resnet18_mnist_baseline.pt"
STATE_PATH = RETRAIN_DIR / "retrain_state.json"
HISTORY_PATH = RETRAIN_DIR / "retrain_history.jsonl"

SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8000")


# ---- state + audit trail -----------------------------------------------------

def read_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_retrain_time": 0.0}


def write_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def append_history(record: dict) -> None:
    """One JSON line per cycle: what triggered it and what was decided, and why."""
    record = {"timestamp": time.time(), **record}
    with HISTORY_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---- server interaction --------------------------------------------------------

def reload_server(timeout: float = 20) -> str:
    """'reloaded' | 'unreachable' | 'failed: ...'. Also flushes the server's cache."""
    try:
        resp = requests.post(f"{SERVER_URL}/admin/reload-model", timeout=timeout)
    except requests.ConnectionError:
        return "unreachable"
    except requests.RequestException as exc:
        return f"failed: {exc}"
    if resp.status_code != 200:
        return f"failed: HTTP {resp.status_code}"
    return "reloaded"


def smoke_test_server(png_bytes: bytes, timeout: float = 20) -> bool:
    """
    Real request through the real serving path. Right after a reload the cache
    is empty, so this cannot be answered from a stale cache entry: it proves the
    NEW model loads, runs and returns a well-formed answer.
    """
    try:
        resp = requests.post(
            f"{SERVER_URL}/predict",
            files={"file": ("smoke.png", png_bytes, "image/png")},
            headers={"X-Traffic-Source": "deploy_smoke_test"},
            timeout=timeout,
        )
        body = resp.json()
        probs = body["probabilities"]
        return resp.status_code == 200 and len(probs) == 10 and abs(sum(probs) - 1.0) < 1e-3
    except Exception:
        return False


# ---- backup + rollback ------------------------------------------------------------

def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def backup_live_model() -> None:
    if ONNX_PATH.exists():
        _copy_atomic(ONNX_PATH, ONNX_PREV_PATH)
    if WEIGHTS_PATH.exists():
        _copy_atomic(WEIGHTS_PATH, WEIGHTS_PREV_PATH)


def rollback() -> dict:
    """Restore the previous model + weights and tell the server to reload them."""
    if not ONNX_PREV_PATH.exists():
        return {"rolled_back": False, "reason": "no backup exists (nothing was ever deployed)"}
    _copy_atomic(ONNX_PREV_PATH, ONNX_PATH)
    if WEIGHTS_PREV_PATH.exists():
        _copy_atomic(WEIGHTS_PREV_PATH, WEIGHTS_PATH)
    return {"rolled_back": True, "server": reload_server()}
