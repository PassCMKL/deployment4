"""
Pure monitoring math (no Streamlit, no database), so it can be unit-tested
and reused. The dashboard calls these; nothing here touches I/O.
"""

import numpy as np
import pandas as pd

# Common industry rule of thumb for PSI: <0.1 stable, 0.1-0.25 moderate shift,
# >0.25 major shift. We flag on the major-shift line.
PSI_DRIFT_THRESHOLD = 0.25
# A z-test on a handful of requests is noise; don't report drift from them.
MIN_SAMPLES_FOR_DRIFT = 30
_EPS = 1e-4


def baseline_histogram(values: np.ndarray, n_bins: int = 10) -> tuple[list[float], list[float]]:
    """
    Quantile bins from the TRAINING values: each bin holds ~1/n_bins of the
    baseline by construction, so PSI compares live traffic against 'what
    fraction of training images fell in each decile'. Returns (inner edges, probs).
    """
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    probs = _bin_probs(values, edges)
    return edges.tolist(), probs.tolist()


def _bin_probs(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(values, edges)  # 0..len(edges); outer bins are open-ended
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def psi(live_values: np.ndarray, edges: list[float], baseline_probs: list[float]) -> float:
    """Population Stability Index of live values vs. the stored baseline bins."""
    live = np.clip(_bin_probs(np.asarray(live_values, dtype=float), np.asarray(edges)), _EPS, None)
    base = np.clip(np.asarray(baseline_probs, dtype=float), _EPS, None)
    return float(np.sum((live - base) * np.log(live / base)))


def mean_shift_z(live_values: np.ndarray, baseline_avg: float, baseline_std: float) -> float:
    """
    z-test of the live MEAN against the training mean. Divides by the standard
    error (baseline_std / sqrt(n)), not by baseline_std itself: the mean of n
    images is far less noisy than one image, so using the per-image spread
    would make real drift look tiny.
    """
    n = len(live_values)
    if n == 0 or baseline_std <= 0:
        return 0.0
    return float((np.mean(live_values) - baseline_avg) / (baseline_std / np.sqrt(n)))


def drift_report(live_values: np.ndarray, feature_baseline: dict) -> dict:
    n = len(live_values)
    if n < MIN_SAMPLES_FOR_DRIFT:
        return {"n": n, "enough_data": False, "psi": None, "z": None, "drifted": False}
    p = psi(live_values, feature_baseline["edges"], feature_baseline["probs"])
    z = mean_shift_z(live_values, feature_baseline["avg"], feature_baseline["std"])
    return {"n": n, "enough_data": True, "psi": p, "z": z, "drifted": p > PSI_DRIFT_THRESHOLD}


def api_summary(api_df: pd.DataFrame) -> dict:
    """HTTP-level health from the api_requests table."""
    if api_df.empty:
        return {"total": 0, "client_errors": 0, "server_errors": 0, "error_rate": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0}
    status = api_df["status_code"]
    client = int(((status >= 400) & (status < 500)).sum())
    server = int((status >= 500).sum())
    lat = api_df["latency_ms"]
    return {
        "total": int(len(api_df)),
        "client_errors": client,
        "server_errors": server,
        "error_rate": (client + server) / len(api_df),
        "p50": float(lat.quantile(0.50)),
        "p95": float(lat.quantile(0.95)),
        "p99": float(lat.quantile(0.99)),
    }


def low_confidence_rate(pred_df: pd.DataFrame, threshold: float) -> float:
    if pred_df.empty:
        return 0.0
    return float((pred_df["confidence"] < threshold).mean())
