import numpy as np
import pandas as pd
from metrics import MIN_SAMPLES_FOR_DRIFT, api_summary, baseline_histogram, drift_report, low_confidence_rate, psi

RNG = np.random.default_rng(0)


def _baseline(values):
    edges, probs = baseline_histogram(values)
    return {"avg": float(values.mean()), "std": float(values.std()), "edges": edges, "probs": probs}


def test_psi_near_zero_for_same_distribution():
    train = RNG.normal(0.13, 0.01, 5000)
    base = _baseline(train)
    assert psi(RNG.normal(0.13, 0.01, 500), base["edges"], base["probs"]) < 0.1


def test_psi_large_for_shifted_distribution():
    train = RNG.normal(0.13, 0.01, 5000)
    base = _baseline(train)
    assert psi(RNG.normal(0.18, 0.01, 500), base["edges"], base["probs"]) > 0.25


def test_drift_report_flags_shift_and_not_stable():
    train = RNG.normal(0.13, 0.01, 5000)
    base = _baseline(train)
    assert drift_report(RNG.normal(0.18, 0.01, 200), base)["drifted"]
    assert not drift_report(RNG.normal(0.13, 0.01, 200), base)["drifted"]


def test_drift_report_refuses_tiny_samples():
    base = _baseline(RNG.normal(0.13, 0.01, 1000))
    report = drift_report(RNG.normal(0.5, 0.01, MIN_SAMPLES_FOR_DRIFT - 1), base)
    assert not report["enough_data"] and not report["drifted"]


def test_api_summary_counts_errors_and_percentiles():
    df = pd.DataFrame({"status_code": [200, 200, 200, 400, 500], "latency_ms": [10, 20, 30, 40, 50]})
    s = api_summary(df)
    assert s["client_errors"] == 1 and s["server_errors"] == 1 and s["error_rate"] == 0.4
    assert s["p50"] == 30


def test_api_summary_empty():
    assert api_summary(pd.DataFrame())["total"] == 0


def test_low_confidence_rate():
    assert low_confidence_rate(pd.DataFrame({"confidence": [0.2, 0.9, 0.4, 0.99]}), 0.5) == 0.5
