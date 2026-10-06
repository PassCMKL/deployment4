"""
Monitoring dashboard (Streamlit). Reads backend/predictions.db and
backend/baseline.json directly, read-only. It never calls the FastAPI app or
the model, so the dashboard can be slow, crash or restart with zero effect on
live serving: monitoring must not be able to take down what it monitors.

Four sections, one question each:
  1. API health      Is the service fast and is it failing?   (api_requests)
  2. Model behaviour Is the model sure of its answers?        (predictions)
  3. Data drift      Has the input changed vs. training?      (predictions + baseline)
  4. Retraining      What did the pipeline decide and why?    (retrain_history.jsonl)

Run with: streamlit run dashboard/app.py
"""

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from metrics import (
    MIN_SAMPLES_FOR_DRIFT,
    PSI_DRIFT_THRESHOLD,
    api_summary,
    drift_report,
    low_confidence_rate,
)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backend" / "predictions.db"
BASELINE_PATH = ROOT / "backend" / "baseline.json"
HISTORY_PATH = ROOT / "retrain" / "retrain_history.jsonl"

# A PROXY for model error, not the real thing: a live request never comes with
# its true label, so true accuracy cannot be computed in production. Low
# confidence is the closest observable signal.
LOW_CONFIDENCE_THRESHOLD = 0.5

st.set_page_config(page_title="Digit Recognizer - Monitoring", layout="wide")
st.title("Digit Recognizer - Monitoring Dashboard")

auto_refresh = st.sidebar.checkbox("Auto-refresh every 5s", value=False)
source_filter = st.sidebar.selectbox("Traffic source (model + drift sections)", ["all", "live", "synthetic_normal", "synthetic_drift"])


def read_table(query: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:  # table not created yet
        df = pd.DataFrame()
    finally:
        conn.close()
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


api_df = read_table("SELECT timestamp, endpoint, status_code, latency_ms FROM api_requests ORDER BY id")
pred_df = read_table(
    "SELECT id, timestamp, digit, confidence, latency_ms, pixel_mean, pixel_std, cache_hit, source, "
    "true_label, used_in_retrain FROM predictions ORDER BY id"
)
baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else None

if api_df.empty and pred_df.empty:
    st.warning(
        "No requests logged yet. Start the stack, then run "
        "`python scripts/workload_generator.py` (or use the web app) and refresh."
    )
    st.stop()

# ---------------------------------------------------------------- 1. API health
st.header("1. API health")
s = api_summary(api_df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests", s["total"])
c2.metric("Error rate", f"{s['error_rate'] * 100:.1f}%")
c3.metric("p50 latency", f"{s['p50']:.1f} ms")
c4.metric("p95 latency", f"{s['p95']:.1f} ms")
c5.metric("p99 latency", f"{s['p99']:.1f} ms")
st.caption(
    f"{s['client_errors']} client errors (4xx: bad upload, bad header) and "
    f"{s['server_errors']} server errors (5xx: the service failed). "
    "A rising 5xx rate is an outage signal; 4xx mostly means bad callers."
)

if not api_df.empty:
    api_df["is_error"] = (api_df["status_code"] >= 400).astype(int)
    bucketed = api_df.set_index("datetime").resample("10s").agg(
        {"latency_ms": "mean", "is_error": "mean", "status_code": "count"}
    ).rename(columns={"status_code": "requests"})
    left, mid, right = st.columns(3)
    left.subheader("Mean latency / 10s (ms)")
    left.line_chart(bucketed["latency_ms"], height=200)
    mid.subheader("Error rate / 10s")
    mid.line_chart(bucketed["is_error"], height=200)
    right.subheader("Requests / 10s")
    right.bar_chart(bucketed["requests"], height=200)
    st.dataframe(
        api_df["status_code"].value_counts().rename_axis("HTTP status").reset_index(name="count"),
        hide_index=True,
        width="content",
    )

# ------------------------------------------------------------ 2. Model behaviour
view = pred_df if source_filter == "all" or pred_df.empty else pred_df[pred_df["source"] == source_filter]
st.header("2. Model behaviour")
if view.empty:
    st.info("No model predictions for this traffic source yet.")
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Predictions logged", len(view))
    m2.metric("Cache hit rate", f"{view['cache_hit'].mean() * 100:.1f}%")
    m3.metric(
        f"Low-confidence rate (<{LOW_CONFIDENCE_THRESHOLD})",
        f"{low_confidence_rate(view, LOW_CONFIDENCE_THRESHOLD) * 100:.1f}%",
    )
    st.caption(
        "Low-confidence rate is a proxy for model error. Real accuracy needs true "
        "labels, which live traffic never carries."
    )
    st.line_chart(view.set_index("datetime")["confidence"].rolling(10, min_periods=1).mean(), height=200)
    st.caption("Model confidence, rolling mean over 10 predictions.")

# ------------------------------------------------------------------ 3. Data drift
st.header("3. Data drift vs. training baseline")
if baseline is None:
    st.warning("No baseline yet. Run `python scripts/compute_baseline.py` once to create backend/baseline.json.")
elif pred_df.empty:
    st.info("No predictions yet.")
else:
    st.caption(
        f"Baseline: {baseline['n_samples']} real MNIST training images through the same preprocess() "
        f"as live traffic. PSI > {PSI_DRIFT_THRESHOLD} flags a major distribution shift "
        f"(needs >= {MIN_SAMPLES_FOR_DRIFT} cache-miss requests; cache hits skip preprocessing so carry no pixel stats)."
    )
    misses = pred_df[(pred_df["cache_hit"] == 0)].dropna(subset=["pixel_mean", "pixel_std"])
    for source, group in misses.groupby("source"):
        st.subheader(source)
        cols = st.columns(2)
        for col, feature in zip(cols, ("pixel_mean", "pixel_std")):
            report = drift_report(group[feature].to_numpy(), baseline["features"][feature])
            with col:
                if not report["enough_data"]:
                    st.metric(feature, "not enough data", f"{report['n']} samples")
                    continue
                st.metric(f"{feature} PSI", f"{report['psi']:.3f}", f"mean shift z = {report['z']:+.1f}", delta_color="off")
                if report["drifted"]:
                    st.error("DRIFT DETECTED: this input distribution no longer matches training.")
                else:
                    st.success("Stable.")

# ------------------------------------------------------------------ 4. Retraining
st.header("4. Retraining history")
if HISTORY_PATH.exists():
    records = [json.loads(line) for line in HISTORY_PATH.read_text().splitlines() if line.strip()]
    hist = pd.DataFrame(records)
    hist["time"] = pd.to_datetime(hist["timestamp"], unit="s")
    st.dataframe(hist.drop(columns=["timestamp"]).iloc[::-1], hide_index=True)
else:
    st.info("No retraining cycle has run yet. Run `python retrain/run_retrain_cycle.py --force`.")

if not pred_df.empty:
    labeled = pred_df["true_label"].notna()
    st.caption(
        f"Labeled rows: {int(labeled.sum())} total, "
        f"{int((labeled & (pred_df['used_in_retrain'] == 0)).sum())} not yet used for retraining."
    )

if auto_refresh:
    time.sleep(5)
    st.rerun()
