# SYS-304 Milestone 4: Continual Learning & Final Demo

Digit recognizer (ResNet18 -> ONNX) served by FastAPI, with request logging, a monitoring dashboard,
drift detection, and an automated retrain -> evaluate -> deploy loop with rollback.
Design document: `docs/SYS304_M4_Design_Document.pdf`.

## Layout
```
backend/     FastAPI service (cache, batching, ONNX) + logging_db.py + preprocessing.py
frontend/    nginx web app (port 8080)
dashboard/   Streamlit monitoring dashboard + metrics.py (PSI, latency, error rate)
scripts/     workload_generator.py, compute_baseline.py, drift_utils.py
retrain/     run_retrain_cycle.py, decision.py, deploy_utils.py, rollback.py
models/      resnet18_mnist_baseline.pt (Git LFS)   backend/model_fp32.onnx (Git LFS)
tests/       pytest suite (49 tests)    e2e/  Playwright UI tests    .github/workflows/ci.yml
```

## One-time setup
1. Put the real files in place: `backend/model_fp32.onnx` and `models/resnet18_mnist_baseline.pt` (from deployment3).
2. `pip install -r scripts/requirements.txt -r dashboard/requirements.txt -r retrain/requirements.txt`
3. `python scripts/compute_baseline.py`   (writes backend/baseline.json from real MNIST training images)

## Run the stack
```
docker compose up --build        # redis + backend :8000 + frontend :8080
streamlit run dashboard/app.py   # dashboard :8501
```

## Demo script
1. Show CI green on GitHub (Actions tab): lint, 49 tests, frontend tests, e2e.
2. Open http://localhost:8080, upload a digit.
3. `python scripts/workload_generator.py --mode normal --count 200 --rate 20 --label-rate 1.0 --error-rate 0.05`
   -> dashboard: latency percentiles, error rate, drift "Stable".
4. `python scripts/workload_generator.py --mode drift --count 200 --rate 20 --label-rate 1.0`
   -> dashboard: PSI jumps, "DRIFT DETECTED", confidence falls.
5. `python retrain/run_retrain_cycle.py --force` -> trigger, fetch, fine-tune, evaluate, decision, deploy, reload, smoke test.
6. Dashboard "Retraining history" shows the cycle. Run the drift workload again to show recovery.
7. Failure drill: `python retrain/rollback.py` restores the previous model.

Options: `--dry-run` trains and evaluates without deploying. Without `--force` the script retrains only if
24 h passed or average confidence over the last hour is < 0.70 (10-minute cooldown).

## Tests
`pip install -r requirements-dev.txt && ruff check . && pytest` (needs the real `backend/model_fp32.onnx`).
