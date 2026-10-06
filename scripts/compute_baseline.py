"""
Computes the "what does normal input data look like" baseline that drift
gets compared against. Run ONCE, offline, against the real MNIST training
set -- this is not something that runs per-request or even periodically;
it only needs to change if the model is retrained on a meaningfully
different dataset.

Why a baseline is needed at all: a drift score is always a COMPARISON.
"Today's incoming pixel_mean is 0.31" means nothing on its own -- you
need "...and the model was trained on data averaging 0.13" to call that
drift. This script produces that reference number once; the dashboard
and the Week 8 retrain trigger both read it from backend/baseline.json
rather than recomputing it live.

Uses the exact same preprocess() function the live backend uses (imported
from backend/preprocessing.py, not reimplemented here) -- if this script
preprocessed images even slightly differently than main.py does, any
"drift" the dashboard reports could just be measuring that mismatch
instead of anything real happening to live traffic.
"""

import json
import sys
from pathlib import Path

import numpy as np
from torchvision import datasets

# Import the backend's own preprocess() rather than reimplementing it here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from preprocessing import preprocess  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))
from metrics import baseline_histogram  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "backend" / "baseline.json"
N_SAMPLES = 2000


def image_to_png_bytes(pil_image) -> bytes:
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def main():
    ds = datasets.MNIST(root="./data", train=True, download=True)
    n = min(N_SAMPLES, len(ds))

    pixel_means, pixel_stds = [], []
    for i in range(n):
        pil_image, _label = ds[i]
        # Round-trip through PNG bytes so this goes through EXACTLY the
        # same code path (decode -> resize -> polarity check -> normalize)
        # that a real uploaded image goes through in main.py.
        tensor = preprocess(image_to_png_bytes(pil_image))
        pixel_means.append(float(tensor.mean()))
        pixel_stds.append(float(tensor.std()))

    # One entry per monitored input feature: its average, its spread across
    # images, and its decile histogram (what PSI compares live traffic against).
    features = {}
    for name, values in (("pixel_mean", pixel_means), ("pixel_std", pixel_stds)):
        arr = np.asarray(values)
        edges, probs = baseline_histogram(arr)
        features[name] = {
            "avg": float(arr.mean()),
            "std": float(arr.std()),
            "edges": edges,
            "probs": probs,
        }

    baseline = {"n_samples": n, "features": features}
    OUTPUT_PATH.write_text(json.dumps(baseline, indent=2))
    print(f"Wrote baseline from {n} training images to {OUTPUT_PATH}")
    for name, f in features.items():
        print(f"  {name}: avg={f['avg']:.4f} std={f['std']:.4f}")


if __name__ == "__main__":
    main()
