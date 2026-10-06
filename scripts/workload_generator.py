"""
Week 7 workload generator.

Purpose: hit the live /predict endpoint repeatedly so there's actually
something in predictions.db for the dashboard and the retraining trigger
to read. Two traffic modes, because "just send requests" isn't enough to
demo drift detection -- you need to be able to turn drift ON and OFF on
demand:

  --mode normal   real MNIST test-set digits, unmodified. This is what
                   "healthy" traffic looks like -- it should match the
                   training-time baseline closely, by construction.

  --mode drift     the SAME MNIST digits, but each one is rotated, has
                   noise added, and has its brightness shifted before
                   being sent. This is a stand-in for "the real world
                   changed" (e.g. users photographing handwriting at an
                   angle, in bad lighting, with a noisy camera) -- the
                   model never saw inputs like this during training.

Every request carries an X-Traffic-Source header so the backend's log
can tell this synthetic traffic apart from anything else hitting the API.
The header does not affect the prediction at all -- main.py only reads
it for logging.

Also sends X-True-Label on a configurable fraction of requests (--label-rate,
default 1.0), since this generator is the stand-in for "a batch of newly
labeled data arrived" that the Week 8 retraining pipeline fine-tunes on.
Real production traffic wouldn't carry this -- a live user's upload has
no known true label, which is exactly why this generator exists: without
it, there would be no labeled data for the retraining pipeline to use at all.
"""

import argparse
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_utils import apply_drift  # noqa: E402

# MNIST test images 0..999 are RESERVED as the retrain pipeline's held-out
# evaluation set (keep in sync with N_HELDOUT in retrain/run_retrain_cycle.py).
# This generator only samples from 1000+, so the images used to fine-tune the
# model can never also be the images used to judge it.
HELDOUT_RESERVED = 1000


def load_mnist_test_images(n: int) -> list[tuple[np.ndarray, int]]:
    # download=True only fetches once; torchvision caches to ./data after
    # the first run. Real labeled test-set images -- the label travels
    # alongside each image now, used below to populate X-True-Label.
    ds = datasets.MNIST(root="./data", train=False, download=True)
    pool = range(HELDOUT_RESERVED, len(ds))
    indices = random.sample(pool, min(n, len(pool)))
    return [(np.array(ds[i][0], dtype=np.float32), int(ds[i][1])) for i in indices]


def to_png_bytes(pixels: np.ndarray) -> bytes:
    image = Image.fromarray(pixels.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Send synthetic traffic to /predict.")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--count", type=int, default=50, help="number of requests to send")
    parser.add_argument("--rate", type=float, default=5.0, help="requests per second")
    parser.add_argument(
        "--label-rate", type=float, default=1.0,
        help="fraction of requests that carry X-True-Label (simulates partial labeling)",
    )
    parser.add_argument(
        "--error-rate", type=float, default=0.0,
        help="fraction of extra requests that are deliberately invalid (HTTP 400), to exercise the error-rate panel",
    )
    args = parser.parse_args()

    source_tag = "synthetic_normal" if args.mode == "normal" else "synthetic_drift"
    images = load_mnist_test_images(args.count)

    print(f"Sending {args.count} requests, mode={args.mode}, tag={source_tag}, rate={args.rate}/s")
    delay = 1.0 / args.rate if args.rate > 0 else 0

    for i, (pixels, true_label) in enumerate(images):
        if random.random() < args.error_rate:
            # Not an image at all -> backend answers 400. Logged in api_requests.
            bad = requests.post(
                args.url,
                files={"file": ("notes.txt", b"not an image", "text/plain")},
                headers={"X-Traffic-Source": source_tag},
                timeout=5,
            )
            print(f"[{i+1}/{args.count}] deliberate bad request -> HTTP {bad.status_code}")

        if args.mode == "drift":
            pixels = apply_drift(pixels)

        png_bytes = to_png_bytes(pixels)
        headers = {"X-Traffic-Source": source_tag}
        if random.random() < args.label_rate:
            headers["X-True-Label"] = str(true_label)

        try:
            resp = requests.post(
                args.url,
                files={"file": ("digit.png", png_bytes, "image/png")},
                headers=headers,
                timeout=5,
            )
            result = resp.json()
            print(
                f"[{i+1}/{args.count}] digit={result.get('digit')} "
                f"confidence={result.get('confidence'):.3f} true_label={headers.get('X-True-Label', '-')}"
            )
        except Exception as exc:
            print(f"[{i+1}/{args.count}] request failed: {exc}")

        if delay:
            time.sleep(delay)


if __name__ == "__main__":
    main()
