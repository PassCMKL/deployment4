"""
Shared image preprocessing -- used by main.py (live requests) AND by
scripts/compute_baseline.py (computing the training-data baseline that
drift gets compared against).

This is split out from main.py specifically so the baseline script can
import it without also importing onnxruntime and redis, which main.py
needs but a pure-preprocessing script doesn't. It also means there is
exactly one place this logic is defined -- if drift comparisons used a
second, slightly different reimplementation of preprocessing, any drift
signal detected could just be an artifact of that mismatch, not real
drift in the incoming data.
"""

import io

import numpy as np
from PIL import Image

IMAGE_SIZE = 28


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    pixels = np.array(image, dtype=np.float32)

    # The model was trained on dark-background digits; canvas drawings and
    # arbitrary uploads can land on either polarity, so flip to match.
    if pixels.mean() > 127:
        pixels = 255.0 - pixels

    pixels = pixels / 255.0
    return pixels[None, None, :, :]  # (1, 1, 28, 28)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()
