"""
The one definition of "drifted input", shared by the workload generator
(which sends drifted traffic) and the retrain pipeline (which builds a
held-out DRIFTED evaluation set from the same distortion). Keeping a single
definition means the model is judged on the same kind of change it is being
retrained to handle, not on a lookalike that could quietly differ.
"""

import numpy as np
from PIL import Image


def apply_drift(pixels: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Distort a 28x28 uint8-range digit: random rotation, Gaussian pixel noise,
    brightness shift. Stand-in for real causes of drift (camera angle, sensor
    noise, lighting). Pass a seeded `rng` for a reproducible result, which is
    what the fixed drifted evaluation set relies on.
    """
    rng = rng or np.random.default_rng()

    image = Image.fromarray(pixels.astype(np.uint8), mode="L")
    image = image.rotate(float(rng.uniform(-35, 35)), fillcolor=0)

    out = np.array(image, dtype=np.float32)
    out = out + rng.normal(loc=0, scale=25, size=out.shape)
    out = out + float(rng.uniform(-40, 40))
    return np.clip(out, 0, 255)
