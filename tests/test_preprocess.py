import io

import numpy as np
from PIL import Image
from preprocessing import IMAGE_SIZE, preprocess, softmax


def _png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_shape_and_dtype():
    out = preprocess(_png(Image.new("L", (280, 280), 0)))
    assert out.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE) and out.dtype == np.float32


def test_values_in_unit_interval():
    out = preprocess(_png(Image.new("L", (280, 280), 255)))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_light_background_is_inverted():
    assert preprocess(_png(Image.new("L", (280, 280), 240))).mean() < 0.5


def test_dark_background_is_unchanged():
    out = preprocess(_png(Image.new("L", (280, 280), 15)))
    assert np.isclose(out.mean(), 15 / 255, atol=1e-3)


def test_color_is_converted_to_grayscale():
    out = preprocess(_png(Image.new("RGB", (280, 280), (10, 20, 30))))
    assert out.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_softmax_sums_to_one_and_is_stable_for_large_logits():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isclose(p.sum(), 1.0) and not np.isnan(p).any()
