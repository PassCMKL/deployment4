import pathlib
import re

import requests
from playwright.sync_api import Page, expect

FRONTEND_URL = "http://localhost:8080"
BACKEND_HEALTH_URL = "http://localhost:8000/health"
FIXTURE_IMAGE = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_digit.png"


def test_backend_health_is_reachable():
    response = requests.get(BACKEND_HEALTH_URL, timeout=5)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_image_and_see_a_prediction(page: Page):
    page.goto(FRONTEND_URL)

    page.set_input_files("#fileInput", str(FIXTURE_IMAGE))

    digit_locator = page.locator("#digit")
    expect(digit_locator).to_have_text(re.compile(r"^\d$"), timeout=10_000)

    confidence_text = page.locator("#confidence").inner_text()
    assert "confidence" in confidence_text

    probability_rows = page.locator("#probTable tr")
    expect(probability_rows).to_have_count(10)


def test_clear_button_resets_the_result(page: Page):
    page.goto(FRONTEND_URL)

    page.set_input_files("#fileInput", str(FIXTURE_IMAGE))
    expect(page.locator("#digit")).to_have_text(re.compile(r"^\d$"), timeout=10_000)

    page.click("#clearBtn")

    expect(page.locator("#digit")).to_have_text("")
    expect(page.locator("#probTable tr")).to_have_count(0)
