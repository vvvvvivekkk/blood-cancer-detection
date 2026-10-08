"""Capture docs/screenshots/<view>[_dark].png from a running web app.

    python run.py --no-browser &          # serve the artifacts you want to show
    python docs/capture_screenshots.py [--base http://localhost:8000] [--theme light|dark] [--suffix _dark]
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
VIEWS = ["home", "preview", "performance", "chart", "about", "data", "train"]


def predict_image():
    """A held-out test image if exported (python run.py --cli export-test-images), else a bundled sample."""
    for root in (REPO / "test_images" / "[Malignant] Pro-B", REPO / "data" / "sample" / "[Malignant] Pro-B"):
        if root.is_dir():
            return next(root.iterdir())
    raise FileNotFoundError("no image to predict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    out = DOCS / "screenshots"
    out.mkdir(exist_ok=True)
    shot = lambda page, name: (page.screenshot(path=str(out / f"{name}{a.suffix}.png"), full_page=True),  # noqa: E731
                               print("saved", out / f"{name}{a.suffix}.png"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 960}, color_scheme=a.theme)
        page.goto(f"{a.base}/#/home")
        page.wait_for_selector("#login-screen:not([hidden])")
        page.fill("#login-user", "admin")
        page.fill("#login-pass", "admin")
        shot(page, "login")
        page.click(".login-submit")
        page.wait_for_selector('[data-view="home"]:not([hidden])')
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(5500)  # let the "Login successful" toast fade
        for v in VIEWS:
            if v in ("data", "train") and page.get_attribute("#pipeline-toggle", "aria-expanded") != "true":
                page.click("#pipeline-toggle")
            page.click(f'[data-nav="{v}"]')
            page.wait_for_load_state("networkidle")
            if v == "preview":
                page.set_input_files("#file-input", str(predict_image()))
                page.click("#btn-submit")
                page.wait_for_selector("#result-body:not([hidden])", timeout=120_000)
            page.wait_for_timeout(900)  # let charts finish animating
            shot(page, v)
        browser.close()


if __name__ == "__main__":
    main()
