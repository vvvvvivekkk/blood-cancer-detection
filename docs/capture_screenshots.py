"""Capture docs/screenshots/{overview,data,train,evaluate,predict,about}.png from a running web app.

    python run.py --no-browser &          # serve the artifacts you want to show
    python docs/capture_screenshots.py [--base http://localhost:8000] [--theme light|dark]
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
VIEWS = ["overview", "data", "train", "evaluate", "predict", "about"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sample = next((REPO / "data" / "sample" / "[Malignant] Pro-B").iterdir())
    out = DOCS / "screenshots"
    out.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 960}, color_scheme=a.theme)
        page.goto(f"{a.base}/#/overview")
        page.wait_for_load_state("networkidle")
        for v in VIEWS:
            page.click(f'[data-nav="{v}"]')
            page.wait_for_load_state("networkidle")
            if v == "predict":
                page.set_input_files("#file-input", str(sample))
                page.click("#btn-analyze")
                page.wait_for_selector("#result-body:not([hidden])", timeout=120_000)
            page.wait_for_timeout(800)  # let charts finish animating
            page.screenshot(path=str(out / f"{v}{a.suffix}.png"), full_page=True)
            print("saved", out / f"{v}{a.suffix}.png")
        browser.close()


if __name__ == "__main__":
    main()
