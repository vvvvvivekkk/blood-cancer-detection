"""Browser tests: headless Chromium against the real FastAPI app on a free port. Offline:
the Chart.js CDN request is answered with a stub so no network is needed."""
import socket
import threading
import time

import pytest

sync_api = pytest.importorskip(
    "playwright.sync_api",
    reason="Playwright not installed: pip install -r requirements-dev.txt && playwright install chromium")

from api.main import app  # noqa: E402
from src import config as C  # noqa: E402

VIEWS = ["overview", "data", "train", "evaluate", "predict", "about"]
SAMPLE = next((C.SAMPLE_DIR / C.CLASS_NAMES[0]).iterdir())
CHART_STUB = "window.Chart = class { constructor() {} destroy() {} update() {} };"


@pytest.fixture(scope="module")
def server():
    import uvicorn
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    end = time.time() + 30
    while not srv.started:
        assert time.time() < end, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(10)


@pytest.fixture(scope="module")
def pw_browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium not available ({e}). Run: playwright install chromium")
        yield b
        b.close()


def _open(browser, base, **ctx):
    context = browser.new_context(**ctx)
    page = context.new_page()
    errors = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("https://cdn.jsdelivr.net/**", lambda r: r.fulfill(body=CHART_STUB, content_type="application/javascript"))
    page.goto(f"{base}/#/overview")
    page.wait_for_load_state("networkidle")
    return context, page, errors


@pytest.fixture
def ui(pw_browser, server):
    context, page, errors = _open(pw_browser, server, viewport={"width": 1366, "height": 900})
    yield page, errors
    context.close()


def visit_all(page, errors):
    for v in VIEWS:
        page.click(f'[data-nav="{v}"]')
        page.locator(f'[data-view="{v}"]').wait_for(state="visible")
        page.wait_for_load_state("networkidle")
        assert page.locator(f'[data-nav="{v}"]').get_attribute("aria-current") == "page"
        assert "Not for clinical use" in page.inner_text(".footer")
        assert not errors, (v, errors)


def test_every_view_renders_with_smoke_artifacts(smoke, ui):
    page, errors = ui
    visit_all(page, errors)
    page.click('[data-nav="evaluate"]')
    page.wait_for_selector("#cmp-table tbody tr")
    assert page.locator("#cmp-table tbody tr").count() == 5
    assert page.locator("#cmp-table .badge", has_text="selected").count() == 1
    assert page.locator("#smoke-banner").is_visible()


def test_every_view_renders_empty_with_fix_buttons(empty_root, ui):
    page, errors = ui
    visit_all(page, errors)
    for v, label in (("train", "Use demo data"), ("evaluate", "Use demo data"), ("predict", "Use demo data")):
        page.click(f'[data-nav="{v}"]')
        empty = page.locator(f'[data-view="{v}"] .empty')
        empty.wait_for()
        assert empty.locator("button", has_text=label).is_enabled()


def test_predict_flow_shows_class(smoke, ui):
    page, errors = ui
    page.click('[data-nav="predict"]')
    page.set_input_files("#file-input", str(SAMPLE))
    page.click("#btn-analyze")
    page.wait_for_selector("#result-body:not([hidden])", timeout=120_000)
    assert page.inner_text("#result-class") in C.CLASS_NAMES
    assert page.locator("#votes tbody tr").count() == 3
    assert page.get_attribute("#img-cam", "src").startswith("data:image/png;base64,")
    page.fill("#cam-opacity", "30")
    assert page.inner_text("#cam-out") == "30%"
    page.set_input_files("#file-input", {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"hi"})
    assert "Unsupported" in page.inner_text("#predict-error")
    assert not errors, errors


def test_run_full_pipeline_reaches_all_done(empty_root, ui):
    page, errors = ui
    assert page.locator('[data-step][data-state="done"]').count() == 0
    assert page.is_checked("#all-smoke")
    page.click("#run-all")
    page.wait_for_selector("#job-chip.running")
    page.wait_for_function(
        "document.querySelectorAll('[data-step][data-state=\"done\"]').length === 5", timeout=400_000)
    assert page.inner_text("#job-state") == "done"
    assert "job finished" in page.inner_text("#job-log")
    assert not errors, errors


def test_dark_mode_toggle_flips_theme(ui):
    page, _ = ui
    before = page.get_attribute("html", "data-theme")
    page.click("#theme-toggle")
    after = page.get_attribute("html", "data-theme")
    assert {before, after} == {"light", "dark"}
    assert page.get_attribute("#theme-toggle", "aria-pressed") == str(after == "dark").lower()


def test_mobile_sidebar_collapses_to_menu(pw_browser, server):
    context, page, errors = _open(pw_browser, server, viewport={"width": 390, "height": 844})
    try:
        assert page.is_visible("#menu-btn") and not page.is_visible("#sidebar")
        page.click("#menu-btn")
        page.wait_for_selector("#sidebar.open")
        assert page.is_visible('[data-nav="data"]')
        assert page.get_attribute("#menu-btn", "aria-expanded") == "true"
        page.click('[data-nav="data"]')
        page.wait_for_selector("#sidebar", state="hidden")
        page.locator('[data-view="data"]').wait_for(state="visible")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "horizontal scroll"
        assert not errors, errors
    finally:
        context.close()
