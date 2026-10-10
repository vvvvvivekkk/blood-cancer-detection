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

# Sidebar order from the examiner walkthrough; Data and Train sit in the Pipeline submenu.
VIEWS = ["home", "preview", "performance", "chart", "about", "data", "train"]
TITLES = {"home": "Home", "preview": "Preview", "performance": "Performance Analysis", "chart": "Chart"}
SAMPLE = next((C.SAMPLE_DIR / C.CLASS_NAMES[0]).iterdir())
CHART_STUB = "window.Chart = class { constructor() {} destroy() {} update() {} };"
LOGGED_IN = "sessionStorage.setItem('bce_auth', '1');"


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


def _open(browser, base, logged_in=True, **ctx):
    context = browser.new_context(**ctx)
    if logged_in:
        context.add_init_script(LOGGED_IN)
    page = context.new_page()
    errors = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("https://cdn.jsdelivr.net/**", lambda r: r.fulfill(body=CHART_STUB, content_type="application/javascript"))
    page.goto(f"{base}/#/home")
    page.wait_for_load_state("networkidle")
    return context, page, errors


@pytest.fixture
def ui(pw_browser, server):
    context, page, errors = _open(pw_browser, server, viewport={"width": 1366, "height": 900})
    yield page, errors
    context.close()


def go(page, view):
    """Click a sidebar item, opening the Pipeline submenu first for Data/Train."""
    if view in ("data", "train") and page.get_attribute("#pipeline-toggle", "aria-expanded") != "true":
        page.click("#pipeline-toggle")
    page.click(f'[data-nav="{view}"]')
    page.locator(f'[data-view="{view}"]').wait_for(state="visible")
    page.wait_for_load_state("networkidle")


def visit_all(page, errors):
    for v in VIEWS:
        go(page, v)
        assert page.locator(f'[data-nav="{v}"]').get_attribute("aria-current") == "page"
        if v in TITLES:
            assert page.inner_text("#view-title") == TITLES[v]
        assert "Not for clinical use" in page.inner_text(".footer")
        assert not errors, (v, errors)


def test_sidebar_order_matches_walkthrough(ui):
    page, _ = ui
    labels = [t.strip() for t in page.locator(".nav > li > a, .nav > li > button").all_inner_texts()]
    assert labels == ["Home", "Preview", "Performance Analysis", "Chart", "About", "Pipeline", "Logout"]
    assert not page.is_visible('[data-nav="data"]')
    page.click("#pipeline-toggle")
    assert page.is_visible('[data-nav="data"]') and page.is_visible('[data-nav="train"]')


def test_login_logout_flow(pw_browser, server):
    context, page, errors = _open(pw_browser, server, logged_in=False, viewport={"width": 1366, "height": 900})
    try:
        assert page.is_visible("#login-screen") and not page.is_visible("#app")
        page.fill("#login-user", "admin")
        page.fill("#login-pass", "wrong")
        page.click(".login-submit")
        page.wait_for_function("document.querySelector('#login-error').textContent.length > 0")
        assert "Invalid" in page.inner_text("#login-error") and not page.is_visible("#app")

        page.fill("#login-pass", "admin")
        page.click(".login-submit")
        page.locator('[data-view="home"]').wait_for(state="visible")
        assert "Login successful" in page.inner_text("#toast")
        assert page.evaluate("sessionStorage.getItem('bce_auth')") == "1"
        page.reload()
        page.locator('[data-view="home"]').wait_for(state="visible")  # session survives a reload

        page.click("#logout")
        page.locator("#login-screen").wait_for(state="visible")
        assert not page.is_visible("#app") and page.evaluate("sessionStorage.getItem('bce_auth')") is None
        page.goto(page.url.split("#")[0] + "#/performance")
        page.wait_for_load_state("networkidle")
        assert page.is_visible("#login-screen")  # deep links still need the login
        assert not [e for e in errors if "401" not in e], errors  # the wrong-password 401 is expected
    finally:
        context.close()


def test_every_view_renders_with_smoke_artifacts(smoke, ui):
    page, errors = ui
    visit_all(page, errors)
    go(page, "performance")
    page.wait_for_selector("#cmp-table tbody tr")
    assert page.locator("#cmp-table tbody tr").count() == 5
    assert page.locator("#cmp-table .badge", has_text="selected").count() == 1
    assert page.locator("#smoke-banner").is_visible()


def test_performance_analysis_tiles_then_confusion_matrix(smoke, ui):
    page, errors = ui
    go(page, "performance")
    page.wait_for_selector("#perf-tiles .kpi")
    assert page.locator("#perf-tiles .kpi-label").all_text_contents() == ["Accuracy", "Precision", "Recall", "F-measure"]
    assert page.evaluate("document.querySelector('#perf-tiles').nextElementSibling.contains(document.querySelector('#cm-panel'))")
    assert page.locator("#cm-panel table.cm tbody tr").count() == 4
    assert not errors, errors


def test_chart_view_curves_per_backbone(smoke, ui):
    page, errors = ui
    go(page, "chart")
    page.wait_for_selector("#chart-tabs .tab")
    assert page.locator("#chart-tabs .tab").all_inner_texts() == ["MobileNetV2", "EfficientNetB0", "DenseNet121"]
    assert "epochs" in page.inner_text("#chart-meta")
    page.click('#chart-tabs .tab[data-key="densenet121"]')
    page.wait_for_selector('#chart-tabs .tab[data-key="densenet121"][aria-selected="true"]')
    assert page.locator("#chart-model-loss").count() == 1 and page.locator("#chart-model-acc").count() == 1
    assert not errors, errors


def test_every_view_renders_empty_with_fix_buttons(empty_root, ui):
    page, errors = ui
    visit_all(page, errors)
    for v in ("train", "performance", "preview", "chart"):
        go(page, v)
        empty = page.locator(f'[data-view="{v}"] .empty')
        empty.wait_for()
        assert empty.locator("button", has_text="Use demo data").is_enabled()


def test_preview_choose_submit_shows_class(smoke, ui):
    page, errors = ui
    go(page, "preview")
    assert page.is_disabled("#btn-submit")
    page.set_input_files("#file-input", str(SAMPLE))
    page.wait_for_selector("#preview:not([hidden])")  # image preview appears before submitting
    page.click("#btn-submit")
    page.wait_for_selector("#result-body:not([hidden])", timeout=120_000)
    assert page.inner_text("#result-class") in C.CLASS_NAMES
    assert page.locator("#votes tbody tr").count() == 3
    assert page.get_attribute("#img-cam", "src").startswith("data:image/png;base64,")
    page.fill("#cam-opacity", "30")
    assert page.inner_text("#cam-out") == "30%"
    page.set_input_files("#file-input", str(next((C.SAMPLE_DIR / C.CLASS_NAMES[1]).iterdir())))
    assert page.is_hidden("#result-body")  # a new image never sits next to the old prediction
    page.set_input_files("#file-input", {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"hi"})
    assert "Unsupported" in page.inner_text("#predict-error")
    assert not errors, errors


def test_preview_try_a_test_image(smoke, ui):
    page, errors = ui
    try:
        go(page, "preview")
        page.wait_for_selector("#btn-export-tests:not([hidden])")  # nothing exported yet in this root
        page.click("#btn-export-tests")
        page.wait_for_function("document.querySelectorAll('#test-image-select option[value^=\"/media/test/\"]').length === 4")
        value = page.locator('#test-image-select option[value^="/media/test/"]').first.get_attribute("value")
        page.select_option("#test-image-select", value)
        page.wait_for_selector("#preview:not([hidden])")
        page.wait_for_selector("#btn-submit:not([disabled])")  # enabled only once the picked image is loaded
        page.click("#btn-submit")
        page.wait_for_selector("#result-body:not([hidden])", timeout=120_000)
        assert page.inner_text("#result-class") in C.CLASS_NAMES
        assert not errors, errors
    finally:
        import shutil
        shutil.rmtree(C.TEST_IMAGES_DIR, ignore_errors=True)


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
        assert page.is_visible('[data-nav="chart"]')
        assert page.get_attribute("#menu-btn", "aria-expanded") == "true"
        page.click('[data-nav="chart"]')
        page.wait_for_selector("#sidebar", state="hidden")
        page.click("#menu-btn")
        page.wait_for_selector("#sidebar.open")
        page.click('[data-nav="chart"]')  # the current page: no hashchange, drawer must still close
        page.wait_for_selector("#sidebar", state="hidden")
        page.locator('[data-view="chart"]').wait_for(state="visible")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "horizontal scroll"
        assert not errors, errors
    finally:
        context.close()


def test_single_model_toggle_changes_predict_and_evaluate(smoke, ui):
    page, errors = ui
    try:
        assert not page.is_checked("#single-mode")
        page.click('label:has(#single-mode)')
        page.wait_for_function("document.querySelector('#mode-label').textContent.includes('MobileNetV2 (single)')")
        assert C.ACTIVE_MODE_FILE.read_text() == "mobilenetv2"

        go(page, "performance")
        page.wait_for_selector("#cmp-table tbody tr")
        hl = page.locator("#cmp-table tr.highlight")
        assert hl.count() == 1 and "MobileNetV2" in hl.inner_text()
        assert "MobileNetV2" in page.locator("#cmp-table tr", has=page.locator(".badge")).inner_text()

        go(page, "preview")
        page.set_input_files("#file-input", str(SAMPLE))
        page.click("#btn-submit")
        page.wait_for_selector("#result-body:not([hidden])", timeout=120_000)
        assert page.inner_text("#mode-badge") == "MobileNetV2 (single)"
        assert page.locator("#votes tbody tr").count() == 1

        page.click('label:has(#single-mode)')  # back to ensemble
        page.wait_for_function("!document.querySelector('#mode-label').textContent.includes('single')")
        go(page, "performance")
        page.wait_for_function("document.querySelectorAll('#cmp-table tr.highlight').length === 2")
        assert not errors, errors
    finally:
        C.ACTIVE_MODE_FILE.unlink(missing_ok=True)


def test_preview_shows_mc_dropout_confidence_flag(smoke, ui):
    """The badge, the spark-bars and the two metrics must all render on a real prediction."""
    page, errors = ui
    go(page, "preview")
    page.set_input_files("#file-input", str(SAMPLE))
    page.wait_for_selector("#preview:not([hidden])")
    page.click("#btn-submit")
    page.wait_for_selector("#result-body:not([hidden])", timeout=120_000)

    page.wait_for_selector("#conf-flag:not([hidden])", timeout=120_000)
    flag = page.get_attribute("#conf-flag", "class")
    assert any(f"flag-{lvl}" in flag for lvl in ("high", "medium", "low")), flag
    assert page.inner_text("#conf-flag").strip()

    assert page.is_visible("#uncertainty")
    assert f"{C.MC_PASSES_PREVIEW} passes" in page.inner_text("#unc-passes")
    # one spark bar per stochastic pass
    assert page.locator("#unc-spark .spark-bar").count() == C.MC_PASSES_PREVIEW
    for sel in ("#unc-entropy", "#unc-mi"):
        assert float(page.inner_text(sel)) >= 0
    assert float(page.inner_text("#unc-mi")) <= float(page.inner_text("#unc-entropy")) + 1e-6
    assert not errors, errors
