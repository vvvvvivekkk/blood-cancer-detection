"""Examiner-flow audit against a running app: every check from docs/ui_walkthrough.md plus edge cases.

    python app.py --no-browser &
    python scripts/audit_walkthrough.py [--base http://localhost:8000]

Writes docs/audit/results.json and docs/audit/<check>.png for every failing check (and a few
reference shots). Exit code 1 if any check fails. Uses the real CDN (the examiner's machine does too).
"""
import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OUT = DOCS / "audit"
sys.path.insert(0, str(REPO))
from src.config import CLASS_NAMES  # noqa: E402

VIEWS = ["home", "preview", "performance", "chart", "about", "data", "train"]
# A view has finished its async render when one of these exists (content or its empty state).
READY = {
    "home": "#stepper [data-step]",
    "preview": "#test-image-select option:nth-child(2), #test-image-select:disabled, #predict-empty .empty",
    "performance": "#perf-tiles .kpi, #eval-empty .empty",
    "chart": "#chart-tabs .tab, #chart-empty .empty",
    "about": "#mapping tbody tr",
    "data": "#chart-counts, #counts-empty",
    "train": "#backbone-checks input, #train-empty .empty",
}
results = []


class Watch:
    """Console errors, failed requests and 4xx/5xx responses since the last reset."""

    def __init__(self, page, expected=()):
        self.expected, self.items = expected, []
        page.on("console", lambda m: m.type == "error" and self.items.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: self.items.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: self.items.append(f"failed: {r.url} {r.failure}"))
        page.on("response", lambda r: r.status >= 400 and self.items.append(f"HTTP {r.status}: {r.url}"))

    def take(self, allow=()):
        out = [i for i in self.items if not any(a in i for a in allow)]
        self.items = []
        return out


def check(name, page, ok, detail="", shot=False):
    status = "PASS" if ok else "FAIL"
    entry = {"check": name, "status": status, "detail": detail}
    if not ok or shot:
        OUT.mkdir(exist_ok=True)
        f = OUT / f"{name}.png"
        page.screenshot(path=str(f), full_page=True)
        entry["screenshot"] = f"audit/{f.name}"
    results.append(entry)
    print(f"[{status}] {name} {detail}", flush=True)


def broken_images(page):
    return page.evaluate("""() => [...document.querySelectorAll('img')]
        .filter(i => i.offsetParent && i.getAttribute('src') && i.complete && i.naturalWidth === 0)
        .map(i => i.getAttribute('src').slice(0, 80))""")


def visible_text(page, view):
    return page.locator(f'[data-view="{view}"]').inner_text().strip()


def go(page, view):
    if view in ("data", "train") and page.get_attribute("#pipeline-toggle", "aria-expanded") != "true":
        page.click("#pipeline-toggle")
    page.click(f'[data-nav="{view}"]')
    page.locator(f'[data-view="{view}"]').wait_for(state="visible")
    page.locator(f'[data-view="{view}"]').locator(READY[view]).first.wait_for(state="attached", timeout=30_000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(600)  # chart animation


def login(page, base, user="admin", pw="admin"):
    page.goto(f"{base}/#/home")
    page.wait_for_selector("#login-screen:not([hidden])")
    page.fill("#login-user", user)
    page.fill("#login-pass", pw)
    page.click(".login-submit")


def view_check(page, w, view, min_chars=40):
    go(page, view)
    text, broken, errs = visible_text(page, view), broken_images(page), w.take()
    ok = len(text) >= min_chars and not broken and not errs
    check(f"view_{view}", page, ok, f"{len(text)} chars; broken={broken}; errors={errs}", shot=True)


def predict_flow(page, w, base):
    go(page, "preview")
    opts = page.evaluate("""() => [...document.querySelectorAll('#test-image-select option')]
        .filter(o => o.value.startsWith('/media/test/')).map(o => o.value)""")
    for cls in CLASS_NAMES:
        name = "preview_" + cls.replace("[Malignant] ", "").replace(" ", "_").replace("-", "")
        pick = next((o for o in opts if f"/media/test/{cls}/" in o.replace("%20", " ").replace("%5B", "[").replace("%5D", "]")), None)
        if not pick:
            check(name, page, False, f"no test image for {cls} in the picker ({len(opts)} options)")
            continue
        page.select_option("#test-image-select", pick)
        page.wait_for_selector("#preview:not([hidden])")
        page.wait_for_selector("#btn-submit:not([disabled])")
        page.click("#btn-submit")
        try:
            page.wait_for_function("""() => !document.querySelector('#result-body').hidden
                || document.querySelector('#predict-error').textContent.trim().length > 0""", timeout=180_000)
        except Exception as e:  # noqa: BLE001
            check(name, page, False, f"timeout: {e}")
            continue
        err = page.inner_text("#predict-error").strip()
        if err:
            check(name, page, False, f"error shown: {err}; {w.take()}")
            continue
        got = page.inner_text("#result-class")
        cam = (page.get_attribute("#img-cam", "src") or "").startswith("data:image/png")
        votes = page.locator("#votes tbody tr").count()
        errs = w.take()
        check(name, page, got in CLASS_NAMES and cam and not errs,
              f"true={cls} predicted={got} {'(correct)' if got == cls else '(MISCLASSIFIED)'} gradcam={cam} votes={votes} errors={errs}", shot=True)


def upload_errors(page, w):
    go(page, "preview")
    cases = {
        "upload_non_image_txt": {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"hello"},
        "upload_oversize_11mb": {"name": "big.png", "mimeType": "image/png", "buffer": b"\x89PNG" + b"0" * (11 * 1024 * 1024)},
        "upload_corrupt_png": {"name": "bad.png", "mimeType": "image/png", "buffer": b"\x89PNGnotreally"},
    }
    for name, f in cases.items():
        page.evaluate("document.querySelector('#predict-error').textContent = ''")
        page.set_input_files("#file-input", f)
        if not page.is_disabled("#btn-submit"):
            page.click("#btn-submit")
        try:
            page.wait_for_function("document.querySelector('#predict-error').textContent.trim().length > 0", timeout=60_000)
        except Exception:  # noqa: BLE001
            pass
        msg = page.inner_text("#predict-error").strip()
        errs = w.take(allow=("HTTP 400", "400 (Bad Request)"))
        ok = bool(msg) and "Traceback" not in msg and not errs
        check(name, page, ok, f"message={msg!r}; errors={errs}", shot=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    base = ap.parse_args().base
    with sync_playwright() as p:
        browser = p.chromium.launch()

        # --- desktop walkthrough ---
        ctx = browser.new_context(viewport={"width": 1440, "height": 960})
        page = ctx.new_page()
        w = Watch(page)
        login(page, base, pw="wrong")
        page.wait_for_function("document.querySelector('#login-error').textContent.length > 0", timeout=10_000)
        check("login_wrong_password", page, page.is_visible("#login-screen") and not page.is_visible("#app"),
              page.inner_text("#login-error"), shot=True)
        w.take(allow=("401",))
        page.fill("#login-pass", "admin")
        page.click(".login-submit")
        page.locator('[data-view="home"]').wait_for(state="visible")
        page.wait_for_load_state("networkidle")
        check("login_admin", page, "Login successful" in page.inner_text("#toast"), page.inner_text("#toast"))
        for v in ("home", "performance", "chart", "about", "data", "train"):
            view_check(page, w, v)
        predict_flow(page, w, base)
        upload_errors(page, w)

        go(page, "chart")
        page.reload()
        page.wait_for_load_state("networkidle")
        check("refresh_mid_session", page,
              page.is_visible('[data-view="chart"]') and not page.is_visible("#login-screen"), page.url)
        w.take()

        page.click("#theme-toggle")
        page.wait_for_timeout(300)
        dark = page.get_attribute("html", "data-theme") == "dark"
        for v in ("home", "performance", "chart", "preview"):
            go(page, v)
            OUT.mkdir(exist_ok=True)
            page.screenshot(path=str(OUT / f"dark_{v}.png"), full_page=True)
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        check("dark_mode", page, dark and not w.take(), f"theme={page.get_attribute('html', 'data-theme')} body={bg}")
        page.click("#theme-toggle")

        page.click("#logout")
        page.locator("#login-screen").wait_for(state="visible")
        page.goto(f"{base}/#/performance")
        page.wait_for_load_state("networkidle")
        check("logout_and_deep_link_gated", page, page.is_visible("#login-screen") and not page.is_visible("#app"), page.url)
        ctx.close()

        # --- mobile 390px ---
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        page = ctx.new_page()
        w = Watch(page)
        login(page, base)
        page.locator('[data-view="home"]').locator(READY["home"]).first.wait_for(state="attached")
        page.wait_for_load_state("networkidle")
        overflow = []
        for v in VIEWS:
            page.click("#menu-btn")
            page.wait_for_selector("#sidebar.open")
            if v in ("data", "train") and page.get_attribute("#pipeline-toggle", "aria-expanded") != "true":
                page.click("#pipeline-toggle")
            page.click(f'[data-nav="{v}"]')
            page.locator(f'[data-view="{v}"]').locator(READY[v]).first.wait_for(state="attached", timeout=30_000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(600)
            if page.get_attribute("#menu-btn", "aria-expanded") == "true":
                overflow.append(f"{v}: drawer still open after navigating")
            if not page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"):
                overflow.append(v)
            page.screenshot(path=str(OUT / f"mobile_{v}.png"), full_page=True)
        check("mobile_390_no_horizontal_scroll", page, not overflow and not w.take(), f"overflowing views={overflow}")
        ctx.close()
        browser.close()

    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    fails = [r for r in results if r["status"] == "FAIL"]
    print(f"{len(results) - len(fails)}/{len(results)} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
