import re
from html.parser import HTMLParser

from src import config as C

FRONT = C.REPO / "frontend"
SERVED_BY_API = ("/api/", "/media/", "/figures/", "/docs-assets/")


class Refs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets, self.links = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "img") and a.get("src"):
            self.assets.append(a["src"])
        if tag == "link" and a.get("href"):
            self.assets.append(a["href"])
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])


def refs():
    p = Refs()
    p.feed((FRONT / "index.html").read_text(encoding="utf-8"))
    return p


def test_index_references_only_existing_files():
    p = refs()
    local = [u for u in p.assets + p.links
             if not u.startswith(("http://", "https://", "#", "mailto:")) and not u.startswith(SERVED_BY_API)]
    assert local, "expected local css/js references"
    missing = [u for u in local if not (FRONT / u.lstrip("/")).is_file()]
    assert not missing, missing
    # ES module imports inside the JS must resolve too
    for js in (FRONT / "js").glob("*.js"):
        for imp in re.findall(r'from\s+"(\./[^"]+)"', js.read_text(encoding="utf-8")):
            assert (js.parent / imp).is_file(), f"{js.name} imports missing {imp}"


def test_cdn_assets_pinned_to_a_version():
    external = [u for u in refs().assets if u.startswith(("http://", "https://"))]
    assert external, "Chart.js CDN script expected"
    for u in external:
        assert u.startswith("https://"), u
        assert re.search(r"@\d+\.\d+\.\d+/", u), f"unpinned CDN asset: {u}"


def test_no_unused_css_variables():
    css = (FRONT / "css" / "styles.css").read_text(encoding="utf-8")
    js = "".join(p.read_text(encoding="utf-8") for p in (FRONT / "js").glob("*.js"))
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    used = set(re.findall(r"var\((--[\w-]+)", css + js)) | set(re.findall(r'"(--[\w-]+)"', js))
    assert defined and not (defined - used), sorted(defined - used)
    assert not (used - defined), f"used but never defined: {sorted(used - defined)}"


def test_every_view_has_a_nav_link_and_footer_disclaimer():
    html = (FRONT / "index.html").read_text(encoding="utf-8")
    views = re.findall(r'data-view="(\w+)"', html)
    navs = re.findall(r'data-nav="(\w+)"', html)
    assert navs == ["home", "preview", "performance", "chart", "about", "data", "train"]  # walkthrough order
    assert sorted(views) == sorted(navs)
    assert "Not for clinical use" in html
