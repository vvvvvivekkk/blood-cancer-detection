import io
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import ui
from src import config as C

HOME = str(C.REPO / "app" / "Home.py")
PAGES = [None, "pages/1_Data.py", "pages/2_Train.py", "pages/3_Evaluate.py", "pages/4_Predict.py", "pages/5_About.py"]


def render(page):
    at = AppTest.from_file(HOME, default_timeout=180)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    return at


def text(at):
    return " ".join(e.value for kind in ("markdown", "caption", "warning", "info", "success", "subheader", "title")
                    for e in getattr(at, kind) if isinstance(e.value, str))


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert "Not for clinical use" in text(at)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_nothing_on_disk(empty_root, page):
    at = render(page)
    assert_clean(at)
    if page in ("pages/2_Train.py", "pages/3_Evaluate.py", "pages/4_Predict.py"):
        assert at.warning, "should say what is missing"


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_smoke_artifacts(smoke, page):
    at = render(page)
    assert_clean(at)
    assert "SMOKE" in text(at)


def test_evaluate_shows_comparison_table(smoke):
    at = render("pages/3_Evaluate.py")
    assert at.dataframe and "Selected ensemble" in text(at)


def test_predict_sample_image_shows_class(smoke):
    at = render("pages/4_Predict.py")
    at.selectbox(key="image_choice").select_index(1).run()
    at.button(key="analyse").click().run()
    assert_clean(at)
    assert any(f"Prediction: {c}" in text(at) for c in C.CLASS_NAMES)
    assert at.metric[0].label == "Confidence"


def test_demo_banner_after_demo_data(empty_root):
    at = render("pages/1_Data.py")
    next(b for b in at.button if b.label == "Use demo data").click().run()
    assert_clean(at)
    assert "DEMO DATA" in text(at)
    next(b for b in at.button if b.label == "Make splits").click().run()
    assert_clean(at)
    assert "DEMO DATA" in text(at) and at.dataframe  # split counts table


def _upload(name, data):
    return SimpleNamespace(name=name, getvalue=lambda: data)


def test_upload_validation():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    assert ui.load_upload(_upload("a.png", buf.getvalue())).size == (8, 8)
    with pytest.raises(ValueError, match="Unsupported"):
        ui.load_upload(_upload("a.gif", buf.getvalue()))
    with pytest.raises(ValueError, match="corrupt"):
        ui.load_upload(_upload("a.jpg", b"not an image"))
    with pytest.raises(ValueError, match="MB"):
        ui.load_upload(_upload("a.png", b"0" * (C.MAX_UPLOAD_MB * 1024 * 1024 + 1)))


def test_train_button_runs_with_live_progress(empty_root):
    from src.data import make_splits
    make_splits(smoke=True)
    at = render("pages/2_Train.py")
    at.multiselect[0].set_value(["mobilenetv2"]).run()
    at.checkbox(key="train_smoke").check().run()
    at.button(key="train").click().run()
    assert_clean(at)
    assert C.model_path("mobilenetv2").is_file()
    assert "Training finished" in text(at)
