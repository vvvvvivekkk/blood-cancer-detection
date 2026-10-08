"""Shared Streamlit helpers: banners, footer, guarded actions, upload validation."""
import io
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src import config as C  # noqa: E402
from src.status import status  # noqa: E402

log = logging.getLogger("app")
DISCLAIMER = "⚠️ Research / educational project. Not a medical device. **Not for clinical use.**"
PAGES = {"data": "pages/1_Data.py", "train": "pages/2_Train.py", "evaluate": "pages/3_Evaluate.py",
         "predict": "pages/4_Predict.py"}


def header(title, icon="🩸"):
    st.set_page_config(page_title=f"{title} | ALL hybrid ensemble", page_icon=icon, layout="wide")
    s = status()
    st.title(f"{icon} {title}")
    if s["demo"]:
        st.warning("**DEMO DATA**: synthetic coloured shapes, not blood cells. Metrics and predictions mean nothing "
                   "about leukemia. Download the Kaggle dataset on the Data page for real results.")
    if s["smoke"]:
        st.info("**SMOKE artifacts**: built from 8 sample images, 1 epoch per phase, untrained weights. "
                "This only shows the pipeline works.")
    return s


def footer():
    st.divider()
    st.caption(DISCLAIMER)


def friendly(e):
    log.exception("action failed")  # full traceback goes to the console, never to the page
    if isinstance(e, (FileNotFoundError, ValueError)):
        return str(e)
    return f"Something went wrong ({type(e).__name__}): {e}"


def guard(main):
    """Run a page body; any exception becomes an st.error message, never a traceback."""
    try:
        main()
    except Exception as e:  # noqa: BLE001  (StopException/RerunException are BaseException)
        st.error(friendly(e))


def flash():
    msg = st.session_state.pop("_flash", None)
    if msg:
        getattr(st, msg[0])(msg[1])


def action(label, fn, key, spinner=None, rerun=True, **button_kwargs):
    """Button that runs fn(); result or error is flashed after a rerun (so status badges refresh)."""
    if not st.button(label, key=key, **button_kwargs):
        return
    with st.spinner(spinner or f"{label}..."):
        try:
            result = fn()
            st.session_state["_flash"] = ("success", f"Done: {result}" if isinstance(result, str) else "Done.")
        except Exception as e:  # noqa: BLE001
            st.session_state["_flash"] = ("error", friendly(e))
    if rerun:
        st.rerun()
    flash()


def go(page, label):
    st.page_link(PAGES[page], label=label, icon="➡️")


def missing(s, need):
    """Show what is missing for this page and the control that fixes it. Returns True if something is missing."""
    from src import pipeline
    if need in ("splits", "models", "ensemble", "metrics") and not s["splits"]:
        if not s["has_data"]:
            st.warning("No dataset yet. Download it from Kaggle (Data page) or generate demo data.")
            action("Use demo data (200 synthetic images)", lambda: pipeline.run("demo-data"), key=f"fix_demo_{need}")
            go("data", "Data page: Kaggle download")
        else:
            st.warning("Dataset found but no train/val/test splits yet.")
            action("Make splits", lambda: pipeline.run("splits"), key=f"fix_splits_{need}")
        return True
    if need in ("models", "ensemble", "metrics") and not any(s["models"].values()):
        st.warning("No trained backbones in `models/` yet.")
        go("train", "Train page: train the backbones")
        return True
    if need in ("ensemble", "metrics") and not (s["voting"] and s["stacking"]):
        st.warning("The ensemble has not been fitted yet.")
        action("Fit ensemble on val split", lambda: pipeline.run("ensemble"), key=f"fix_ens_{need}")
        return True
    if need == "metrics" and not s["metrics"]:
        st.warning("No evaluation results yet.")
        action("Run evaluation on test split", lambda: pipeline.run("evaluate"), key=f"fix_eval_{need}")
        return True
    return False


def load_upload(uploaded):
    """Validate a Streamlit UploadedFile and return an RGB PIL image, or raise ValueError."""
    ext = Path(uploaded.name).suffix.lower().lstrip(".")
    if ext not in C.UPLOAD_EXTS:
        raise ValueError(f"Unsupported file type .{ext}. Use one of: {', '.join(C.UPLOAD_EXTS)}.")
    data = uploaded.getvalue()
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {C.MAX_UPLOAD_MB} MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise ValueError("Could not read the image. The file may be corrupt.") from e


def has_gpu():
    import tensorflow as tf
    return bool(tf.config.list_physical_devices("GPU"))
