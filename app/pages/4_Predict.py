from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

import ui
from src import config as C
from src.data import list_images
from src.ensemble import load_voting
from src.predict import predict_image

UPLOAD = "Upload my own image"


def main():
    s = ui.header("Predict", "🔬")
    if not any(s["models"].values()):
        st.warning("No trained backbones in `models/` yet.")
        ui.go("train", "Train page: train the backbones")
        return
    if not (s["voting"] and s["stacking"]):
        st.info("Ensemble not fitted yet: predictions use the plain mean of the backbones.")

    samples = {f"sample: {lab} / {Path(p).name}": p for p, lab in list_images(C.SAMPLE_DIR)}
    choice = st.selectbox("Image", [UPLOAD, *samples], key="image_choice")
    img = None
    if choice == UPLOAD:
        up = st.file_uploader("Peripheral blood smear image", type=C.UPLOAD_EXTS,
                              help=f"{', '.join(C.UPLOAD_EXTS)} · max {C.MAX_UPLOAD_MB} MB")
        if up is not None:
            try:
                img = ui.load_upload(up)
            except ValueError as e:
                st.error(str(e))
    else:
        img = Image.open(samples[choice]).convert("RGB")
    if img is None:
        st.caption("Choose a bundled sample or upload an image to analyse it.")
        return

    if not st.button("🔍 Analyse", type="primary", key="analyse"):
        st.image(img, width=256)
        return
    with st.spinner("Running 3 backbones + ensemble + Grad-CAM..."):
        r = predict_image(img, with_gradcam=True)

    st.subheader(f"Prediction: {r['class']}")
    c1, c2, c3 = st.columns([1, 1, 1.3])
    c1.image(img, caption="Input", width="stretch")
    c2.image(r["gradcam"], caption=f"Grad-CAM: {C.DISPLAY_NAMES[r['gradcam_model']]} (highest ensemble weight)",
             width="stretch")
    with c3:
        st.metric("Confidence", f"{r['confidence']:.1%}")
        st.caption(f"Model: **{r['mode_label']}**")
        st.bar_chart(pd.Series(r["probs"], name="probability"), horizontal=True)

    st.subheader("Per-model votes")
    weights = (load_voting() or {}).get("weights", {})
    st.dataframe(pd.DataFrame([{
        "model": C.DISPLAY_NAMES[n], "vote": r["votes"][n], "confidence": max(p.values()),
        "voting weight": weights.get(n), **p,
    } for n, p in r["per_model_probs"].items()]).set_index("model"), width="stretch")
    st.caption("Research use only. A prediction here is not a diagnosis.")


ui.guard(main)
ui.footer()
