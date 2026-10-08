import pandas as pd
import streamlit as st

import ui
from src import config as C
from src import pipeline
from src.data import data_source, list_images
from src.download import save_credentials


def main():
    s = ui.header("Data", "🗂️")
    ui.flash()

    st.subheader("1 · Kaggle dataset")
    st.markdown(f"[`{C.KAGGLE_DATASET}`](https://www.kaggle.com/datasets/{C.KAGGLE_DATASET}): 3,242 images in 4 classes, "
                "downloaded with kagglehub into `data/raw/`.")
    if s["credentials"]:
        st.success("Kaggle credentials found (env vars or `~/.kaggle/kaggle.json`).")
    with st.form("kaggle_creds"):
        st.caption(f"Saved to `{C.KAGGLE_JSON}`. Create a key at kaggle.com → Settings → API → Create New Token.")
        user = st.text_input("Kaggle username")
        key = st.text_input("Kaggle API key", type="password")
        if st.form_submit_button("Save credentials"):
            try:
                st.session_state["_flash"] = ("success", f"Saved to {save_credentials(user, key)}")
            except Exception as e:  # noqa: BLE001
                st.session_state["_flash"] = ("error", ui.friendly(e))
            st.rerun()
    ui.action("⬇ Download dataset from Kaggle", lambda: pipeline.run("download"), key="download",
              spinner="Downloading (a few minutes)...", disabled=not s["credentials"])
    if s["kaggle_images"]:
        st.success(f"{s['kaggle_images']} Kaggle images in `data/raw/`.")

    st.subheader("2 · No Kaggle account? Demo data")
    st.caption(f"{C.DEMO_PER_CLASS * C.NUM_CLASSES} synthetic images (one colour and shape per class) in `data/demo/`, so the "
               "pipeline can run end to end. Kaggle data takes priority when both exist.")
    ui.action("Use demo data", lambda: pipeline.run("demo-data"), key="demo",
              type="secondary" if s["credentials"] else "primary")
    if s["demo_images"]:
        st.info(f"{s['demo_images']} demo images in `data/demo/`.")

    st.subheader("3 · Train / val / test splits")
    src = data_source()
    smoke = st.checkbox("Smoke splits from the 8 bundled sample images", value=False, key="split_smoke")
    if not (src or smoke):
        st.warning("No dataset yet: download it or use demo data above (or tick smoke splits).")
    else:
        st.caption(f"Source: **{'sample' if smoke else src[0]}**. 70/15/15 stratified, seed {C.SEED}.")
    ui.action("Make splits", lambda: pipeline.run("splits", smoke=smoke), key="splits", disabled=not (src or smoke))

    counts = C.RESULTS_DIR / "split_counts.csv"
    if s["splits"] and counts.is_file():
        df = pd.read_csv(counts, index_col="class")
        st.dataframe(df.assign(total=df.sum(axis=1)), width="stretch")
        st.bar_chart(df)

    if src:
        st.subheader("Examples")
        items = list_images(src[1])
        cols = st.columns(C.NUM_CLASSES)
        for col, name in zip(cols, C.CLASS_NAMES):
            paths = [p for p, lab in items if lab == name][:2]
            with col:
                st.caption(name)
                for p in paths:
                    st.image(p, width="stretch")


ui.guard(main)
ui.footer()
