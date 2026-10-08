import streamlit as st

import ui
from src import config as C
from src import pipeline


def badge(ok, label, detail=""):
    st.markdown(f"{'✅' if ok else '⬜'} **{label}**  \n<small>{detail}</small>", unsafe_allow_html=True)


def main():
    s = ui.header("ALL Hybrid Ensemble")
    st.markdown("Classifies peripheral blood smear images into four Acute Lymphoblastic Leukemia classes with three "
                "fine-tuned CNNs and a **hybrid ensemble** (weighted soft voting + stacked logistic regression). "
                "Replication of Jayachitra & Umarkathaf, *ICEARS 2023*.")
    ui.flash()

    st.subheader("Pipeline status")
    data_detail = (f"{s['kaggle_images']} Kaggle images" if s["kaggle_images"]
                   else f"{s['demo_images']} demo images" if s["demo_images"] else "none")
    cols = st.columns(7)
    with cols[0]:
        badge(s["has_data"], "Data", data_detail)
    with cols[1]:
        badge(s["splits"], "Splits", (s["split_meta"] or {}).get("source", "not made"))
    for col, name in zip(cols[2:5], C.BACKBONES):
        with col:
            badge(s["models"][name], C.DISPLAY_NAMES[name], "trained" if s["models"][name] else "not trained")
    with cols[5]:
        badge(s["voting"] and s["stacking"], "Ensemble", "fitted" if s["voting"] else "not fitted")
    with cols[6]:
        badge(s["metrics"], "Evaluation", f"selected: {s['ensemble_mode']}" if s["ensemble_mode"] else "not run")
    st.info(f"**Next step:** {s['next_step']}")

    st.subheader("Run full pipeline")
    st.caption("data (download or demo) → splits → train 3 backbones → fit ensemble → evaluate")
    smoke = st.checkbox("Smoke mode: 8 bundled sample images, 1 epoch per phase, about 1 minute on CPU",
                        value=True, key="home_smoke")
    if not smoke:
        st.caption("Full mode trains at 224 px for 15 epochs. Use a GPU (see the Colab notebook); on CPU it takes hours.")
    if st.button("▶ Run full pipeline", type="primary", key="run_all"):
        bar, line = st.progress(0.0), st.empty()

        def cb(stage, epoch, logs):
            total = logs.get("epochs_total") if isinstance(logs, dict) else None
            if total:
                bar.progress(min(epoch / total, 1.0), text=f"{stage}: epoch {epoch}/{total}")
            line.caption(f"[{stage}] step {epoch}")

        try:
            with st.spinner("Running pipeline..."):
                out = pipeline.run("all", smoke=smoke, progress_cb=cb)
            st.session_state["_flash"] = ("success", "Pipeline finished.")
            st.session_state["last_run"] = out
        except Exception as e:  # noqa: BLE001
            st.session_state["_flash"] = ("error", ui.friendly(e))
        st.rerun()
    if "last_run" in st.session_state:
        with st.expander("Last pipeline output"):
            st.code(st.session_state["last_run"])

    st.subheader("Pages")
    for page, text in [("data", "Data: download or demo data, make splits"),
                       ("train", "Train: fine-tune backbones with live charts"),
                       ("evaluate", "Evaluate: fit ensemble, test metrics, figures"),
                       ("predict", "Predict: upload an image, get class + Grad-CAM")]:
        ui.go(page, text)


ui.guard(main)
ui.footer()
