import json

import pandas as pd
import streamlit as st

import ui
from src import config as C
from src import pipeline

KEYS = C.BACKBONES + C.ENSEMBLES


def figure(name, caption=None):
    p = C.FIGURES_DIR / name
    if p.is_file():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.caption(f"`{name}` not generated yet.")


def main():
    s = ui.header("Evaluate", "📊")
    ui.flash()
    if ui.missing(s, "ensemble"):
        return

    c1, c2 = st.columns(2)
    with c1:
        ui.action("Re-fit ensemble on val split", lambda: pipeline.run("ensemble"), key="fit_ens")
    with c2:
        ui.action("▶ Run evaluation on test split", lambda: pipeline.run("evaluate"), key="run_eval", type="primary")
    if ui.missing(s, "metrics"):
        return

    m = json.loads(C.METRICS_JSON.read_text())
    best = m["best_ensemble"]
    meta = m["meta"]
    st.caption(f"Test split: {meta['n_test']} images · source **{meta['source']}** · generated {meta['generated_at']}")
    st.success(f"Selected ensemble for Predict: **{C.DISPLAY_NAMES[best]}** (higher test macro-F1)")

    st.subheader("Comparison table")
    table = pd.DataFrame({C.DISPLAY_NAMES[k]: {
        "accuracy": v["accuracy"], "macro P": v["macro_precision"], "macro R": v["macro_recall"],
        "macro F1": v["macro_f1"], "weighted F1": v["weighted_f1"], "kappa": v["kappa"], "ROC-AUC": v["roc_auc_ovr"],
    } for k, v in m["models"].items()}).T
    st.dataframe(table.style.format("{:.4f}", na_rep="–").highlight_max(axis=0, color="#c8e6c9"), width="stretch")
    st.caption("Voting weights (∝ val macro-F1): " + ", ".join(
        f"{C.DISPLAY_NAMES[k]} {w:.3f}" for k, w in meta["voting_weights"].items()))
    notes = C.RESULTS_DIR / "NOTES.md"
    if notes.is_file():
        with st.expander("Target check (ENDGOAL.md)"):
            st.markdown(notes.read_text(encoding="utf-8"))
    figure("model_comparison_bar.png")

    st.subheader("Per-class metrics")
    key = st.selectbox("Model", [k for k in KEYS if k in m["models"]], index=KEYS.index(best),
                       format_func=C.DISPLAY_NAMES.get)
    st.dataframe(pd.DataFrame(m["models"][key]["per_class"]).T.style.format("{:.3f}"), width="stretch")

    st.subheader("Confusion matrices")
    cols = st.columns(len(KEYS))
    for col, k in zip(cols, KEYS):
        with col:
            figure(f"cm_{k}.png")

    st.subheader("ROC (one-vs-rest) and Grad-CAM")
    c1, c2 = st.columns([1, 2])
    with c1:
        figure("roc_ensemble.png", f"{C.DISPLAY_NAMES[best]}")
    with c2:
        figure("gradcam_examples.png", "Row 1: correct · Row 2: misclassified")

    st.subheader("Training curves")
    for n in C.BACKBONES:
        figure(f"training_curves_{n}.png")


ui.guard(main)
ui.footer()
