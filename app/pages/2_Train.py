import pandas as pd
import streamlit as st

import ui
from src import config as C
from src.train import train_backbone


def history_chart(name):
    f = C.HISTORY_DIR / f"{name}.csv"
    if not f.is_file():
        return False
    h = pd.read_csv(f)
    h["epoch"] = h["epoch"] + 1
    h = h.set_index("epoch")
    c1, c2 = st.columns(2)
    c1.line_chart(h[[c for c in ("loss", "val_loss") if c in h]], x_label="epoch", y_label="loss")
    c2.line_chart(h[[c for c in ("accuracy", "val_accuracy") if c in h]], x_label="epoch", y_label="accuracy")
    return True


def main():
    s = ui.header("Train", "🏋️")
    ui.flash()
    if ui.missing(s, "splits"):
        return

    st.markdown("Two-phase fine-tuning per backbone: **phase 1** trains the head with the base frozen "
                f"(lr {C.LR_HEAD}); **phase 2** unfreezes the top {C.UNFREEZE_FRACTION:.0%} of the base (lr {C.LR_FINE}). "
                "Adam, class weights, early stopping, LR-on-plateau, best checkpoint by val loss.")
    names = st.multiselect("Backbones", C.BACKBONES, default=C.BACKBONES, format_func=C.DISPLAY_NAMES.get)
    smoke = st.checkbox("Smoke mode (96 px, 1 epoch per phase, no ImageNet download)",
                        value=bool(s["smoke"]) or not ui.has_gpu(), key="train_smoke")
    epochs = st.slider("Total epochs (both phases)", 2, 50, C.EPOCHS, disabled=smoke)
    if not smoke and not ui.has_gpu():
        st.warning("No GPU detected. Full training on CPU takes hours; consider the Colab notebook.")

    if st.button("▶ Train", type="primary", disabled=not names, key="train"):
        overall = st.progress(0.0, text="starting")
        for i, name in enumerate(names):
            st.markdown(f"**{C.DISPLAY_NAMES[name]}**")
            bar, chart, rows = st.progress(0.0), st.empty(), []

            def cb(stage, epoch, logs, bar=bar, chart=chart, rows=rows):
                bar.progress(epoch / logs["epochs_total"],
                             text=f"phase {logs['phase']} · epoch {epoch}/{logs['epochs_total']} · "
                                  f"loss {logs.get('loss', 0):.4f} · val_acc {logs.get('val_accuracy', 0):.3f}")
                rows.append({k: logs.get(k) for k in ("loss", "val_loss", "accuracy", "val_accuracy")})
                chart.line_chart(pd.DataFrame(rows, index=range(1, len(rows) + 1)))

            try:
                train_backbone(name, smoke=smoke, epochs=epochs, progress_cb=cb)
            except Exception as e:  # noqa: BLE001
                st.error(ui.friendly(e))
                return
            overall.progress((i + 1) / len(names), text=f"{i + 1}/{len(names)} backbones done")
        st.success("Training finished. Next: fit the ensemble on the Evaluate page.")
        ui.go("evaluate", "Evaluate page")

    st.subheader("Training history")
    for name in C.BACKBONES:
        with st.expander(f"{C.DISPLAY_NAMES[name]}: {'trained ✅' if s['models'][name] else 'not trained'}",
                         expanded=s["models"][name]):
            if not history_chart(name):
                st.caption("No history yet. Curves appear here after training.")


ui.guard(main)
ui.footer()
