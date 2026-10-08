"""Regenerate docs/architecture.png: python docs/make_assets.py (screenshots: docs/capture_screenshots.py)"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DOCS = Path(__file__).resolve().parent


def box(ax, x, y, w, h, text, colour):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", fc=colour, ec="#333"))
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#333"))


def architecture():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set(xlim=(0, 13), ylim=(0, 5))
    ax.axis("off")
    box(ax, 1.15, 2.5, 2.1, 1.0, "PBS image\n224×224 RGB\n+ backbone preprocess", "#fde2e4")
    for y, name in ((4.0, "MobileNetV2"), (2.5, "EfficientNetB0"), (1.0, "DenseNet121")):
        box(ax, 4.0, y, 2.6, 0.9, f"{name}\nImageNet → GAP → Dropout\n→ Dense(4, softmax)", "#dbe7f5")
        arrow(ax, (2.25, 2.5), (2.7, y))
        arrow(ax, (5.35, y), (6.0, 2.5))
    box(ax, 6.7, 2.5, 1.4, 1.0, "3 × softmax\n(12 probs)", "#eeeeee")
    box(ax, 8.9, 3.6, 2.3, 1.0, "Weighted soft voting\nw ∝ val macro-F1", "#e2f0d9")
    box(ax, 8.9, 1.4, 2.3, 1.0, "Stacked meta-classifier\nLogReg on 12 probs", "#e2f0d9")
    arrow(ax, (7.4, 2.7), (7.75, 3.5))
    arrow(ax, (7.4, 2.3), (7.75, 1.5))
    box(ax, 11.6, 2.5, 2.2, 1.2, "Better test macro-F1\n→ class + confidence\n+ Grad-CAM", "#fff2cc")
    arrow(ax, (10.05, 3.6), (10.5, 2.8))
    arrow(ax, (10.05, 1.4), (10.5, 2.2))
    ax.text(6.5, 0.15, "Fine-tuning: phase 1 frozen base (lr 1e-3) → phase 2 top 30 % unfrozen (lr 1e-5) · "
            "ensembles fit on val only · test split read once by src/evaluate.py", ha="center", fontsize=8.5)
    ax.set_title("Hybrid ensemble for 4-class ALL classification", fontsize=12)
    fig.savefig(DOCS / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
