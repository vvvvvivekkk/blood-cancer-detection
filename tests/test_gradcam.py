import numpy as np

from src import config as C
from src import gradcam
from src.models import GRADCAM_LAYER, build_backbone, preprocess_fn


def test_heatmap_and_overlay_shapes():
    s = C.SMOKE_IMG_SIZE
    m = build_backbone("mobilenetv2", weights=None, img_size=s)
    raw = np.random.default_rng(0).uniform(0, 255, (s, s, 3)).astype("float32")
    h = gradcam.make_heatmap(m, preprocess_fn("mobilenetv2")(raw[None].copy()), GRADCAM_LAYER["mobilenetv2"])
    assert h.ndim == 2 and h.min() >= 0 and h.max() <= 1 + 1e-6
    out = gradcam.overlay(raw, h)
    assert out.shape == (s, s, 3) and out.dtype == np.uint8


def test_overlay_blends_toward_heatmap():
    img = np.zeros((10, 20, 3), np.uint8)
    hot = gradcam.overlay(img, np.ones((2, 2)), alpha=0.5)
    assert hot.shape == (10, 20, 3) and hot.max() > 0  # colour added onto black
    assert (gradcam.overlay(img, np.ones((2, 2)), alpha=0.0) == 0).all()
