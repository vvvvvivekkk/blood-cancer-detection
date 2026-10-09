"""Out-of-distribution check: does an upload look like a stained peripheral blood smear?

Two cheap HSV statistics, compared with the training split (computed once by src/evaluate.py,
stored in models/ood_stats.json):
  * mean saturation, must fall inside the training range [p_low, p_high];
  * saturation-weighted hue histogram, Hellinger distance to the mean training histogram
    must not exceed the training-set percentile.
ponytail: a colour heuristic, not a learned OOD detector; it flags photos, grayscale scans and
other stains, not a smear of a different leukaemia. Upgrade path: Mahalanobis on backbone features.
"""
import json

import numpy as np
from PIL import Image

from src import config as C


def features(img):
    """PIL image or path -> (mean saturation in [0, 1], normalised hue histogram)."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    hsv = np.asarray(img.convert("RGB").resize((C.OOD_PX, C.OOD_PX)).convert("HSV"), np.float32) / 255
    hue, sat = hsv[..., 0].ravel(), hsv[..., 1].ravel()
    hist = np.histogram(hue, bins=C.OOD_BINS, range=(0, 1), weights=sat)[0]
    total = hist.sum()
    hist = hist / total if total > 1e-6 else np.full(C.OOD_BINS, 1 / C.OOD_BINS)  # grayscale: flat
    return float(sat.mean()), hist


def hellinger(p, q):
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def compute_stats(paths):
    feats = [features(p) for p in paths]
    sats = np.array([f[0] for f in feats])
    mean_hist = np.mean([f[1] for f in feats], axis=0)
    dists = np.array([hellinger(f[1], mean_hist) for f in feats])
    lo, hi = 100 - C.OOD_PERCENTILE, C.OOD_PERCENTILE
    return {"n_images": len(feats), "mean_hist": mean_hist.tolist(),
            "saturation_range": [float(np.percentile(sats, lo)), float(np.percentile(sats, hi))],
            "hist_distance_max": float(np.percentile(dists, hi)), "percentile": C.OOD_PERCENTILE}


def write_stats(paths):
    stats = compute_stats(paths)
    C.OOD_STATS.parent.mkdir(parents=True, exist_ok=True)
    C.OOD_STATS.write_text(json.dumps(stats, indent=2))
    return stats


def load_stats():
    return json.loads(C.OOD_STATS.read_text()) if C.OOD_STATS.is_file() else None


def check(img, stats=None):
    """{ood, reasons, saturation, hist_distance}; ood is None when no training stats exist yet."""
    stats = stats or load_stats()
    sat, hist = features(img)
    if not stats:
        return {"ood": None, "reasons": [], "saturation": sat, "hist_distance": None}
    dist = hellinger(hist, np.array(stats["mean_hist"]))
    lo, hi = stats["saturation_range"]
    reasons = []
    if not lo <= sat <= hi:
        reasons.append(f"colour saturation {sat:.2f} is outside the training range {lo:.2f}–{hi:.2f}")
    if dist > stats["hist_distance_max"]:
        reasons.append(f"colour histogram distance {dist:.2f} exceeds the training maximum {stats['hist_distance_max']:.2f}")
    return {"ood": bool(reasons), "reasons": reasons, "saturation": sat, "hist_distance": dist}
