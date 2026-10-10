"""Single-image inference with the chosen ensemble, or MobileNetV2 alone in single-model mode."""
import numpy as np

from src import config as C
from src.data import load_image_array
from src.ensemble import img_size, load_models, load_stacking, load_voting, stacking_probs, voting_probs
from src.models import preprocess_fn


def ensemble_mode():
    mode = C.MODE_FILE.read_text().strip() if C.MODE_FILE.is_file() else C.DEFAULT_ENSEMBLE_MODE
    return mode if mode in C.ENSEMBLES else C.DEFAULT_ENSEMBLE_MODE


def mode_label(mode):
    """Badge text: 'MobileNetV2 (single)', 'Ensemble-Voting', 'Ensemble-Stacking' or 'Mean of backbones'."""
    if mode == "single":
        return f"{C.DISPLAY_NAMES[C.SINGLE_MODEL]} (single)"
    return C.DISPLAY_NAMES.get(mode, "Mean of backbones")


def _as_dict(p):
    return {c: float(v) for c, v in zip(C.CLASS_NAMES, p)}


def predict_image(img, with_gradcam=False, with_uncertainty=True, mc_passes=None):
    """img: path, PIL image or uint8 array.

    Returns {class, class_idx, confidence, probs, per_model_probs, votes, mode, mode_label, active_mode,
    gradcam_model[, gradcam][, uncertainty]}. mode is 'single' when models/active_mode.txt selects
    MobileNetV2 alone, else 'stacking' / 'voting' (from models/ensemble_mode.txt) or 'mean' when no
    ensemble is fitted.

    with_uncertainty adds MC Dropout entropy / mutual information / flag. Preview uses
    config.MC_PASSES_PREVIEW (10) rather than the full MC_PASSES (30) so the page stays responsive;
    evaluate and Batch Test use the full count. The reported `passes` says which was used.
    """
    active = C.get_active_mode()
    single = active == C.SINGLE_MODEL
    models = load_models([C.SINGLE_MODEL] if single else None)
    if single and not models:
        raise FileNotFoundError(f"Single-model mode needs models/{C.SINGLE_MODEL}.keras. Train "
                                f"{C.DISPLAY_NAMES[C.SINGLE_MODEL]} first, or switch back to ensemble mode.")
    if not models:
        raise FileNotFoundError("No trained models found. Train the backbones first (Train page).")
    arrays, per = {}, {}
    for n, m in models.items():
        x = preprocess_fn(n)(load_image_array(img, img_size(m))[None].copy())
        arrays[n] = x
        per[n] = np.asarray(m(x, training=False))  # (1, 4)

    voting, stacking = load_voting(), load_stacking()
    weights = voting["weights"] if voting else {n: 1.0 for n in models}
    mode = ensemble_mode()
    if single:
        mode, p = "single", per[C.SINGLE_MODEL][0]
    elif mode == "stacking" and stacking and all(n in per for n in stacking["models"]):
        p = stacking_probs(stacking, per)[0]
    elif voting and any(n in per for n in weights):
        mode, p = "voting", voting_probs(per, weights)[0]
    else:
        mode, p = "mean", np.mean([v[0] for v in per.values()], axis=0)

    idx = int(np.argmax(p))
    cam_model = max((n for n in per), key=lambda n: weights.get(n, 0))
    out = {
        "class": C.CLASS_NAMES[idx],
        "class_idx": idx,
        "confidence": float(p[idx]),
        "probs": _as_dict(p),
        "per_model_probs": {n: _as_dict(v[0]) for n, v in per.items()},
        "votes": {n: C.CLASS_NAMES[int(np.argmax(v[0]))] for n, v in per.items()},
        "mode": mode,
        "mode_label": mode_label(mode),
        "active_mode": active,
        "gradcam_model": cam_model,
    }
    if with_uncertainty:
        from src.uncertainty import uncertainty_for_image
        u = uncertainty_for_image(img, models=models, T=mc_passes)
        u.pop("mean_probs", None)  # the reported probs stay the deterministic ones, for consistency
        out["uncertainty"] = u

    if with_gradcam:
        from src.gradcam import gradcam_overlay
        m = models[cam_model]
        out["gradcam"] = gradcam_overlay(m, cam_model, arrays[cam_model], load_image_array(img, img_size(m)), idx)
    return out


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(predict_image(sys.argv[1]), indent=2))
