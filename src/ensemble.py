"""Hybrid ensemble: weighted soft voting + stacked logistic regression, both fit on the val split.

python -m src.ensemble
"""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src import config as C
from src.data import make_dataset, load_split
from src.models import preprocess_fn

_CACHE = {}


def load_models(names=None):
    """{name: model} for every trained backbone on disk (cached by path + mtime)."""
    from tensorflow import keras
    out = {}
    for n in names or C.BACKBONES:
        p = C.model_path(n)
        if p.is_file():
            key = (str(p), p.stat().st_mtime)
            if key not in _CACHE:
                _CACHE[key] = keras.models.load_model(str(p), compile=False)
            out[n] = _CACHE[key]
    return out


def img_size(model):
    return int(model.input_shape[1])


def predict_probs(models, df):
    """{name: (N, 4) softmax} for a split dataframe. Eager calls: no graph tracing per model."""
    out = {}
    for n, m in models.items():
        ds = make_dataset(df, preprocess_fn(n), img_size(m), C.BATCH_SIZE)
        out[n] = np.concatenate([np.asarray(m(x, training=False)) for x, _ in ds])
    return out


def voting_probs(probs, weights):
    names = [n for n in weights if n in probs]
    w = np.array([weights[n] for n in names], dtype=float)
    return sum(wi * probs[n] for wi, n in zip(w / w.sum(), names))


def stacking_probs(bundle, probs):
    X = np.concatenate([probs[n] for n in bundle["models"]], axis=1)
    out = np.zeros((len(X), C.NUM_CLASSES))
    out[:, bundle["clf"].classes_] = bundle["clf"].predict_proba(X)
    return out


def macro_f1(y, p):
    return float(f1_score(y, p.argmax(1), average="macro", labels=list(range(C.NUM_CLASSES)), zero_division=0))


def fit_voting(probs, y):
    """Weights proportional to val macro-F1, normalised to sum to 1 (uniform if all zero)."""
    f1 = {n: macro_f1(y, p) for n, p in probs.items()}
    total = sum(f1.values())
    weights = {n: (v / total if total else 1 / len(f1)) for n, v in f1.items()}
    info = {"models": list(probs), "weights": weights, "val_macro_f1": f1}
    C.WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.WEIGHTS_JSON.write_text(json.dumps(info, indent=2))
    return info


def fit_stacking(probs, y):
    """Multinomial logistic regression on the concatenated (N, 4 * n_models) val probabilities."""
    names = list(probs)
    X = np.concatenate([probs[n] for n in names], axis=1)
    # lbfgs is multinomial by default; the multi_class arg is deprecated in sklearn >= 1.5
    clf = LogisticRegression(C=C.META_C, max_iter=C.META_MAX_ITER, random_state=C.SEED).fit(X, y)
    bundle = {"models": names, "clf": clf}
    C.META_CLF.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, C.META_CLF)
    return bundle


def load_voting():
    return json.loads(C.WEIGHTS_JSON.read_text()) if C.WEIGHTS_JSON.is_file() else None


def load_stacking():
    return joblib.load(C.META_CLF) if C.META_CLF.is_file() else None


def fit_all(progress_cb=None):
    models = load_models()
    if not models:
        raise FileNotFoundError("No trained backbones in models/. Train them first.")
    val = load_split("val")
    y = val["label_idx"].values
    probs = predict_probs(models, val)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.RESULTS_DIR / "val_probs.npy", np.stack([probs[n] for n in models]))
    np.save(C.RESULTS_DIR / "val_labels.npy", y)
    info = fit_voting(probs, y)
    fit_stacking(probs, y)
    if progress_cb:
        progress_cb("ensemble", 1, {"val_macro_f1": info["val_macro_f1"], "weights": info["weights"]})
    return info


if __name__ == "__main__":
    print(json.dumps(fit_all(), indent=2))
