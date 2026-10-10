"""MC Dropout uncertainty: how much the ensemble's answer moves when dropout is left on.

A single softmax score says nothing about whether the model would answer the same way twice.
Running the network T times with `head_dropout` still active samples T slightly different
sub-networks; if their answers scatter, the prediction is unreliable.

Two numbers come out of the T passes:

    predictive entropy  H  = -sum_c p_bar(c) log p_bar(c)      total uncertainty
    mutual information  MI = H - mean_t[ -sum_c p_t(c) log p_t(c) ]   the part caused by
                                                               model disagreement (epistemic)

Both are divided by log(NUM_CLASSES) so they land in [0, 1] and are comparable across runs.
H is high both when the classes genuinely overlap and when the model is unsure; MI isolates the
second case, so MI near 0 with H high means "these classes look alike", not "the model is guessing".

Thresholds come from `calibrate()` on the VALIDATION split only, never test.

No FastAPI, no I/O beyond models/ — this module is importable from the CLI, the API and tests.
"""
import json

import numpy as np

from src import config as C
from src.data import load_image_array, make_dataset
from src.ensemble import img_size, load_models, load_stacking, load_voting, stacking_probs, voting_probs
from src.models import preprocess_fn

FLAGS = ("HIGH", "MEDIUM", "LOW")  # confidence, not uncertainty: HIGH = trust it


def assert_stochastic(model, img_size_=None, rtol=1e-6):
    """Fail loudly if dropout is not actually sampling.

    Without this, MC Dropout silently degenerates to T identical copies of the deterministic
    prediction — entropy still looks plausible, MI collapses to ~0, and every image reads HIGH.

    The check is made on the Dropout layer's own output, not on the final softmax. An
    under-trained model (one smoke epoch, random init) emits a near-uniform softmax whatever
    dropout does, so comparing predictions there reports "dropout is off" for a model whose
    dropout is working perfectly well. The layer's activations are the honest signal.
    """
    import tensorflow as tf
    from tensorflow import keras

    drops = [l for l in model.layers if isinstance(l, keras.layers.Dropout)]
    if not drops:
        raise RuntimeError(f"{model.name}: no Dropout layer, so MC Dropout cannot sample. "
                           f"Check models.build_backbone.")
    if not any(float(getattr(l, "rate", 0)) > 0 for l in drops):
        raise RuntimeError(f"{model.name}: every Dropout layer has rate 0; MC Dropout would be a no-op.")

    n = img_size_ or int(model.input_shape[1])
    # Random, not zeros: all-zero pixels can pool to all-zero features, and dropping zeros is a no-op.
    # A wide batch so a sparse internal-dropout mask is very unlikely to fire identically twice.
    x = np.random.default_rng(0).normal(size=(16, n, n, 3)).astype("float32")
    # Prefer the classification head's own dropout (GRADCAM-free MC path drops here); fall back to
    # the highest-rate layer. EfficientNet's internal stochastic-depth `Dropout`s can emit identical
    # output on a small batch, so they are a poor probe.
    layer = next((l for l in drops if l.name == "head_dropout"),
                 max(drops, key=lambda l: float(getattr(l, "rate", 0))))
    try:
        probe = keras.Model(model.inputs, layer.output)
        a = np.asarray(probe(x, training=True))
        b = np.asarray(probe(x, training=True))
    except (ValueError, TypeError, AttributeError, tf.errors.OpError):
        a = np.asarray(model(x, training=True))      # fall back to the prediction
        b = np.asarray(model(x, training=True))
    if np.allclose(a, b, rtol=rtol, atol=0):
        raise RuntimeError(
            f"{model.name}: two training=True passes through '{layer.name}' are identical, so dropout "
            f"is not sampling. MC Dropout needs a Dropout layer that stays active.")
    return True


def mc_predict(model, x, T=None, batch_size=None, seed=None):
    """T stochastic forward passes over x. Returns (T, N, NUM_CLASSES).

    x is a preprocessed float array (N, H, W, 3). Images are batched, never fed one at a time.
    """
    T = T or C.MC_PASSES
    bs = batch_size or C.BATCH_SIZE
    x = np.asarray(x)
    out = np.empty((T, len(x), C.NUM_CLASSES), dtype="float32")
    for t in range(T):
        if seed is not None:
            import tensorflow as tf
            tf.random.set_seed(seed + t)
        chunks = [np.asarray(model(x[i:i + bs], training=True)) for i in range(0, len(x), bs)]
        out[t] = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return out


def _entropy(p, axis=-1):
    """Shannon entropy in nats, normalised to [0, 1] by log(NUM_CLASSES)."""
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis) / np.log(C.NUM_CLASSES)


def metrics_from_passes(probs):
    """probs: (T, N, C) per-pass ensemble probabilities -> per-image uncertainty.

    Returns {mean_probs (N,C), entropy (N,), mutual_information (N,), top_std (N,)},
    with entropy and MI in [0, 1] and MI <= entropy by construction.
    """
    probs = np.asarray(probs, dtype="float64")
    mean_p = probs.mean(axis=0)                      # (N, C)
    h_total = _entropy(mean_p)                       # (N,)
    h_each = _entropy(probs, axis=-1).mean(axis=0)   # mean over passes -> (N,)
    mi = np.maximum(h_total - h_each, 0.0)           # clamp float noise at 0
    top = mean_p.argmax(axis=1)
    top_std = probs[:, np.arange(probs.shape[1]), top].std(axis=0)
    return {"mean_probs": mean_p, "entropy": h_total, "mutual_information": mi, "top_std": top_std}


# ---------- ensemble-level MC ----------

def _combine(per_pass, voting, stacking, mode):
    """One pass of per-backbone probs -> one pass of ensemble probs, using the SELECTED head."""
    if mode == "stacking" and stacking and all(n in per_pass for n in stacking["models"]):
        return stacking_probs(stacking, per_pass)
    if voting:
        w = {n: v for n, v in voting["weights"].items() if n in per_pass}
        if w:
            return voting_probs(per_pass, w)
    return np.mean(list(per_pass.values()), axis=0)


def _mc_over_split(model, df, name, T):
    """T stochastic passes over a whole split, one batch at a time. Returns (T, N, C).

    Batch-at-a-time on purpose: materialising the whole split first (487 x 224 x 224 x 3 float32 is
    ~290 MB per backbone, before TensorFlow's own working set) exhausted memory on CPU and killed a
    run with a MKL "could not create a memory object" abort. Here only one batch of pixels is live,
    and each batch is decoded once and reused for all T passes.
    """
    ds = make_dataset(df, preprocess_fn(name), img_size(model), C.BATCH_SIZE)
    chunks = []
    for x, _ in ds:
        x = np.asarray(x)
        chunks.append(np.stack([np.asarray(model(x, training=True)) for _ in range(T)]))
        del x
    return np.concatenate(chunks, axis=1) if chunks else np.empty((T, 0, C.NUM_CLASSES), dtype="float32")


def mc_ensemble_probs(df, T=None, models=None, progress_cb=None, check_stochastic=True):
    """MC Dropout over a split dataframe, combined per pass by the selected ensemble.

    Returns (T, N, C). Each backbone is sampled T times, then pass t of every backbone is
    combined into ensemble pass t — so the ensemble itself is what varies, not just one network.
    """
    T = T or C.MC_PASSES
    models = models or load_models()
    if not models:
        raise FileNotFoundError("No trained models found. Train the backbones first.")
    mode = C.MODE_FILE.read_text().strip() if C.MODE_FILE.is_file() else C.DEFAULT_ENSEMBLE_MODE
    voting, stacking = load_voting(), load_stacking()

    import gc

    per_backbone = {}
    for i, (n, m) in enumerate(models.items()):
        if check_stochastic:
            assert_stochastic(m)
        per_backbone[n] = _mc_over_split(m, df, n, T)
        gc.collect()  # release the previous backbone's batches before loading the next
        if progress_cb:
            progress_cb("uncertainty", i + 1, {"backbone": n, "passes": T})
    return np.stack([_combine({n: per_backbone[n][t] for n in per_backbone}, voting, stacking, mode)
                     for t in range(T)])


# ---------- thresholds ----------

def flag_for(entropy, thresholds):
    """entropy -> HIGH / MEDIUM / LOW confidence."""
    t = thresholds
    if entropy >= t["low_confidence"]:
        return "LOW"
    if entropy >= t["review"]:
        return "MEDIUM"
    return "HIGH"


def _youden(entropy, wrong):
    """Threshold maximising (fraction of wrong flagged) - (fraction of correct flagged).

    This is Youden's J for 'is this prediction wrong?', so it is the point where the flag buys
    the most error detection per unit of reviewer effort.
    """
    order = np.unique(np.round(entropy, 6))
    cands = np.concatenate([[0.0], (order[:-1] + order[1:]) / 2, [1.0]]) if len(order) > 1 else np.array([0.5])
    n_w, n_c = max(wrong.sum(), 1), max((~wrong).sum(), 1)
    best, best_j = 0.5, -np.inf
    for c in cands:
        flagged = entropy >= c
        j = flagged[wrong].sum() / n_w - flagged[~wrong].sum() / n_c
        if j > best_j:
            best, best_j = float(c), float(j)
    return best, float(best_j)


def auroc(score, positive):
    """AUROC of `score` as a detector of `positive` (rank-based, no sklearn import needed here)."""
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return float("nan")
    r = np.empty(len(score), dtype="float64")
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks, i = np.empty(len(s)), 0
    while i < len(s):                      # average ranks within ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r[order] = ranks
    n_p, n_n = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


MIN_WRONG_FOR_YOUDEN = 10  # below this, one image moves the threshold; use the correct-side spread instead
REVIEW_PCTL, LOW_PCTL = 95.0, 99.0


def calibrate(df=None, T=None, save=True, progress_cb=None):
    """Pick the two entropy thresholds on the VALIDATION split. Never touches test.

    Two methods, chosen by how many validation errors there are to learn from:

    youden      — the requested criterion: the threshold maximising (wrong flagged) - (correct
                  flagged). Sound when there are enough wrong predictions to define a rate.
    percentile  — fallback when the split has fewer than MIN_WRONG_FOR_YOUDEN errors. This model
                  gets ~1 validation image wrong, so Youden would be fitted to a single point and
                  would move wherever that one image happened to land. Instead both thresholds come
                  from the spread of the ~485 CORRECT predictions: review at the 95th percentile,
                  low_confidence at the 99th. That fixes the review budget (flag ~5% / ~1% of
                  confident cases) and is estimated from hundreds of samples rather than one.

    Whichever is used, both candidates and the error count are recorded so the choice is auditable.
    """
    from src.data import load_split
    df = load_split("val") if df is None else df
    T = T or C.MC_PASSES
    probs = mc_ensemble_probs(df, T, progress_cb=progress_cb)
    m = metrics_from_passes(probs)
    y = df["label"].map({c: i for i, c in enumerate(C.CLASS_NAMES)}).to_numpy()
    pred = m["mean_probs"].argmax(1)
    wrong = pred != y
    ent = m["entropy"]

    n_wrong = int(wrong.sum())
    correct_ent = ent[~wrong]
    youden_t, j = _youden(ent, wrong)
    pctl_review = float(np.percentile(correct_ent, REVIEW_PCTL)) if len(correct_ent) else 0.25
    pctl_low = float(np.percentile(correct_ent, LOW_PCTL)) if len(correct_ent) else 0.60

    enough = n_wrong >= MIN_WRONG_FOR_YOUDEN
    method = "youden" if enough else "percentile"
    review = youden_t if enough else pctl_review
    low = max(pctl_low, review)  # the escalation level can never sit below the review level

    flags = np.array([flag_for(e, {"review": review, "low_confidence": low}) for e in ent])
    flagged = flags != "HIGH"
    small = (None if enough else
             f"Only {n_wrong} validation error(s): too few for Youden's J, which would be fitted to "
             f"{n_wrong} point(s). Thresholds taken from the {REVIEW_PCTL:g}th/{LOW_PCTL:g}th percentile "
             f"of the {len(correct_ent)} correct predictions instead. Error-detection rates measured on "
             f"this split are not meaningful at this sample size — see the test-split numbers.")
    out = {
        "passes": int(T),
        "split": "val",
        "n_images": int(len(df)),
        "n_wrong": n_wrong,
        "method": method,
        "review": round(float(review), 6),
        "low_confidence": round(float(low), 6),
        "rationale": {
            "review": (f"Youden's J on 'prediction is wrong'" if enough else
                       f"{REVIEW_PCTL:g}th percentile of entropy among correct validation predictions"),
            "low_confidence": f"{LOW_PCTL:g}th percentile of entropy among correct validation predictions",
            "caveat": small,
        },
        "candidates": {
            "youden": round(float(youden_t), 6),
            "youden_j": round(j, 4),
            f"pctl{REVIEW_PCTL:g}_correct": round(pctl_review, 6),
            f"pctl{LOW_PCTL:g}_correct": round(pctl_low, 6),
        },
        "val": {
            "accuracy": round(float((~wrong).mean()), 4),
            "errors_flagged_pct": round(float(flagged[wrong].mean() * 100), 1) if wrong.any() else None,
            "correct_flagged_pct": round(float(flagged[~wrong].mean() * 100), 1),
            "mean_entropy_correct": round(float(ent[~wrong].mean()), 4),
            "mean_entropy_wrong": round(float(ent[wrong].mean()), 4) if wrong.any() else None,
            "entropy_auroc_vs_wrong": round(auroc(ent, wrong), 4) if n_wrong else None,
            "mi_auroc_vs_wrong": round(auroc(m["mutual_information"], wrong), 4) if n_wrong else None,
            "auroc_note": None if enough else f"AUROC estimated from {n_wrong} wrong prediction(s); not reliable.",
        },
    }
    if save:
        # Keep the per-image arrays so thresholds can be revisited without re-running 19 min of MC.
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(C.RESULTS_DIR / "uncertainty_val.npz", entropy=ent,
                            mutual_information=m["mutual_information"], top_std=m["top_std"],
                            wrong=wrong, y=y, pred=pred)
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        C.UNCERTAINTY_THRESHOLDS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def load_thresholds():
    """Calibrated thresholds, or a neutral fallback if calibration has not been run yet."""
    if C.UNCERTAINTY_THRESHOLDS.is_file():
        try:
            d = json.loads(C.UNCERTAINTY_THRESHOLDS.read_text(encoding="utf-8"))
            if "review" in d and "low_confidence" in d:
                return d
        except (ValueError, OSError):
            pass
    return {"review": 0.25, "low_confidence": 0.60, "passes": C.MC_PASSES, "uncalibrated": True}


# ---------- single image (Preview) ----------

def uncertainty_for_image(img, models=None, T=None):
    """MC Dropout for one image. Returns entropy / MI / flag / passes plus the averaged probs."""
    T = T or C.MC_PASSES_PREVIEW
    models = models or load_models()
    if not models:
        raise FileNotFoundError("No trained models found. Train the backbones first.")
    mode = C.MODE_FILE.read_text().strip() if C.MODE_FILE.is_file() else C.DEFAULT_ENSEMBLE_MODE
    voting, stacking = load_voting(), load_stacking()

    per = {}
    for n, m in models.items():
        x = preprocess_fn(n)(load_image_array(img, img_size(m))[None].copy())
        per[n] = mc_predict(m, x, T, seed=C.MC_SEED)          # (T, 1, C)
    passes = np.stack([_combine({n: per[n][t] for n in per}, voting, stacking, mode) for t in range(T)])
    m_ = metrics_from_passes(passes)
    th = load_thresholds()
    ent = float(m_["entropy"][0])
    return {
        "entropy": round(ent, 4),
        "mutual_information": round(float(m_["mutual_information"][0]), 4),
        "top_std": round(float(m_["top_std"][0]), 4),
        "flag": flag_for(ent, th),
        "passes": int(T),
        "mean_probs": m_["mean_probs"][0],
        "pass_top_probs": [round(float(v), 4) for v in passes[:, 0, :].max(axis=1)],
        "thresholds": {"review": th["review"], "low_confidence": th["low_confidence"]},
    }
