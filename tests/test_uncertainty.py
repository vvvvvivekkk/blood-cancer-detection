"""MC Dropout uncertainty: the maths, the calibration file, and the flag mapping.

No trained models and no network: a tiny randomly-initialised backbone is enough to prove the
passes are stochastic, and the metric tests run on hand-built probability arrays where the right
answer is known analytically.
"""
import json

import numpy as np
import pytest

from src import config as C
from src import uncertainty as U
from src.models import build_backbone


@pytest.fixture(scope="module")
def tiny():
    """An untrained MobileNetV2 at smoke resolution — offline, and it still has head_dropout."""
    return build_backbone("mobilenetv2", weights=None, img_size=C.SMOKE_IMG_SIZE)


# ---------- the passes must actually be stochastic ----------

def test_two_mc_passes_differ(tiny):
    # Random input, not zeros: zero pixels can pool to zero features, and dropping zeros is a no-op.
    x = np.random.default_rng(0).normal(size=(2, C.SMOKE_IMG_SIZE, C.SMOKE_IMG_SIZE, 3)).astype("float32")
    a = np.asarray(tiny(x, training=True))
    b = np.asarray(tiny(x, training=True))
    assert not np.allclose(a, b), "dropout is not sampling — MC Dropout would be T identical copies"


def test_deterministic_pass_is_stable(tiny):
    x = np.random.default_rng(1).normal(size=(2, C.SMOKE_IMG_SIZE, C.SMOKE_IMG_SIZE, 3)).astype("float32")
    assert np.allclose(np.asarray(tiny(x, training=False)), np.asarray(tiny(x, training=False)))


def test_assert_stochastic_accepts_a_real_backbone(tiny):
    assert U.assert_stochastic(tiny, C.SMOKE_IMG_SIZE) is True


def test_assert_stochastic_rejects_a_model_with_no_dropout():
    from tensorflow import keras
    inp = keras.Input((C.SMOKE_IMG_SIZE, C.SMOKE_IMG_SIZE, 3))
    x = keras.layers.GlobalAveragePooling2D()(inp)
    out = keras.layers.Dense(C.NUM_CLASSES, activation="softmax")(x)
    with pytest.raises(RuntimeError, match="no Dropout layer"):
        U.assert_stochastic(keras.Model(inp, out, name="nodrop"))


def test_assert_stochastic_rejects_rate_zero_dropout():
    from tensorflow import keras
    inp = keras.Input((C.SMOKE_IMG_SIZE, C.SMOKE_IMG_SIZE, 3))
    x = keras.layers.GlobalAveragePooling2D()(inp)
    x = keras.layers.Dropout(0.0, name="head_dropout")(x)
    out = keras.layers.Dense(C.NUM_CLASSES, activation="softmax")(x)
    with pytest.raises(RuntimeError, match="rate 0"):
        U.assert_stochastic(keras.Model(inp, out, name="zerodrop"))


def test_assert_stochastic_passes_on_an_undertrained_model(tiny):
    """A near-uniform softmax must not be mistaken for 'dropout is off' — the guard probes the
    Dropout layer's activations, which still move even when the head has learned nothing."""
    assert U.assert_stochastic(tiny, C.SMOKE_IMG_SIZE) is True


def test_mc_predict_shape_and_batches(tiny):
    n, T = 5, 3
    x = np.random.default_rng(2).normal(size=(n, C.SMOKE_IMG_SIZE, C.SMOKE_IMG_SIZE, 3)).astype("float32")
    p = U.mc_predict(tiny, x, T=T, batch_size=2)       # n > batch_size: exercises the chunking
    assert p.shape == (T, n, C.NUM_CLASSES)
    assert np.allclose(p.sum(axis=-1), 1, atol=1e-4)   # still softmax rows
    assert not np.allclose(p[0], p[1])                 # passes differ


# ---------- entropy / mutual information ----------

def test_entropy_bounds_and_endpoints():
    certain = np.tile(np.array([[1.0, 0, 0, 0]]), (4, 3, 1))
    assert U.metrics_from_passes(certain)["entropy"].max() == pytest.approx(0, abs=1e-6)
    uniform = np.full((4, 3, C.NUM_CLASSES), 1 / C.NUM_CLASSES)
    assert U.metrics_from_passes(uniform)["entropy"].max() == pytest.approx(1, abs=1e-6)


def test_mutual_information_separates_aleatoric_from_epistemic():
    # every pass agrees it is uniform -> ambiguity is in the data, not the model
    uniform = np.full((8, 1, C.NUM_CLASSES), 1 / C.NUM_CLASSES)
    m = U.metrics_from_passes(uniform)
    assert m["entropy"][0] == pytest.approx(1, abs=1e-6)
    assert m["mutual_information"][0] == pytest.approx(0, abs=1e-6)

    # each pass is confident but about a different class -> model disagreement
    disagree = np.zeros((C.NUM_CLASSES, 1, C.NUM_CLASSES))
    for t in range(C.NUM_CLASSES):
        disagree[t, 0, t] = 1.0
    m = U.metrics_from_passes(disagree)
    assert m["entropy"][0] == pytest.approx(1, abs=1e-6)
    assert m["mutual_information"][0] == pytest.approx(1, abs=1e-6)


def test_metric_invariants_on_random_probabilities():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(C.NUM_CLASSES), size=(12, 150))
    m = U.metrics_from_passes(p)
    assert ((m["entropy"] >= 0) & (m["entropy"] <= 1)).all()
    assert (m["mutual_information"] >= 0).all()
    assert (m["mutual_information"] <= m["entropy"] + 1e-9).all()
    assert m["mean_probs"].shape == (150, C.NUM_CLASSES)
    assert (m["top_std"] >= 0).all()


def test_auroc_matches_sklearn_including_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    s, lab = rng.random(400), rng.random(400) < 0.25
    assert U.auroc(s, lab) == pytest.approx(sk.roc_auc_score(lab, s), abs=1e-9)
    s2 = np.round(rng.random(300), 1)                      # heavy ties
    lab2 = rng.random(300) < 0.3
    assert U.auroc(s2, lab2) == pytest.approx(sk.roc_auc_score(lab2, s2), abs=1e-9)


def test_youden_finds_a_clean_split():
    ent = np.concatenate([np.full(50, 0.05), np.full(10, 0.9)])
    wrong = np.arange(60) >= 50
    t, j = U._youden(ent, wrong)
    assert 0.05 < t <= 0.9
    assert j == pytest.approx(1.0)


# ---------- flags and the thresholds file ----------

@pytest.mark.parametrize("entropy,expected", [
    (0.0, "HIGH"), (0.29, "HIGH"), (0.30, "MEDIUM"), (0.69, "MEDIUM"), (0.70, "LOW"), (1.0, "LOW"),
])
def test_flag_mapping_is_inclusive_at_the_thresholds(entropy, expected):
    assert U.flag_for(entropy, {"review": 0.3, "low_confidence": 0.7}) == expected


def test_flags_are_monotonic_in_entropy():
    th = {"review": 0.3, "low_confidence": 0.7}
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    seq = [order[U.flag_for(e, th)] for e in np.linspace(0, 1, 50)]
    assert seq == sorted(seq)


def test_thresholds_file_round_trips(empty_root):
    payload = {"passes": 7, "review": 0.4, "low_confidence": 0.8, "split": "val", "method": "youden"}
    C.UNCERTAINTY_THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    C.UNCERTAINTY_THRESHOLDS.write_text(json.dumps(payload), encoding="utf-8")
    got = U.load_thresholds()
    assert got["review"] == 0.4 and got["low_confidence"] == 0.8 and got["passes"] == 7
    assert not got.get("uncalibrated")
    assert U.flag_for(0.5, got) == "MEDIUM" and U.flag_for(0.9, got) == "LOW"


def test_load_thresholds_falls_back_when_missing_or_corrupt(empty_root):
    assert U.load_thresholds().get("uncalibrated") is True          # no file
    C.UNCERTAINTY_THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    C.UNCERTAINTY_THRESHOLDS.write_text("{not json", encoding="utf-8")
    assert U.load_thresholds().get("uncalibrated") is True          # unparseable
    C.UNCERTAINTY_THRESHOLDS.write_text('{"passes": 3}', encoding="utf-8")
    assert U.load_thresholds().get("uncalibrated") is True          # missing the keys


def test_fallback_thresholds_still_order_correctly(empty_root):
    th = U.load_thresholds()
    assert 0 < th["review"] < th["low_confidence"] <= 1
