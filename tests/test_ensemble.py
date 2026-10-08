import numpy as np

from src import config as C
from src import ensemble


def _fake_probs(rng, y, acc):
    """Probabilities that pick the true class with probability ~acc."""
    p = rng.dirichlet(np.ones(C.NUM_CLASSES), len(y))
    hit = rng.random(len(y)) < acc
    p[hit, y[hit]] += 2
    return p / p.sum(1, keepdims=True)


def test_voting_weights_sum_to_one_and_follow_f1(empty_root):
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(C.NUM_CLASSES), 25)
    probs = {"mobilenetv2": _fake_probs(rng, y, 0.9), "efficientnetb0": _fake_probs(rng, y, 0.6),
             "densenet121": _fake_probs(rng, y, 0.3)}
    info = ensemble.fit_voting(probs, y)
    w = info["weights"]
    assert abs(sum(w.values()) - 1) < 1e-9
    assert w["mobilenetv2"] > w["efficientnetb0"] > w["densenet121"]
    assert ensemble.load_voting() == info
    v = ensemble.voting_probs(probs, w)
    assert v.shape == (100, 4) and np.allclose(v.sum(1), 1)


def test_stacking_shape_and_reload(empty_root):
    rng = np.random.default_rng(1)
    y = np.tile(np.arange(C.NUM_CLASSES), 25)
    probs = {n: _fake_probs(rng, y, 0.7) for n in C.BACKBONES}
    ensemble.fit_stacking(probs, y)
    bundle = ensemble.load_stacking()
    assert bundle["models"] == C.BACKBONES
    assert bundle["clf"].coef_.shape == (4, 12)  # 12 features = 3 models x 4 probs
    p = ensemble.stacking_probs(bundle, probs)
    assert p.shape == (100, 4) and np.allclose(p.sum(1), 1)


def test_smoke_ensemble_artifacts(smoke):
    assert C.WEIGHTS_JSON.is_file() and C.META_CLF.is_file()
    vp = np.load(C.RESULTS_DIR / "val_probs.npy")
    assert vp.shape == (3, 4, 4)  # 3 models x 4 val images x 4 classes
