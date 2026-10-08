import numpy as np
import pytest
from tensorflow import keras

from src import config as C
from src import models


@pytest.mark.parametrize("name", C.BACKBONES)
def test_backbone_builds_outputs_probs_and_gradcam_layer(name):
    m = models.build_backbone(name, weights=None, img_size=C.SMOKE_IMG_SIZE)
    x = models.preprocess_fn(name)(np.random.uniform(0, 255, (1, C.SMOKE_IMG_SIZE, C.SMOKE_IMG_SIZE, 3)).astype("float32"))
    p = np.asarray(m(x, training=False))
    assert p.shape == (1, C.NUM_CLASSES)
    assert abs(p.sum() - 1) < 1e-4
    # GRADCAM_LAYER must match what iterating model.layers finds
    assert models.GRADCAM_LAYER[name] == models.last_spatial_layer(m)
    assert m.layers[-1].name == "head_predictions"
    keras.backend.clear_session()


def test_phase2_unfreezes_top_30_percent_without_batchnorm():
    m = models.build_backbone("mobilenetv2", weights=None, img_size=C.SMOKE_IMG_SIZE)
    base = m.layers[:-models.HEAD_LAYERS]
    models.set_phase(m, 1)
    assert not any(l.trainable for l in base)
    models.set_phase(m, 2)
    cut = len(base) - int(len(base) * C.UNFREEZE_FRACTION)
    assert not any(l.trainable for l in base[:cut])
    top = base[cut:]
    assert all(l.trainable != isinstance(l, keras.layers.BatchNormalization) for l in top)
    assert all(l.trainable for l in m.layers[-models.HEAD_LAYERS:])
