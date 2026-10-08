import pytest
from PIL import Image

from src import config as C
from src import predict

SAMPLE = next((C.SAMPLE_DIR / C.CLASS_NAMES[0]).iterdir())


def test_no_models_raises_friendly_error(empty_root):
    with pytest.raises(FileNotFoundError, match="Train"):
        predict.predict_image(SAMPLE)


@pytest.mark.parametrize("mode", ["voting", "stacking"])
def test_predict_sample_returns_valid_class(smoke, mode):
    C.MODE_FILE.write_text(mode)
    r = predict.predict_image(Image.open(SAMPLE))
    assert r["class"] in C.CLASS_NAMES and r["mode"] == mode
    assert 0 <= r["confidence"] <= 1 and abs(sum(r["probs"].values()) - 1) < 1e-4
    assert set(r["per_model_probs"]) == set(C.BACKBONES) == set(r["votes"])
    assert r["gradcam_model"] in C.BACKBONES
