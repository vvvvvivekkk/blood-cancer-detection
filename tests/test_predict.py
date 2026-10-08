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


def test_active_mode_default_persistence_and_validation(empty_root):
    assert C.get_active_mode() == C.ACTIVE_MODE == "ensemble"
    C.set_active_mode("mobilenetv2")
    assert C.ACTIVE_MODE_FILE.read_text() == "mobilenetv2" and C.get_active_mode() == "mobilenetv2"
    with pytest.raises(ValueError, match="Unknown mode"):
        C.set_active_mode("vgg16")
    C.ACTIVE_MODE_FILE.write_text("garbage")
    assert C.get_active_mode() == C.ACTIVE_MODE  # bad file falls back to the config default
    from src.status import status
    assert status()["active_mode"] == "ensemble"


def test_single_mode_uses_only_mobilenetv2(single_mode):
    r = predict.predict_image(SAMPLE)
    assert r["mode"] == "single" and r["mode_label"] == "MobileNetV2 (single)" and r["active_mode"] == "mobilenetv2"
    assert list(r["per_model_probs"]) == ["mobilenetv2"] == list(r["votes"])
    assert r["gradcam_model"] == "mobilenetv2"
    assert r["probs"] == pytest.approx(r["per_model_probs"]["mobilenetv2"])


def test_single_mode_without_mobilenet_is_a_readable_error(empty_root):
    C.set_active_mode("mobilenetv2")
    with pytest.raises(FileNotFoundError, match="MobileNetV2"):
        predict.predict_image(SAMPLE)
