from src import config as C


def test_classes_and_constants():
    assert C.NUM_CLASSES == 4 == len(set(C.CLASS_NAMES))
    assert C.CLASS_NAMES[0] == "Benign"
    assert C.IMG_SIZE == 224 and C.SEED == 42
    assert abs(sum(C.SPLIT_FRACTIONS.values()) - 1) < 1e-9
    assert set(C.BACKBONES) == {"mobilenetv2", "efficientnetb0", "densenet121"}


def test_set_root_moves_every_path(tmp_path):
    old = C.ROOT
    try:
        C.set_root(tmp_path)
        assert C.MODELS_DIR == tmp_path / "models"
        assert C.SPLITS_DIR == tmp_path / "data" / "splits"
        assert C.model_path("densenet121") == tmp_path / "models" / "densenet121.keras"
    finally:
        C.set_root(old)


def test_sample_images_committed():
    for name in C.CLASS_NAMES:
        files = list((C.SAMPLE_DIR / name).iterdir())
        assert len(files) == C.SAMPLE_PER_CLASS
        assert all(f.stat().st_size < 50_000 for f in files)
