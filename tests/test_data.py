import pandas as pd
import pytest

from src import config as C
from src import data, download


@pytest.fixture
def demo_splits(empty_root):
    download.make_demo_data()
    data.make_splits()
    return {n: pd.read_csv(C.SPLITS_DIR / f"{n}.csv") for n in ("train", "val", "test")}


def test_splits_disjoint_sized_and_stratified(demo_splits):
    sets = {n: set(df["path"]) for n, df in demo_splits.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    total = sum(map(len, sets.values()))
    assert total == 200
    for name, frac in C.SPLIT_FRACTIONS.items():
        df = demo_splits[name]
        assert abs(len(df) / total - frac) < 0.02
        share = df["label"].value_counts(normalize=True)
        assert all(abs(share[c] - 0.25) < 0.02 for c in C.CLASS_NAMES)
    assert list(demo_splits["train"].columns) == ["path", "label", "label_idx"]
    assert (C.RESULTS_DIR / "split_counts.csv").is_file()
    assert data.split_meta()["source"] == "demo"


def test_test_split_is_off_limits(demo_splits):
    with pytest.raises(ValueError):
        data.load_split("test")


def test_pipeline_shapes_and_dtypes(empty_root):
    import tensorflow as tf
    data.make_splits(smoke=True)
    df = data.load_split("train")
    for training in (True, False):
        x, y = next(iter(data.make_dataset(df, lambda t: t / 255.0, img_size=32, batch_size=4, training=training)))
        assert x.shape == (4, 32, 32, 3) and x.dtype == tf.float32
        assert y.dtype == tf.int32 and set(y.numpy()) <= set(range(C.NUM_CLASSES))
