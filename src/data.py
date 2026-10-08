"""Image listing, stratified splits, tf.data pipelines."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C

_CLASS_KEY = {" ".join(n.lower().split()): n for n in C.CLASS_NAMES}


def list_images(root):
    """[(path, class_name)] for every image under a folder named after a class, anywhere below root."""
    out = []
    for dirpath, _, files in os.walk(root):
        label = _CLASS_KEY.get(" ".join(Path(dirpath).name.lower().split()))
        if label:
            out += [(str(Path(dirpath) / f), label) for f in files if Path(f).suffix.lower() in C.IMAGE_EXTS]
    return sorted(out)


def data_source(smoke=False):
    """(name, dir) of the dataset to split: sample (smoke) > kaggle raw > demo; None if nothing."""
    if smoke:
        return "sample", C.SAMPLE_DIR
    for name, d in (("kaggle", C.RAW_DIR), ("demo", C.DEMO_DIR)):
        if d.is_dir() and list_images(d):
            return name, d
    return None


def _rel(p):
    try:
        return Path(p).resolve().relative_to(C.ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def make_splits(smoke=False):
    """Write data/splits/{train,val,test}.csv + meta.json and results/split_counts.csv."""
    src = data_source(smoke)
    if src is None:
        raise FileNotFoundError("No dataset found. Download it or generate demo data first.")
    source, root = src
    df = pd.DataFrame(list_images(root), columns=["path", "label"])
    df["path"] = df["path"].map(_rel)
    df["label_idx"] = df["label"].map(C.CLASS_NAMES.index)
    if smoke:
        # ponytail: 2 images/class can't fill 3 disjoint stratified splits; smoke trains on all,
        # val/test take one image per class each. Plumbing only, numbers are meaningless.
        first = df.groupby("label").cumcount() == 0
        splits = {"train": df, "val": df[first], "test": df[~first].groupby("label").head(1)}
    else:
        f = C.SPLIT_FRACTIONS
        train, rest = train_test_split(df, train_size=f["train"], stratify=df["label_idx"], random_state=C.SEED)
        val, test = train_test_split(rest, test_size=f["test"] / (f["val"] + f["test"]),
                                     stratify=rest["label_idx"], random_state=C.SEED)
        splits = {"train": train, "val": val, "test": test}
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        s.sort_values("path").to_csv(C.SPLITS_DIR / f"{name}.csv", index=False)
    counts = pd.DataFrame({k: v["label"].value_counts() for k, v in splits.items()})
    counts = counts.reindex(C.CLASS_NAMES).fillna(0).astype(int)
    counts.index.name = "class"
    counts.to_csv(C.RESULTS_DIR / "split_counts.csv")
    (C.SPLITS_DIR / "meta.json").write_text(json.dumps({"source": source, "smoke": smoke, "n_images": len(df)}))
    return counts


def split_meta():
    p = C.SPLITS_DIR / "meta.json"
    return json.loads(p.read_text()) if p.is_file() else None


def load_split(name):
    """train/val only. The test split is read exclusively by src/evaluate.py."""
    if name not in ("train", "val"):
        raise ValueError("test split is only read by src/evaluate.py")
    return pd.read_csv(C.SPLITS_DIR / f"{name}.csv")


def resolve(p):
    return str(C.ROOT / p)  # an absolute p stays absolute


def _augmenter():
    from tensorflow import keras
    L = keras.layers
    return keras.Sequential([
        L.RandomFlip("horizontal_and_vertical", seed=C.SEED),
        L.RandomRotation(C.AUG_ROTATION_DEG / 360, seed=C.SEED),
        L.RandomZoom(C.AUG_ZOOM, seed=C.SEED),
        L.RandomBrightness(C.AUG_BRIGHTNESS, value_range=(0, 255), seed=C.SEED),
        L.RandomContrast(C.AUG_CONTRAST, seed=C.SEED),
    ])


def make_dataset(df, preprocess, img_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE, training=False):
    """tf.data of (B, img_size, img_size, 3) float32 preprocessed images and int labels."""
    import tensorflow as tf

    paths = [resolve(p) for p in df["path"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, df["label_idx"].astype("int32").values))
    if training:
        ds = ds.shuffle(len(paths), seed=C.SEED, reshuffle_each_iteration=True)

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32), (img_size, img_size))
        return img, y

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        aug = _augmenter()
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0, 255), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def load_image_array(img, img_size):
    """Path / PIL image / uint8 array -> (img_size, img_size, 3) float32 in [0, 255]."""
    from PIL import Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    elif not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.make_splits:
        print(make_splits(a.smoke))
