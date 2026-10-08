"""Two-phase fine-tuning of each backbone.

python -m src.train --model {mobilenetv2,efficientnetb0,densenet121,all} [--epochs N] [--smoke]
"""
import argparse

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src import config as C
from src.data import load_split, make_dataset
from src.models import build_backbone, compile_model, preprocess_fn, set_phase


class _Progress(keras.callbacks.Callback):
    def __init__(self, stage, total, phase, cb):
        super().__init__()
        self.stage, self.total, self.phase, self.cb = stage, total, phase, cb

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        self.cb(self.stage, epoch + 1, {**logs, "epochs_total": self.total, "phase": self.phase})


def train_backbone(name, smoke=False, epochs=None, progress_cb=None):
    """Train one backbone; saves models/<name>.keras and results/history/<name>.csv."""
    C.seed_everything()
    img = C.SMOKE_IMG_SIZE if smoke else C.IMG_SIZE
    bs = C.SMOKE_BATCH_SIZE if smoke else C.BATCH_SIZE
    head = C.SMOKE_EPOCHS_PER_PHASE if smoke else min(C.HEAD_EPOCHS, epochs or C.EPOCHS)
    total = 2 * C.SMOKE_EPOCHS_PER_PHASE if smoke else (epochs or C.EPOCHS)

    train_df, val_df = load_split("train"), load_split("val")
    pp = preprocess_fn(name)
    train_ds = make_dataset(train_df, pp, img, bs, training=True)
    val_ds = make_dataset(val_df, pp, img, bs)
    y = train_df["label_idx"].values
    classes = np.unique(y)
    class_weight = dict(zip(classes.tolist(), compute_class_weight("balanced", classes=classes, y=y).tolist()))

    # Smoke runs offline: no ImageNet download.
    model = build_backbone(name, weights=None if smoke else "imagenet", img_size=img)
    path = C.model_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    C.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist = C.HISTORY_DIR / f"{name}.csv"
    hist.unlink(missing_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", save_best_only=True)  # shared: best across phases

    for phase, lr, start, end in ((1, C.LR_HEAD, 0, head), (2, C.LR_FINE, head, total)):
        if end <= start:
            continue
        set_phase(model, phase)
        compile_model(model, lr, eager=smoke)
        callbacks = [
            ckpt,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=C.LR_PLATEAU_FACTOR,
                                              patience=C.LR_PLATEAU_PATIENCE, min_lr=C.MIN_LR),
            keras.callbacks.CSVLogger(str(hist), append=True),
        ]
        if progress_cb:
            callbacks.append(_Progress(name, total, phase, progress_cb))
        model.fit(train_ds, validation_data=val_ds, epochs=end, initial_epoch=start,
                  class_weight=class_weight, callbacks=callbacks, verbose=0 if progress_cb else 2)
    if not path.is_file():  # val_loss never finite
        model.save(str(path))
    keras.backend.clear_session()
    return path


def train_all(names=None, smoke=False, epochs=None, progress_cb=None):
    return [train_backbone(n, smoke, epochs, progress_cb) for n in (names or C.BACKBONES)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=C.BACKBONES + ["all"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(train_all(None if a.model == "all" else [a.model], a.smoke, a.epochs))
