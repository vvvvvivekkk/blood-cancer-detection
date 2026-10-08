import pandas as pd

from src import config as C


def test_smoke_training_saves_models_history_and_reports_progress(smoke):
    _, calls = smoke
    for name in C.BACKBONES:
        assert C.model_path(name).is_file()
        hist = pd.read_csv(C.HISTORY_DIR / f"{name}.csv")
        assert len(hist) == 2 * C.SMOKE_EPOCHS_PER_PHASE  # one row per epoch, both phases
        assert {"loss", "accuracy", "val_loss", "val_accuracy"} <= set(hist.columns)
        epochs = [(e, logs) for stage, e, logs in calls if stage == name]
        assert [e for e, _ in epochs] == [1, 2]
        assert [l["phase"] for _, l in epochs] == [1, 2]
        assert all(l["epochs_total"] == 2 and "loss" in l for _, l in epochs)
