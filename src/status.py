"""What exists on disk. Cheap: no TensorFlow import."""
from src import config as C
from src.data import list_images, split_meta
from src.download import has_credentials


def status():
    meta = split_meta()
    kaggle = len(list_images(C.RAW_DIR)) if C.RAW_DIR.is_dir() else 0
    demo = len(list_images(C.DEMO_DIR)) if C.DEMO_DIR.is_dir() else 0
    models = {m: C.model_path(m).is_file() for m in C.BACKBONES}
    s = {
        "kaggle_images": kaggle,
        "demo_images": demo,
        "credentials": has_credentials(),
        "splits": all((C.SPLITS_DIR / f"{n}.csv").is_file() for n in ("train", "val", "test")),
        "split_meta": meta,
        "models": models,
        "voting": C.WEIGHTS_JSON.is_file(),
        "stacking": C.META_CLF.is_file(),
        "metrics": C.METRICS_JSON.is_file(),
        "ensemble_mode": C.MODE_FILE.read_text().strip() if C.MODE_FILE.is_file() else None,
        "active_mode": C.get_active_mode(),
    }
    source = meta["source"] if meta else ("kaggle" if kaggle else "demo" if demo else None)
    s["source"] = source
    s["demo"] = source == "demo"
    s["smoke"] = bool(meta and meta.get("smoke"))
    s["has_data"] = bool(kaggle or demo)
    s["next_step"] = next_step(s)
    return s


def next_step(s):
    if not s["has_data"] and not s["splits"]:
        return "Get data: download from Kaggle or use demo data (Data page)."
    if not s["splits"]:
        return "Make splits (Data page)."
    if not all(s["models"].values()):
        return "Train the backbones (Train page)."
    if not (s["voting"] and s["stacking"]):
        return "Fit the ensemble (Evaluate page)."
    if not s["metrics"]:
        return "Run the evaluation (Evaluate page)."
    return "All done. Try the Predict page."


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2))
