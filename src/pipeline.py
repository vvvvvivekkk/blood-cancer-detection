"""Stage orchestration shared by run.py and the Streamlit Home page."""
from src import config as C

STAGES = ["download", "demo-data", "splits", "train", "ensemble", "evaluate", "all"]


def run(stage, smoke=False, progress_cb=None, models=None, epochs=None):
    """Run one stage (or 'all'). Returns a short human-readable result string."""
    if stage == "download":
        from src.download import download_dataset
        return "downloaded %d images to %s" % download_dataset(progress_cb=progress_cb)[::-1]
    if stage == "demo-data":
        from src.download import make_demo_data
        return "demo data: %d images in %s" % make_demo_data()[::-1]
    if stage == "splits":
        from src.data import make_splits
        counts = make_splits(smoke)
        return "splits written:\n" + counts.to_string()
    if stage == "train":
        from src.train import train_all
        paths = train_all(models or C.BACKBONES, smoke=smoke, epochs=epochs, progress_cb=progress_cb)
        return "trained: " + ", ".join(str(p) for p in paths)
    if stage == "ensemble":
        from src.ensemble import fit_all
        info = fit_all(progress_cb=progress_cb)
        return "ensemble weights: " + ", ".join(f"{k}={v:.3f}" for k, v in info["weights"].items())
    if stage == "evaluate":
        from src.evaluate import run as evaluate
        m = evaluate(progress_cb=progress_cb)
        return ((C.RESULTS_DIR / "comparison_table.md").read_text(encoding="utf-8")
                + f"\nbest ensemble: {m['best_ensemble']}")
    if stage == "all":
        out = []
        if not smoke:
            from src.data import data_source
            from src.download import has_credentials
            if data_source() is None:
                out.append(run("download" if has_credentials() else "demo-data", progress_cb=progress_cb))
        for s in ("splits", "train", "ensemble", "evaluate"):
            out.append(run(s, smoke, progress_cb, models, epochs))
        return "\n\n".join(out)
    raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
