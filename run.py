"""Single entry point.

    python run.py                                   -> web app (FastAPI + HTML/JS) at http://localhost:8000
    python run.py --cli {download,demo-data,splits,train,ensemble,evaluate,all} [--smoke]
    python run.py --cli export-test-images          -> 10 test images per class into test_images/
    python run.py --fetch-models [URL]              -> trained models into models/ (default: pinned release)
    python run.py --check                           -> verify models/ against models/MODELS_SHA256.txt
"""
import argparse
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _has_real_artifacts():
    meta = ROOT / "data" / "splits" / "meta.json"
    return (meta.is_file() and not json.loads(meta.read_text()).get("smoke", True)
            and any((ROOT / "models").glob("*.keras")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", choices=["download", "demo-data", "splits", "train", "ensemble", "evaluate", "all",
                                      "export-test-images"])
    ap.add_argument("--fetch-models", nargs="?", const="", metavar="URL",
                    help="download + extract models.zip into models/ (default: the pinned release)")
    ap.add_argument("--check", action="store_true", help="verify models/ against models/MODELS_SHA256.txt")
    ap.add_argument("--smoke", action="store_true", help="data/sample, 1 epoch/phase, batch 4, CPU, offline")
    ap.add_argument("--models", nargs="+", help="backbones to train (default: all three)")
    ap.add_argument("--epochs", type=int, help="total epochs per backbone (default from config)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(ROOT))
    for _stream in (sys.stdout, sys.stderr):  # tables use ≥; a cp1252 Windows console would crash on print
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if a.fetch_models is not None:
        from scripts.package_models import fetch_models
        return fetch_models(a.fetch_models or None)
    if a.check:
        from scripts.package_models import check
        return check()

    if not a.cli:
        import uvicorn
        url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
        print(f"ALL Hybrid Ensemble web app: {url}", flush=True)
        if not a.no_browser:
            threading.Timer(1.5, webbrowser.open, args=(url,)).start()
        uvicorn.run("api.main:app", host=a.host, port=a.port, log_level="info")
        return 0

    if a.cli == "export-test-images":
        from scripts.export_test_images import export_test_images
        print("exported %d test images to %s" % export_test_images()[::-1])
        return 0

    if a.smoke and "BCE_ROOT" not in os.environ and _has_real_artifacts():
        os.environ["BCE_ROOT"] = str(ROOT / ".smoke")
        print(f"Real-data models exist: smoke run writes to {ROOT / '.smoke'} instead.", flush=True)
    from src import pipeline

    def log(stage, epoch, logs):
        shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in (logs or {}).items()}
        print(f"[{stage}] {epoch} {shown}", flush=True)

    print(pipeline.run(a.cli, smoke=a.smoke, progress_cb=log, models=a.models, epochs=a.epochs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
