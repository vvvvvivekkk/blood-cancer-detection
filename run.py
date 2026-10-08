"""Single entry point.

    python run.py                                   -> web app (FastAPI + HTML/JS) at http://localhost:8000
    python run.py --streamlit                       -> fallback Streamlit app at http://localhost:8501
    python run.py --cli {download,demo-data,splits,train,ensemble,evaluate,all} [--smoke]
    python run.py --cli export-test-images          -> 10 test images per class into test_images/
"""
import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", choices=["download", "demo-data", "splits", "train", "ensemble", "evaluate", "all",
                                      "export-test-images"])
    ap.add_argument("--streamlit", action="store_true", help="launch the fallback Streamlit app")
    ap.add_argument("--smoke", action="store_true", help="data/sample, 1 epoch/phase, batch 4, CPU, offline")
    ap.add_argument("--models", nargs="+", help="backbones to train (default: all three)")
    ap.add_argument("--epochs", type=int, help="total epochs per backbone (default from config)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(ROOT))

    if a.streamlit:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
        cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py")]
        return subprocess.call(cmd, cwd=ROOT, env=env)

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

    from src import pipeline

    def log(stage, epoch, logs):
        shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in (logs or {}).items()}
        print(f"[{stage}] {epoch} {shown}", flush=True)

    print(pipeline.run(a.cli, smoke=a.smoke, progress_cb=log, models=a.models, epochs=a.epochs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
