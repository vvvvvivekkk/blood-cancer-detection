"""Single entry point.

    python run.py                                   -> Streamlit UI
    python run.py --cli {download,demo-data,splits,train,ensemble,evaluate,all} [--smoke]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", choices=["download", "demo-data", "splits", "train", "ensemble", "evaluate", "all"])
    ap.add_argument("--smoke", action="store_true", help="data/sample, 1 epoch/phase, batch 4, CPU, offline")
    ap.add_argument("--models", nargs="+", help="backbones to train (default: all three)")
    ap.add_argument("--epochs", type=int, help="total epochs per backbone (default from config)")
    a = ap.parse_args()

    if not a.cli:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
        cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py")]
        return subprocess.call(cmd, cwd=ROOT, env=env)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(ROOT))
    from src import pipeline

    def log(stage, epoch, logs):
        shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in (logs or {}).items()}
        print(f"[{stage}] {epoch} {shown}", flush=True)

    print(pipeline.run(a.cli, smoke=a.smoke, progress_cb=log, models=a.models, epochs=a.epochs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
