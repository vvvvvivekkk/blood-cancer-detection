"""Build the shareable client package: dist/blood-cancer-detection-client.zip

Contents:
  * every git-tracked file (the app, docs, the split model parts, tests, ...)
  * the UNPACKED models/ (3 .keras + ensemble files + uncertainty_thresholds.json
    + MODELS_SHA256.txt) so the client runs offline with no assembly step
  * START_HERE.txt and the setup/run scripts

Excluded: .git, data/raw, data/demo, data/splits, dist/ (the parts and the monolith),
verify/, .venv, __pycache__, .pytest_cache, models/history.db, and any Kaggle credential.

    python scripts/make_client_zip.py

Prints the zip's path, size and SHA-256.
"""
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "blood-cancer-detection-client.zip"

# Unpacked model files the client needs at inference time (shipped loose, not as zip parts).
MODEL_FILES = ["mobilenetv2.keras", "efficientnetb0.keras", "densenet121.keras",
               "ensemble_weights.json", "meta_clf.pkl", "ensemble_mode.txt", "ood_stats.json",
               "uncertainty_thresholds.json", "MODELS_SHA256.txt"]

# Loose (possibly untracked) files that must be in the client package.
EXTRA_FILES = ["START_HERE.txt", "setup.bat", "setup.sh", "run.bat", "run.sh"]

# Never ship these, even if a glob would catch them.
EXCLUDE_PREFIXES = ("dist/", "data/raw/", "data/demo/", "data/splits/", "verify/",
                    ".git/", ".venv/", ".agents/", ".claude/", "brag-output/")
EXCLUDE_NAMES = {"history.db", "kaggle.json", "access_token", "access_token.txt",
                 "skills-lock.json"}
EXCLUDE_SUFFIXES = (".pyc", ".token")


def _excluded(rel):
    rel = rel.replace("\\", "/")
    if any(rel.startswith(p) for p in EXCLUDE_PREFIXES):
        return True
    if "__pycache__/" in rel or "/.pytest_cache/" in rel or rel.startswith(".pytest_cache/"):
        return True
    if Path(rel).name in EXCLUDE_NAMES or rel.endswith(EXCLUDE_SUFFIXES):
        return True
    return False


def _sha256(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    members = {}  # arcname -> source path, deduped

    for rel in tracked:
        if _excluded(rel):
            continue
        src = ROOT / rel
        if src.is_file():
            members[rel] = src

    for m in MODEL_FILES:
        src = ROOT / "models" / m
        if not src.is_file():
            raise FileNotFoundError(
                f"models/{m} is missing. Run `python run.py --fetch-models` first so the client "
                f"zip ships ready-to-run models.")
        members[f"models/{m}"] = src

    for rel in EXTRA_FILES:
        src = ROOT / rel
        if src.is_file():
            members[rel] = src

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for arc in sorted(members):
            z.write(members[arc], arcname=f"blood-cancer-detection/{arc}")

    return OUT, len(members)


if __name__ == "__main__":
    # Safety: never package a real Kaggle credential that slipped into the tree.
    leaked = [m for m in EXCLUDE_NAMES
              if (ROOT / m).is_file() and m in {"kaggle.json", "access_token", "access_token.txt"}]
    if leaked:
        print(f"Refusing to build: credential file(s) present in repo root: {leaked}", file=sys.stderr)
        sys.exit(1)
    path, n = build()
    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {path}")
    print(f"  files : {n}")
    print(f"  size  : {size_mb:.1f} MB")
    print(f"  sha256: {_sha256(path)}")
