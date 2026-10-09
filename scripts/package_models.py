"""Package the trained models for a GitHub release, and verify a download.

    python scripts/package_models.py              # build dist/models.zip + models/MODELS_SHA256.txt
    python run.py --fetch-models <url>            # download + extract a release models.zip into models/
    python run.py --check                         # verify models/ against models/MODELS_SHA256.txt

The .keras backbones and meta_clf.pkl are gitignored (too big / regenerable), so they ship as a
release asset instead. MODELS_SHA256.txt is committed so a fresh clone can verify the download.
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DIST = ROOT / "dist"
ZIP = DIST / "models.zip"
SHA_FILE = MODELS / "MODELS_SHA256.txt"

# Everything the ensemble needs at inference time (no training history / caches).
MEMBERS = ["mobilenetv2.keras", "efficientnetb0.keras", "densenet121.keras",
           "ensemble_weights.json", "meta_clf.pkl", "ensemble_mode.txt", "ood_stats.json"]


def _sha256(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected():
    """{name: sha256} from the committed MODELS_SHA256.txt (sha256sum format)."""
    out = {}
    if SHA_FILE.is_file():
        for line in SHA_FILE.read_text().splitlines():
            if "  " in line:
                h, name = line.split("  ", 1)
                out[name.strip()] = h.strip()
    return out


def write_sha256():
    lines = [f"{_sha256(MODELS / m)}  {m}" for m in MEMBERS if (MODELS / m).is_file()]
    if ZIP.is_file():
        lines.append(f"{_sha256(ZIP)}  models.zip")
    SHA_FILE.write_text("\n".join(lines) + "\n")
    return SHA_FILE


def package():
    missing = [m for m in MEMBERS if not (MODELS / m).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot package; missing {', '.join(missing)}. "
                                "Train the backbones and fit + evaluate the ensemble first.")
    DIST.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for m in MEMBERS:
            z.write(MODELS / m, arcname=m)
    write_sha256()
    return ZIP, ZIP.stat().st_size


def check():
    """Verify models/ against MODELS_SHA256.txt. Returns 0 (PASS) or 1 (FAIL)."""
    exp = _expected()
    if not exp:
        print("No models/MODELS_SHA256.txt to check against — run scripts/package_models.py first.", flush=True)
        return 1
    missing = [m for m in MEMBERS if not (MODELS / m).is_file()]
    bad = [m for m in MEMBERS if (MODELS / m).is_file() and m in exp and _sha256(MODELS / m) != exp[m]]
    for m in missing:
        print(f"MISSING  {m}", flush=True)
    for m in bad:
        print(f"MISMATCH {m}", flush=True)
    ok = not missing and not bad
    print("CHECK PASS" if ok else "CHECK FAIL", flush=True)
    return 0 if ok else 1


def fetch_models(url):
    """Download a release models.zip (http(s):// or file://), verify it, extract into models/."""
    DIST.mkdir(exist_ok=True)
    print(f"Downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as r:  # noqa: S310 - user supplies their own release URL
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    exp = _expected().get("models.zip")
    if exp and got != exp:
        raise ValueError(f"models.zip sha256 mismatch: expected {exp[:12]}…, got {got[:12]}…")
    ZIP.write_bytes(data)
    MODELS.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(MODELS)
    print(f"Extracted into {MODELS} ({len(data) / 1e6:.1f} MB){' — sha256 verified' if exp else ''}.", flush=True)
    return check()


if __name__ == "__main__":
    zp, size = package()
    print(f"Wrote {zp} ({size / 1e6:.1f} MB) and {SHA_FILE}")
    sys.exit(0)
