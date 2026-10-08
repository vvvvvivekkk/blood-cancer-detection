"""Dataset acquisition: kagglehub download, credential storage, synthetic demo data."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src import config as C
from src.data import list_images

# One look per class: (shape, RGB colour). Deliberately easy to tell apart.
DEMO_STYLE = {
    C.CLASS_NAMES[0]: ("circle", (120, 60, 160)),
    C.CLASS_NAMES[1]: ("square", (40, 80, 200)),
    C.CLASS_NAMES[2]: ("triangle", (40, 150, 70)),
    C.CLASS_NAMES[3]: ("ring", (220, 120, 30)),
}


def resolve_credentials(kaggle_json=None, access_token=None):
    """Which Kaggle credential will be used, in kagglehub's order; None if there is none.

    1. KAGGLE_API_TOKEN env var  2. ~/.kaggle/access_token  3. ~/.kaggle/kaggle.json
    (4. legacy KAGGLE_USERNAME + KAGGLE_KEY env vars). Never returns the secret itself.
    """
    token_file = Path(access_token or C.KAGGLE_ACCESS_TOKEN)
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return "KAGGLE_API_TOKEN env var"
    if token_file.is_file() and token_file.read_text().strip():
        return str(token_file)
    if Path(kaggle_json or C.KAGGLE_JSON).is_file():
        return str(Path(kaggle_json or C.KAGGLE_JSON))
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY env vars"
    return None


def has_credentials(kaggle_json=None, access_token=None):
    return resolve_credentials(kaggle_json, access_token) is not None


def save_access_token(token, path=None):
    """Write a new-style Kaggle access token (KGAT_...) to ~/.kaggle/access_token, never into the repo."""
    token = (token or "").strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("Paste the access token as one line, with no spaces.")
    path = Path(path or C.KAGGLE_ACCESS_TOKEN)
    if C.REPO.resolve() in path.resolve().parents:
        raise ValueError("Refusing to store a Kaggle token inside the project folder.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass  # Windows ignores POSIX modes
    return path


def save_credentials(username, key, kaggle_json=None):
    username, key = username.strip(), key.strip()
    if not username or not key:
        raise ValueError("Both Kaggle username and key are required.")
    path = Path(kaggle_json or C.KAGGLE_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"username": username, "key": key}))
    try:
        path.chmod(0o600)
    except OSError:
        pass  # Windows ignores POSIX modes
    return path


def download_dataset(dest=None, progress_cb=None):
    """Fetch the Kaggle dataset via kagglehub and copy its 4 class folders into data/raw/."""
    import kagglehub

    dest = Path(dest or C.RAW_DIR)
    if progress_cb:
        progress_cb("download", 0, {"msg": "downloading from Kaggle"})
    src_root = Path(kagglehub.dataset_download(C.KAGGLE_DATASET))
    items = list_images(src_root)
    if not items:
        raise RuntimeError(f"No class folders {C.CLASS_NAMES} found in {src_root}")
    for path, label in items:
        out = dest / label / Path(path).name
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
    if progress_cb:
        progress_cb("download", 1, {"images": len(items)})
    return dest, len(items)


def _draw(shape, colour, px, rng):
    bg = rng.normal(225, 12, (px, px, 3)) * np.array([1.0, 0.85, 0.9])
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    r = int(px * rng.uniform(0.18, 0.3))
    cx, cy = int(px * rng.uniform(0.35, 0.65)), int(px * rng.uniform(0.35, 0.65))
    col = tuple(int(np.clip(c + rng.integers(-20, 21), 0, 255)) for c in colour)
    box = [cx - r, cy - r, cx + r, cy + r]
    if shape == "circle":
        d.ellipse(box, fill=col)
    elif shape == "square":
        d.rectangle(box, fill=col)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    else:
        d.ellipse(box, outline=col, width=max(2, r // 3))
    return img


def make_synthetic(dest, per_class, px, seed=C.SEED, ext=".jpg"):
    """Write per_class images per class to dest/<class>/. Returns image count."""
    rng = np.random.default_rng(seed)
    dest = Path(dest)
    for label, (shape, colour) in DEMO_STYLE.items():
        (dest / label).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            _draw(shape, colour, px, rng).save(dest / label / f"{shape}_{i:03d}{ext}")
    return per_class * len(DEMO_STYLE)


def make_demo_data(dest=None):
    dest = Path(dest or C.DEMO_DIR)
    return dest, make_synthetic(dest, C.DEMO_PER_CLASS, C.DEMO_IMG_PX)


def make_sample_data(dest=None):
    return make_synthetic(dest or C.SAMPLE_DIR, C.SAMPLE_PER_CLASS, C.SAMPLE_IMG_PX, seed=0, ext=".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-data", action="store_true", help="generate synthetic demo images")
    ap.add_argument("--sample-data", action="store_true", help="regenerate data/sample/")
    a = ap.parse_args()
    if a.sample_data:
        print("sample images:", make_sample_data())
    elif a.demo_data:
        print("demo data: %s (%d images)" % make_demo_data())
    else:
        print("downloaded: %s (%d images)" % download_dataset())
