"""Copy EXPORT_PER_CLASS images per class from the TEST split into test_images/<class>/ for live demos.

    python run.py --cli export-test-images

Sanctioned exception to "the test split is read only by src/evaluate.py": this copies files for the
Preview demo. Nothing is trained, fitted or tuned on them, so evaluation stays clean.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data import resolve  # noqa: E402


def export_test_images(per_class=None, dest=None):
    """Returns (dest, n_copied). Re-running replaces the folder, so it always mirrors the current split."""
    per_class = per_class or C.EXPORT_PER_CLASS
    test_csv = C.SPLITS_DIR / "test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError("No test split yet. Make the splits first.")
    df = pd.read_csv(test_csv)
    picked = df.sample(frac=1, random_state=C.SEED).groupby("label", sort=False).head(per_class)
    dest = Path(dest or C.TEST_IMAGES_DIR)
    shutil.rmtree(dest, ignore_errors=True)
    for path, label in zip(picked["path"], picked["label"]):
        out = dest / label / Path(path).name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve(path), out)
    return dest, len(picked)


if __name__ == "__main__":
    print("exported %d test images to %s" % export_test_images()[::-1])
