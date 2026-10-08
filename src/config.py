"""Single source of truth: class names, paths, image size, seed, hyperparameters."""
import os
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Index -> name. Matches the Kaggle folder names.
CLASS_NAMES = ["Benign", "[Malignant] early Pre-B", "[Malignant] Pre-B", "[Malignant] Pro-B"]
NUM_CLASSES = len(CLASS_NAMES)

BACKBONES = ["mobilenetv2", "efficientnetb0", "densenet121"]
DISPLAY_NAMES = {
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNetB0",
    "densenet121": "DenseNet121",
    "voting": "Ensemble-Voting",
    "stacking": "Ensemble-Stacking",
}
ENSEMBLES = ["voting", "stacking"]
DEFAULT_ENSEMBLE_MODE = "voting"

SEED = 42
IMG_SIZE = 224
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
UPLOAD_EXTS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
MAX_UPLOAD_MB = 10

# Training
BATCH_SIZE = 32
EPOCHS = 15            # total, both phases
HEAD_EPOCHS = 5        # phase 1 (frozen base)
LR_HEAD = 1e-3
LR_FINE = 1e-5
UNFREEZE_FRACTION = 0.30
DROPOUT = 0.3
EARLY_STOP_PATIENCE = 5
LR_PLATEAU_PATIENCE = 2
LR_PLATEAU_FACTOR = 0.2
MIN_LR = 1e-7

# Augmentation (train only)
AUG_ROTATION_DEG = 20
AUG_ZOOM = 0.10
AUG_BRIGHTNESS = 0.10
AUG_CONTRAST = 0.10

# Ensemble
META_C = 1.0
META_MAX_ITER = 1000

# Targets from ENDGOAL.md, checked in results/NOTES.md
TARGET_ACCURACY = 0.96
TARGET_MACRO_F1 = 0.95
TARGET_MARGIN_PP = 0.5
TARGET_CLASS_RECALL = 0.90

# Smoke mode: plumbing check on data/sample, CPU, offline
SMOKE_IMG_SIZE = 96
SMOKE_BATCH_SIZE = 4
SMOKE_EPOCHS_PER_PHASE = 1

# Data sources
KAGGLE_DATASET = "mohammadamireshraghi/blood-cell-cancer-all-4class"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
DEMO_PER_CLASS = 50
DEMO_IMG_PX = 128
SAMPLE_PER_CLASS = 2
SAMPLE_IMG_PX = 64
SAMPLE_DIR = REPO / "data" / "sample"
README_PATH = REPO / "README.md"
GRADCAM_ALPHA = 0.4


def set_root(root):
    """Point every artifact path at `root` (tests use a tmp dir)."""
    global ROOT, DATA_DIR, RAW_DIR, DEMO_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR
    global HISTORY_DIR, FIGURES_DIR, WEIGHTS_JSON, META_CLF, MODE_FILE, METRICS_JSON
    ROOT = Path(root)
    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    DEMO_DIR = DATA_DIR / "demo"
    SPLITS_DIR = DATA_DIR / "splits"
    MODELS_DIR = ROOT / "models"
    RESULTS_DIR = ROOT / "results"
    HISTORY_DIR = RESULTS_DIR / "history"
    FIGURES_DIR = RESULTS_DIR / "figures"
    WEIGHTS_JSON = MODELS_DIR / "ensemble_weights.json"
    META_CLF = MODELS_DIR / "meta_clf.pkl"
    MODE_FILE = MODELS_DIR / "ensemble_mode.txt"
    METRICS_JSON = RESULTS_DIR / "metrics.json"


set_root(os.environ.get("BCE_ROOT", REPO))


def model_path(name):
    return MODELS_DIR / f"{name}.keras"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
