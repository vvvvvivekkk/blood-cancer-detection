import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "app")]

from src import config as C  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_kaggle_credentials(tmp_path_factory, monkeypatch):
    """Tests never see the developer's real Kaggle token/kaggle.json, so nothing hits the network."""
    home = tmp_path_factory.mktemp("kaggle_home")
    monkeypatch.setattr(C, "KAGGLE_JSON", home / "kaggle.json")
    monkeypatch.setattr(C, "KAGGLE_ACCESS_TOKEN", home / "access_token")
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def empty_root(tmp_path):
    """Point every artifact path at an empty dir."""
    old = C.ROOT
    C.set_root(tmp_path)
    yield tmp_path
    C.set_root(old)


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory):
    """Run the full smoke pipeline once into a tmp root; yields (root, progress-callback calls)."""
    from src import pipeline
    root = tmp_path_factory.mktemp("smoke")
    old = C.ROOT
    C.set_root(root)
    calls = []
    pipeline.run("all", smoke=True, progress_cb=lambda *a: calls.append(a))
    C.set_root(old)
    return root, calls


@pytest.fixture
def smoke(smoke_root):
    old = C.ROOT
    C.set_root(smoke_root[0])
    yield smoke_root
    C.set_root(old)


@pytest.fixture
def single_mode(smoke):
    """Smoke artifacts with Predict switched to MobileNetV2 alone; always switched back afterwards."""
    C.set_active_mode(C.SINGLE_MODEL)
    yield smoke
    C.ACTIVE_MODE_FILE.unlink(missing_ok=True)
