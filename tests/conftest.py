import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "app")]

from src import config as C  # noqa: E402


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
    for stage in ("splits", "train", "ensemble"):
        pipeline.run(stage, smoke=True, progress_cb=lambda *a: calls.append(a))
    C.set_root(old)
    return root, calls


@pytest.fixture
def smoke(smoke_root):
    old = C.ROOT
    C.set_root(smoke_root[0])
    yield smoke_root
    C.set_root(old)
