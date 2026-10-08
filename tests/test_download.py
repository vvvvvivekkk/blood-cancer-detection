import json
import os
import re
from pathlib import Path

import pytest

from src import config as C
from src import download
from src.data import list_images


def test_demo_data_has_200_images_4_classes(empty_root):
    dest, n = download.make_demo_data()
    items = list_images(dest)
    assert n == len(items) == 200
    assert {lab for _, lab in items} == set(C.CLASS_NAMES)


def test_credentials_roundtrip(tmp_path):
    f = tmp_path / ".kaggle" / "kaggle.json"
    assert not download.has_credentials(f)
    download.save_credentials(" alice ", "k3y", f)
    assert json.loads(f.read_text()) == {"username": "alice", "key": "k3y"}
    assert download.has_credentials(f)


def test_download_uses_kagglehub_and_copies_class_folders(empty_root, tmp_path, monkeypatch):
    import kagglehub
    fake = tmp_path / "cache" / "Blood cell Cancer [ALL]"
    download.make_synthetic(fake, 1, 16)
    monkeypatch.setattr(kagglehub, "dataset_download", lambda handle: str(fake.parent))
    dest, n = download.download_dataset()
    assert n == 4 and dest == C.RAW_DIR
    assert len(list_images(C.RAW_DIR)) == 4


def test_resolver_order_env_token_then_access_token_then_kaggle_json(tmp_path, monkeypatch):
    token_file, kaggle_json = tmp_path / "access_token", tmp_path / "kaggle.json"
    resolve = lambda: download.resolve_credentials(kaggle_json, token_file)  # noqa: E731
    assert resolve() is None
    download.save_credentials("alice", "k3y", kaggle_json)
    assert resolve() == str(kaggle_json)
    download.save_access_token("tok-from-file", token_file)
    assert resolve() == str(token_file)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok-from-env")
    assert resolve() == "KAGGLE_API_TOKEN env var"  # env var wins over both files
    assert "tok" not in resolve()  # never returns the secret


def test_access_token_validation_and_never_inside_repo(tmp_path):
    p = download.save_access_token("  abc123  ", tmp_path / "access_token")
    assert p.read_text() == "abc123\n"
    with pytest.raises(ValueError, match="one line"):
        download.save_access_token("two words", tmp_path / "t")
    with pytest.raises(ValueError, match="inside the project"):
        download.save_access_token("abc123", C.REPO / "access_token")
    assert not (C.REPO / "access_token").exists()


# A real token is KGAT_ + 32 hex characters. The pattern is split so this file does not match itself.
TOKEN_RE = re.compile(rb"KGAT" rb"_[0-9A-Za-z]{16,}")
SKIP_DIRS = {".git", "raw", "demo", "__pycache__", ".pytest_cache"}
BINARY = {".keras", ".pkl", ".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".zip"}


def test_no_kaggle_token_anywhere_in_the_repo():
    hits = []
    for dirpath, dirs, files in os.walk(C.REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() not in BINARY and TOKEN_RE.search(p.read_bytes()):
                hits.append(str(p.relative_to(C.REPO)))
    assert not hits, f"Kaggle token found in: {hits}"
