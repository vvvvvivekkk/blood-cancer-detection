import json

from src import config as C
from src import download
from src.data import list_images


def test_demo_data_has_200_images_4_classes(empty_root):
    dest, n = download.make_demo_data()
    items = list_images(dest)
    assert n == len(items) == 200
    assert {lab for _, lab in items} == set(C.CLASS_NAMES)


def test_credentials_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
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
