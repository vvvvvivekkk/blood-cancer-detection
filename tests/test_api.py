import base64
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src import config as C

client = TestClient(app)
SAMPLE = next((C.SAMPLE_DIR / C.CLASS_NAMES[3]).iterdir())


def wait(job_id, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed"):
            assert j["state"] == "done", j["error"]
            return j
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish")


def png(data):
    return Image.open(io.BytesIO(base64.b64decode(data))).format == "PNG"


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_status_shape_with_nothing_on_disk(empty_root):
    s = client.get("/api/status").json()
    assert [x["key"] for x in s["steps"]] == ["data", "train", "ensemble", "evaluate", "predict"]
    assert all(x["state"] == "pending" for x in s["steps"])
    assert s["demo_mode"] is False and s["active_job"] is None and s["models"] == {m: False for m in C.BACKBONES}
    assert s["next_step"] and s["class_names"] == C.CLASS_NAMES


def test_demo_job_then_splits_and_samples(empty_root):
    r = client.post("/api/data/demo")
    assert r.status_code == 202
    j = wait(r.json()["job_id"])
    assert j["progress"] == 100 and j["logs"]
    assert client.get("/api/status").json()["demo_mode"] is True

    counts = client.post("/api/data/splits", json={"smoke": False}).json()["counts"]
    assert set(counts) == {"train", "val", "test"}
    assert all(set(c) == set(C.CLASS_NAMES) for c in counts.values())
    assert sum(sum(c.values()) for c in counts.values()) == 200

    samples = client.get("/api/data/samples").json()
    assert samples["source"] == "demo" and len(samples["samples"]) == 16
    assert {s["label"] for s in samples["samples"]} == set(C.CLASS_NAMES)
    img = client.get(samples["samples"][0]["url"])
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/")


def test_train_smoke_one_backbone_and_busy_409(empty_root):
    client.post("/api/data/splits", json={"smoke": True})
    r = client.post("/api/train", json={"backbones": ["mobilenetv2"], "epochs": 2, "smoke": True})
    assert r.status_code == 202
    busy = client.post("/api/ensemble")
    assert busy.status_code == 409 and "running" in busy.json()["detail"]
    assert client.post("/api/data/splits", json={"smoke": True}).status_code == 409
    j = wait(r.json()["job_id"])
    assert C.model_path("mobilenetv2").is_file()
    assert [h["epoch"] for h in j["history"]["mobilenetv2"]] == [1, 2]
    assert j["models"]["mobilenetv2"]["progress"] == 100


def test_train_rejects_bad_requests(empty_root):
    assert client.post("/api/train", json={"backbones": ["vgg16"]}).status_code == 400
    assert client.post("/api/train", json={"smoke": True}).status_code == 409  # no splits yet
    r = client.post("/api/train", json={"epochs": "many"})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    r = client.post("/api/data/download")
    assert r.status_code == 400 and "access token" in r.json()["detail"]  # no credentials in tests


def test_evaluate_job_then_results(smoke):
    wait(client.post("/api/evaluate").json()["job_id"])
    r = client.get("/api/results").json()
    assert r["available"] and len(r["rows"]) == 5
    assert sum(row["selected"] for row in r["rows"]) == 1
    assert {"gradcam_examples.png", "roc_ensemble.png"} <= set(r["figures"])
    assert all(len(cm) == 4 for cm in r["confusion"].values())
    assert set(r["histories"]) == set(C.BACKBONES)
    assert client.get(r["figures"]["roc_ensemble.png"]).status_code == 200
    assert client.get("/api/results/files/metrics.json").status_code == 200


def test_predict_sample_image(smoke):
    with open(SAMPLE, "rb") as f:
        r = client.post("/api/predict", files={"file": (SAMPLE.name, f, "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert d["class"] in C.CLASS_NAMES and 0 <= d["confidence"] <= 1
    assert set(d["probs"]) == set(C.CLASS_NAMES) and set(d["per_model_probs"]) == set(C.BACKBONES)
    assert d["mode"] in ("voting", "stacking")
    assert png(d["gradcam_png_base64"]) and png(d["original_png_base64"])


@pytest.mark.parametrize("name,size,msg", [
    ("notes.txt", 5, "Unsupported"),
    ("broken.png", 16, "corrupt"),
    ("huge.png", C.MAX_UPLOAD_MB * 1024 * 1024 + 1, "MB"),
], ids=["wrong-type", "corrupt", "oversize"])
def test_predict_bad_files_400(smoke, name, size, msg):
    data = b"0" * size
    r = client.post("/api/predict", files={"file": (name, data, "application/octet-stream")})
    assert r.status_code == 400 and msg in r.json()["detail"]


def test_predict_without_models_is_readable_409(empty_root):
    with open(SAMPLE, "rb") as f:
        r = client.post("/api/predict", files={"file": (SAMPLE.name, f, "image/png")})
    assert r.status_code == 409 and "Train" in r.json()["detail"]


def test_unknown_job_404():
    r = client.get("/api/jobs/doesnotexist")
    assert r.status_code == 404 and r.json() == {"detail": "No job with id doesnotexist."}


def test_stream_sends_job_state(empty_root):
    j = wait(client.post("/api/data/demo").json()["job_id"])
    with client.stream("GET", f"/api/jobs/{j['id']}/stream") as s:
        assert s.headers["content-type"].startswith("text/event-stream")
        first = next(line for line in s.iter_lines() if line.startswith("data: "))
    assert '"state": "done"' in first


def test_frontend_and_media_are_served():
    assert "ALL Hybrid Ensemble" in client.get("/").text
    assert client.get("/media/sample/../../README.md").status_code == 404
    assert client.get("/figures/..%2Fmetrics.json").status_code == 404


def test_settings_roundtrip_and_validation(empty_root):
    assert client.get("/api/settings").json()["active_mode"] == "ensemble"
    r = client.post("/api/settings", json={"active_mode": "mobilenetv2"})
    assert r.status_code == 200 and r.json()["label"] == "MobileNetV2 (single)"
    assert C.ACTIVE_MODE_FILE.read_text() == "mobilenetv2"
    s = client.get("/api/status").json()
    assert s["active_mode"] == "mobilenetv2" and s["settings"]["single_model_trained"] is False
    bad = client.post("/api/settings", json={"active_mode": "vgg16"})
    assert bad.status_code == 400 and "Unknown mode" in bad.json()["detail"]


def test_single_mode_predict_and_results_highlight(single_mode):
    with open(SAMPLE, "rb") as f:
        d = client.post("/api/predict", files={"file": (SAMPLE.name, f, "image/png")}).json()
    assert d["mode"] == "single" and d["mode_label"] == "MobileNetV2 (single)"
    assert list(d["per_model_probs"]) == ["mobilenetv2"]
    r = client.get("/api/results").json()
    assert [x["key"] for x in r["rows"] if x["highlighted"]] == ["mobilenetv2"]
    assert [x["key"] for x in r["rows"] if x["selected"]] == ["mobilenetv2"] and r["active_key"] == "mobilenetv2"


def test_ensemble_mode_highlights_ensemble_rows(smoke):
    r = client.get("/api/results").json()
    assert [x["key"] for x in r["rows"] if x["highlighted"]] == C.ENSEMBLES
    assert [x["key"] for x in r["rows"] if x["selected"]] == [r["best_ensemble"]]


def test_kaggle_token_endpoint_saves_outside_repo(empty_root):
    r = client.post("/api/data/kaggle-token", json={"token": "test-token-value"})
    assert r.status_code == 200 and C.KAGGLE_ACCESS_TOKEN.read_text().strip() == "test-token-value"
    s = client.get("/api/status").json()
    assert s["credentials"] and s["credential_source"] == str(C.KAGGLE_ACCESS_TOKEN)
    assert client.post("/api/data/kaggle-token", json={"token": "has spaces"}).status_code == 400


def test_demo_login_accepts_only_configured_credentials():
    ok = client.post("/api/login", json={"username": C.DEMO_USERNAME, "password": C.DEMO_PASSWORD})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    for user, pw in (("admin", "wrong"), ("root", "admin"), ("", "")):
        r = client.post("/api/login", json={"username": user, "password": pw})
        assert r.status_code == 401 and r.json()["detail"] == "Invalid username or password."


def test_export_test_images_from_test_split(smoke):
    import shutil
    try:
        assert client.get("/api/data/test-images").json()["images"] == []
        r = client.post("/api/data/export-test-images")
        assert r.status_code == 200 and r.json()["count"] == 4  # smoke test split: 1 image per class
        imgs = client.get("/api/data/test-images").json()["images"]
        assert {i["label"] for i in imgs} == set(C.CLASS_NAMES)
        test_paths = {p.rsplit("/", 1)[-1] for p in __import__("pandas").read_csv(C.SPLITS_DIR / "test.csv")["path"]}
        assert {i["name"] for i in imgs} <= test_paths  # only test-split images
        assert client.get(imgs[0]["url"]).status_code == 200
    finally:
        shutil.rmtree(C.TEST_IMAGES_DIR, ignore_errors=True)


def test_export_test_images_needs_splits(empty_root):
    r = client.post("/api/data/export-test-images")
    assert r.status_code == 409 and "splits" in r.json()["detail"]


def test_app_py_is_the_same_entry_point():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, str(C.REPO / "app.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "export-test-images" in out.stdout


def test_smoke_run_refuses_to_overwrite_real_models(empty_root):
    (C.SPLITS_DIR).mkdir(parents=True)
    (C.SPLITS_DIR / "meta.json").write_text('{"source": "kaggle", "smoke": false, "n_images": 3242}')
    C.MODELS_DIR.mkdir()
    C.model_path("mobilenetv2").write_bytes(b"real")
    for url, body in (("/api/pipeline/all", {"smoke": True}), ("/api/data/splits", {"smoke": True}),
                      ("/api/train", {"backbones": ["mobilenetv2"], "smoke": True})):
        r = client.post(url, json=body)
        assert r.status_code == 409 and "overwrite" in r.json()["detail"], (url, r.text)
    assert C.model_path("mobilenetv2").read_bytes() == b"real"


def test_predict_returns_mc_dropout_uncertainty(smoke):
    with open(SAMPLE, "rb") as f:
        d = client.post("/api/predict", files={"file": (SAMPLE.name, f, "image/png")}).json()
    u = d["uncertainty"]
    assert set(u) >= {"entropy", "mutual_information", "flag", "passes", "pass_top_probs", "thresholds"}
    assert 0 <= u["entropy"] <= 1 and u["mutual_information"] >= 0
    assert u["mutual_information"] <= u["entropy"] + 1e-6
    assert u["flag"] in ("HIGH", "MEDIUM", "LOW")
    assert u["passes"] == C.MC_PASSES_PREVIEW and len(u["pass_top_probs"]) == u["passes"]
    assert all(0 <= p <= 1 for p in u["pass_top_probs"])
    assert u["thresholds"]["review"] <= u["thresholds"]["low_confidence"]


def test_results_exposes_uncertainty_key(smoke):
    wait(client.post("/api/evaluate").json()["job_id"])
    r = client.get("/api/results").json()
    assert "uncertainty" in r  # None until thresholds are calibrated; the key must always be present
    u = r["uncertainty"]
    if u:
        assert u["passes"] >= 1 and 0 <= u["correct_flagged_pct"] <= 100
        assert set(u["flag_counts"]) == {"HIGH", "MEDIUM", "LOW"}
        assert sum(u["flag_counts"].values()) == u["n_test"]
