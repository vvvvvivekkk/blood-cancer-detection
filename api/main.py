"""FastAPI backend: JSON under /api/, images under /figures /media /docs-assets, frontend at /."""
import asyncio
import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import List
from urllib.parse import quote

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.jobs import Busy, JobRunner, friendly
from src import config as C
from src import pipeline
from src.data import data_source, list_images, load_split
from src.download import has_credentials, save_access_token, save_credentials
from src.status import status

log = logging.getLogger("api")
FRONTEND = C.REPO / "frontend"
SAMPLES_PER_CLASS = 4
# Relative cost of each stage for the overall progress bar (DenseNet121 is the slowest backbone).
STAGE_WEIGHT = {"download": 2, "demo-data": 1, "splits": 1, "mobilenetv2": 2, "efficientnetb0": 3,
                "densenet121": 4, "ensemble": 1, "evaluate": 2}
# Rough CPU training cost (seconds per training image per epoch at 224 px), for the UI estimate only.
CPU_SEC_PER_IMAGE_EPOCH = {"mobilenetv2": 0.05, "efficientnetb0": 0.08, "densenet121": 0.15}
SMOKE_SEC_PER_BACKBONE = 15
STEP_OF_STAGE = {"download": "data", "demo-data": "data", "splits": "data", "ensemble": "ensemble",
                 "evaluate": "evaluate", **{b: "train" for b in C.BACKBONES}}
MEDIA_DIRS = {"sample": lambda: C.SAMPLE_DIR, "demo": lambda: C.DEMO_DIR, "kaggle": lambda: C.RAW_DIR}

app = FastAPI(title="ALL Hybrid Ensemble API", version="1.0")
jobs = JobRunner()


# ---------- errors: always {"detail": "<readable message>"} ----------

@app.exception_handler(StarletteHTTPException)
async def _http(_, exc):
    return JSONResponse({"detail": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation(_, exc):
    msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
    return JSONResponse({"detail": f"Invalid request: {msg}"}, status_code=422)


@app.exception_handler(Busy)
async def _busy(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(FileNotFoundError)
async def _missing(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(ValueError)
async def _bad(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def _crash(_, exc):
    log.exception("unhandled error")
    return JSONResponse({"detail": f"Internal error: {friendly(exc)}"}, status_code=500)


# ---------- request bodies ----------

class Creds(BaseModel):
    username: str = Field(min_length=1)
    key: str = Field(min_length=1)


class TokenBody(BaseModel):
    token: str = Field(min_length=1)


class SmokeBody(BaseModel):
    smoke: bool = False


class SettingsBody(BaseModel):
    active_mode: str


class TrainBody(BaseModel):
    backbones: List[str] = Field(default_factory=lambda: list(C.BACKBONES), min_length=1)
    epochs: int = Field(default=C.EPOCHS, ge=2, le=200)
    smoke: bool = True


# ---------- jobs ----------

def _stage(name, **kw):
    """(stage, weight, fn) running one pipeline stage with the job's progress callback."""
    return name, STAGE_WEIGHT.get(name, 1), lambda job: pipeline.run(name, progress_cb=job.callback, **kw)


def _train_steps(backbones, smoke, epochs=None):
    return [(b, STAGE_WEIGHT[b], lambda job, b=b: pipeline.run("train", smoke=smoke, progress_cb=job.callback,
                                                                models=[b], epochs=epochs)) for b in backbones]


def _start(kind, steps):
    job = jobs.start(kind, steps)
    return {"job_id": job.id, "job": job.to_dict()}


@app.post("/api/data/download", status_code=202)
def download():
    if not has_credentials():
        raise HTTPException(400, "No Kaggle credentials. Paste an access token or save your username and API key first.")
    return _start("download", [_stage("download")])


@app.post("/api/data/demo", status_code=202)
def demo():
    return _start("demo-data", [_stage("demo-data")])


@app.post("/api/train", status_code=202)
def train(body: TrainBody):
    unknown = [b for b in body.backbones if b not in C.BACKBONES]
    if unknown:
        raise HTTPException(400, f"Unknown backbone(s): {', '.join(unknown)}. Choose from {', '.join(C.BACKBONES)}.")
    if not status()["splits"]:
        raise HTTPException(409, "No train/val/test splits yet. Make the splits on the Data view first.")
    return _start("train", _train_steps(body.backbones, body.smoke, body.epochs))


@app.post("/api/ensemble", status_code=202)
def ensemble():
    return _start("ensemble", [_stage("ensemble")])


@app.post("/api/evaluate", status_code=202)
def evaluate():
    return _start("evaluate", [_stage("evaluate")])


@app.post("/api/pipeline/all", status_code=202)
def pipeline_all(body: SmokeBody = SmokeBody()):
    steps = []
    if not body.smoke and data_source() is None:
        steps.append(_stage("download" if has_credentials() else "demo-data"))
    steps.append(_stage("splits", smoke=body.smoke))
    steps += _train_steps(C.BACKBONES, body.smoke)
    steps += [_stage("ensemble"), _stage("evaluate")]
    return _start("pipeline", steps)


def _job(job_id):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"No job with id {job_id}.")
    return job


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str):
    return _job(job_id).to_dict()


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    job = _job(job_id)

    async def events():
        seen, idle = -1, 0
        while True:
            if job.version != seen:
                seen, idle = job.version, 0
                d = job.to_dict()
                yield f"data: {json.dumps(d)}\n\n"
                if d["state"] in ("done", "failed"):
                    return
            else:
                idle += 1
                if idle % 20 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- status / data ----------

def _steps(s, active):
    stage = active.stage if active and active.state in ("queued", "running") else None
    running = STEP_OF_STAGE.get(stage)
    done = {"data": s["splits"], "train": all(s["models"].values()), "ensemble": s["voting"] and s["stacking"],
            "evaluate": s["metrics"], "predict": any(s["models"].values())}
    labels = {"data": "Data", "train": "Train", "ensemble": "Ensemble", "evaluate": "Evaluate", "predict": "Predict"}
    return [{"key": k, "label": labels[k], "state": "running" if k == running else "done" if done[k] else "pending"}
            for k in labels]


@app.get("/api/status")
def api_status():
    s = status()
    active = jobs.active if jobs.active_id() else None
    n_train = len(load_split("train")) if s["splits"] else 0
    return {**s, "demo_mode": s["demo"], "active_job": active.id if active else None,
            "active_job_kind": active.kind if active else None, "steps": _steps(s, active),
            "train_images": n_train, "class_names": C.CLASS_NAMES, "backbones": C.BACKBONES,
            "display_names": C.DISPLAY_NAMES, "default_epochs": C.EPOCHS,
            "estimate": {"cpu_sec_per_image_epoch": CPU_SEC_PER_IMAGE_EPOCH,
                         "smoke_sec_per_backbone": SMOKE_SEC_PER_BACKBONE},
            "max_upload_mb": C.MAX_UPLOAD_MB, "upload_exts": C.UPLOAD_EXTS, "settings": _settings()}


@app.get("/api/health")
def health():
    s = status()
    return {"status": "ok", "models_loaded": [m for m in C.BACKBONES if C.model_path(m).is_file()],
            "ensemble_mode": s["ensemble_mode"], "active_mode": s["active_mode"], "active_job": jobs.active_id()}


def _settings():
    from src.predict import ensemble_mode, mode_label
    active = C.get_active_mode()
    return {"active_mode": active, "options": C.ACTIVE_MODES,
            "label": mode_label("single" if active == C.SINGLE_MODEL else ensemble_mode()),
            "single_model_trained": C.model_path(C.SINGLE_MODEL).is_file()}


@app.get("/api/settings")
def get_settings():
    return _settings()


@app.post("/api/settings")
def update_settings(body: SettingsBody):
    C.set_active_mode(body.active_mode)  # ValueError -> 400
    return _settings()


@app.post("/api/data/kaggle-creds")
def kaggle_creds(body: Creds):
    return {"saved": str(save_credentials(body.username, body.key)), "credentials": True}


@app.post("/api/data/kaggle-token")
def kaggle_token(body: TokenBody):
    return {"saved": str(save_access_token(body.token)), "credentials": True}


def _counts():
    f = C.RESULTS_DIR / "split_counts.csv"
    if not f.is_file():
        return None
    df = pd.read_csv(f, index_col="class")
    return {split: {c: int(df.loc[c, split]) for c in C.CLASS_NAMES} for split in df.columns}


@app.post("/api/data/splits")
def splits(body: SmokeBody = SmokeBody()):
    if jobs.active_id():
        raise Busy("A job is running. Make the splits after it finishes.")
    pipeline.run("splits", smoke=body.smoke)
    return {"counts": _counts(), "source": status()["source"]}


def _media_url(source, path):
    rel = Path(path).resolve().relative_to(MEDIA_DIRS[source]().resolve()).as_posix()
    return f"/media/{source}/{quote(rel)}"


@app.get("/api/data/samples")
def samples():
    src = data_source()
    source, root = src if src else ("sample", C.SAMPLE_DIR)
    items = list_images(root)
    out = []
    for i, name in enumerate(C.CLASS_NAMES):
        for p in [p for p, lab in items if lab == name][:SAMPLES_PER_CLASS]:
            out.append({"url": _media_url(source, p), "label": name, "label_idx": i, "name": Path(p).name})
    return {"source": source, "samples": out, "counts": _counts()}


@app.get("/media/{source}/{rel:path}")
def media(source: str, rel: str):
    if source not in MEDIA_DIRS:
        raise HTTPException(404, "Unknown image source.")
    base = MEDIA_DIRS[source]().resolve()
    p = (base / rel).resolve()
    if base not in p.parents or not p.is_file() or p.suffix.lower() not in C.IMAGE_EXTS:
        raise HTTPException(404, "Image not found.")
    return FileResponse(p)


# ---------- results ----------

def _history(name):
    f = C.HISTORY_DIR / f"{name}.csv"
    if not f.is_file():
        return None
    h = pd.read_csv(f)
    h["epoch"] = h["epoch"] + 1
    return json.loads(h.to_json(orient="records"))


@app.get("/api/results")
def results():
    histories = {n: _history(n) for n in C.BACKBONES}
    if not C.METRICS_JSON.is_file():
        return {"available": False, "histories": histories}
    m = json.loads(C.METRICS_JSON.read_text())
    best = m["best_ensemble"]
    single = C.get_active_mode() == C.SINGLE_MODEL
    active_key = C.SINGLE_MODEL if single else best  # the row Predict uses
    rows = [{"key": k, "name": C.DISPLAY_NAMES[k], "is_ensemble": k in C.ENSEMBLES, "selected": k == active_key,
             "highlighted": k == C.SINGLE_MODEL if single else k in C.ENSEMBLES,
             **{f: v[f] for f in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                                  "kappa", "roc_auc_ovr")}} for k, v in m["models"].items()]
    backbones = [r for r in rows if not r["is_ensemble"]]
    top = max(backbones, key=lambda r: r["accuracy"]) if backbones else None
    figs = {p.name: f"/figures/{p.name}?v={int(p.stat().st_mtime)}" for p in sorted(C.FIGURES_DIR.glob("*.png"))}
    notes = C.RESULTS_DIR / "NOTES.md"
    return {"available": True, "meta": m["meta"], "best_ensemble": best, "ensemble_mode": status()["ensemble_mode"],
            "active_mode": C.get_active_mode(), "active_key": active_key,
            "rows": rows, "per_class": {k: v["per_class"] for k, v in m["models"].items()},
            "confusion": {k: v["confusion_matrix"] for k, v in m["models"].items()},
            "best_backbone": top["key"] if top else None,
            "gain_pp": 100 * (m["models"][best]["accuracy"] - top["accuracy"]) if top else None,
            "figures": figs, "histories": histories, "class_names": C.CLASS_NAMES,
            "notes": notes.read_text(encoding="utf-8") if notes.is_file() else None,
            "downloads": {"metrics.json": "/api/results/files/metrics.json",
                          "comparison_table.md": "/api/results/files/comparison_table.md"}}


@app.get("/api/results/files/{name}")
def result_file(name: str):
    if name not in ("metrics.json", "comparison_table.md", "NOTES.md"):
        raise HTTPException(404, "Unknown results file.")
    p = C.RESULTS_DIR / name
    if not p.is_file():
        raise HTTPException(404, f"{name} has not been generated yet. Run the evaluation.")
    return FileResponse(p, filename=name)


@app.get("/figures/{name}")
def figure(name: str):
    p = C.FIGURES_DIR / name
    if "/" in name or "\\" in name or p.suffix != ".png" or not p.is_file():
        raise HTTPException(404, "Figure not found.")
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


@app.get("/docs-assets/architecture.png")
def architecture():
    return FileResponse(C.REPO / "docs" / "architecture.png")


def _md_table(text):
    rows = [ln.strip().strip("|").split("|") for ln in text.splitlines() if ln.strip().startswith("|")]
    rows = [[c.strip() for c in r] for r in rows if not re.fullmatch(r"[\s|:-]+", "|".join(r))]
    return {"header": rows[0], "rows": rows[1:]} if rows else {"header": [], "rows": []}


@app.get("/api/about")
def about():
    return {"mapping": _md_table((C.REPO / "docs" / "paper_mapping.md").read_text(encoding="utf-8")),
            "architecture_url": "/docs-assets/architecture.png", "dataset": C.KAGGLE_DATASET,
            "class_names": C.CLASS_NAMES}


# ---------- predict ----------

def _png64(arr):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    from src.predict import predict_image

    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in C.UPLOAD_EXTS:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Use one of: {', '.join(C.UPLOAD_EXTS)}.")
    data = file.file.read(C.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"File is larger than {C.MAX_UPLOAD_MB} MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Could not read the image. The file may be corrupt.") from None
    r = predict_image(img, with_gradcam=True)
    cam = r.pop("gradcam")
    original = img.resize(cam.shape[1::-1], Image.BILINEAR)
    weights = (json.loads(C.WEIGHTS_JSON.read_text()) if C.WEIGHTS_JSON.is_file() else {}).get("weights", {})
    return {**r, "gradcam_png_base64": _png64(cam), "original_png_base64": _png64(original),
            "voting_weights": weights, "display_names": C.DISPLAY_NAMES}


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
