# CLAUDE.md — Blood Cancer Identification using Hybrid Ensemble Deep Learning

## What this project is
Replication of the IEEE paper **"Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique"** (J. Jayachitra, N. Umarkathaf, ICEARS 2023, IEEE Xplore doc 10084996).
Goal: classify peripheral blood smear (PBS) images into 4 classes of Acute Lymphoblastic Leukemia (ALL) using a **hybrid ensemble of transfer-learning CNNs**, beating every single backbone, and drive the whole pipeline from a **web dashboard** (FastAPI backend + vanilla HTML/CSS/JS frontend). A Streamlit app remains as a fallback.

Read `ENDGOAL.md` before starting any task. It is the definition of done.

## Dataset (fixed — do not substitute)
- Kaggle: `mohammadamireshraghi/blood-cell-cancer-all-4class` ("Blood Cells Cancer (ALL)", Aria et al.)
- 3,242 PBS images, 4 classes: `Benign`, `[Malignant] early Pre-B`, `[Malignant] Pre-B`, `[Malignant] Pro-B`
- Downloaded automatically by `src/download.py` with **kagglehub** into `data/raw/` (git-ignored). Never commit images (except `data/sample/`).
- Credentials: env `KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json` (the Data page can write it).
- No credentials → **demo data**: 200 synthetic 4-class images (distinct colour/shape per class) in `data/demo/`. Everything downstream runs on it; the UI shows a "DEMO DATA" banner.
- Canonical class order is defined ONCE in `src/config.py` as `CLASS_NAMES`. Never hard-code class lists elsewhere.
- Split: stratified 70 / 15 / 15 (train / val / test), seed 42 → `data/splits/{train,val,test}.csv` (+ `meta.json`). Test split is read **only** by `src/evaluate.py`.

## Architecture (the "hybrid ensemble")
Three ImageNet-pretrained backbones fine-tuned separately at 224×224: `MobileNetV2`, `EfficientNetB0`, `DenseNet121`.
Two ensemble heads over their softmax outputs — the pair is what makes it "hybrid":
- **Weighted soft voting** — weights ∝ each model's validation macro-F1 (normalised).
- **Stacked meta-classifier** — logistic regression on concatenated val-set probabilities (12 features).

Both are evaluated on test; the better (macro-F1) is persisted to `models/ensemble_mode.txt` and used by Predict.

**Single-model mode:** `config.ACTIVE_MODE` (`"ensemble"` | `"mobilenetv2"`, default `"ensemble"`) is overridden by `models/active_mode.txt` (written via `config.set_active_mode`, the sidebar switch, or `POST /api/settings`). In `mobilenetv2` mode `predict_image` loads only MobileNetV2 and returns `mode="single"`, `mode_label="MobileNetV2 (single)"`.

## Stack
- Python 3.10–3.12, TensorFlow/Keras (2.15 on py ≤ 3.11, 2.21 on py 3.12 — pinned with markers), scikit-learn, NumPy, Pandas, Pillow, OpenCV (headless), Matplotlib, Seaborn
- **Web UI**: FastAPI + uvicorn (`api/`), vanilla HTML/CSS/JS (`frontend/`), Chart.js from a pinned CDN, no build step, no npm
- Fallback **Streamlit** UI (`app/`), kagglehub for download
- Dev: Playwright for browser tests and screenshots (`requirements-dev.txt`)
- Grad-CAM hand-rolled in `src/gradcam.py` (no extra deps)
- Config in `src/config.py`; no magic numbers in scripts

## Repo layout
```
.
├── run.py               single entry point (UI or --cli stages)
├── CLAUDE.md README.md ENDGOAL.md LICENSE requirements.txt .gitignore
├── data/   raw/ demo/ (gitignored)  splits/ (generated)  sample/ (8 committed tiny images)
├── src/
│   ├── config.py      paths, CLASS_NAMES, IMG_SIZE, SEED, hyperparams, set_root()
│   ├── status.py      what exists on disk (used by UI + CLI)
│   ├── download.py    kagglehub download, credentials, demo-data generator
│   ├── data.py        make_splits, tf.data pipelines, train-only augmentation
│   ├── models.py      build_backbone, preprocess_fn, GRADCAM_LAYER
│   ├── train.py       two-phase fine-tune, callbacks, progress_cb
│   ├── ensemble.py    val probs, fit_voting, fit_stacking
│   ├── predict.py     predict_image
│   ├── evaluate.py    test metrics, figures, comparison table, README injection
│   ├── gradcam.py     make_heatmap, overlay
│   └── pipeline.py    stage orchestration shared by run.py and the UI
├── api/       main.py (routes, /api/*), jobs.py (one background job at a time)
├── frontend/  index.html  css/styles.css  js/{app,api,charts}.js  img/
├── app/   fallback Streamlit: Home.py  ui.py  pages/{1_Data,2_Train,3_Evaluate,4_Predict,5_About}.py
├── models/  results/  notebooks/train_colab.ipynb  docs/  tests/
```

## Commands
```bash
pip install -r requirements.txt
python run.py                                   # web UI → http://localhost:8000
python run.py --streamlit                       # fallback Streamlit UI → http://localhost:8501
python docs/capture_screenshots.py              # Playwright screenshots of every view (app must be running)
python run.py --cli download                    # kagglehub → data/raw/
python run.py --cli demo-data                   # 200 synthetic images → data/demo/
python run.py --cli splits|train|ensemble|evaluate [--smoke]
python run.py --cli all --smoke                 # whole pipeline on data/sample, CPU, ~1-2 min
python -m src.evaluate --write-readme-table     # refresh README results table
python -m pytest -q                             # must pass before any commit
```
Smoke mode: `data/sample/` (8 images), 1 epoch per phase, batch 4, 96 px, `weights=None` (offline). Plumbing check only.

## Rules for working in this repo
1. Pipeline must run end-to-end in `--smoke` mode on CPU before you claim anything works. Real training is done on Colab via `notebooks/train_colab.ipynb`.
2. Never read `data/splits/test.csv` outside `evaluate.py` (`data.load_split("test")` raises). Ensemble weights/meta-clf are fit on **val** only.
3. Every metric in `results/` comes from a script run. `results/metrics.json` is the single source of truth; the README table is generated from it.
4. Fixed seeds (`SEED=42`) for numpy, tf, random, sklearn.
5. Save models as `.keras`. Load with `compile=False` for inference.
6. `src/` never imports streamlit or fastapi. Long stages accept `progress_cb(stage, epoch, logs)`; `api/jobs.py` and the Streamlit pages consume it. UI changes must not change logic in `src/`.
7. Every page renders with no data and no models: show what's missing and the button that fixes it. No tracebacks reach the UI.
8. Uploads: extension jpg/jpeg/png/bmp/tif/tiff, max 10 MB, decoded in memory to RGB, resized to model input. Nothing written to disk.
9. Grad-CAM target layer lives in `models.GRADCAM_LAYER`; it is verified by iterating `model.layers` (test). Don't guess layer names.
10. Keep `requirements.txt` pinned. Don't add a dependency without a reason in the commit message.
11. Tests use `data/sample/` and `build_backbone(name, weights=None)`; no network.
12. No notebooks with real logic. Notebooks only call `run.py`.
13. Conventional commits (`feat:`, `fix:`, `docs:`, `test:`).
14. Research/educational project: UI and README carry a "not for clinical use" notice.
15. API errors are always JSON `{detail: "<readable message>"}` with a proper 4xx/5xx; the frontend shows `detail` inline. No tracebacks reach the browser.
16. Frontend: CSS variables for every colour (light + dark), no unused variables, only pinned CDN assets, keyboard-navigable, visible focus rings, contrast ≥ 4.5:1. `tests/test_static_assets.py` enforces the mechanical parts.

## Reference repos (ideas, not copy-paste)
- https://github.com/MAmirEshraghi/Lightweight-Deep-CNN-Based-Mobile-App-in-the-Screening-of-ALL — dataset author
- https://github.com/Parisaroozgarian/aml-detection — EfficientNetB0 on this dataset (MIT)
- https://github.com/P-MCC/Leukemia-Detection — ensemble notebook pattern
- https://arxiv.org/abs/2105.03995 — Mondal et al., weighted CNN ensemble for ALL

## Build order
P0 scaffold → P1 data → P2 models+train → P3 ensemble+predict → P4 evaluate → P5 gradcam → P6 Streamlit app → P7 Colab → P8 docs → P9 web UI (FastAPI + HTML/CSS/JS).
