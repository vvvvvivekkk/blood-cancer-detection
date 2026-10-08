# ENDGOAL.md — Definition of Done

The project is finished when every box below is checked and `pytest -q` is green.

## The one-sentence goal
A reproducible pipeline that trains three CNN backbones on the 4-class ALL blood-smear dataset, combines them with a hybrid ensemble (weighted soft-voting + stacked meta-classifier) that **outperforms each single backbone on the held-out test set**, all driven from a **web dashboard** (FastAPI + HTML/CSS/JS, Streamlit fallback) where a user can download data, train, evaluate, and upload a smear image to get the class, confidence, per-model votes and a Grad-CAM overlay.

## Target numbers (test set, 15 % stratified hold-out, real Kaggle data)
| Metric | Minimum | Paper claim |
|---|---|---|
| Ensemble accuracy | ≥ 96 % | > 95 % |
| Ensemble macro-F1 | ≥ 0.95 | — |
| Ensemble beats best single backbone | by ≥ 0.5 pp accuracy | implied |
| Per-class recall (all 4 classes) | ≥ 0.90 | — |

`src/evaluate.py` writes `results/NOTES.md` stating whether each target is met. If the ensemble does not beat the best single model, that is a finding — report it, then try kappa-weighted voting (Mondal et al.), TTA, or swapping a backbone.

## Phases

### P0 — Scaffold
- [x] Layout matches `CLAUDE.md`; `requirements.txt` pinned; `.gitignore`; MIT `LICENSE`
- [x] `src/config.py` holds `CLASS_NAMES`, `IMG_SIZE=224`, `SEED=42`, paths, hyperparams
- [x] `run.py` launches Streamlit; `run.py --cli <stage> [--smoke]` runs stages headless
- [x] `src/status.py` reports what exists on disk
- [x] `data/sample/` has 8 generated images (2 per class); `tests/test_config.py`

### P1 — Data
- [x] `src/download.py`: kagglehub download → `data/raw/`; credentials saved to `~/.kaggle/kaggle.json`; `--demo-data` writes 200 synthetic images to `data/demo/`
- [x] `make_splits` → `data/splits/{train,val,test}.csv` (`path,label,label_idx`, 70/15/15 stratified, seed 42) + `results/split_counts.csv`
- [x] `tf.data` pipeline: decode → resize → backbone preprocess; train-only augmentation (flip H/V, rotate ±20°, brightness/contrast ±10 %, zoom ±10 %)
- [x] `tests/test_download.py`, `tests/test_data.py` (disjoint, stratified ±2 %, `(B,H,W,3)` float + int labels)

### P2 — Models + training
- [x] `build_backbone(name, weights)`: base → GAP → Dropout(0.3) → Dense(4, softmax); `preprocess_fn`; `GRADCAM_LAYER`
- [x] Two-phase fine-tune: frozen base lr 1e-3, then top 30 % unfrozen lr 1e-5; Adam; class weights
- [x] ModelCheckpoint, EarlyStopping(5), ReduceLROnPlateau, CSVLogger → `results/history/<name>.csv`; saves `models/<name>.keras`
- [x] `progress_cb(stage, epoch, logs)` called every epoch
- [x] `tests/test_models.py`, `tests/test_train.py`

### P3 — Hybrid ensemble
- [x] Val probs → `results/val_probs.npy`; `fit_voting` → `models/ensemble_weights.json` (weights ∝ val macro-F1, sum to 1); `fit_stacking` → `models/meta_clf.pkl`
- [x] `predict_image` → `{class, confidence, probs, per_model_probs, mode}`
- [x] `tests/test_ensemble.py`, `tests/test_predict.py`

### P4 — Evaluation
- [x] 3 backbones + 2 ensembles on test once → `results/metrics.json` (acc, macro/weighted P/R/F1, kappa, per-class, confusion matrices, OvR ROC-AUC)
- [x] Better ensemble persisted to `models/ensemble_mode.txt`; `results/comparison_table.md`; `--write-readme-table`
- [x] Figures: `cm_*.png` ×5, `training_curves_*.png` ×3, `roc_ensemble.png`, `model_comparison_bar.png`, `gradcam_examples.png`
- [x] `tests/test_evaluate.py`

### P5 — Grad-CAM
- [x] `make_heatmap(model, img_array, layer_name)`, `overlay(img, heatmap, alpha)`; hand-rolled
- [x] Predict page shows Grad-CAM from the highest-weight backbone
- [x] `tests/test_gradcam.py`

### P6 — Streamlit app
- [x] Home: status badges, "Run full pipeline", demo banner, not-for-clinical-use footer
- [x] Data: Kaggle credential form, download, "Use demo data", make splits, class counts
- [x] Train: model picker, live progress bars + loss/accuracy charts
- [x] Evaluate: fit ensemble, run evaluation, comparison table, confusion matrices, ROC, per-class metrics, Grad-CAM grid
- [x] Predict: upload → class, confidence, per-model votes, Grad-CAM overlay
- [x] About: paper, method, disclaimer
- [x] Every page renders with no artifacts; `tests/test_app.py` (AppTest) covers empty + smoke states

### P7 — Colab
- [x] `notebooks/train_colab.ipynb`: clone, pip install, Kaggle key, `python run.py --cli all`, zip `models/` + `results/`

### P8 — Docs
- [x] README: 3-step quick start, demo note, Colab link, Mermaid diagram, generated results table, screenshots, limitations, citations
- [x] `docs/paper_mapping.md`, `docs/architecture.png`, `docs/screenshots/` (placeholders until a real run)
- [x] Final `pytest -q` green (< 3 min), `python run.py --cli all --smoke` green
- [ ] Real-data results (from Colab) committed in `results/` — **requires GPU run by the user**

### P9 — Web UI (FastAPI + HTML/CSS/JS)
- [x] `python run.py` serves the app at http://localhost:8000 and opens the browser; `--streamlit` keeps the old UI; `--cli` unchanged
- [x] `api/main.py`: every endpoint in the README API table; frontend served at `/`, figures/samples/architecture served as images
- [x] `api/jobs.py`: background thread, id/state/progress/stage/log/error, fed by `progress_cb`; second job → 409
- [x] `/api/jobs/{id}/stream` Server-Sent Events, UI falls back to 1 s polling
- [x] Errors are `{detail}` with proper 4xx/5xx (400 bad upload, 404 unknown job, 409 busy / missing prerequisite, 422 invalid body)
- [x] Overview: hero + citation, stepper, Run full pipeline (smoke toggle), live job panel, DEMO banner, footer on every view
- [x] Data: credential form (masked key), Download / Use demo data / Make splits, class counts chart, sample grid (4 per class)
- [x] Train: backbone checkboxes, epochs slider, smoke toggle, CPU estimate, per-backbone progress, live loss/accuracy charts, history on load
- [x] Evaluate: KPI tiles, comparison table (ensembles highlighted, selected badged), metric chart, 5 confusion-matrix tabs, ROC, per-class table, Grad-CAM grid, downloads
- [x] Predict: drag-and-drop + sample strip, inline errors, class, confidence ring, probability bars, vote table, Grad-CAM with opacity slider, mode badge
- [x] About: paper summary, architecture diagram, mapping table from `docs/paper_mapping.md`, dataset, limitations
- [x] Every view renders with nothing trained and offers the fixing job
- [x] Light/dark themes (CSS variables, prefers-color-scheme default), responsive to 768 px, sidebar → menu on phones, keyboard + focus rings + aria-labels
- [x] `tests/test_api.py`, `tests/test_frontend.py` (Playwright), `tests/test_static_assets.py` green
- [x] `docs/ui_walkthrough.md`; screenshots of every view in `docs/screenshots/` (smoke artifacts, re-capture after a real run)

### P10 — Examiner walkthrough parity
- [x] Static login (admin / admin, no database), *Login successful* toast, Logout in the sidebar, flag in `sessionStorage`
- [x] Sidebar order Home, Preview, Performance Analysis, Chart, About, Logout; Data and Train under a Pipeline submenu
- [x] Performance Analysis: Accuracy / Precision / Recall / F-measure tiles for the selected ensemble, confusion matrix directly below
- [x] Chart view: Model Loss and Model Accuracy, train vs val, backbone selector, from `results/history/*.csv`
- [x] Preview flow: Choose image → preview → Submit → result; *Try a test image* dropdown
- [x] `python run.py --cli export-test-images` → `test_images/<class>/` (10 per class, test split, gitignored)
- [x] `python app.py` starts the web app
- [x] API + Playwright tests for login, renamed views, Chart, test-image export

## Explicitly out of scope
- Cell segmentation (dataset is pre-cropped); mobile app; C-NMC binary task; any clinical claim

## How to demo
See `docs/ui_walkthrough.md` (10-step viva script for the web UI).
