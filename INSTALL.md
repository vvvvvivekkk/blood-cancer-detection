# Install & run

Blood Cancer Identification using Hybrid Ensemble Deep Learning — clone, fetch the pretrained
models from the release, and start the web app. Python 3.10–3.12, no GPU required (inference runs on CPU).

> **Not for clinical use.** Research / educational replication only.

## Client quick start (5 commands)

```bash
git clone https://github.com/vvvvvivekkk/blood-cancer-detection.git
cd blood-cancer-detection
pip install -r requirements.txt
python run.py --fetch-models https://github.com/vvvvvivekkk/blood-cancer-detection/releases/download/v1.0.0/models.zip
python app.py
```

`python app.py` is the same as `python run.py`: it serves the dashboard at <http://localhost:8000>
and opens a browser tab. Log in with **admin / admin** (a fixed demo gate for the walkthrough, not
security).

## What `--fetch-models` does

The three `.keras` backbones and the stacked meta-classifier are too large to keep in git, so they
ship as the **v1.0.0** release asset `models.zip` (~105 MB). `--fetch-models <url>`:

1. downloads the zip (the URL above, or any `file://` path),
2. verifies its SHA-256 against the committed `models/MODELS_SHA256.txt`,
3. unpacks `mobilenetv2.keras`, `efficientnetb0.keras`, `densenet121.keras`, `ensemble_weights.json`,
   `meta_clf.pkl`, `ensemble_mode.txt`, `ood_stats.json` into `models/`,
4. re-checks every extracted file's SHA-256.

Verify an existing `models/` at any time:

```bash
python run.py --check        # prints CHECK PASS / CHECK FAIL, exit code 0 / 1
```

## Running without the release

You can skip `--fetch-models` and produce the models yourself:

```bash
python run.py --cli all --smoke     # CPU plumbing check on the 8 bundled sample images (~2 min)
# or, with a Kaggle account, the real pipeline (GPU recommended — use the Colab notebook):
python run.py --cli all             # download -> splits -> train -> ensemble -> evaluate
```

Or press **Run full pipeline** on the Home view. No Kaggle account → **Use demo data** on the Data
view generates 200 synthetic images so every page still works (shown with a **DEMO DATA** banner).

## Maintainer: cut a release

```bash
python scripts/package_models.py                     # -> dist/models.zip + models/MODELS_SHA256.txt
# then attach dist/models.zip to a GitHub release tagged v1.0.0 (gh release create ...)
```

## Troubleshooting

- **`CHECK FAIL` / SHA mismatch** — the download is corrupt or stale; re-run `--fetch-models`.
- **Port in use** — `python run.py --port 8010`; add `--no-browser` to not open a tab.
- **Browser tests** — `pip install -r requirements-dev.txt && playwright install chromium`.
- **Streamlit fallback UI** — `python run.py --streamlit` (<http://localhost:8501>).
