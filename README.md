# Blood Cancer Identification using Hybrid Ensemble Deep Learning

Replication of the IEEE paper **"Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique"**, J. Jayachitra and N. Umarkathaf, *ICEARS 2023* ([IEEE Xplore 10084996](https://ieeexplore.ieee.org/document/10084996)).

Three transfer-learning CNNs (MobileNetV2, EfficientNetB0, DenseNet121) are combined by a **hybrid ensemble** (weighted soft voting + a stacked logistic-regression meta-classifier) to classify peripheral blood smear images into four Acute Lymphoblastic Leukemia (ALL) classes. The whole pipeline runs from a web dashboard.

> ⚠️ **Research / educational project. Not a medical device. Not for clinical use.**

## Quick start

```bash
git clone https://github.com/vvvvvivekkk/blood-cancer-detection.git
cd blood-cancer-detection
pip install -r requirements.txt
python run.py --fetch-models https://github.com/vvvvvivekkk/blood-cancer-detection/releases/download/v1.0.0/models.zip
python app.py                        # opens http://localhost:8000
```

Log in with **admin / admin**. No Kaggle account? Press *Use demo data*. No GPU? `python run.py --cli all --smoke` runs the pipeline on CPU in ~2 minutes.

## Results

<!-- RESULTS:START -->
| Model | Accuracy | Macro-F1 | Weighted-F1 | Kappa | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| MobileNetV2 | 96.92% | 0.9662 | 0.9695 | 0.9583 | 0.9981 |
| EfficientNetB0 | 91.38% | 0.9114 | 0.9126 | 0.8828 | 0.9973 |
| DenseNet121 | 98.97% | 0.9890 | 0.9898 | 0.9861 | 0.9994 |
| Ensemble-Voting | 98.36% | 0.9809 | 0.9837 | 0.9777 | 0.9995 |
| **Ensemble-Stacking** (selected) | **98.97%** | **0.9883** | **0.9898** | **0.9861** | **0.9997** |

_487 test images, Kaggle data, 2026-10-09. Regenerate with `python -m src.evaluate --write-readme-table`._
<!-- RESULTS:END -->

## Architecture

![architecture](docs/architecture.png)

```mermaid
flowchart LR
    A[PBS image 224×224] --> B1[MobileNetV2]
    A --> B2[EfficientNetB0]
    A --> B3[DenseNet121]
    B1 --> P[3 × softmax 4]
    B2 --> P
    B3 --> P
    P --> V[Weighted soft voting]
    P --> S[Stacked meta-classifier]
    V --> O{better macro-F1}
    S --> O
    O --> R[Class + confidence]
    B1 -.-> G[Grad-CAM]
```

## Project structure

```
run.py      single entry point: UI, or --cli stages
src/        config · status · download · data · models · train · ensemble · predict · evaluate · gradcam · pipeline
api/        main.py (FastAPI routes), jobs.py (background job runner)
frontend/   index.html, css/styles.css, js/{app,api,charts}.js
models/     <backbone>.keras, ensemble_weights.json, meta_clf.pkl, ensemble_mode.txt
results/    metrics.json, comparison_table.md, NOTES.md, split_counts.csv, history/, figures/
notebooks/  train_colab.ipynb
docs/       HOW_IT_WORKS.pdf (client guide), project_report.pdf, screenshots/
tests/      pytest suite (no network, no GPU)
```

## Key commands

```bash
python run.py                                # web app at http://localhost:8000
python run.py --cli all --smoke              # full pipeline on CPU (~2 min)
python run.py --cli export-test-images       # 10 test images per class for demo
python run.py --fetch-models <URL>           # download + verify pre-trained models
python run.py --check                        # verify model SHA-256 checksums
python -m pytest -q                          # run test suite
```

## Documentation

| Document | Audience | Description |
|---|---|---|
| [`docs/HOW_IT_WORKS.pdf`](docs/HOW_IT_WORKS.pdf) | Client | 29-page plain-English explanation with figures, installation, and walkthrough |
| [`docs/HOW_IT_WORKS.docx`](docs/HOW_IT_WORKS.docx) | Client | Same content, editable Word format |
| [`docs/project_report.pdf`](docs/project_report.pdf) | Examiner | Academic project report |
| [`docs/ieee_paper.pdf`](docs/ieee_paper.pdf) | Academic | IEEE-formatted paper |
| [`docs/DOCUMENTATION.pdf`](docs/DOCUMENTATION.pdf) | Developer | Technical reference documentation |
| `CLAUDE.md` | Developer | Engineering rules and conventions |

## Limitations

- One public dataset of pre-cropped single cells; no segmentation or external validation.
- The ensemble ties the best single backbone (DenseNet121) rather than beating it.
- Not clinically validated. Do not use for diagnosis.

## References

- J. Jayachitra, N. Umarkathaf, "Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique," *ICEARS*, IEEE, 2023.
- M. Aria et al., "Acute Lymphoblastic Leukemia (ALL) image dataset," Kaggle, 2021. [Dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)
- R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017.

## Licence

MIT, see `LICENSE`. The dataset licence is set on its Kaggle page.
