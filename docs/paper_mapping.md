# Paper → code mapping

Paper: J. Jayachitra, N. Umarkathaf, "Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique", *ICEARS 2023* (IEEE Xplore 10084996).

The left column follows the usual structure of the paper. The right column names the code that implements each part here. Where this repo picks a specific setting (backbones, split, hyperparameters), the setting is listed in the middle column and defined once in `src/config.py`.

| Paper section | What this repo does | File / function |
|---|---|---|
| **Abstract**: hybrid ensemble of deep CNNs for blood cancer (ALL) identification, > 95 % accuracy | Three fine-tuned CNNs + two ensemble heads, compared on a held-out test set; target check written to `results/NOTES.md` | `src/pipeline.py::run("all")`, `src/evaluate.py::notes` |
| **Dataset** | Kaggle "Blood Cells Cancer (ALL)", 3,242 images, 4 classes; kagglehub download; synthetic demo fallback | `src/download.py::download_dataset`, `make_demo_data`; `src/config.py::CLASS_NAMES` |
| **Data split** | 70 / 15 / 15 stratified, seed 42; test read only at evaluation | `src/data.py::make_splits`, `load_split` (refuses `"test"`), `src/evaluate.py::run` |
| **Preprocessing** | Decode → resize 224×224 → backbone-specific `preprocess_input` | `src/data.py::make_dataset`, `src/models.py::preprocess_fn` |
| **Augmentation** | Train only: H/V flip, ±20° rotation, ±10 % zoom, ±10 % brightness and contrast | `src/data.py::_augmenter` |
| **CNN backbones (transfer learning)** | MobileNetV2, EfficientNetB0, DenseNet121 (ImageNet); head GAP → Dropout(0.3) → Dense(4, softmax) | `src/models.py::build_backbone` |
| **Training** | Phase 1 frozen base, Adam lr 1e-3; phase 2 top 30 % unfrozen (BatchNorm frozen), lr 1e-5; class weights; EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger | `src/train.py::train_backbone`, `src/models.py::set_phase` |
| **Hybrid ensemble: voting** | Weighted soft voting, weights ∝ val macro-F1, normalised | `src/ensemble.py::fit_voting`, `voting_probs` |
| **Hybrid ensemble: stacking** | Multinomial logistic regression on 12 concatenated val probabilities | `src/ensemble.py::fit_stacking`, `stacking_probs` |
| **Results: metrics** | Accuracy, macro/weighted P/R/F1, Cohen's κ, per-class report, confusion matrix, OvR ROC-AUC for all 5 models | `src/evaluate.py::metrics_for` → `results/metrics.json` |
| **Results: tables and figures** | Comparison table, confusion matrices, training curves, ROC, comparison bar chart | `src/evaluate.py::comparison_table`, `plot_*` → `results/` |
| **Results: model selection** | Ensemble with higher test macro-F1 is persisted and used for prediction | `src/evaluate.py::run` → `models/ensemble_mode.txt`, `src/predict.py::ensemble_mode` |
| **Explainability** (extension) | Grad-CAM on the highest-weight backbone | `src/gradcam.py::make_heatmap`, `overlay`; `results/figures/gradcam_examples.png` |
| **Application / deployment** | Streamlit app: data, training with live charts, evaluation, single-image prediction | `app/Home.py`, `app/pages/*.py`, `src/predict.py::predict_image` |
| **Conclusion** | Whether the ensemble beats every backbone, stated from the numbers | `results/NOTES.md` (generated), README results table (`python -m src.evaluate --table-only`) |
