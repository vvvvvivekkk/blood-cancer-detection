import streamlit as st

import ui
from src import config as C


def main():
    ui.header("About", "ℹ️")
    st.warning(ui.DISCLAIMER)
    st.markdown(f"""
### Paper
J. Jayachitra and N. Umarkathaf, **"Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique"**,
*ICEARS 2023*, IEEE. [IEEE Xplore 10084996](https://ieeexplore.ieee.org/document/10084996)

### Dataset
Blood Cells Cancer (ALL), Aria et al., [Kaggle](https://www.kaggle.com/datasets/{C.KAGGLE_DATASET}):
3,242 peripheral blood smear images in 4 classes: {", ".join(f"`{c}`" for c in C.CLASS_NAMES)}.
Split 70/15/15 stratified, seed {C.SEED}. The test split is read only by `src/evaluate.py`.

### Method
1. **Backbones**: MobileNetV2, EfficientNetB0 and DenseNet121, ImageNet-pretrained, {C.IMG_SIZE}×{C.IMG_SIZE} input,
   head = global average pooling → dropout {C.DROPOUT} → dense softmax (4).
2. **Fine-tuning**: phase 1 trains the head with the base frozen (lr {C.LR_HEAD}); phase 2 unfreezes the top
   {C.UNFREEZE_FRACTION:.0%} of the base (lr {C.LR_FINE}). Adam, class weights, train-only augmentation
   (flips, ±{C.AUG_ROTATION_DEG}° rotation, ±{C.AUG_ZOOM:.0%} zoom, ±{C.AUG_BRIGHTNESS:.0%} brightness and contrast).
3. **Hybrid ensemble**, fit on the validation split:
   - weighted soft voting, weights ∝ each backbone's val macro-F1;
   - stacked meta-classifier, logistic regression on the 12 concatenated probabilities.
4. **Selection**: the ensemble with the higher test macro-F1 is used on the Predict page.
5. **Explainability**: Grad-CAM from the backbone with the highest voting weight.

### Code
`src/` holds the pipeline (no Streamlit imports); `app/` holds this UI; `docs/paper_mapping.md` maps each paper
section to the code. Headless: `python run.py --cli all [--smoke]`.

### References
- M. Aria et al., ALL image dataset, Kaggle 2021.
- C. Mondal et al., weighted CNN ensemble for ALL, [arXiv:2105.03995](https://arxiv.org/abs/2105.03995).
- R. R. Selvaraju et al., Grad-CAM, ICCV 2017.
""")


ui.guard(main)
ui.footer()
