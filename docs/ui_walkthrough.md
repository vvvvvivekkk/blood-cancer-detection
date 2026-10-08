# Viva walkthrough: web UI

About 10 minutes, following the order of the reference project video. Before the viva:

```bash
python run.py --cli export-test-images     # 10 test-split images per class -> test_images/<class>/
```

Train for real first (on Colab or on CPU, see README), then run `python -m src.evaluate --table-only`. If only smoke or demo artifacts exist, say so: the banner on every view says it anyway.

Fallback if anything breaks: `python run.py --streamlit` (same features, no login), or `python run.py --cli all --smoke` in a terminal.

1. **Start.** Run `python app.py`. It is the same as `python run.py` and opens http://localhost:8000.

2. **Open the URL and log in.** The login screen appears first. Enter **admin / admin** and press **Login**. A *Login successful* toast appears and you land on **Home**. Say plainly that this is a demo gate fixed in `src/config.py`, not security.

3. **Home.** Show the paper title and citation, and the pipeline stepper (Data → Train → Ensemble → Evaluate → Predict, all done). Point out the footer, *Not for clinical use*, which appears on every view.

4. **Preview: one test image per class.** Open **Preview**. In *Try a test image*, pick an image from the **Benign** group: its preview appears. Press **Submit**. The result shows the predicted class in large type, then the confidence ring, the four probability bars, the Grad-CAM overlay (drag the opacity slider) and the per-backbone votes. Repeat for **[Malignant] early Pre-B**, **[Malignant] Pre-B** and **[Malignant] Pro-B**. These images come from the held-out test split, so the model never trained on them.

5. **Preview: one image from the dataset folder.** Press **Choose image** and pick any file from `data/raw/<class>/`. Note that it may come from the training split, which is why step 4 uses `test_images/`. Then drop in a `.txt` file to show the inline error, which does not crash anything.

6. **Performance Analysis.** The four large tiles show **Accuracy, Precision, Recall and F-measure** of the selected ensemble on the test split. The confusion matrix of that ensemble is directly below; click the tabs to compare backbones (arrow keys work). Scroll down to the kappa / ROC-AUC / gain tiles, the comparison table (ensemble rows highlighted, *selected* badge), ROC curves, per-class metrics, the target check and the Grad-CAM grid.

7. **Chart.** Open **Chart**. *Model Loss* and *Model Accuracy* show train vs validation per epoch. Switch between MobileNetV2, EfficientNetB0 and DenseNet121 with the tabs. The note under the charts says where phase 2 (top 30 % unfrozen, lower learning rate) starts.

8. **Optional: About and Pipeline.** **About** has the architecture diagram and the paper → code mapping table. **Pipeline → Data / Train** shows how the dataset was downloaded and split and how the backbones were trained.

9. **Optional: settings.** The sidebar switch *Single model: MobileNetV2* makes Preview use MobileNetV2 alone (badge "MobileNetV2 (single)"), and Performance Analysis then highlights the MobileNetV2 row. Switch it back.

10. **Logout.** Press **Logout** in the sidebar. You are back on the login screen, and deep links such as `#/performance` also require logging in again.
