# Viva walkthrough: web UI

About 10 minutes. Before the viva, train for real on Colab (`notebooks/train_colab.ipynb`), unzip `models/` and `results/` into the repo, and run `python -m src.evaluate --table-only`. If you only have smoke or demo artifacts, say so: the banner on every view says it anyway.

Fallback if anything breaks: `python run.py --streamlit` (same features), or `python run.py --cli all --smoke` in a terminal.

1. **Start.** `python run.py`. The browser opens http://localhost:8000. Point out the footer, *Not for clinical use*, which appears on every view.

2. **Overview: the paper.** Read the title and citation in the hero. One sentence on the method: three fine-tuned CNNs plus a hybrid ensemble of weighted soft voting and a stacked meta-classifier.

3. **Overview: the pipeline.** The stepper shows Data → Train → Ensemble → Evaluate → Predict, each marked done or pending. Explain that every stage is a background job.

4. **Data.** Show the Kaggle form. The key is masked and the eye button reveals it. Show the class-count chart: 70/15/15, stratified, seed 42. Say that the test split is read only by `src/evaluate.py`, and that `data.load_split("test")` raises an error. Scroll to the example grid.

5. **Train.** Tick MobileNetV2 only, keep smoke on, and note the CPU estimate. Press **Start training**. The per-backbone bar and the live loss and accuracy curves update each epoch, and the job chip in the top bar follows along. Explain the two phases: frozen base at lr 1e-3, then the top 30 % unfrozen at lr 1e-5.

6. **Evaluate: headline.** The KPI tiles show ensemble accuracy, macro-F1, kappa and the gain over the best single backbone. In the comparison table, the shaded rows are the two ensembles. The *selected* badge marks the one used for prediction, picked by higher test macro-F1.

7. **Evaluate: detail.** Click through the confusion-matrix tabs (arrow keys work), then show the ROC, switch the per-class table to a backbone and back, and show the Grad-CAM grid (correct row and misclassified row). Open **Target check**, which says plainly whether each ENDGOAL target is met. Download `metrics.json` to show that the numbers come from a file a script wrote.

8. **Predict.** Drop in a test image, or click a sample in the strip, and press **Analyze**. Walk through the predicted class, the confidence ring, the four probability bars and the per-backbone votes with their voting weights. Drag the Grad-CAM opacity slider to compare the heatmap with the cell.

9. **Robustness.** Drop a `.txt` file (or one over 10 MB) to show the inline error and that nothing crashes. Toggle dark mode in the sidebar. Narrow the window to show the sidebar turning into a menu.

10. **Traceability and tests.** Open **About** to show the architecture diagram and the paper → code mapping table. In a terminal, run `python -m pytest -q` to show the API, browser and pipeline tests passing.
