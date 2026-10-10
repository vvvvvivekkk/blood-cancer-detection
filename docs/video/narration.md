# Narration script — demo_walkthrough.mp4

**[00:00:00] Scene 1** — This is a hybrid ensemble deep learning system that identifies blood cancer subtypes from microscope images of blood smears.

**[00:00:05] Scene 2** — Acute Lymphoblastic Leukemia is the most common childhood cancer. Diagnosis starts with a blood smear under a microscope, which is slow, and the three malignant stages look very similar. Our goal is to classify an image into Benign, early Pre-B, Pre-B or Pro-B automatically, and to show why.

**[00:00:13] Scene 3** — Every image is sent to three separately fine-tuned convolutional networks: MobileNetV2, EfficientNetB0 and DenseNet121. Each outputs four probabilities. The twelve numbers are combined in two ways, weighted soft voting and a stacked logistic-regression meta-classifier. That pair is the hybrid ensemble.

**[00:00:22] Scene 4** — On the Preview page the user chooses a blood smear image and presses Submit. Within about two seconds the ensemble returns the predicted class and a confidence. This image is Benign at 97 percent.

**[00:00:29] Scene 5** — The probability bars show how the confidence is shared across all four classes, so a close call is visible at a glance. This image is an early Pre-B malignant cell.

**[00:00:35] Scene 6** — Underneath, each backbone's own vote and confidence are listed. If the three networks disagree, the app flags the result for manual review.

**[00:00:42] Scene 7** — The Grad-CAM heatmap explains the decision. Red marks the regions that most increased the predicted class score: the large nucleated blast cells, not the background.

**[00:00:49] Scene 8** — Across the test set, correct predictions show focused heat on the cell; the few misclassified cases show diffuse heat, which is itself a useful warning sign.

**[00:00:56] Scene 9** — On 487 unseen test images, MobileNetV2 reached 96.92 %, EfficientNetB0 91.38 % and DenseNet121 98.97 %. The stacked ensemble reached 98.97 % accuracy with a macro F1 of 0.988.

**[00:01:04] Scene 10** — The confusion matrix shows only five errors in 487 images. Every class is recalled above 98 percent, and no benign image was ever called malignant.

**[00:01:11] Scene 11** — The ROC curves confirm it: an area under the curve of 1.000, meaning the ensemble separates all four classes almost perfectly.

**[00:01:16] Scene 12** — In summary: a stacked hybrid ensemble reaching 98.97 % accuracy on held-out data, with explanations on every prediction, delivered as a complete web application. This is a research project and is not for clinical use. Thank you.
