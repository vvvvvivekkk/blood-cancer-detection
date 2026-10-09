# Target check: Ensemble-Stacking

Data source: **kaggle**, 487 test images.

| Target | Value | Met |
|---|---|---|
| Ensemble accuracy ≥ 96% | 0.9897 | yes |
| Ensemble macro-F1 ≥ 0.95 | 0.9883 | yes |
| Beats best backbone (DenseNet121) by ≥ 0.5 pp | +0.00 pp | NO |
| Every class recall ≥ 0.9 | min 0.9796 | yes |

The ensemble does not beat the best single backbone by the target margin. Next things to try (ENDGOAL.md): kappa-weighted voting, test-time augmentation, swapping a backbone.

## Fallback attempts (ENDGOAL)

The selected ensemble ties/loses to the best backbone (DenseNet121, test acc 0.9897). Variants below are **selected on val macro-F1**; test is reported only.

| Variant | Val macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|
| F1-weighted voting (baseline) ← selected (best val) | 0.9926 | 0.9836 | 0.9809 |
| Kappa-weighted voting | 0.9926 | 0.9836 | 0.9809 |
| F1²-weighted voting (sharper) | 0.9926 | 0.9836 | 0.9809 |
| Drop EfficientNetB0 (F1-weighted) | 0.9926 | 0.9856 | 0.9841 |

Val-selected fallback: **F1-weighted voting (baseline)** → test acc 0.9836 (-0.62 pp vs DenseNet121); still does not clear the +0.5 pp target.

TTA (test-time augmentation) deferred: ~N× model passes over the test set is CPU-prohibitive here and expected gain is marginal with one backbone already near-ceiling; run on Colab GPU if needed.

**Finding:** with one near-ceiling backbone (DenseNet121) and one much weaker (EfficientNetB0, test acc 0.914), no val-selected soft-voting or stacking blend exceeds the best backbone by 0.5 pp. The hybrid ensemble matches the top backbone and raises precision on the minority Benign class; it does not beat it outright on this split.
