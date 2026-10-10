# Target check: Ensemble-Stacking

Data source: **kaggle**, 487 test images.

| Target | Value | Met |
|---|---|---|
| Ensemble accuracy ≥ 96% | 0.9897 | yes |
| Ensemble macro-F1 ≥ 0.95 | 0.9883 | yes |
| Beats best backbone (DenseNet121) by ≥ 0.5 pp | +0.00 pp | NO |
| Every class recall ≥ 0.9 | min 0.9796 | yes |

The ensemble does not beat the best single backbone by the target margin. Next things to try: kappa-weighted voting, test-time augmentation, swapping a backbone.
