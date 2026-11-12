# Generated training outputs

- `model_comparison_spatial_cv.csv`: mean and SD of MAE, RMSE, and R² for each model across five spatial CV folds.
- `spatial_cv_metrics_by_fold.csv`: fold-level benchmark metrics.
- `heldout_spatial_test_metrics.csv`: final metrics on 25 held-out 0.5° blocks.
- `heldout_spatial_test_predictions.csv.gz`: held-out labels, predictions, and absolute errors.
- `vif_diagnostic.csv`: VIF calculated from a reproducible sample of model predictors; diagnostic only, not a weight calculation method.
- `feature_correlation_spearman.csv`: predictor rank-correlation matrix.
- `permutation_feature_importance.csv`: held-out permutation importance for reproducing the pseudo-target.
- `target_summary.csv`: descriptive statistics of V3 target scores.
- `spatial_split_summary.csv`: number of rows and blocks in train/test partitions.
- `model_metadata.json`: model name, scoring weights, validation metrics, features, and limitations.

These outputs describe the current V3 pseudo-target emulation, not an independent field validation.
