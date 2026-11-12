# Model Card — LaoAgri Arabica Suitability V3

## Intended use

A regional screening and research interface for exploring a provisional, rule-derived Arabica suitability index across the Lao PDR Dataset V2 grid. It is intended for educational/research exploration, visualization, sensitivity review, and locating cells for further field investigation.

## Not intended for

- Claiming field-validated Arabica suitability.
- Predicting coffee yield, bean quality, farm profitability, or climate resilience as measured outcomes.
- Recommending a farm plot without on-site soil, drainage, water, access, land-use, and farmer-level checks.
- Treating score bands as FAO S1/S2/S3/N land evaluation classes.

## Dataset

- 287,066 grid cells in the supplied Dataset V2.
- Approximate grid resolution: 30 arc-seconds (~928 m nominal; geographic cell size varies with latitude).
- Study area boundary: FAO GAUL 2024, Lao PDR.
- Environmental predictors: Copernicus DEM GLO-30 2024_1, WorldClim 2.1 (BIO1/BIO12/BIO14/BIO15), and SoilGrids 2.0.
- Climate normals: WorldClim 1970–2000.

## Target definition

`target_y` is a weighted geometric mean of eight component suitability scores. It is not an independent observation. The target is derived from environmental predictors, and the ML model is trained to emulate it.

### Active target weights

| Component | Weight |
|---|---:|
| Temperature | 28% |
| Elevation | 12% |
| Annual rainfall | 20% |
| Soil pH | 15% |
| Slope | 12% |
| BIO14 | 5% |
| SOC relative percentile | 5% |
| Aspect | 3% |
| BIO15 | 0% (diagnostic only) |

The 1-point floor in the geometric mean is a numerical/modeling assumption. SOC uses a percentile rank within the Laos grid because a validated absolute threshold compatible with the SoilGrids 0–30 cm concentration was not established. BIO15 is excluded from target scoring because a defensible monotonic/optimum suitability direction was not established.

## Model selection

Five regressors were compared with 5-fold GroupKFold spatial cross-validation. Blocks were 0.5° × 0.5°. The training side was sampled to at most 8,000 rows per fold, preserving representation across blocks. Model choice used lowest mean spatial-CV RMSE, with MAE as a secondary diagnostic.

| Model | Mean RMSE | SD RMSE | Mean MAE | Mean R² |
|---|---:|---:|---:|---:|
| XGBoost | **1.093** | 0.287 | 0.643 | 0.9951 |
| HistGradientBoosting | 1.106 | 0.320 | **0.613** | 0.9949 |
| RandomForest | 1.598 | 0.410 | 0.976 | 0.9895 |
| Ridge | 8.999 | 0.681 | 6.999 | 0.6830 |
| DummyMean | 16.803 | 1.244 | 15.215 | -0.0788 |

XGBoost was selected due to the lowest mean spatial-CV RMSE. It narrowly outperformed HistGradientBoosting on RMSE, while HistGradientBoosting had a slightly better MAE. This is a close model-selection result, not strong evidence that XGBoost is universally superior.

## Held-out spatial test

- Selected model: XGBoost.
- Test size: 54,294 cells.
- Test groups: 25 held-out spatial blocks.
- Test-fit sample: 30,000 training cells.
- MAE: 0.483 index points.
- RMSE: 0.812 index points.
- R²: 0.9971.

The final deployment model is refit on a group-aware 50,000-cell sample and applied to the full Dataset V2 grid.

## Key risk: circular / pseudo-target evaluation

Because the target is calculated from the same environmental variables supplied to ML, a high R² and low RMSE primarily show that the model can approximate the scoring equation. They do not demonstrate crop performance or real land suitability. Independent Arabica presence data, field measurements, expert validation, and/or yield data are required for external validity.

## Known feature limitations

- Elevation and temperature are strongly correlated (high VIF). A 40% combined target group limits their shared weight, but both remain model features.
- Annual rainfall, BIO14, and BIO15 overlap as climate descriptors.
- SoilGrids pH and SOC are spatial predictions.
- SOC is a relative within-Laos percentile score, not an absolute agronomic threshold.
- BIO14 describes precipitation in the driest month, not dry-season duration or PET deficit.
- The grid resolution is too coarse for plot-level decisions.

## Ethical and practical considerations

The map can influence decisions about land use and livelihoods. Users should not treat an index class as a guarantee or as a substitute for field surveys, community land rights, protected-area checks, environmental safeguards, or local agronomist advice.
