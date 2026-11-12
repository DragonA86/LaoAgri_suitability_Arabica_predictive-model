# LaoAgri-Suitability — Arabica Land Suitability Explorer

A local, VS Code–ready Streamlit application for exploring a provisional Arabica coffee land-suitability screening index for Lao PDR. The interface uses a green-and-white theme and provides a national grid map, suitability classes, a site simulator, model diagnostics, and downloadable prediction data.

> **Research limitation:** `target_y` is a rule-derived pseudo-target calculated from the environmental predictors. The ML model learns to reproduce that index. Its MAE, RMSE, and R² are measures of **rule emulation**, not independently validated agricultural suitability, coffee yield, or farm profitability. No farm-level Arabica GPS or yield observations are used as labels in this version.

## 1. What is included

- `app.py` — Streamlit user interface.
- `src/scoring.py` — V3 component curves, weights, target formula, score bands, and single-site scoring.
- `src/train.py` — V3 model benchmarking with spatial cross-validation and held-out spatial testing.
- `src/finalize_selected_model.py` — fit the selected model and refresh the app artifacts after model selection.
- `models/laoagri_best_model_v3.joblib` — fitted XGBoost model bundle plus scoring reference distribution.
- `data/processed/LaoAgri_Arabica_Suitability_Predictions_V3.csv.gz` — national grid predictions for the app.
- `outputs/` — CV results, held-out test results, VIF, correlation matrix, target summary, feature importance, and metadata.
- `data/raw/LaoAgri_Arabica_Environmental_Features_V2.csv` — Dataset V2 used for model training. It is included in this ZIP for local reproducibility but is ignored by `.gitignore` to avoid accidentally committing a large source dataset to a public repository.
- `docs/LaoAgri_V3_Training_Notebook.ipynb` — the V3 scoring/training notebook used as the methodological reference.
- `README_Lao.md` — Lao-language setup and research notes.

## 2. Recommended environment

Use Python 3.11 or 3.12 (Python 3.13 is also supported by the current Streamlit installation guide). Use a virtual environment to isolate dependencies. See the [official Streamlit command-line installation guide](https://docs.streamlit.io/get-started/installation/command-line).

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Once the app starts, Streamlit prints a local URL (normally `http://localhost:8501`). Open it in your browser. In VS Code, open the project folder and use **Run and Debug → LaoAgri — Streamlit App** or run `run_app.bat` / `run_app.ps1` / `run_app.sh` after creating the virtual environment.

## 3. Model selected from spatial validation

Five candidates were compared using the same 5-fold `GroupKFold` spatial folds. Spatial blocks are 0.5° × 0.5° and the model-selection rule is **lowest mean spatial-CV RMSE**.

| Model | Mean spatial CV RMSE | SD | Mean spatial CV MAE | Mean R² |
|---|---:|---:|---:|---:|
| **XGBoost** | **1.093** | 0.287 | 0.643 | 0.9951 |
| HistGradientBoosting | 1.106 | 0.320 | **0.613** | 0.9949 |
| RandomForest | 1.598 | 0.410 | 0.976 | 0.9895 |
| Ridge | 8.999 | 0.681 | 6.999 | 0.6830 |
| DummyMean | 16.803 | 1.244 | 15.215 | -0.0788 |

**Selected model: XGBoost.** It has the lowest mean spatial-CV RMSE. HistGradientBoosting is a close second and has a slightly lower mean MAE, so the selection is not a large or decisive performance gap. The app shows this comparison rather than hiding the near-tie.

Held-out spatial test (blocks not used in model fitting):

- Model: XGBoost
- Test cells: 54,294
- Held-out blocks: 25
- MAE: 0.483 points
- RMSE: 0.812 points
- R²: 0.9971

Training protocol used for the included benchmark: 5 spatial folds, up to 8,000 sampled training rows per fold, 30,000 rows for the held-out benchmark fit, and 50,000 rows for the final app model. The spatial held-out test was kept separate from the CV folds.

> These strong metrics are expected because XGBoost is learning a deterministic, rule-derived target from the same environmental predictors. Do not describe these values as real-world prediction accuracy.

## 4. V3 pseudo-target methodology

### Weights

| Factor | Weight |
|---|---:|
| Temperature | 28% |
| Elevation | 12% |
| Annual rainfall | 20% |
| Soil pH | 15% |
| Slope | 12% |
| BIO14 — driest-month rainfall | 5% |
| Soil organic carbon (SOC) | 5% |
| Aspect | 3% |
| BIO15 — rainfall seasonality | 0% (diagnostic / model predictor only) |
| **Total active weight** | **100%** |

Temperature + Elevation have a combined target weight of 40%. This is a project-level way to cap the contribution of a highly correlated thermal-altitude group; it does **not** remove multicollinearity from the ML predictors.

### Aggregation

The target uses a weighted geometric mean with a 1-point numerical floor:

\[
S = 100\exp\left(\sum_i \frac{w_i}{100}\ln\left(\frac{\max(s_i,1)}{100}\right)\right)
\]

where `s_i` is the component score and `w_i` its percentage weight. The 1-point floor is a modelling choice and is checked in the V3 notebook's floor-sensitivity analysis.

### Suitability bands

- Low: 0–49 (`[0, 50)`)
- Fair: 50–69 (`[50, 70)`)
- Good: 70–84 (`[70, 85)`)
- Very Good: 85–100 (`[85, 100]`)

These are project-defined index bands, **not** FAO S1/S2/S3/N land-suitability classes.

## 5. Run model training again

The pre-trained model and predictions are already included, so the app can open without retraining. To reproduce the benchmark and refresh the model/prediction outputs, make sure the Dataset V2 CSV exists at `data/raw/` and run:

```bash
python src/train.py --cv-train-rows 8000 --test-train-rows 30000 --final-train-rows 50000
```

This command performs 5-fold spatial CV, selects the model by mean CV RMSE, evaluates it on held-out spatial blocks, fits the deployment model, and writes updated files to `models/`, `outputs/`, and `data/processed/`.

For a faster software smoke test only (3 folds and fewer trees):

```bash
python src/train.py --quick --cv-train-rows 8000 --test-train-rows 30000 --final-train-rows 50000
```

**Warning:** the `--quick` run is not the final five-fold benchmark and may select a different model. Re-run the non-quick command to recreate the included five-fold selection.

## 6. Test the scoring code

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## 7. Main app pages

1. **Overview** — national score distribution, selected model, CV comparison, spatial snapshot.
2. **Suitability Map** — filters by suitability band and minimum score; download filtered cells.
3. **Site Simulator** — enter environmental conditions and compare the V3 rule index with the ML model estimate.
4. **Model & Methods** — model benchmark, held-out test metrics, VIF, permutation importance, scoring weights, and caveats.
5. **Data & Downloads** — export national prediction grid and diagnostics.

## 8. Important research limitations

- The V2 grid is approximately 928 m and is intended for national/regional screening, not farm-plot selection.
- FAO GAUL 2024 is the administrative boundary source used to define the study area; it is not a source of Arabica suitability weights.
- WorldClim 2.1 BIO variables represent the 1970–2000 historical baseline.
- Copernicus DEM GLO-30 2024_1 is a DSM, not a bare-earth DTM.
- SoilGrids pH and SOC are spatial predictions. SOC is a relative percentile within this Laos dataset, not an absolute agronomic threshold.
- Elevation and temperature remain highly collinear (high VIF). Grouping weights addresses index double-counting; it does not solve predictor collinearity for every model interpretation.
- The model emulates the pseudo-target. Independent local Arabica observations, field verification, and expert agronomic assessment are still needed for real-world validation.

## 9. Source references

See [`DATA_SOURCES.md`](DATA_SOURCES.md) and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the source list, methodological role, and model card.
"#LaoAgri_suitability_Arabica_predictive-model" 
