# LaoAgri-Suitability — ລະບົບສຳຫຼວດຄວາມເໝາະສົມສຳລັບປູກ Arabica

ໂຄງການນີ້ແມ່ນ Streamlit Web App ສຳລັບສຳຫຼວດຄະແນນຄວາມເໝາະສົມຂອງ Arabica Coffee ໃນ ສປປ ລາວ. ໜ້າຕາໃຊ້ Theme ສີຂຽວ-ຂາວ ແລະສາມາດເປີດແກ້ໄຂຕໍ່ໃນ VS Code ໄດ້.

> **ຂໍ້ຈຳກັດສຳຄັນ:** `target_y` ແມ່ນ Pseudo-target ທີ່ສ້າງຈາກ Rule-based Scoring. ML Model ຮຽນຮູ້ຈຳລອງສູດນີ້. MAE, RMSE ແລະ R² ຈຶ່ງວັດຄວາມສາມາດໃນການຈຳລອງສູດ ບໍ່ແມ່ນຄວາມແມ່ນຍຳຂອງຄວາມເໝາະສົມຈິງ, ຜົນຜະລິດ ຫຼືກຳໄລຂອງສວນກາເຟ.

## 1. ໄຟລ໌ສຳຄັນໃນ Project

- `app.py` — Streamlit User Interface.
- `src/scoring.py` — V3 scoring functions, weights ແລະ target formula.
- `src/train.py` — ຝຶກ Model, ປຽບທຽບ 5 Algorithms ແລະ Spatial Validation.
- `src/finalize_selected_model.py` — ຝຶກ Final Model ແລະສ້າງ App artifacts ຫຼັງຈາກເລືອກ Model.
- `models/laoagri_best_model_v3.joblib` — XGBoost model ທີ່ຝຶກແລ້ວ.
- `data/processed/` — National prediction layer ສຳລັບເປີດໃນ App.
- `outputs/` — Spatial CV metrics, test metrics, VIF, correlation, permutation importance ແລະ metadata.
- `data/raw/` — Dataset V2 CSV ທີ່ໃຊ້ຝຶກ Model.
- `docs/LaoAgri_V3_Training_Notebook.ipynb` — Notebook ອ້າງອີງດ້ານ methodology.
- `README.md` — ຄູ່ມືພາສາອັງກິດ.

## 2. ວິທີເປີດໃນ VS Code (Windows)

ເປີດ Folder `LaoAgri_Arabica_Suitability_App` ໃນ VS Code ແລ້ວເປີດ Terminal.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

ຈາກນັ້ນເປີດ URL ທີ່ Streamlit ສະແດງ (ປົກກະຕິແມ່ນ `http://localhost:8501`).

ສາມາດກົດ **Run and Debug → LaoAgri — Streamlit App** ໃນ VS Code ໄດ້ ຫຼືໃຊ້ `run_app.bat` ຫຼັງຈາກຕິດຕັ້ງ Dependencies ແລ້ວ.

## 3. Model ທີ່ຖືກເລືອກ

ທົດສອບ 5 Algorithms ດ້ວຍ 5-Fold `GroupKFold` Spatial Cross-Validation. ຂະໜາດ Spatial Block = 0.5° × 0.5°. ເກນເລືອກ Model ແມ່ນ Mean RMSE ທີ່ຕ່ຳທີ່ສຸດ.

| Model | Mean Spatial CV RMSE | Mean MAE | Mean R² |
|---|---:|---:|---:|
| **XGBoost** | **1.093** | 0.643 | 0.9951 |
| HistGradientBoosting | 1.106 | **0.613** | 0.9949 |
| RandomForest | 1.598 | 0.976 | 0.9895 |
| Ridge | 8.999 | 6.999 | 0.6830 |
| DummyMean | 16.803 | 15.215 | -0.0788 |

**Selected Model = XGBoost** ເພາະ Mean RMSE ຕ່ຳທີ່ສຸດ. ແຕ່ HistGradientBoosting ຢູ່ໃກ້ກັນຫຼາຍ ແລະ Mean MAE ຕ່ຳກວ່າ. ດັ່ງນັ້ນຜົນບໍ່ໄດ້ຊີ້ວ່າ XGBoost ດີກວ່າທຸກມິຕິ.

Held-out Spatial Test:

- Test grid cells: 54,294
- Test spatial blocks: 25
- MAE: 0.483 points
- RMSE: 0.812 points
- R²: 0.9971

Metrics ເຫຼົ່ານີ້ວັດວ່າ XGBoost ຈຳລອງ Rule-based Target ໄດ້ດີພຽງໃດ. ບໍ່ແມ່ນ Real-world accuracy.

## 4. V3 Weights

| Factor | Weight |
|---|---:|
| Temperature | 28% |
| Elevation | 12% |
| Annual Rainfall | 20% |
| Soil pH | 15% |
| Slope | 12% |
| BIO14 | 5% |
| SOC | 5% |
| Aspect | 3% |
| BIO15 | 0% (Diagnostic only) |
| **Total** | **100%** |

Temperature + Elevation ລວມກັນ 40%. ນີ້ແມ່ນ Project Weighting Choice ເພື່ອຈຳກັດອິດທິພົນຂອງກຸ່ມທີ່ມີ Correlation ສູງ. ມັນບໍ່ໄດ້ແກ້ Multicollinearity ຂອງ ML Input ໂດຍກົງ.

ໃຊ້ Weighted Geometric Mean:

\[
S = 100\exp\left(\sum_i \frac{w_i}{100}\ln\left(\frac{\max(s_i,1)}{100}\right)\right)
\]

## 5. Suitability Bands

- Low: 0–49
- Fair: 50–69
- Good: 70–84
- Very Good: 85–100

ເກນນີ້ແມ່ນ Project-defined Index Bands, ບໍ່ແມ່ນ FAO S1/S2/S3/N.

## 6. ວິທີ Train Model ໃໝ່

Dataset V2 ຕ້ອງຢູ່ໃນ `data/raw/LaoAgri_Arabica_Environmental_Features_V2.csv`.

```bash
python src/train.py --cv-train-rows 8000 --test-train-rows 30000 --final-train-rows 50000
```

ຄຳສັ່ງນີ້ໃຊ້ 5-Fold Spatial CV ແລະສ້າງ Metrics, Final Model ແລະ Prediction CSV ໃໝ່. ຜົນຈະຖືກຂຽນໃນ `outputs/`, `models/` ແລະ `data/processed/`.

ສຳລັບການທົດສອບ Code ແບບໄວເທົ່ານັ້ນ:

```bash
python src/train.py --quick --cv-train-rows 8000 --test-train-rows 30000 --final-train-rows 50000
```

**ລະວັງ:** `--quick` ໃຊ້ 3-Fold ແລະຈຳນວນ Tree ນ້ອຍກວ່າ. ຜົນທີ່ໄດ້ບໍ່ຄວນນຳໄປແທນ 5-Fold benchmark ສະບັບນີ້.

## 7. ໜ້າຫຼັກຂອງ App

1. **Overview:** ສະຫຼຸບຄະແນນ ແລະ Distribution.
2. **Suitability Map:** ສຳຫຼວດແຜນທີ່ ແລະກັ່ນຕອງ Grid Cells.
3. **Site Simulator:** ປ້ອນ Environmental Features ເພື່ອຄຳນວນ Rule Index ແລະ ML Estimate.
4. **Model & Methods:** ຜົນ CV, Held-out Test, VIF ແລະ Feature Importance.
5. **Data & Downloads:** ດາວໂຫຼດ Prediction Data ແລະ Diagnostics.

## 8. ຂໍ້ຈຳກັດທາງວິຊາການ

- Dataset V2 ມີ Resolution ປະມານ 928 m; ເໝາະກັບ National/Regional Screening ບໍ່ແມ່ນລະດັບສວນຍ່ອຍ.
- FAO GAUL 2024 ໃຊ້ກຳນົດ AOI ເທົ່ານັ້ນ; ບໍ່ໄດ້ກຳນົດ Weight ຂອງ Arabica.
- WorldClim 2.1 ແມ່ນຄ່າ Climate Normal ຂອງ 1970–2000.
- Elevation ແລະ Temperature ຍັງມີ VIF ສູງ. ການລວມ Weight ບໍ່ໄດ້ລຶບຄວາມຊ້ຳຊ້ອນນີ້ອອກຈາກ Predictor.
- SOC ເປັນ Relative Percentile ພາຍໃນລາວ ບໍ່ແມ່ນ Absolute Agronomic Threshold.
- ຍັງຕ້ອງການ Arabica Farm GPS, Field Validation ແລະ Expert Assessment ເພື່ອຢືນຢັນຄວາມເໝາະສົມໃນໂລກຈິງ.

ເບິ່ງ `DATA_SOURCES.md` ແລະ `docs/MODEL_CARD.md` ສຳລັບແຫຼ່ງອ້າງອີງ ແລະ Model Documentation.
