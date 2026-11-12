"""Fit the selected CV winner on all Dataset V2 cells and refresh app artifacts."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring import MODEL_FEATURES, WEIGHTS, prepare_environmental_data
from src.train import sample_by_spatial_group

SEED=42
RAW=ROOT/'data/raw/LaoAgri_Arabica_Environmental_Features_V2.csv'
CV_PATH=ROOT/'outputs/model_comparison_spatial_cv.csv'
TEST_PATH=ROOT/'outputs/heldout_spatial_test_metrics.csv'

cv=pd.read_csv(CV_PATH).sort_values(['Mean_RMSE','Mean_MAE']).reset_index(drop=True)
best_name=str(cv.iloc[0]['model'])
if best_name != 'XGBoost':
    raise RuntimeError(f'Expected CV selected XGBoost based on current benchmark, got {best_name}')
raw=pd.read_csv(RAW)
data, soc_sorted, scoring_reference=prepare_environmental_data(raw)
data['block_lon']=np.floor(data['lon']/0.5).astype(int)
data['block_lat']=np.floor(data['lat']/0.5).astype(int)
data['spatial_block_id']=data['block_lon'].astype(str)+'_'+data['block_lat'].astype(str)
train_sample=sample_by_spatial_group(data, min(50000,len(data)), 'spatial_block_id', SEED+500)

# Same tree depth/learning rate/regularisation as the CV benchmark; capped threads for local reproducibility.
model=XGBRegressor(
    n_estimators=260, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.0, reg_lambda=2.0,
    objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
    n_jobs=2, random_state=SEED,
)
print(f'Fitting final {best_name} on {len(train_sample):,} spatially balanced grid cells...')
model.fit(train_sample[MODEL_FEATURES].astype('float32'), train_sample.target_y.astype('float32'))
print('Predicting all grid cells...')
pred=np.clip(model.predict(data[MODEL_FEATURES].astype('float32')),0,100)

bundle={
    'model':model,
    'model_name':best_name,
    'model_features':MODEL_FEATURES,
    'weights':WEIGHTS,
    'scoring_reference':scoring_reference,
    'target_version':'V3 weighted geometric mean; score floor=1',
    'score_bands':{'Low':'[0,50)','Fair':'[50,70)','Good':'[70,85)','Very Good':'[85,100]'},
    'selected_by':'Lowest mean RMSE in 5-fold GroupKFold spatial CV on 0.5-degree blocks',
}
joblib.dump(bundle, ROOT/'models/laoagri_best_model_v3.joblib', compress=3)

out=data[[
    'grid_id','lon','lat','elevation_m','temperature_c','rainfall_mm_year','soil_ph','slope_pct',
    'bio14_driest_month_mm','bio15_seasonality_cv','soc_0_30cm_g_kg','aspect_deg','aspect_sin','aspect_cos','aspect_is_defined',
    'score_elevation','score_temperature','score_rainfall','score_soil_ph','score_slope','score_bio14','score_soc','score_aspect',
    'target_y','target_y_wlc','suitability_band','limiting_feature'
]].copy()
out['ml_predicted_suitability']=pred
out['ml_suitability_band']=pd.cut(pred,bins=[0,50,70,85,100.000001],labels=['Low','Fair','Good','Very Good'],right=False,include_lowest=True).astype(str)
out['rule_vs_ml_abs_difference']=np.abs(out.target_y-out.ml_predicted_suitability)
out.to_csv(ROOT/'data/processed/LaoAgri_Arabica_Suitability_Predictions_V3.csv.gz',index=False,compression='gzip')

cv_summary=cv.to_dict(orient='records')
test_metrics=pd.read_csv(TEST_PATH).iloc[0].to_dict()
metadata={
    'project':'LaoAgri-Suitability',
    'app_title':'Arabica Land Suitability Explorer',
    'target':'V3 Weighted Geometric Screening Index 0-100 (pseudo-target, not field truth)',
    'study_area':'Lao PDR; Dataset V2 AOI derived from FAO GAUL 2024',
    'dataset_rows':int(len(data)),
    'model_name':best_name,
    'model_features':MODEL_FEATURES,
    'weights_percent':WEIGHTS,
    'spatial_block_size_degrees':0.5,
    'spatial_cv_folds':5,
    'cv_train_cap_per_fold':8000,
    'cv_summary':cv_summary,
    'heldout_spatial_test_metrics':test_metrics,
    'final_model_training_rows':int(len(train_sample)),
    'random_seed':SEED,
    'score_floor':1.0,
    'selection_rule':'Lowest mean RMSE across 5 spatial GroupKFold folds; MAE is secondary diagnostic.',
    'limitations':[
        'The target y is constructed from the environmental predictors; metrics describe emulation of the V3 scoring rule, not independently validated Arabica suitability.',
        'No Lao farm-level Arabica GPS or yield observations are used as labels.',
        'Elevation and temperature remain highly collinear; their combined target weight is 40%, but both remain model predictors.',
        'SOC is a within-Laos percentile score, not an absolute agronomic threshold.',
        'BIO15 is retained as a predictor/diagnostic but excluded from target scoring.',
        'Grid cells are approximately 928 m and support regional screening, not plot-level decisions.'
    ],
}
(ROOT/'outputs/model_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved final model, metadata and national prediction layer.')
print('Model:',best_name,'CV mean RMSE:',cv.iloc[0]['Mean_RMSE'],'held-out test RMSE:',test_metrics['RMSE_points'])
