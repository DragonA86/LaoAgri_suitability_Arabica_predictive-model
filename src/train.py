"""Train and select the best V3 model using spatial cross-validation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring import (  # noqa: E402
    MODEL_FEATURES, WEIGHTS, prepare_environmental_data,
)

SEED = 42
BLOCK_SIZE_DEG = 0.5


def sample_by_spatial_group(frame: pd.DataFrame, max_rows: int | None, group_col: str, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()
    sizes = frame[group_col].value_counts().sort_index()
    n_groups = len(sizes)
    if max_rows < n_groups:
        raise ValueError("max_rows is lower than the number of spatial blocks")
    raw = sizes / sizes.sum() * max_rows
    quotas = np.floor(raw).astype(int).clip(lower=1)
    quotas = pd.Series(np.minimum(quotas.values, sizes.values), index=sizes.index, dtype=int)
    while int(quotas.sum()) > max_rows:
        can_reduce = quotas[quotas > 1]
        if can_reduce.empty:
            break
        quotas.loc[can_reduce.idxmax()] -= 1
    while int(quotas.sum()) < max_rows:
        capacity = sizes - quotas
        capacity = capacity[capacity > 0]
        if capacity.empty:
            break
        quotas.loc[capacity.idxmax()] += 1
    parts = []
    for group_id, group in frame.groupby(group_col, sort=False):
        parts.append(group.sample(n=int(quotas.loc[group_id]), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def metrics(y_true, y_pred) -> dict:
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, 100)
    truth = np.asarray(y_true, dtype=float)
    return {
        "MAE_points": float(mean_absolute_error(truth, pred)),
        "RMSE_points": float(np.sqrt(mean_squared_error(truth, pred))),
        "R2": float(r2_score(truth, pred)),
    }


def build_models(seed: int, quick: bool = False):
    n_rf = 80 if quick else 140
    n_xgb = 140 if quick else 260
    n_hgb = 100 if quick else 180
    models = {
        "DummyMean": lambda: DummyRegressor(strategy="mean"),
        "Ridge": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=n_rf, max_depth=18, min_samples_leaf=3,
            max_features=0.85, n_jobs=min(2, os.cpu_count() or 1), random_state=seed,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=n_hgb, learning_rate=0.07, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed,
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=n_xgb, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.0, reg_lambda=2.0,
            objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist", n_jobs=min(2, os.cpu_count() or 1),
            random_state=seed,
        )
    except ImportError:
        print("WARNING: xgboost is not installed; XGBoost will be skipped.")
    return models


def main():
    parser = argparse.ArgumentParser(description="Train the LaoAgri V3 pseudo-target emulator.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/LaoAgri_Arabica_Environmental_Features_V2.csv")
    parser.add_argument("--quick", action="store_true", help="Use smaller model settings and 3-fold CV for a smoke test.")
    parser.add_argument("--cv-train-rows", type=int, default=40000)
    parser.add_argument("--final-train-rows", type=int, default=160000)
    parser.add_argument("--test-train-rows", type=int, default=120000)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    out_dir = ROOT / "outputs"
    model_dir = ROOT / "models"
    processed_dir = ROOT / "data/processed"
    for d in (out_dir, model_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"Loading data: {args.data}")
    raw = pd.read_csv(args.data)
    print(f"Input rows: {len(raw):,}")
    data, soc_sorted, scoring_reference = prepare_environmental_data(raw)
    data["block_lon"] = np.floor(data["lon"] / BLOCK_SIZE_DEG).astype(int)
    data["block_lat"] = np.floor(data["lat"] / BLOCK_SIZE_DEG).astype(int)
    data["spatial_block_id"] = data["block_lon"].astype(str) + "_" + data["block_lat"].astype(str)

    corr = data[MODEL_FEATURES].corr(method="spearman")
    corr.to_csv(out_dir / "feature_correlation_spearman.csv")

    # VIF is a multicollinearity diagnostic only; it does not determine agronomic weights.
    vif_frame = data[MODEL_FEATURES].sample(
        n=min(30000, len(data)), random_state=SEED
    ).dropna().reset_index(drop=True)
    vif_rows = []
    for feature in MODEL_FEATURES:
        others = [c for c in MODEL_FEATURES if c != feature]
        aux = LinearRegression().fit(vif_frame[others], vif_frame[feature])
        aux_r2 = float(aux.score(vif_frame[others], vif_frame[feature]))
        vif_value = float("inf") if aux_r2 >= 0.999999 else 1.0 / (1.0 - aux_r2)
        vif_rows.append({"feature": feature, "auxiliary_R2": aux_r2, "VIF": vif_value})
    pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).to_csv(
        out_dir / "vif_diagnostic.csv", index=False
    )
    data["target_y"].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).to_frame().to_csv(out_dir / "target_summary.csv")

    # Group-aware spatial holdout. No 0.5-degree block appears in both partitions.
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_pos, te_pos = next(split.split(data, data["target_y"], groups=data["spatial_block_id"]))
    train = data.iloc[tr_pos].copy().reset_index(drop=True)
    test = data.iloc[te_pos].copy().reset_index(drop=True)
    assert set(train.spatial_block_id).isdisjoint(set(test.spatial_block_id))
    split_summary = pd.DataFrame([
        {"split": "train_pool", "rows": len(train), "blocks": train.spatial_block_id.nunique()},
        {"split": "heldout_test", "rows": len(test), "blocks": test.spatial_block_id.nunique()},
    ])
    split_summary.to_csv(out_dir / "spatial_split_summary.csv", index=False)

    models = build_models(SEED, quick=args.quick)
    folds = min(3 if args.quick else args.folds, train.spatial_block_id.nunique())
    cv_cap = min(args.cv_train_rows, 20000) if args.quick else args.cv_train_rows
    gkf = GroupKFold(n_splits=folds)
    cv_rows = []
    for fold_no, (fit_idx, val_idx) in enumerate(
        gkf.split(train, train.target_y, groups=train.spatial_block_id), start=1
    ):
        fold_fit = train.iloc[fit_idx]
        fold_val = train.iloc[val_idx]
        fold_sample = sample_by_spatial_group(
            fold_fit, cv_cap, "spatial_block_id", SEED + fold_no
        )
        x_fit = fold_sample[MODEL_FEATURES].astype("float32")
        y_fit = fold_sample.target_y.astype("float32")
        x_val = fold_val[MODEL_FEATURES].astype("float32")
        y_val = fold_val.target_y.astype("float32")
        print(f"\nFold {fold_no}/{folds}: train={len(fold_sample):,}; validation={len(fold_val):,}")
        for name, factory in models.items():
            estimator = factory()
            estimator.fit(x_fit, y_fit)
            pred = estimator.predict(x_val)
            row = {"model": name, "fold": fold_no, "fit_rows": len(fold_sample), "validation_rows": len(fold_val), **metrics(y_val, pred)}
            cv_rows.append(row)
            print(f"  {name:23s} RMSE={row['RMSE_points']:.4f} | MAE={row['MAE_points']:.4f} | R2={row['R2']:.5f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "spatial_cv_metrics_by_fold.csv", index=False)
    summary = cv_df.groupby("model").agg(
        Mean_MAE=("MAE_points", "mean"), SD_MAE=("MAE_points", "std"),
        Mean_RMSE=("RMSE_points", "mean"), SD_RMSE=("RMSE_points", "std"),
        Mean_R2=("R2", "mean"), SD_R2=("R2", "std"),
    ).sort_values(["Mean_RMSE", "Mean_MAE"]).reset_index()
    summary.to_csv(out_dir / "model_comparison_spatial_cv.csv", index=False)
    best_name = str(summary.iloc[0].model)
    print("\nSelected model based on lowest Mean Spatial CV RMSE:", best_name)

    # Separate untouched spatial holdout for a final emulation check.
    test_cap = min(args.test_train_rows, len(train))
    test_train = sample_by_spatial_group(train, test_cap, "spatial_block_id", SEED + 100)
    selected_test_model = models[best_name]()
    selected_test_model.fit(test_train[MODEL_FEATURES].astype("float32"), test_train.target_y.astype("float32"))
    test_pred = np.clip(selected_test_model.predict(test[MODEL_FEATURES].astype("float32")), 0, 100)
    test_metrics = metrics(test.target_y, test_pred)
    test_metrics.update({
        "selected_model": best_name,
        "train_rows_used": int(len(test_train)),
        "test_rows": int(len(test)),
        "test_blocks": int(test.spatial_block_id.nunique()),
    })
    pd.DataFrame([test_metrics]).to_csv(out_dir / "heldout_spatial_test_metrics.csv", index=False)
    test_out = test[["grid_id", "lon", "lat", "target_y", "suitability_band"]].copy()
    test_out["ml_predicted_suitability"] = test_pred
    test_out["absolute_error_points"] = np.abs(test.target_y.to_numpy() - test_pred)
    test_out.to_csv(out_dir / "heldout_spatial_test_predictions.csv.gz", index=False, compression="gzip")

    # Model-agnostic permutation importance on a reproducible subset of held-out blocks.
    perm_n = min(2500 if args.quick else 4000, len(test))
    perm_sample = test.sample(n=perm_n, random_state=SEED)
    perm = permutation_importance(
        selected_test_model, perm_sample[MODEL_FEATURES].astype("float32"), perm_sample.target_y,
        scoring="neg_root_mean_squared_error", n_repeats=2,
        random_state=SEED, n_jobs=min(2, os.cpu_count() or 1),
    )
    importance = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(out_dir / "permutation_feature_importance.csv", index=False)

    # Fit deployment model on a group-aware sample from the complete labelled grid.
    final_sample_cap = min(args.final_train_rows, len(data))
    final_sample = sample_by_spatial_group(data, final_sample_cap, "spatial_block_id", SEED + 500)
    final_model = models[best_name]()
    final_model.fit(final_sample[MODEL_FEATURES].astype("float32"), final_sample.target_y.astype("float32"))
    full_pred = np.clip(final_model.predict(data[MODEL_FEATURES].astype("float32")), 0, 100)
    result = data[[
        "grid_id", "lon", "lat", "elevation_m", "temperature_c", "rainfall_mm_year",
        "soil_ph", "slope_pct", "bio14_driest_month_mm", "bio15_seasonality_cv",
        "soc_0_30cm_g_kg", "aspect_deg", "aspect_sin", "aspect_cos", "aspect_is_defined",
        "score_elevation", "score_temperature", "score_rainfall", "score_soil_ph",
        "score_slope", "score_bio14", "score_soc", "score_aspect", "target_y", "target_y_wlc",
        "suitability_band", "limiting_feature",
    ]].copy()
    result["ml_predicted_suitability"] = full_pred
    result["ml_suitability_band"] = pd.cut(
        full_pred, bins=[0, 50, 70, 85, 100.000001],
        labels=["Low", "Fair", "Good", "Very Good"], right=False, include_lowest=True,
    ).astype(str)
    result["rule_vs_ml_abs_difference"] = np.abs(result.target_y - result.ml_predicted_suitability)
    result.to_csv(processed_dir / "LaoAgri_Arabica_Suitability_Predictions_V3.csv.gz", index=False, compression="gzip")

    model_bundle = {
        "model": final_model,
        "model_name": best_name,
        "model_features": MODEL_FEATURES,
        "weights": WEIGHTS,
        "scoring_reference": scoring_reference,
        "target_version": "V3 weighted geometric mean; score floor=1",
        "score_bands": {"Low": "[0,50)", "Fair": "[50,70)", "Good": "[70,85)", "Very Good": "[85,100]"},
        "selected_by": "Lowest mean RMSE in GroupKFold spatial CV on 0.5-degree blocks",
    }
    joblib.dump(model_bundle, model_dir / "laoagri_best_model_v3.joblib", compress=3)

    metadata = {
        "project": "LaoAgri-Suitability",
        "app_title": "Arabica Land Suitability Explorer",
        "target": "V3 Weighted Geometric Screening Index 0-100 (pseudo-target, not field truth)",
        "study_area": "Lao PDR; Dataset V2 AOI derived from FAO GAUL 2024",
        "dataset_rows": int(len(data)),
        "model_name": best_name,
        "model_features": MODEL_FEATURES,
        "weights_percent": WEIGHTS,
        "spatial_block_size_degrees": BLOCK_SIZE_DEG,
        "spatial_cv_folds": folds,
        "cv_summary": summary.to_dict(orient="records"),
        "heldout_spatial_test_metrics": test_metrics,
        "final_model_training_rows": int(len(final_sample)),
        "random_seed": SEED,
        "score_floor": 1.0,
        "limitations": [
            "The y target is constructed from the same environmental predictors; scores and metrics describe emulation of the scoring rule, not independently validated agricultural suitability.",
            "No Lao farm-level Arabica GPS or yield observations are used as labels.",
            "Elevation and temperature remain highly collinear; their combined target weight is capped at 40%, but both remain model predictors.",
            "SOC score is a relative percentile within the Lao dataset, not an absolute agronomic threshold.",
            "BIO15 is retained as a predictor/diagnostic but excluded from target scoring.",
            "Spatial cells are around 928 m; this is regional screening, not plot-level farm advice.",
        ],
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTraining complete.")
    print("Selected model:", best_name)
    print("Held-out spatial test:", test_metrics)
    print(f"Elapsed: {(time.time() - started)/60:.1f} minutes")


if __name__ == "__main__":
    main()
