"""V3 Arabica suitability scoring functions.

The target is a project-defined pseudo-target built from environmental features.
It is not a field-validated label or a measured yield.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THERMAL_ALTITUDE_GROUP_WEIGHT = 40.0
TEMPERATURE_SHARE_IN_GROUP = 0.70
ELEVATION_SHARE_IN_GROUP = 0.30
SCORE_FLOOR = 1.0

OTHER_FACTOR_BASE_WEIGHTS = {
    "rainfall": 20.0,
    "bio14": 5.0,
    "soil_ph": 15.0,
    "soc": 5.0,
    "slope": 12.0,
    "aspect": 3.0,
}

SCORE_COLUMN_BY_FACTOR = {
    "elevation": "score_elevation",
    "temperature": "score_temperature",
    "rainfall": "score_rainfall",
    "soil_ph": "score_soil_ph",
    "slope": "score_slope",
    "bio14": "score_bio14",
    "soc": "score_soc",
    "aspect": "score_aspect",
}

MODEL_FEATURES = [
    "elevation_m",
    "temperature_c",
    "rainfall_mm_year",
    "soil_ph",
    "slope_pct",
    "bio14_driest_month_mm",
    "bio15_seasonality_cv",
    "soc_0_30cm_g_kg",
    "aspect_sin",
    "aspect_cos",
    "aspect_is_defined",
]

REQUIRED_COLUMNS = [
    "grid_id", "lon", "lat", "elevation_m", "temperature_c",
    "rainfall_mm_year", "soil_ph", "slope_pct",
    "bio14_driest_month_mm", "bio15_seasonality_cv",
    "soc_0_30cm_g_kg", "aspect_deg", "aspect_sin", "aspect_cos",
    "aspect_is_defined",
]

BAND_LABELS = ["Low", "Fair", "Good", "Very Good"]
BAND_BINS = [0.0, 50.0, 70.0, 85.0, 100.000001]


def build_weights(
    thermal_group_weight: float = THERMAL_ALTITUDE_GROUP_WEIGHT,
    temperature_share: float = TEMPERATURE_SHARE_IN_GROUP,
) -> dict[str, float]:
    if not 0 < thermal_group_weight < 100:
        raise ValueError("thermal_group_weight must be between 0 and 100")
    if not 0 < temperature_share < 1:
        raise ValueError("temperature_share must be between 0 and 1")
    other_total = sum(OTHER_FACTOR_BASE_WEIGHTS.values())
    other_scale = (100.0 - thermal_group_weight) / other_total
    weights = {
        "elevation": thermal_group_weight * (1.0 - temperature_share),
        "temperature": thermal_group_weight * temperature_share,
    }
    weights.update({k: v * other_scale for k, v in OTHER_FACTOR_BASE_WEIGHTS.items()})
    return weights


WEIGHTS = build_weights()


def interp_score(values, x_points, y_points) -> np.ndarray:
    return np.interp(np.asarray(values, dtype=float), x_points, y_points).clip(0, 100)


def score_elevation(x) -> np.ndarray:
    return interp_score(x, [0, 400, 800, 1000, 2800, 3000], [0, 0, 60, 100, 100, 0])


def score_temperature(x) -> np.ndarray:
    return interp_score(x, [0, 12, 15, 19, 22.2, 25, 28, 30, 40], [0, 0, 0, 100, 100, 75, 30, 0, 0])


def score_rainfall(x) -> np.ndarray:
    return interp_score(x, [0, 750, 1200, 1500, 2500, 3534, 4200, 5000], [0, 0, 60, 85, 100, 100, 0, 0])


def score_soil_ph(x) -> np.ndarray:
    return interp_score(x, [0, 4.3, 4.7, 5.0, 5.5, 5.75, 6.0, 6.5, 7.0, 8.4, 14], [0, 0, 40, 85, 95, 100, 95, 70, 40, 0, 0])


def score_slope(x) -> np.ndarray:
    return interp_score(x, [0, 15, 30, 45, 60, 100, 200], [100, 100, 70, 35, 0, 0, 0])


def score_bio14(x) -> np.ndarray:
    return interp_score(x, [0, 5, 15, 25, 40, 60, 100], [100, 100, 80, 60, 25, 0, 0])


def score_soc_relative(values, missing_mask=None) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    scores = series.rank(method="average", pct=True) * 100.0
    if missing_mask is not None:
        scores.loc[np.asarray(missing_mask, dtype=bool)] = 50.0
    return scores.to_numpy(dtype=float)


def score_soc_from_reference(values, sorted_reference: np.ndarray, missing_mask=None) -> np.ndarray:
    """Map new SOC observations to the fitted reference empirical percentile.

    Uses mid-rank behavior for ties, matching pandas rank(method='average', pct=True)
    when values are evaluated against the same reference sample.
    """
    x = np.asarray(values, dtype=float)
    ref = np.asarray(sorted_reference, dtype=float)
    left = np.searchsorted(ref, x, side="left")
    right = np.searchsorted(ref, x, side="right")
    scores = ((left + right + 1.0) / 2.0) / max(len(ref), 1) * 100.0
    if missing_mask is not None:
        scores[np.asarray(missing_mask, dtype=bool)] = 50.0
    return np.clip(scores, 0, 100)


def score_aspect(aspect_deg, aspect_is_defined) -> np.ndarray:
    aspect = np.asarray(aspect_deg, dtype=float)
    defined = np.asarray(aspect_is_defined).astype(bool)
    d_east = np.abs((aspect - 90 + 180) % 360 - 180)
    d_south = np.abs((aspect - 180 + 180) % 360 - 180)
    best_cosine = np.maximum(np.cos(np.deg2rad(d_east)), np.cos(np.deg2rad(d_south)))
    scores = 50 + 50 * best_cosine
    scores = np.where(defined & np.isfinite(aspect), scores, 75.0)
    return np.clip(scores, 0, 100)


def classify_suitability(values) -> np.ndarray:
    return pd.cut(
        np.asarray(values, dtype=float), bins=BAND_BINS,
        labels=BAND_LABELS, right=False, include_lowest=True,
    ).astype(str)


def prepare_environmental_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Apply V3 imputation and derive component scores/target using full dataset refs."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    work = df.copy()
    work["soil_ph_was_missing"] = work["soil_ph"].isna().astype("int8")
    work["soc_was_missing"] = work["soc_0_30cm_g_kg"].isna().astype("int8")
    pH_median = float(work["soil_ph"].median())
    soc_median = float(work["soc_0_30cm_g_kg"].median())
    work["soil_ph"] = work["soil_ph"].fillna(pH_median)
    work["soc_0_30cm_g_kg"] = work["soc_0_30cm_g_kg"].fillna(soc_median)

    missing_angle = work["aspect_deg"].isna() & (work["aspect_is_defined"].astype(int) == 1)
    if missing_angle.any():
        reconstructed = (
            np.degrees(np.arctan2(work.loc[missing_angle, "aspect_sin"], work.loc[missing_angle, "aspect_cos"])) + 360
        ) % 360
        work.loc[missing_angle, "aspect_deg"] = reconstructed

    core = [
        "lon", "lat", "elevation_m", "temperature_c", "rainfall_mm_year",
        "soil_ph", "slope_pct", "bio14_driest_month_mm", "bio15_seasonality_cv",
        "soc_0_30cm_g_kg", "aspect_sin", "aspect_cos", "aspect_is_defined",
    ]
    if work[core].isna().any().any():
        bad = work[core].isna().sum()
        raise ValueError(f"Missing values remain in required fields: {bad[bad > 0].to_dict()}")

    work["score_elevation"] = score_elevation(work["elevation_m"])
    work["score_temperature"] = score_temperature(work["temperature_c"])
    work["score_rainfall"] = score_rainfall(work["rainfall_mm_year"])
    work["score_soil_ph"] = score_soil_ph(work["soil_ph"])
    work["score_slope"] = score_slope(work["slope_pct"])
    work["score_bio14"] = score_bio14(work["bio14_driest_month_mm"])
    work["score_soc"] = score_soc_relative(
        work["soc_0_30cm_g_kg"].to_numpy(), work["soc_was_missing"].to_numpy()
    )
    work["score_aspect"] = score_aspect(work["aspect_deg"], work["aspect_is_defined"])

    score_cols = [SCORE_COLUMN_BY_FACTOR[k] for k in WEIGHTS]
    matrix = work[score_cols].to_numpy(dtype="float64")
    weight_vec = np.array([WEIGHTS[k] for k in WEIGHTS], dtype="float64") / 100.0
    target = 100.0 * np.exp(np.sum(np.log(np.maximum(matrix, SCORE_FLOOR) / 100.0) * weight_vec, axis=1))
    work["target_y"] = np.clip(target, 0, 100)
    work["target_y_wlc"] = np.clip(np.sum(matrix * weight_vec, axis=1), 0, 100)
    work["suitability_band"] = classify_suitability(work["target_y"])
    work["limiting_feature"] = work[score_cols].idxmin(axis=1).str.replace("score_", "", regex=False)

    reference = {
        "soil_ph_median": pH_median,
        "soc_median": soc_median,
        "soc_sorted_reference": np.sort(work["soc_0_30cm_g_kg"].to_numpy(dtype=float)),
    }
    return work, reference["soc_sorted_reference"], reference


def score_single_site(payload: dict, scoring_reference: dict) -> tuple[float, dict[str, float], str]:
    """Calculate V3 weighted-geometric screening index for one hypothetical site."""
    components = {
        "elevation": float(score_elevation([payload["elevation_m"]])[0]),
        "temperature": float(score_temperature([payload["temperature_c"]])[0]),
        "rainfall": float(score_rainfall([payload["rainfall_mm_year"]])[0]),
        "soil_ph": float(score_soil_ph([payload["soil_ph"]])[0]),
        "slope": float(score_slope([payload["slope_pct"]])[0]),
        "bio14": float(score_bio14([payload["bio14_driest_month_mm"]])[0]),
        "soc": float(score_soc_from_reference(
            [payload["soc_0_30cm_g_kg"]], scoring_reference["soc_sorted_reference"]
        )[0]),
        "aspect": float(score_aspect([payload["aspect_deg"]], [payload["aspect_is_defined"]])[0]),
    }
    weights = WEIGHTS
    log_score = sum((weights[k] / 100.0) * np.log(max(components[k], SCORE_FLOOR) / 100.0) for k in weights)
    target = float(np.clip(100.0 * np.exp(log_score), 0, 100))
    band = str(classify_suitability([target])[0])
    return target, components, band
