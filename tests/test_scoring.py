import numpy as np

from src.scoring import (
    WEIGHTS,
    build_weights,
    classify_suitability,
    score_aspect,
    score_elevation,
    score_soc_from_reference,
    score_soc_relative,
    score_single_site,
)


def test_weights_sum_to_100_and_thermal_group_is_40():
    assert abs(sum(WEIGHTS.values()) - 100.0) < 1e-9
    assert abs(WEIGHTS["elevation"] + WEIGHTS["temperature"] - 40.0) < 1e-9
    assert abs(WEIGHTS["temperature"] - 28.0) < 1e-9
    assert abs(WEIGHTS["elevation"] - 12.0) < 1e-9


def test_other_group_weights_normalize_for_alternate_thermal_weight():
    weights = build_weights(thermal_group_weight=35.0, temperature_share=0.7)
    assert abs(sum(weights.values()) - 100.0) < 1e-9
    assert abs(weights["temperature"] + weights["elevation"] - 35.0) < 1e-9


def test_score_bands_use_requested_cutoffs():
    values = [0, 49.999, 50, 69.999, 70, 84.999, 85, 100]
    labels = classify_suitability(values).tolist()
    assert labels == ["Low", "Low", "Fair", "Fair", "Good", "Good", "Very Good", "Very Good"]


def test_elevation_score_is_bounded():
    scores = score_elevation([-100, 0, 400, 800, 1000, 2800, 4000])
    assert np.all(scores >= 0)
    assert np.all(scores <= 100)
    assert scores[0] == 0
    assert scores[4] == 100


def test_undefined_aspect_gets_neutral_score():
    result = score_aspect([np.nan], [0])
    assert result[0] == 75.0


def test_soc_reference_percentile_matches_reference_rank():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    a = score_soc_relative(ref)
    b = score_soc_from_reference(ref, np.sort(ref))
    assert np.allclose(a, b)


def test_site_score_returns_valid_range_and_band():
    reference = {"soc_sorted_reference": np.linspace(1, 30, 100)}
    payload = {
        "elevation_m": 1100,
        "temperature_c": 21,
        "rainfall_mm_year": 2800,
        "soil_ph": 5.7,
        "slope_pct": 12,
        "bio14_driest_month_mm": 18,
        "soc_0_30cm_g_kg": 20,
        "aspect_deg": 150,
        "aspect_is_defined": 1,
    }
    score, components, band = score_single_site(payload, reference)
    assert 0 <= score <= 100
    assert set(components) == set(WEIGHTS)
    assert band in {"Low", "Fair", "Good", "Very Good"}
