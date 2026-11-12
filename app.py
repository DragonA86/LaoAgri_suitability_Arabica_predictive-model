from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.scoring import MODEL_FEATURES, WEIGHTS, score_single_site  # noqa: E402

st.set_page_config(
    page_title="LaoAgri | Arabica Suitability",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme / Styling -----------------------------
st.markdown(
    """
    <style>
    :root {
      --forest: #14532d;
      --green: #16834a;
      --mint: #e8f5ec;
      --soft: #f6faf7;
      --ink: #183326;
      --muted: #62756a;
      --line: #dce9df;
    }
    .stApp { background: #f7faf8; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,250,248,.92); }
    [data-testid="stSidebar"] { background: #eff7f1; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: #183326; }
    h1, h2, h3 { color: var(--forest) !important; letter-spacing: -0.02em; }
    .hero {
      background: linear-gradient(115deg, #14532d 0%, #16834a 64%, #38a169 100%);
      color: white; padding: 1.5rem 1.7rem; border-radius: 22px;
      margin: .2rem 0 1.2rem 0; box-shadow: 0 10px 26px rgba(20,83,45,.13);
    }
    .hero h1 { color: white !important; margin: 0 0 .35rem 0; font-size: 2.0rem; }
    .hero p { color: #e5f4e9; margin: 0; font-size: .98rem; }
    .eyebrow { color: #c7f0d2; font-size: .72rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    .section-kicker { color: #16834a; font-weight: 800; text-transform: uppercase; letter-spacing: .1em; font-size: .72rem; }
    .soft-card {
      background: white; border: 1px solid var(--line); border-radius: 16px;
      padding: 1rem 1.1rem; box-shadow: 0 3px 12px rgba(21,65,42,.035);
    }
    .metric-label { color: var(--muted); font-size: .78rem; font-weight: 700; }
    .metric-value { color: var(--forest); font-size: 1.75rem; font-weight: 800; line-height: 1.25; }
    .metric-help { color: var(--muted); font-size: .73rem; }
    .notice {
      background: #fff9e9; border-left: 4px solid #d69e2e; border-radius: 10px;
      padding: .85rem 1rem; color: #654c16; font-size: .88rem;
    }
    .success-note {
      background: #e8f5ec; border-left: 4px solid #26965b; border-radius: 10px;
      padding: .85rem 1rem; color: #174c30; font-size: .88rem;
    }
    div[data-testid="stMetric"] {
      background: white; border: 1px solid var(--line); border-radius: 16px;
      padding: .75rem 1rem; box-shadow: 0 3px 12px rgba(21,65,42,.035);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .stButton button, .stDownloadButton button {
      background: #16834a; color: white; border: 1px solid #16834a;
      border-radius: 10px; font-weight: 700;
    }
    .stButton button:hover, .stDownloadButton button:hover {
      background: #14532d; color: white; border-color: #14532d;
    }
    .small-muted { color: #718278; font-size: .78rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Language -----------------------------------
LANG = st.sidebar.selectbox("Language / ພາສາ", ["ລາວ", "English"], index=0)
LAO = LANG == "ລາວ"

TEXT = {
    "nav": "ເມນູຫຼັກ" if LAO else "NAVIGATION",
    "overview": "ພາບລວມ" if LAO else "Overview",
    "map": "ແຜນທີ່ຄວາມເໝາະສົມ" if LAO else "Suitability Map",
    "site": "ທົດລອງພື້ນທີ່" if LAO else "Site Simulator",
    "model": "Model & ວິທີການ" if LAO else "Model & Methods",
    "data": "ຂໍ້ມູນ & ດາວໂຫຼດ" if LAO else "Data & Downloads",
    "subtitle": "GIS + Machine Learning ສຳລັບການຄັດກອງພື້ນທີ່ປູກ Arabica ໃນ ສປປ ລາວ" if LAO else "GIS + Machine Learning for Arabica coffee land-suitability screening in Lao PDR",
    "pseudo": "ຄະແນນນີ້ແມ່ນ Provisional Rule-based Index; ບໍ່ແມ່ນ Field-validated suitability ຫຼື Yield prediction." if LAO else "This is a provisional rule-based index. It is not field-validated suitability or a yield prediction.",
    "cells": "Grid cells" if LAO else "Grid cells",
    "mean": "ຄະແນນສະເລ່ຍ" if LAO else "Mean index",
    "top": "Good + Very Good" if LAO else "Good + Very Good",
    "best": "Model ທີ່ເລືອກ" if LAO else "Selected model",
}

nav_items = [TEXT["overview"], TEXT["map"], TEXT["site"], TEXT["model"], TEXT["data"]]
page = st.sidebar.radio(TEXT["nav"], nav_items, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**LaoAgri-Suitability (LAOS)**")
st.sidebar.caption("Dataset V2 · FAO GAUL 2024 AOI · V3 scoring")

# --------------------------------- Load --------------------------------------
PRED_PATH = ROOT / "data/processed/LaoAgri_Arabica_Suitability_Predictions_V3.csv.gz"
MODEL_PATH = ROOT / "models/laoagri_best_model_v3.joblib"
META_PATH = ROOT / "outputs/model_metadata.json"


@st.cache_data(show_spinner="Loading national grid predictions…")
def load_predictions(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Loading trained model…")
def load_bundle(path: str, modified: float):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_metadata(path: str, modified: float) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def show_missing_assets():
    st.error("Required model files are missing. Run the training command first." if not LAO else "ບໍ່ພົບໄຟລ໌ Model. ກະລຸນາ Train Model ກ່ອນ.")
    st.code("python src/train.py --cv-train-rows 8000 --final-train-rows 50000 --test-train-rows 30000", language="bash")
    st.stop()

if not PRED_PATH.exists() or not MODEL_PATH.exists() or not META_PATH.exists():
    show_missing_assets()

pred_df = load_predictions(str(PRED_PATH), PRED_PATH.stat().st_mtime)
bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
metadata = load_metadata(str(META_PATH), META_PATH.stat().st_mtime)
model = bundle["model"]
selected_model_name = bundle.get("model_name", metadata.get("model_name", "Unknown"))

SCORE_COL = "ml_predicted_suitability"
if SCORE_COL not in pred_df.columns:
    SCORE_COL = "target_y"

pred_df["display_band"] = pd.cut(
    pred_df[SCORE_COL], bins=[0, 50, 70, 85, 100.000001],
    labels=["Low", "Fair", "Good", "Very Good"], right=False, include_lowest=True,
).astype(str)

# --------------------------------- Header ------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">LAOAGRI-SUITABILITY · ARABICA · LAO PDR</div>
      <h1>🌱 Arabica Land Suitability Explorer</h1>
      <p>{TEXT['subtitle']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<div class="notice">⚠️ {TEXT["pseudo"]}</div>', unsafe_allow_html=True)
st.write("")

# -------------------------------- Overview -----------------------------------
if page == TEXT["overview"]:
    st.markdown(f'<div class="section-kicker">01 / OVERVIEW</div>', unsafe_allow_html=True)
    st.header("ພາບລວມຄວາມເໝາະສົມ" if LAO else "National suitability snapshot")

    n_cells = len(pred_df)
    mean_score = float(pred_df[SCORE_COL].mean())
    very_good_share = float(pred_df["display_band"].isin(["Good", "Very Good"]).mean() * 100)
    test_metrics = metadata.get("heldout_spatial_test_metrics", {})
    cv_summary = pd.read_csv(ROOT / "outputs/model_comparison_spatial_cv.csv") if (ROOT / "outputs/model_comparison_spatial_cv.csv").exists() else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(TEXT["cells"], f"{n_cells:,}")
    c2.metric(TEXT["mean"], f"{mean_score:.1f}/100")
    c3.metric(TEXT["top"], f"{very_good_share:.1f}%")
    c4.metric(TEXT["best"], selected_model_name)

    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("ການກະຈາຍຂອງ Suitability Classes" if LAO else "Suitability class distribution")
        dist = pred_df["display_band"].value_counts().reindex(["Low", "Fair", "Good", "Very Good"], fill_value=0)
        dist_df = pd.DataFrame({"Class": dist.index, "Grid cells": dist.values})
        dist_df["Percent"] = dist_df["Grid cells"] / n_cells * 100
        fig = px.bar(dist_df, x="Class", y="Percent", text=dist_df["Percent"].map(lambda v: f"{v:.1f}%"),
                     color="Class", color_discrete_map={"Low": "#b6c9bb", "Fair": "#8dc5a0", "Good": "#3c9b67", "Very Good": "#14532d"})
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Share of cells (%)", xaxis_title="", margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dist_df.style.format({"Grid cells": "{:,.0f}", "Percent": "{:.2f}%"}), hide_index=True, use_container_width=True)

    with right:
        st.subheader("Model validation" if not LAO else "ຜົນການປະເມີນ Model")
        m1, m2 = st.columns(2)
        m1.metric("Spatial Test RMSE", f"{test_metrics.get('RMSE_points', float('nan')):.3f}")
        m2.metric("Spatial Test MAE", f"{test_metrics.get('MAE_points', float('nan')):.3f}")
        st.metric("Spatial Test R²", f"{test_metrics.get('R2', float('nan')):.4f}")
        st.caption(f"Held-out test: {int(test_metrics.get('test_rows', 0)):,} cells across {int(test_metrics.get('test_blocks', 0))} unseen spatial blocks.")
        if not cv_summary.empty:
            st.markdown("**Model benchmark — 5-fold spatial CV**")
            view = cv_summary[["model", "Mean_RMSE", "Mean_MAE", "Mean_R2"]].copy()
            st.dataframe(view.style.format({"Mean_RMSE": "{:.3f}", "Mean_MAE": "{:.3f}", "Mean_R2": "{:.4f}"}), hide_index=True, use_container_width=True)
            st.caption("Selected by lowest mean spatial CV RMSE. HistGradientBoosting is close and had a slightly lower MAE in CV.")

    st.write("")
    st.subheader("ພາບລວມແຜນທີ່" if LAO else "Spatial overview")
    map_n = min(14000, len(pred_df))
    map_sample = pred_df.sample(map_n, random_state=42)
    map_fig = go.Figure(go.Scattergeo(
        lon=map_sample["lon"], lat=map_sample["lat"],
        mode="markers", text=map_sample["grid_id"],
        marker=dict(size=4, color=map_sample[SCORE_COL], colorscale="Greens", cmin=0, cmax=100,
                    colorbar=dict(title="Index"), opacity=0.68, line=dict(width=0)),
        customdata=np.column_stack([map_sample[SCORE_COL], map_sample["display_band"]]),
        hovertemplate="Grid: %{text}<br>Index: %{customdata[0]:.1f}<br>Class: %{customdata[1]}<extra></extra>",
    ))
    map_fig.update_geos(
        lonaxis_range=[100, 108.5], lataxis_range=[13.5, 23.0],
        showcountries=True, showland=True, landcolor="#edf5ee", countrycolor="#9db9a5",
        showocean=True, oceancolor="#f3f8f5", projection_type="mercator",
    )
    map_fig.update_layout(height=520, margin=dict(l=0,r=0,t=5,b=0), paper_bgcolor="white", geo=dict(bgcolor="white"))
    st.plotly_chart(map_fig, use_container_width=True)
    st.caption("National display samples grid cells for responsiveness. Use the Suitability Map page to filter and download cells.")

# -------------------------------- Map page -----------------------------------
elif page == TEXT["map"]:
    st.markdown('<div class="section-kicker">02 / SPATIAL EXPLORER</div>', unsafe_allow_html=True)
    st.header("ແຜນທີ່ຄະແນນ Arabica" if LAO else "Explore the suitability surface")
    st.write("ກັ່ນຕອງຕາມ Class ແລະ Score ເພື່ອສຳຫຼວດ Grid Cells." if LAO else "Filter by class and score to explore individual grid cells.")

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        class_filter = st.multiselect("Suitability class", ["Low", "Fair", "Good", "Very Good"], default=["Good", "Very Good"])
    with f2:
        min_score = st.slider("Minimum index", 0, 100, 0, 1)
    with f3:
        map_limit = st.select_slider("Map detail (sample cells)", options=[3000, 8000, 15000, 25000], value=8000)

    filtered = pred_df[pred_df["display_band"].isin(class_filter) & (pred_df[SCORE_COL] >= min_score)]
    st.caption(f"{len(filtered):,} matching cells out of {len(pred_df):,}.")
    if filtered.empty:
        st.warning("No grid cells match the current filters.")
    else:
        sample = filtered.sample(min(map_limit, len(filtered)), random_state=7)
        fig = go.Figure(go.Scattergeo(
            lon=sample["lon"], lat=sample["lat"], mode="markers", text=sample["grid_id"],
            marker=dict(size=5, color=sample[SCORE_COL], colorscale="Greens", cmin=0, cmax=100,
                        colorbar=dict(title="Suitability index"), opacity=.72),
            customdata=np.column_stack([sample[SCORE_COL], sample["display_band"], sample["elevation_m"], sample["temperature_c"]]),
            hovertemplate="%{text}<br>Index: %{customdata[0]:.1f}<br>Class: %{customdata[1]}<br>Elevation: %{customdata[2]:.0f} m<br>Temperature: %{customdata[3]:.1f} °C<extra></extra>",
        ))
        fig.update_geos(lonaxis_range=[100,108.5], lataxis_range=[13.5,23], showcountries=True, showland=True,
                        landcolor="#edf5ee", countrycolor="#9db9a5", showocean=True, oceancolor="#f3f8f5",
                        projection_type="mercator")
        fig.update_layout(height=640, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="white", geo=dict(bgcolor="white"))
        st.plotly_chart(fig, use_container_width=True)
        show_cols = ["grid_id", "lon", "lat", SCORE_COL, "display_band", "elevation_m", "temperature_c", "rainfall_mm_year", "soil_ph", "slope_pct"]
        st.dataframe(filtered[show_cols].head(500), use_container_width=True, hide_index=True)
        st.download_button("Download filtered grid cells (CSV)", filtered[show_cols].to_csv(index=False).encode("utf-8"),
                           file_name="LaoAgri_filtered_suitability_cells.csv", mime="text/csv")

# -------------------------------- Simulator ----------------------------------
elif page == TEXT["site"]:
    st.markdown('<div class="section-kicker">03 / SITE SIMULATOR</div>', unsafe_allow_html=True)
    st.header("ທົດລອງສະພາບພື້ນທີ່" if LAO else "Simulate an environmental profile")
    st.write("ປ້ອນຄ່າ Environmental Features ແລ້ວເບິ່ງ Rule Index ແລະ ML Estimate." if LAO else "Enter environmental conditions to compare the V3 rule index and the selected ML model estimate.")

    defaults = {
        "elevation_m": float(pred_df.elevation_m.median()),
        "temperature_c": float(pred_df.temperature_c.median()),
        "rainfall_mm_year": float(pred_df.rainfall_mm_year.median()),
        "soil_ph": float(pred_df.soil_ph.median()),
        "slope_pct": float(pred_df.slope_pct.median()),
        "bio14_driest_month_mm": float(pred_df.bio14_driest_month_mm.median()),
        "bio15_seasonality_cv": float(pred_df.bio15_seasonality_cv.median()),
        "soc_0_30cm_g_kg": float(pred_df.soc_0_30cm_g_kg.median()),
    }
    with st.form("site_simulator"):
        st.markdown("**Climate & topography**" if not LAO else "**ພູມອາກາດ ແລະພູມສັນຖານ**")
        a, b, c = st.columns(3)
        with a:
            elevation = st.number_input("Elevation (m)", 0.0, 3500.0, defaults["elevation_m"], step=10.0)
            temperature = st.number_input("Annual mean temperature (°C)", 0.0, 40.0, defaults["temperature_c"], step=0.1)
            rainfall = st.number_input("Annual rainfall (mm)", 0.0, 6000.0, defaults["rainfall_mm_year"], step=25.0)
        with b:
            bio14 = st.number_input("BIO14 — driest month (mm)", 0.0, 500.0, defaults["bio14_driest_month_mm"], step=1.0)
            bio15 = st.number_input("BIO15 — seasonality CV", 0.0, 300.0, defaults["bio15_seasonality_cv"], step=1.0)
            slope = st.number_input("Slope (%)", 0.0, 200.0, defaults["slope_pct"], step=1.0)
        with c:
            soil_ph = st.number_input("Soil pH (H₂O)", 3.0, 9.0, defaults["soil_ph"], step=0.05, format="%.2f")
            soc = st.number_input("SOC 0–30 cm (g/kg)", 0.0, 300.0, defaults["soc_0_30cm_g_kg"], step=0.5)
            aspect = st.number_input("Aspect (degrees)", 0.0, 359.9, 180.0, step=5.0)
            aspect_defined = st.checkbox("Aspect is defined (non-flat terrain)", value=True)
        submitted = st.form_submit_button("Calculate suitability estimate")

    if submitted:
        payload = {
            "elevation_m": elevation,
            "temperature_c": temperature,
            "rainfall_mm_year": rainfall,
            "soil_ph": soil_ph,
            "slope_pct": slope,
            "bio14_driest_month_mm": bio14,
            "bio15_seasonality_cv": bio15,
            "soc_0_30cm_g_kg": soc,
            "aspect_deg": aspect,
            "aspect_is_defined": aspect_defined,
        }
        rule_score, components, rule_band = score_single_site(payload, bundle["scoring_reference"])
        aspect_rad = np.deg2rad(aspect)
        site_row = pd.DataFrame([{
            "elevation_m": elevation,
            "temperature_c": temperature,
            "rainfall_mm_year": rainfall,
            "soil_ph": soil_ph,
            "slope_pct": slope,
            "bio14_driest_month_mm": bio14,
            "bio15_seasonality_cv": bio15,
            "soc_0_30cm_g_kg": soc,
            "aspect_sin": float(np.sin(aspect_rad)) if aspect_defined else 0.0,
            "aspect_cos": float(np.cos(aspect_rad)) if aspect_defined else 0.0,
            "aspect_is_defined": int(aspect_defined),
        }], columns=MODEL_FEATURES).astype("float32")
        model_score = float(np.clip(model.predict(site_row)[0], 0, 100))
        model_band = str(pd.cut([model_score], bins=[0,50,70,85,100.000001], labels=["Low","Fair","Good","Very Good"], right=False, include_lowest=True).astype(str)[0])

        r1, r2, r3 = st.columns(3)
        r1.metric("V3 rule index", f"{rule_score:.1f}/100", rule_band)
        r2.metric("ML model estimate", f"{model_score:.1f}/100", model_band)
        r3.metric("Difference", f"{abs(model_score-rule_score):.2f} points")
        st.markdown('<div class="notice">The ML estimate emulates the V3 rule index. A close match does not establish real-world Arabica performance or guarantee a successful crop.</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({"Factor": list(components.keys()), "Component score": list(components.values()), "Weight (%)": [WEIGHTS[k] for k in components]})
        comp_fig = px.bar(comp_df.sort_values("Component score"), x="Component score", y="Factor", orientation="h", color="Component score", color_continuous_scale="Greens", range_color=[0,100])
        comp_fig.update_layout(height=390, plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False, margin=dict(l=8,r=8,t=8,b=8))
        st.plotly_chart(comp_fig, use_container_width=True)

# --------------------------------- Methods -----------------------------------
elif page == TEXT["model"]:
    st.markdown('<div class="section-kicker">04 / MODEL & METHODS</div>', unsafe_allow_html=True)
    st.header("Model selection & methodology" if not LAO else "ການເລືອກ Model ແລະ Methodology")
    st.markdown("**Selected model:** " + selected_model_name)
    st.write("Primary selection criterion: lowest Mean RMSE across 5-fold GroupKFold spatial validation (0.5° × 0.5° blocks).")

    cv_path = ROOT / "outputs/model_comparison_spatial_cv.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        fig = px.bar(cv.sort_values("Mean_RMSE"), x="model", y="Mean_RMSE", error_y="SD_RMSE", color="Mean_RMSE", color_continuous_scale="Greens")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False, xaxis_title="Model", yaxis_title="Mean spatial CV RMSE (points)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cv.style.format({"Mean_MAE":"{:.3f}","SD_MAE":"{:.3f}","Mean_RMSE":"{:.3f}","SD_RMSE":"{:.3f}","Mean_R2":"{:.4f}","SD_R2":"{:.4f}"}), hide_index=True, use_container_width=True)
        xgb = cv[cv.model == "XGBoost"]
        hgb = cv[cv.model == "HistGradientBoosting"]
        if not xgb.empty and not hgb.empty:
            gap = float(hgb.iloc[0].Mean_RMSE - xgb.iloc[0].Mean_RMSE)
            st.info(f"XGBoost's mean RMSE advantage over HistGradientBoosting is {gap:.3f} points. Their fold variability overlaps; model selection is therefore close rather than decisive.")

    st.subheader("Held-out spatial test")
    metrics = metadata.get("heldout_spatial_test_metrics", {})
    st.dataframe(pd.DataFrame([metrics]), hide_index=True, use_container_width=True)
    st.subheader("Predictor multicollinearity (VIF)")
    vif_path = ROOT / "outputs/vif_diagnostic.csv"
    if vif_path.exists():
        vif = pd.read_csv(vif_path)
        fig_vif = px.bar(vif.sort_values("VIF", ascending=True), x="VIF", y="feature", orientation="h", color="VIF", color_continuous_scale="Greens")
        fig_vif.add_vline(x=5, line_dash="dash", line_color="#d69e2e", annotation_text="VIF 5")
        fig_vif.add_vline(x=10, line_dash="dot", line_color="#b83232", annotation_text="VIF 10")
        fig_vif.update_layout(plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False, xaxis_title="Variance Inflation Factor")
        st.plotly_chart(fig_vif, use_container_width=True)
        st.dataframe(vif.style.format({"auxiliary_R2":"{:.4f}","VIF":"{:.3f}"}), hide_index=True, use_container_width=True)
        st.caption("VIF diagnoses redundancy among predictors. It is not a formula for calculating agronomic weights.")

    imp_path = ROOT / "outputs/permutation_feature_importance.csv"
    if imp_path.exists():
        st.subheader("Permutation feature importance")
        st.caption("Measured on held-out blocks; indicates contribution to reproducing the pseudo-target, not causal agronomic importance.")
        imp = pd.read_csv(imp_path).sort_values("importance_mean")
        fig_imp = px.bar(imp, x="importance_mean", y="feature", orientation="h", error_x="importance_std", color="importance_mean", color_continuous_scale="Greens")
        fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False, xaxis_title="RMSE increase after permutation")
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("V3 scoring weights")
    wdf = pd.DataFrame({"Factor": list(WEIGHTS.keys()), "Weight (%)": list(WEIGHTS.values())})
    fig_w = px.bar(wdf.sort_values("Weight (%)"), x="Weight (%)", y="Factor", orientation="h", color="Weight (%)", color_continuous_scale="Greens")
    fig_w.update_layout(plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig_w, use_container_width=True)
    st.markdown("""
    **Important limitations**
    - `target_y` is calculated from the same environmental predictors used by ML. Model metrics measure rule emulation, not independently validated land suitability.
    - FAO GAUL 2024 defines the study-area boundary; it does not prescribe crop weights.
    - The score weights and some interpolation anchors are project assumptions informed by local Arabica agronomy references.
    - Elevation and temperature remain strongly correlated. Grouping their total target weight at 40% controls their combined influence in the index, but does not remove collinearity from model predictors.
    - The approximately 928 m grid supports regional screening, not farm-plot decisions.
    """)

# -------------------------------- Data page ----------------------------------
elif page == TEXT["data"]:
    st.markdown('<div class="section-kicker">05 / DATA & DOWNLOADS</div>', unsafe_allow_html=True)
    st.header("Data, outputs & reproducibility" if not LAO else "ຂໍ້ມູນ, ໄຟລ໌ຜົນລັບ ແລະການເຮັດຊ້ຳ")
    st.write("Dataset V2 fields and generated V3 model outputs are provided with this VS Code project.")
    st.dataframe(pred_df[["grid_id","lon","lat","target_y", "ml_predicted_suitability", "display_band"]].head(30), hide_index=True, use_container_width=True)
    st.download_button("Download national prediction layer (CSV.GZ)", PRED_PATH.read_bytes(),
                       file_name="LaoAgri_Arabica_Suitability_Predictions_V3.csv.gz", mime="application/gzip")
    for filename, label in [
        ("model_comparison_spatial_cv.csv", "Spatial CV model comparison"),
        ("heldout_spatial_test_metrics.csv", "Held-out test metrics"),
        ("vif_diagnostic.csv", "VIF diagnostics"),
        ("permutation_feature_importance.csv", "Permutation feature importance"),
        ("model_metadata.json", "Model metadata"),
    ]:
        fp = ROOT / "outputs" / filename
        if fp.exists():
            st.download_button(f"Download {label}", fp.read_bytes(), file_name=filename, key=f"dl_{filename}")

st.markdown("---")
st.caption("LaoAgri-Suitability · Arabica only · V3 provisional screening index · Not a guarantee of yield or farm profitability")
