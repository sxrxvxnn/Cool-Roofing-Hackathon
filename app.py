import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cool Roofing Predictor", page_icon="🏠", layout="wide")

# ── Category lists (same as notebook) ─────────────────────────────────────────
LOCATION_TYPES   = ["urban", "intermediate", "rural", "dense_urban"]
QUALITY_LEVELS   = ["basic", "intermediate", "high"]
ROOFING_TYPES    = ["concrete_flat", "tiles", "concrete_slanted", "sheet"]
ROOFING_METHODS  = ["basic_coating", "tiles", "tiles_with_coating", "sheet_with_coating", "sheet"]

CAT_COLS = ["location_type", "cool_roofing_quality", "roofing_type", "cool_roofing_method"]
CAT_VALS = {
    "location_type":        LOCATION_TYPES,
    "cool_roofing_quality": QUALITY_LEVELS,
    "roofing_type":         ROOFING_TYPES,
    "cool_roofing_method":  ROOFING_METHODS,
}

BASE_NUM = ["building_size", "building_age", "num_ac_units"]
BASE_CAT = [f"{col}_{val}" for col, vals in CAT_VALS.items() for val in vals]
BASE = BASE_NUM + BASE_CAT

TEMP_FEATS     = BASE + ["avg_roof_temperature_before"]
UNITS_FEATS    = BASE + ["avg_units_before"]
RATING_FEATS   = BASE + ["avg_roof_temperature_before", "avg_units_before"]
REFERRAL_FEATS = BASE + ["avg_roof_temperature_before", "avg_units_before"]


def encode_input(row: dict) -> pd.DataFrame:
    """Convert user input dict to one-hot encoded row matching training columns."""
    # Start with numeric features
    encoded = {col: row.get(col, 0) for col in BASE_NUM}
    # One-hot encode categoricals
    for col, vals in CAT_VALS.items():
        for val in vals:
            encoded[f"{col}_{val}"] = 1 if row.get(col) == val else 0
    # Add continuous pre-install features
    encoded["avg_roof_temperature_before"] = row.get("avg_roof_temperature_before", np.nan)
    encoded["avg_units_before"]            = row.get("avg_units_before", 0)
    return pd.DataFrame([encoded])


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    models = {}
    features = {}
    tasks = ["cost", "temp_drop", "units_saved", "high_rating", "referral"]
    for task in tasks:
        mpath = os.path.join(model_dir, f"model_{task}.joblib")
        fpath = os.path.join(model_dir, f"features_{task}.joblib")
        if os.path.exists(mpath):
            models[task] = joblib.load(mpath)
        if os.path.exists(fpath):
            features[task] = joblib.load(fpath)
    return models, features


def predict_all(row: dict, models: dict, features: dict) -> dict:
    enc = encode_input(row)
    results = {}
    for task, m in models.items():
        feats = features.get(task, BASE)
        # add any missing columns as 0
        for f in feats:
            if f not in enc.columns:
                enc[f] = 0
        X = enc[feats].fillna(0)
        if task in ("high_rating", "referral"):
            results[task] = float(m.predict_proba(X)[0, 1])
        else:
            results[task] = float(m.predict(X)[0])
    return results


# ── Load ──────────────────────────────────────────────────────────────────────
models, features = load_models()

# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Cool Roofing Predictor")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Predict", "Batch & Feature Selection", "Recommend"])

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "Home":
    st.title("Cool Roofing Cost & Impact Predictor")
    st.markdown("""
Predicts five outcomes of a cool roofing installation from building details
available **before** any work starts.

| Task | Type | Metric |
|------|------|--------|
| Installation Cost | Regression | MAE / RMSE |
| Roof Temperature Drop | Regression | MAE / RMSE |
| Electricity Units Saved | Regression | MAE / RMSE |
| High Customer Rating | Classification | Accuracy / MAE |
| Customer Referral | Classification | Accuracy / F1 |
""")
    if models:
        st.success(f"Models loaded: {', '.join(models.keys())}")
    else:
        st.error("No models found. Run the notebook first to generate models/.")

    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "cool_roofing_data.csv"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset rows", f"{len(df):,}")
        c2.metric("Avg install cost", f"₹{df['cool_roofing_cost'].mean():,.0f}")
        c3.metric("Avg temp drop", f"{(df['avg_roof_temperature_before']-df['avg_roof_temperature_after']).mean():.1f}°C")
        c4.metric("Avg units saved", f"{(df['avg_units_before']-df['avg_units_after']).mean():.1f} kWh")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Dashboard":
    st.title("Dataset Analysis Dashboard")

    @st.cache_data
    def load_data():
        path = os.path.join(os.path.dirname(__file__), "cool_roofing_data.csv")
        df = pd.read_csv(path)
        df["avg_roof_temperature_before"] = df["avg_roof_temperature_before"].replace(120, np.nan)
        df["temperature_drop"] = df["avg_roof_temperature_before"] - df["avg_roof_temperature_after"]
        df["units_saved"]      = df["avg_units_before"] - df["avg_units_after"]
        df["high_rating"]      = (df["customer_given_rating"] >= 4).astype(int)
        df["referral"]         = df["customer_referral"].astype(int)
        return df

    df = load_data()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    sel_location = st.sidebar.multiselect("Location type", df["location_type"].unique().tolist(),
                                          default=df["location_type"].unique().tolist())
    sel_quality  = st.sidebar.multiselect("Quality", df["cool_roofing_quality"].unique().tolist(),
                                          default=df["cool_roofing_quality"].unique().tolist())
    sel_method   = st.sidebar.multiselect("Method", df["cool_roofing_method"].unique().tolist(),
                                          default=df["cool_roofing_method"].unique().tolist())
    size_range   = st.sidebar.slider("Building size (sq ft)", int(df["building_size"].min()),
                                     int(df["building_size"].max()),
                                     (int(df["building_size"].min()), int(df["building_size"].max())))

    fdf = df[
        df["location_type"].isin(sel_location) &
        df["cool_roofing_quality"].isin(sel_quality) &
        df["cool_roofing_method"].isin(sel_method) &
        df["building_size"].between(*size_range)
    ]

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Rows",            f"{len(fdf):,}")
    k2.metric("Avg Cost",        f"₹{fdf['cool_roofing_cost'].mean():,.0f}")
    k3.metric("Avg Temp Drop",   f"{fdf['temperature_drop'].mean():.1f}°C")
    k4.metric("Avg Units Saved", f"{fdf['units_saved'].mean():.1f} kWh")
    k5.metric("High Rating %",   f"{fdf['high_rating'].mean():.1%}")
    k6.metric("Referral %",      f"{fdf['referral'].mean():.1%}")

    st.markdown("---")

    # ── Row 1: Cost distribution + Cost by method×quality heatmap ─────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost Distribution")
        color_by = st.selectbox("Color by", ["cool_roofing_quality", "cool_roofing_method",
                                              "roofing_type", "location_type"], key="cost_color")
        fig = px.histogram(fdf, x="cool_roofing_cost", color=color_by, nbins=40,
                           labels={"cool_roofing_cost": "Installation Cost (₹)"},
                           barmode="overlay", opacity=0.7,
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Cost: Method × Quality")
        pivot = fdf.groupby(["cool_roofing_method", "cool_roofing_quality"])["cool_roofing_cost"].mean().reset_index()
        heat  = pivot.pivot(index="cool_roofing_method", columns="cool_roofing_quality", values="cool_roofing_cost")
        fig2  = px.imshow(heat, text_auto=".0f", aspect="auto",
                          color_continuous_scale="Blues",
                          labels=dict(color="Avg Cost (₹)"))
        fig2.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Cost vs building size scatter + Temp drop distribution ─────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Cost vs Building Size")
        scatter_df = fdf.copy()
        scatter_df["num_ac_units"] = scatter_df["num_ac_units"].fillna(1).clip(lower=1)
        fig3 = px.scatter(scatter_df, x="building_size", y="cool_roofing_cost",
                          color="cool_roofing_quality", size="num_ac_units",
                          hover_data=["cool_roofing_method", "location_type"],
                          labels={"building_size": "Building Size (sq ft)",
                                  "cool_roofing_cost": "Cost (₹)"},
                          color_discrete_sequence=px.colors.qualitative.Set1,
                          opacity=0.6)
        fig3.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Temperature Drop by Method")
        fig4 = px.box(fdf.dropna(subset=["temperature_drop"]),
                      x="cool_roofing_method", y="temperature_drop",
                      color="cool_roofing_quality",
                      labels={"temperature_drop": "Temp Drop (°C)",
                              "cool_roofing_method": "Method"},
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_layout(height=380, margin=dict(t=20, b=20), xaxis_tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Units saved vs before + Rating distribution ────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Units Saved vs Baseline Usage")
        fig5 = px.scatter(fdf, x="avg_units_before", y="units_saved",
                          color="cool_roofing_quality",
                          labels={"avg_units_before": "Baseline Usage (kWh)",
                                  "units_saved": "Units Saved (kWh)"},
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          opacity=0.6)
        fig5.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.subheader("Customer Rating Distribution")
        rating_counts = fdf["customer_given_rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        fig6 = px.bar(rating_counts, x="Rating", y="Count",
                      color="Rating", color_continuous_scale="RdYlGn",
                      labels={"Rating": "Customer Rating (1-5)"})
        fig6.update_layout(height=380, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: Categorical breakdowns ─────────────────────────────────────────
    st.subheader("Categorical Breakdowns")
    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
    for col_widget, cat in zip([cat_col1, cat_col2, cat_col3, cat_col4],
                                ["location_type", "roofing_type", "cool_roofing_method", "cool_roofing_quality"]):
        counts = fdf[cat].value_counts().reset_index()
        counts.columns = [cat, "count"]
        fig_pie = px.pie(counts, names=cat, values="count", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(height=280, margin=dict(t=30, b=0, l=0, r=0),
                              title_text=cat.replace("_", " ").title())
        col_widget.plotly_chart(fig_pie, use_container_width=True)

    # ── Row 5: Correlation heatmap ─────────────────────────────────────────────
    st.subheader("Correlation Matrix")
    num_cols = ["building_size", "building_age", "num_ac_units",
                "avg_roof_temperature_before", "avg_units_before",
                "cool_roofing_cost", "temperature_drop", "units_saved",
                "customer_given_rating"]
    corr = fdf[num_cols].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_corr.update_layout(height=450, margin=dict(t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Row 6: Raw data table ──────────────────────────────────────────────────
    with st.expander("Raw data sample"):
        st.dataframe(fdf.head(100), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Predict":
    st.title("Single Building Prediction")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            location_type       = st.selectbox("Location type", LOCATION_TYPES)
            building_size       = st.number_input("Building size (sq ft)", 100, 10000, 1200)
            building_age        = st.number_input("Building age (years)", 0, 100, 10)
            num_ac_units        = st.number_input("Number of AC units", 0, 50, 3)
        with c2:
            roofing_type        = st.selectbox("Roof type", ROOFING_TYPES)
            cool_method         = st.selectbox("Cool roofing method", ROOFING_METHODS)
            cool_quality        = st.selectbox("Cool roofing quality", QUALITY_LEVELS)
            avg_temp_before     = st.number_input("Avg roof temp before (°C)", 20.0, 85.0, 55.0, step=0.5)
            avg_units_before    = st.number_input("Avg monthly electricity (kWh)", 0.0, 2000.0, 250.0)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        row = {
            "location_type":             location_type,
            "building_size":             building_size,
            "building_age":              building_age,
            "num_ac_units":              num_ac_units,
            "roofing_type":              roofing_type,
            "cool_roofing_method":       cool_method,
            "cool_roofing_quality":      cool_quality,
            "avg_roof_temperature_before": avg_temp_before,
            "avg_units_before":          avg_units_before,
        }
        preds = predict_all(row, models, features)

        st.subheader("Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("Installation Cost",   f"₹{preds.get('cost', 0):,.0f}")
        r2.metric("Temp Drop",           f"{preds.get('temp_drop', 0):.2f}°C")
        r3.metric("Units Saved / Month", f"{preds.get('units_saved', 0):.1f} kWh")

        r4, r5 = st.columns(2)
        hr  = preds.get("high_rating", 0)
        ref = preds.get("referral", 0)
        r4.metric("P(High Rating ≥ 4)", f"{hr:.1%}")
        r5.metric("P(Referral)",         f"{ref:.1%}")

        st.subheader("Financial Estimate")
        st.caption("⚠️ Tariff is **not in the dataset** — used only for savings estimate below.")
        tariff = st.number_input("Electricity tariff (₹/kWh) — not in dataset", 4.0, 12.0, 7.0, step=0.5)
        monthly = preds.get("units_saved", 0) * tariff
        annual  = monthly * 12
        payback = preds.get("cost", 0) / annual if annual > 0 else None

        f1, f2, f3 = st.columns(3)
        f1.metric("Monthly Savings", f"₹{monthly:,.0f}")
        f2.metric("Annual Savings",  f"₹{annual:,.0f}")
        f3.metric("Payback Period",  f"{payback:.1f} yrs" if payback else "N/A")

# ─────────────────────────────────────────────────────────────────────────────
# BATCH & FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Batch & Feature Selection":
    st.title("Batch Prediction & Feature Selection")

    # ── Human-readable raw feature names ──────────────────────────────────────
    RAW_FEATURES = BASE_NUM + list(CAT_VALS.keys()) + [
        "avg_roof_temperature_before", "avg_units_before"
    ]
    # importance per raw feature = sum of dummies for that feature
    def get_importance(task_name: str) -> pd.Series:
        m = models.get(task_name)
        if m is None:
            return pd.Series(dtype=float)
        feats = features.get(task_name, [])
        fi = pd.Series(m.feature_importances_, index=feats)
        # collapse dummy columns back to raw feature name
        collapsed = {}
        for feat, imp in fi.items():
            raw = next((c for c in CAT_VALS if feat.startswith(f"{c}_")), feat)
            collapsed[raw] = collapsed.get(raw, 0.0) + imp
        return pd.Series(collapsed).sort_values(ascending=False)

    # ── Feature selection mode ─────────────────────────────────────────────────
    st.subheader("Feature Selection")
    sel_mode = st.radio("Mode", ["Manual", "Automatic (importance-based)"], horizontal=True)

    task_for_fi = st.selectbox(
        "Task to base feature importance on",
        ["cost", "temp_drop", "units_saved", "high_rating", "referral"],
        index=0,
    )
    fi_series = get_importance(task_for_fi)

    if sel_mode == "Manual":
        selected_raw = st.multiselect(
            "Select features to use for prediction",
            options=RAW_FEATURES,
            default=RAW_FEATURES,
        )
        if not selected_raw:
            st.warning("Select at least one feature.")
            selected_raw = RAW_FEATURES
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Feature importances** (collapsed from model):")
            st.bar_chart(fi_series)
        with col2:
            top_n = st.slider(
                "Top N features to keep",
                min_value=2, max_value=len(RAW_FEATURES),
                value=min(6, len(RAW_FEATURES)),
            )
            importance_threshold = st.slider(
                "Minimum importance threshold",
                min_value=0.00, max_value=0.50,
                value=0.00, step=0.01,
            )
        # auto-select
        auto_ranked = fi_series[fi_series >= importance_threshold].head(top_n)
        selected_raw = list(auto_ranked.index)
        st.info(f"Auto-selected {len(selected_raw)} features: **{', '.join(selected_raw)}**")

    # ── Template download ──────────────────────────────────────────────────────
    st.subheader("Upload CSV for Batch Prediction")
    template_cols = BASE_NUM + list(CAT_VALS.keys()) + ["avg_roof_temperature_before", "avg_units_before"]
    template_row  = {
        "building_size": 1200, "building_age": 10, "num_ac_units": 3,
        "location_type": "urban", "cool_roofing_quality": "intermediate",
        "roofing_type": "concrete_flat", "cool_roofing_method": "basic_coating",
        "avg_roof_temperature_before": 55.0, "avg_units_before": 250.0,
    }
    template_df = pd.DataFrame([template_row])
    st.download_button(
        "⬇ Download template CSV",
        data=template_df.to_csv(index=False),
        file_name="batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    # Use sample data button
    use_sample = st.checkbox("Use built-in dataset (cool_roofing_data.csv) instead of uploading")
    if use_sample:
        data_path = os.path.join(os.path.dirname(__file__), "cool_roofing_data.csv")
        if os.path.exists(data_path):
            batch_df = pd.read_csv(data_path).head(200)
            st.info(f"Using first 200 rows of cool_roofing_data.csv")
        else:
            st.error("cool_roofing_data.csv not found in submission folder.")
            batch_df = None
    elif uploaded:
        batch_df = pd.read_csv(uploaded)
    else:
        batch_df = None

    if batch_df is not None:
        st.write(f"Rows: {len(batch_df)} | Columns: {list(batch_df.columns)}")
        with st.expander("Preview data"):
            st.dataframe(batch_df.head(10))

        # ── Row & column selector ──────────────────────────────────────────────
        with st.expander("Select specific rows & columns to predict", expanded=False):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Row selection**")
                row_mode = st.radio("Row mode", ["All rows", "Row range", "Specific row numbers"],
                                    horizontal=True, key="row_mode")
                if row_mode == "Row range":
                    r_start, r_end = st.slider("Row range (0-based index)",
                                               0, len(batch_df)-1,
                                               (0, min(99, len(batch_df)-1)),
                                               key="row_range")
                    batch_df = batch_df.iloc[r_start:r_end+1].reset_index(drop=True)
                    st.caption(f"Using rows {r_start}–{r_end} ({len(batch_df)} rows)")
                elif row_mode == "Specific row numbers":
                    row_input = st.text_input("Enter row numbers (0-based, comma-separated)",
                                              value="0,1,2", key="row_nums")
                    try:
                        row_indices = [int(x.strip()) for x in row_input.split(",")
                                       if x.strip().isdigit()]
                        row_indices = [i for i in row_indices if 0 <= i < len(batch_df)]
                        batch_df = batch_df.iloc[row_indices].reset_index(drop=True)
                        st.caption(f"Using {len(batch_df)} selected rows")
                    except Exception:
                        st.warning("Invalid row numbers — using all rows")
            with rc2:
                st.markdown("**Column selection** (input columns to show in output)")
                all_input_cols = [c for c in template_cols if c in batch_df.columns]
                extra_cols = [c for c in batch_df.columns if c not in template_cols]
                show_cols = st.multiselect("Input columns to include in output",
                                           options=all_input_cols + extra_cols,
                                           default=all_input_cols,
                                           key="show_cols")
                if not show_cols:
                    show_cols = all_input_cols
            st.info(f"Ready: {len(batch_df)} rows × {len(show_cols)} display columns")

        # ── Vectorized batch encode + predict ─────────────────────────────────
        def encode_batch_masked(df: pd.DataFrame, active_raw: list) -> pd.DataFrame:
            """Encode entire DataFrame at once, zero-out non-selected features."""
            enc = pd.DataFrame(index=df.index)
            for col in BASE_NUM:
                enc[col] = df[col].fillna(0) if col in active_raw else 0
            for col, vals in CAT_VALS.items():
                for val in vals:
                    if col in active_raw:
                        enc[f"{col}_{val}"] = (df[col] == val).astype(int) if col in df.columns else 0
                    else:
                        enc[f"{col}_{val}"] = 0
            enc["avg_roof_temperature_before"] = (
                df["avg_roof_temperature_before"] if "avg_roof_temperature_before" in active_raw and "avg_roof_temperature_before" in df.columns
                else np.nan
            )
            enc["avg_units_before"] = (
                df["avg_units_before"].fillna(0) if "avg_units_before" in active_raw and "avg_units_before" in df.columns
                else 0
            )
            return enc

        if st.button("Run Batch Predictions", use_container_width=True):
            progress = st.progress(0)
            try:
                enc_all = encode_batch_masked(batch_df, selected_raw)
                preds = {}
                tasks_list = list(models.keys())
                for idx, task in enumerate(tasks_list):
                    m = models[task]
                    feats = features.get(task, BASE)
                    X = enc_all.reindex(columns=feats, fill_value=0).fillna(0)
                    if task in ("high_rating", "referral"):
                        preds[task] = m.predict_proba(X)[:, 1]
                    else:
                        preds[task] = m.predict(X)
                    progress.progress((idx + 1) / len(tasks_list))

                out_cols = [c for c in show_cols if c in batch_df.columns] if show_cols else [c for c in template_cols if c in batch_df.columns]
                result_df = batch_df[out_cols].copy()
                result_df["pred_cost_inr"]         = preds["cost"].round().astype(int)
                result_df["pred_temp_drop_c"]       = preds["temp_drop"].round(2)
                result_df["pred_units_saved_kwh"]   = preds["units_saved"].round(1)
                result_df["pred_high_rating_prob"]  = preds["high_rating"].round(3)
                result_df["pred_referral_prob"]     = preds["referral"].round(3)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                result_df = pd.DataFrame()
            st.success(f"Done — {len(result_df)} rows predicted.")

            # Summary stats
            pred_cols = ["pred_cost_inr", "pred_temp_drop_c", "pred_units_saved_kwh",
                         "pred_high_rating_prob", "pred_referral_prob"]
            available = [c for c in pred_cols if c in result_df.columns]
            if available:
                st.subheader("Prediction Summary")
                st.dataframe(result_df[available].describe().round(2), use_container_width=True)

            st.subheader("Full Results")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "⬇ Download predictions CSV",
                data=result_df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

        # ── Feature importance comparison: full vs selected ────────────────────
        with st.expander("Feature importance — full vs selected features"):
            fi_full = fi_series.reset_index()
            fi_full.columns = ["feature", "importance"]
            fi_full["selected"] = fi_full["feature"].isin(selected_raw)
            fi_full = fi_full.sort_values("importance", ascending=False)

            import_col = fi_full.set_index("feature")["importance"]
            color_map  = {f: ("#2196F3" if f in selected_raw else "#BDBDBD")
                          for f in fi_full["feature"]}
            st.bar_chart(import_col)
            st.caption("Blue = selected  |  Grey = excluded (approximate; colour not applied in st.bar_chart)")
            st.dataframe(fi_full.rename(columns={"selected": "in_selection"}), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMEND
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Recommend":
    st.title("Best Method Recommendation")
    st.markdown("Scores every compatible method × quality combination and ranks them.")

    METHOD_COMPAT = {
        "concrete_flat":    ["basic_coating", "tiles", "tiles_with_coating", "sheet_with_coating", "sheet"],
        "tiles":            ["basic_coating", "tiles", "tiles_with_coating"],
        "concrete_slanted": ["basic_coating", "sheet_with_coating", "sheet"],
        "sheet":            ["basic_coating", "sheet_with_coating", "sheet"],
    }

    with st.form("rec_form"):
        c1, c2 = st.columns(2)
        with c1:
            location_type    = st.selectbox("Location type", LOCATION_TYPES)
            building_size    = st.number_input("Building size (sq ft)", 100, 10000, 1200)
            building_age     = st.number_input("Building age (years)", 0, 100, 10)
            num_ac_units     = st.number_input("Number of AC units", 0, 50, 3)
        with c2:
            roofing_type     = st.selectbox("Roof type", ROOFING_TYPES)
            avg_temp_before  = st.number_input("Avg roof temp before (°C)", 20.0, 85.0, 55.0, step=0.5)
            avg_units_before = st.number_input("Avg monthly electricity (kWh)", 0.0, 2000.0, 250.0)
            tariff           = st.number_input("Electricity tariff (₹/kWh)", 4.0, 12.0, 7.0, step=0.5)

        run = st.form_submit_button("Find Best Configuration", use_container_width=True)

    if run:
        base_row = {
            "location_type":             location_type,
            "building_size":             building_size,
            "building_age":              building_age,
            "num_ac_units":              num_ac_units,
            "roofing_type":              roofing_type,
            "avg_roof_temperature_before": avg_temp_before,
            "avg_units_before":          avg_units_before,
        }

        rows = []
        for method in METHOD_COMPAT.get(roofing_type, ROOFING_METHODS):
            for quality in QUALITY_LEVELS:
                row = {**base_row, "cool_roofing_method": method, "cool_roofing_quality": quality}
                p = predict_all(row, models, features)
                rows.append({
                    "Method":  method,
                    "Quality": quality,
                    "Cost (₹)":          round(p.get("cost", 0)),
                    "Temp Drop (°C)":    round(p.get("temp_drop", 0), 2),
                    "Units Saved (kWh)": round(p.get("units_saved", 0), 1),
                    "P(High Rating)":    round(p.get("high_rating", 0), 3),
                    "P(Referral)":       round(p.get("referral", 0), 3),
                    "_cost_raw":         p.get("cost", 0),
                    "_temp_raw":         p.get("temp_drop", 0),
                    "_units_raw":        p.get("units_saved", 0),
                    "_rating_raw":       p.get("high_rating", 0),
                    "_referral_raw":     p.get("referral", 0),
                })

        result = pd.DataFrame(rows)

        # Normalized weighted score (cost weight negative)
        def norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

        result["score"] = (
            -0.25 * norm(result["_cost_raw"]) +
             0.25 * norm(result["_temp_raw"]) +
             0.25 * norm(result["_units_raw"]) +
             0.15 * norm(result["_rating_raw"]) +
             0.10 * norm(result["_referral_raw"])
        )
        result = result.sort_values("score", ascending=False).reset_index(drop=True)

        best = result.iloc[0]
        st.subheader("Top Recommendation")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Method",  best["Method"])
        b2.metric("Quality", best["Quality"])
        b3.metric("Cost",    f"₹{best['Cost (₹)']:,}")
        b4.metric("Score",   f"{best['score']:.3f}")

        fin_monthly = best["Units Saved (kWh)"] * tariff
        fin_annual  = fin_monthly * 12
        payback     = best["Cost (₹)"] / fin_annual if fin_annual > 0 else None
        f1, f2, f3  = st.columns(3)
        f1.metric("Monthly Savings", f"₹{fin_monthly:,.0f}")
        f2.metric("Annual Savings",  f"₹{fin_annual:,.0f}")
        f3.metric("Payback Period",  f"{payback:.1f} yrs" if payback else "N/A")

        st.subheader("All Options Ranked")
        display_cols = ["Method", "Quality", "Cost (₹)", "Temp Drop (°C)", "Units Saved (kWh)",
                        "P(High Rating)", "P(Referral)", "score"]
        st.dataframe(result[display_cols], use_container_width=True)
