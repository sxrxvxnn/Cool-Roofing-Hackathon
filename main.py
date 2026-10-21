"""
Cool Roofing Predictor — FastAPI backend
Run: uvicorn main:app --reload
"""
import io
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Cool Roofing Predictor", version="1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ── Feature config ─────────────────────────────────────────────────────────────
CAT_VALS = {
    "location_type":        ["dense_urban", "intermediate", "rural", "urban"],
    "cool_roofing_quality": ["basic", "high", "intermediate"],
    "roofing_type":         ["concrete_flat", "concrete_slanted", "sheet", "tiles"],
    "cool_roofing_method":  ["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"],
}
BASE_NUM = ["building_size", "building_age", "num_ac_units"]
BASE_CAT = [f"{c}_{v}" for c, vals in CAT_VALS.items() for v in vals]
BASE     = BASE_NUM + BASE_CAT

LOCATION_TYPES  = CAT_VALS["location_type"]
QUALITY_LEVELS  = CAT_VALS["cool_roofing_quality"]
ROOFING_TYPES   = CAT_VALS["roofing_type"]
ROOFING_METHODS = CAT_VALS["cool_roofing_method"]

METHOD_COMPAT = {
    "concrete_flat":    ["basic_coating", "tiles", "tiles_with_coating", "sheet_with_coating", "sheet"],
    "tiles":            ["basic_coating", "tiles", "tiles_with_coating"],
    "concrete_slanted": ["basic_coating", "sheet_with_coating", "sheet"],
    "sheet":            ["basic_coating", "sheet_with_coating", "sheet"],
}

# ── Model loading ──────────────────────────────────────────────────────────────
_models: dict = {}
_features: dict = {}

def load_models():
    model_dir = os.path.join(BASE_DIR, "models")
    for task in ["cost", "temp_drop", "units_saved", "high_rating", "referral"]:
        mp = os.path.join(model_dir, f"model_{task}.joblib")
        fp = os.path.join(model_dir, f"features_{task}.joblib")
        if os.path.exists(mp):
            _models[task] = joblib.load(mp)
        if os.path.exists(fp):
            _features[task] = joblib.load(fp)

load_models()

# ── Training statistics (for imputation on arbitrary datasets) ─────────────────
_tstats: dict = {}
def load_training_stats():
    tp = os.path.join(BASE_DIR, "models", "training_stats.joblib")
    if os.path.exists(tp):
        _tstats.update(joblib.load(tp))

load_training_stats()


# ── Column auto-mapper ─────────────────────────────────────────────────────────
KNOWN_ALIASES = {
    "building_size":                 ["size", "area", "sqft", "sq_ft", "floor_area", "bldg_size"],
    "building_age":                  ["age", "year_built", "yrs", "years_old"],
    "num_ac_units":                  ["ac_units", "ac", "air_conditioners", "num_ac", "ac_count"],
    "location_type":                 ["location", "loc_type", "loc", "zone"],
    "roofing_type":                  ["roof_type", "roof", "existing_roof"],
    "cool_roofing_method":           ["method", "cool_method", "roofing_method", "installation_method"],
    "cool_roofing_quality":          ["quality", "grade", "tier", "cool_quality"],
    "avg_roof_temperature_before":   ["temp_before", "roof_temp", "temperature_before", "temp_pre"],
    "avg_units_before":              ["units_before", "kwh_before", "electricity_before", "baseline_kwh"],
}

def auto_map_columns(df_cols: list) -> dict:
    """Map arbitrary CSV columns to known feature names. Returns {known_name: csv_col}."""
    df_cols_lower = {c.lower().replace(" ", "_"): c for c in df_cols}
    mapping = {}
    for known, aliases in KNOWN_ALIASES.items():
        candidates = [known] + aliases
        for cand in candidates:
            if cand in df_cols_lower:
                mapping[known] = df_cols_lower[cand]
                break
    return mapping


def normalize_df(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Rename mapped columns, impute missing features from training medians,
    clamp categoricals to valid values. Returns a DataFrame ready for encode_batch.
    """
    stats = _tstats
    out = pd.DataFrame(index=df.index)

    # Numeric features
    for col in BASE_NUM + ["avg_roof_temperature_before", "avg_units_before"]:
        median = stats.get("numeric", {}).get(col, {}).get("median", 0)
        if col in col_map:
            out[col] = pd.to_numeric(df[col_map[col]], errors="coerce").fillna(median)
        else:
            out[col] = median  # fully missing → training median

    # Categorical features
    for col, valid in CAT_VALS.items():
        mode = stats.get("categorical", {}).get(col, {}).get("mode", valid[0])
        if col in col_map:
            raw = df[col_map[col]].astype(str).str.strip().str.lower()
            # map to valid value (case-insensitive, fallback to mode)
            valid_lower = {v.lower(): v for v in valid}
            out[col] = raw.map(valid_lower).fillna(mode)
        else:
            out[col] = mode  # fully missing → training mode

    return out


def encode_row(row: dict) -> pd.DataFrame:
    enc = {col: float(row.get(col, 0) or 0) for col in BASE_NUM}
    for col, vals in CAT_VALS.items():
        for val in vals:
            enc[f"{col}_{val}"] = 1.0 if row.get(col) == val else 0.0
    enc["avg_roof_temperature_before"] = float(row.get("avg_roof_temperature_before") or np.nan)
    enc["avg_units_before"]            = float(row.get("avg_units_before") or 0)
    return pd.DataFrame([enc])


def predict_row(row: dict) -> dict:
    enc = encode_row(row)
    results = {}
    for task, m in _models.items():
        feats = _features.get(task, BASE)
        X = enc.reindex(columns=feats, fill_value=0).fillna(0).astype(float)
        if task in ("high_rating", "referral"):
            results[task] = float(m.predict_proba(X)[0, 1])
        else:
            results[task] = float(m.predict(X)[0])
    return results


def encode_batch(df: pd.DataFrame) -> pd.DataFrame:
    enc = pd.DataFrame(index=df.index)
    for col in BASE_NUM:
        enc[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    for col, vals in CAT_VALS.items():
        for val in vals:
            enc[f"{col}_{val}"] = (df[col] == val).astype(float) if col in df.columns else 0.0
    enc["avg_roof_temperature_before"] = pd.to_numeric(
        df.get("avg_roof_temperature_before", np.nan), errors="coerce")
    enc["avg_units_before"] = pd.to_numeric(
        df.get("avg_units_before", 0), errors="coerce").fillna(0)
    return enc


def predict_batch(df: pd.DataFrame) -> dict:
    enc = encode_batch(df)
    preds = {}
    for task, m in _models.items():
        feats = _features.get(task, BASE)
        X = enc.reindex(columns=feats, fill_value=0).fillna(0).astype(float)
        if task in ("high_rating", "referral"):
            preds[task] = m.predict_proba(X)[:, 1].tolist()
        else:
            preds[task] = m.predict(X).tolist()
    return preds


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    location_type: str
    building_size: float
    building_age: float
    num_ac_units: float
    roofing_type: str
    cool_roofing_method: str
    cool_roofing_quality: str
    avg_roof_temperature_before: Optional[float] = None
    avg_units_before: float = 0.0


# ── HTML Pages ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    stats = {}
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "cool_roofing_data.csv"))
        df["temperature_drop"] = df["avg_roof_temperature_before"].replace(120, np.nan) - df["avg_roof_temperature_after"]
        df["units_saved"] = df["avg_units_before"] - df["avg_units_after"]
        stats = {
            "rows": len(df),
            "avg_cost": f"{df['cool_roofing_cost'].mean():,.0f}",
            "avg_temp_drop": f"{df['temperature_drop'].mean():.1f}",
            "avg_units_saved": f"{df['units_saved'].mean():.1f}",
            "high_rating_pct": f"{(df['customer_given_rating'] >= 4).mean():.1%}",
        }
    except Exception:
        pass
    return templates.TemplateResponse(request, "home.html", {
        "stats": stats,
        "models_loaded": list(_models.keys()),
    })


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request, "predict.html", {
        "location_types": LOCATION_TYPES,
        "quality_levels": QUALITY_LEVELS,
        "roofing_types": ROOFING_TYPES,
        "roofing_methods": ROOFING_METHODS,
    })


@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    return templates.TemplateResponse(request, "batch.html", {
        "location_types": LOCATION_TYPES,
        "quality_levels": QUALITY_LEVELS,
        "roofing_types": ROOFING_TYPES,
        "roofing_methods": ROOFING_METHODS,
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/recommend", response_class=HTMLResponse)
async def recommend_page(request: Request):
    return templates.TemplateResponse(request, "recommend.html", {
        "location_types": LOCATION_TYPES,
        "quality_levels": QUALITY_LEVELS,
        "roofing_types": ROOFING_TYPES,
        "roofing_methods": ROOFING_METHODS,
    })


# ── REST API endpoints ─────────────────────────────────────────────────────────
@app.post("/api/predict")
async def api_predict(req: PredictRequest, tariff: float = 7.0):
    if not _models:
        raise HTTPException(503, "Models not loaded")
    preds = predict_row(req.model_dump())
    annual_savings = preds.get("units_saved", 0) * tariff * 12
    payback = preds["cost"] / annual_savings if annual_savings > 0 else None
    return {
        "predictions": {
            "cost_inr":           round(preds["cost"]),
            "temp_drop_c":        round(preds["temp_drop"], 2),
            "units_saved_kwh":    round(preds["units_saved"], 1),
            "high_rating_prob":   round(preds["high_rating"], 3),
            "referral_prob":      round(preds["referral"], 3),
        },
        "financial": {
            "tariff_per_kwh":      tariff,
            "monthly_savings_inr": round(preds["units_saved"] * tariff),
            "annual_savings_inr":  round(annual_savings),
            "payback_years":       round(payback, 1) if payback else None,
        },
    }


@app.post("/api/batch")
async def api_batch(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    row_start: int = Form(0),
    row_end: int = Form(-1),
    selected_columns: str = Form(""),
):
    if not _models:
        raise HTTPException(503, "Models not loaded")

    if use_sample or file is None:
        data_path = os.path.join(BASE_DIR, "cool_roofing_data.csv")
        df = pd.read_csv(data_path)
    else:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

    # Row selection
    if row_end == -1:
        row_end = len(df) - 1
    df = df.iloc[row_start: row_end + 1].reset_index(drop=True)

    preds = predict_batch(df)

    result = df.copy()
    result["pred_cost_inr"]        = np.round(preds["cost"]).astype(int)
    result["pred_temp_drop_c"]     = np.round(preds["temp_drop"], 2)
    result["pred_units_saved_kwh"] = np.round(preds["units_saved"], 1)
    result["pred_high_rating_prob"]= np.round(preds["high_rating"], 3)
    result["pred_referral_prob"]   = np.round(preds["referral"], 3)

    # Column filter
    if selected_columns:
        keep = [c.strip() for c in selected_columns.split(",") if c.strip() in result.columns]
        pred_cols = ["pred_cost_inr", "pred_temp_drop_c", "pred_units_saved_kwh",
                     "pred_high_rating_prob", "pred_referral_prob"]
        result = result[[c for c in keep + pred_cols if c in result.columns]]

    return JSONResponse(content={
        "rows": len(result),
        "data": result.to_dict(orient="records"),
        "summary": {
            "avg_cost": float(np.mean(preds["cost"])),
            "avg_temp_drop": float(np.mean(preds["temp_drop"])),
            "avg_units_saved": float(np.mean(preds["units_saved"])),
            "avg_high_rating_prob": float(np.mean(preds["high_rating"])),
            "avg_referral_prob": float(np.mean(preds["referral"])),
        }
    })


@app.post("/api/batch/flexible")
async def api_batch_flexible(
    file: UploadFile = File(...),
    row_start: int = Form(0),
    row_end: int = Form(-1),
):
    """
    Accept ANY CSV. Auto-maps columns to known feature names.
    Missing features filled from training medians/modes.
    Returns predictions + column mapping report.
    """
    if not _models:
        raise HTTPException(503, "Models not loaded")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if row_end == -1:
        row_end = len(df) - 1
    df = df.iloc[row_start: row_end + 1].reset_index(drop=True)

    col_map = auto_map_columns(df.columns.tolist())
    norm_df = normalize_df(df, col_map)

    preds = predict_batch(norm_df)

    result = df.copy()
    result["pred_cost_inr"]         = np.round(preds["cost"]).astype(int)
    result["pred_temp_drop_c"]      = np.round(preds["temp_drop"], 2)
    result["pred_units_saved_kwh"]  = np.round(preds["units_saved"], 1)
    result["pred_high_rating_prob"] = np.round(preds["high_rating"], 3)
    result["pred_referral_prob"]    = np.round(preds["referral"], 3)

    all_features = BASE_NUM + list(CAT_VALS.keys()) + ["avg_roof_temperature_before", "avg_units_before"]
    missing = [f for f in all_features if f not in col_map]
    mapped  = {k: v for k, v in col_map.items()}

    return JSONResponse(content={
        "rows": len(result),
        "column_mapping": {
            "mapped": mapped,
            "missing_filled_with_training_median": missing,
            "coverage_pct": round(len(mapped) / len(all_features) * 100, 1),
        },
        "data": result.to_dict(orient="records"),
        "summary": {
            "avg_cost":             float(np.mean(preds["cost"])),
            "avg_temp_drop":        float(np.mean(preds["temp_drop"])),
            "avg_units_saved":      float(np.mean(preds["units_saved"])),
            "avg_high_rating_prob": float(np.mean(preds["high_rating"])),
            "avg_referral_prob":    float(np.mean(preds["referral"])),
        },
    })


@app.get("/api/batch/csv")
async def api_batch_csv(row_start: int = 0, row_end: int = 199):
    data_path = os.path.join(BASE_DIR, "cool_roofing_data.csv")
    df = pd.read_csv(data_path).iloc[row_start: row_end + 1].reset_index(drop=True)
    preds = predict_batch(df)
    df["pred_cost_inr"]        = np.round(preds["cost"]).astype(int)
    df["pred_temp_drop_c"]     = np.round(preds["temp_drop"], 2)
    df["pred_units_saved_kwh"] = np.round(preds["units_saved"], 1)
    df["pred_high_rating_prob"]= np.round(preds["high_rating"], 3)
    df["pred_referral_prob"]   = np.round(preds["referral"], 3)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.post("/api/recommend")
async def api_recommend(req: PredictRequest, tariff: float = 7.0):
    if not _models:
        raise HTTPException(503, "Models not loaded")
    base = req.model_dump()
    methods = METHOD_COMPAT.get(req.roofing_type, ROOFING_METHODS)
    rows = []
    for method in methods:
        for quality in QUALITY_LEVELS:
            r = {**base, "cool_roofing_method": method, "cool_roofing_quality": quality}
            p = predict_row(r)
            rows.append({**r, **p})

    result_df = pd.DataFrame(rows)

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

    result_df["score"] = (
        -0.25 * norm(result_df["cost"])
        + 0.25 * norm(result_df["temp_drop"])
        + 0.25 * norm(result_df["units_saved"])
        + 0.15 * norm(result_df["high_rating"])
        + 0.10 * norm(result_df["referral"])
    )
    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)

    out = []
    for _, row in result_df.iterrows():
        annual = row["units_saved"] * tariff * 12
        out.append({
            "method": row["cool_roofing_method"],
            "quality": row["cool_roofing_quality"],
            "cost_inr": round(row["cost"]),
            "temp_drop_c": round(row["temp_drop"], 2),
            "units_saved_kwh": round(row["units_saved"], 1),
            "high_rating_prob": round(row["high_rating"], 3),
            "referral_prob": round(row["referral"], 3),
            "annual_savings_inr": round(annual),
            "score": round(row["score"], 4),
        })
    return {"recommendations": out}


@app.get("/api/dataset/stats")
async def dataset_stats():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "cool_roofing_data.csv"))
        df["temperature_drop"] = df["avg_roof_temperature_before"].replace(120, np.nan) - df["avg_roof_temperature_after"]
        df["units_saved"] = df["avg_units_before"] - df["avg_units_after"]
        df["high_rating"] = (df["customer_given_rating"] >= 4).astype(int)
        return {
            "rows": len(df),
            "cost": {"mean": df["cool_roofing_cost"].mean(), "std": df["cool_roofing_cost"].std(),
                     "min": df["cool_roofing_cost"].min(), "max": df["cool_roofing_cost"].max()},
            "temp_drop": {"mean": df["temperature_drop"].mean(), "std": df["temperature_drop"].std()},
            "units_saved": {"mean": df["units_saved"].mean(), "std": df["units_saved"].std()},
            "high_rating_rate": df["high_rating"].mean(),
            "referral_rate": df["customer_referral"].mean(),
            "by_quality": df.groupby("cool_roofing_quality")["cool_roofing_cost"].mean().to_dict(),
            "by_method": df.groupby("cool_roofing_method")["cool_roofing_cost"].mean().to_dict(),
            "by_location": df.groupby("location_type")["cool_roofing_cost"].mean().to_dict(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/dataset/raw")
async def dataset_raw(limit: int = 100, offset: int = 0):
    df = pd.read_csv(os.path.join(BASE_DIR, "cool_roofing_data.csv"))
    slice_df = df.iloc[offset: offset + limit]
    return {"total": len(df), "offset": offset, "limit": limit,
            "data": slice_df.to_dict(orient="records")}
