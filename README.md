# Cool Roofing Cost & Impact Predictor
**Smart Infrastructure & Green Buildings**

A machine learning system that predicts **5 outcomes** of a cool roofing installation from **9 pre-installation inputs** — before a single tile is placed.

---

## What It Does

| Prediction | Model | Performance |
|---|---|---|
| Installation Cost (₹) | XGBoost | R² = 0.9843, MAE ₹69,426 |
| Temperature Drop (°C) | Random Forest | R² = 0.935, MAE 0.46°C |
| Monthly Units Saved (kWh) | Random Forest | R² = 0.944, MAE 5.41 kWh |
| High Rating (≥4 stars) | RF Classifier | 65.5% accuracy, AUC 0.728 |
| Customer Referral | RF Classifier | 62.0% accuracy, F1 0.505 |

**Live example:** 900 sqft urban building, tiles method, basic quality → predicted ₹3,10,742 vs actual ₹3,13,700 = **0.94% error**

---

## Dataset

- 2,500 building records (`cool_roofing_data.csv`)
- 9 input features: `building_size`, `building_age`, `num_ac_units`, `avg_roof_temp_before`, `avg_units_before`, `location_type`, `roofing_type`, `cool_roofing_method`, `quality`
- Split: 70% train (1,730) / 15% val (371) / 15% test (371), `random_state=42`
- After cleaning: 2,472 rows (removed 32 sensor-error rows with temp=120°C)

---

## Models

### XGBoost — Installation Cost
```
n_estimators=2000, max_depth=5, learning_rate=0.06
reg_lambda=0.5, subsample=0.9
```
Chosen because cost is right-skewed (₹55K–₹50.5L range) and non-linear. Outperformed Random Forest by 0.46 R² points.

### Random Forest — All Other Tasks
```
n_estimators=300, max_depth=12, class_weight='balanced' (classifiers)
```
Used for temperature drop, units saved, high rating, and referral prediction.

All models tuned with `RandomizedSearchCV` (40 iterations × 5-fold CV = 200 fits per model).

---

## App — 5 Pages

```bash
pip install -r requirements.txt
streamlit run app.py
```

| Page | Description |
|---|---|
| Home | Live KPIs, model status |
| Dashboard | 8 interactive Plotly charts, 4 filters |
| Predict | Single building → 5 predictions + payback period |
| Batch | Upload CSV → 2,500 rows in 0.23s |
| Recommender | Ranks all 15 method × quality combos by weighted score |

---

## File Structure

```
submission/
├── app.py                  # Streamlit application (5 pages)
├── main.py                 # Training pipeline
├── cool_roofing.ipynb      # EDA + model training notebook
├── cool_roofing_data.csv   # Dataset (2,500 rows)
├── models/
│   ├── model_cost.joblib         # XGBoost cost model
│   ├── model_temp_drop.joblib    # RF temperature model
│   ├── model_units_saved.joblib  # RF units model
│   ├── model_high_rating.joblib  # RF rating classifier
│   ├── model_referral.joblib     # RF referral classifier
│   └── *.joblib                  # Feature lists + encoders
├── README.md
├── QUICK_REFERENCE.md      # Slide-by-slide cheat sheet + glossary
├── README.pdf
├── PRESENTATION_NOTES.pdf  # Full speaker scripts for all 12 slides
└── QUICK_REFERENCE.pdf
```

---

## Key Numbers

| Metric | Value |
|---|---|
| Cost R² (test set) | 0.9843 |
| Cost MAE | ₹69,426 |
| Baseline MAE | ₹6,08,878 |
| MAE improvement | **8.8×** |
| Batch speed | 2,500 rows / 0.23s |
| Example prediction error | 0.94% |
| Average installation cost | ₹10.9L |
| Average temp drop | 8.1°C |
| Average monthly savings | 46 kWh (≈ ₹322 at ₹7/kWh) |
| Payback period (average) | ~2.6 years |

---

## Tech Stack

Python · XGBoost · Scikit-learn · Streamlit · Plotly · Pandas · NumPy · Joblib
