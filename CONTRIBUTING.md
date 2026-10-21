# Contributing

## Setup

```bash
git clone https://github.com/sxrxvxnn/Cool-Roofing-Hackathon.git
cd Cool-Roofing-Hackathon
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — 5 pages |
| `main.py` | FastAPI server |
| `cool_roofing.ipynb` | EDA + model training notebook |
| `models/` | Trained `.joblib` model files |
| `cool_roofing_data.csv` | Dataset (2,500 rows) |

## Retraining Models

Open `cool_roofing.ipynb` and run all cells. New `.joblib` files are written to `models/`. The app loads them at startup automatically.

## Adding a New Prediction Target

1. Add training code in `cool_roofing.ipynb`
2. Save model: `joblib.dump(model, 'models/model_new_target.joblib')`
3. Add feature list: `joblib.dump(features, 'models/features_new_target.joblib')`
4. Load and expose in `app.py`

## Pull Requests

- Keep PRs focused — one change per PR
- Run the CI check locally before pushing:
  ```bash
  python -c "import joblib; [joblib.load(f'models/model_{m}.joblib') for m in ['cost','temp_drop','units_saved','high_rating','referral']]; print('All models OK')"
  ```
- Update `README.md` if metrics or behaviour change

## Issues

Open an issue for bugs, feature requests, or questions about the ML pipeline.
