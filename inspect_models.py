"""Run this to export all model info to readable JSON files in models/inspect/"""
import json, os, joblib, numpy as np

out_dir = os.path.join(os.path.dirname(__file__), "models", "inspect")
os.makedirs(out_dir, exist_ok=True)

tasks = ["cost", "temp_drop", "units_saved", "high_rating", "referral"]
model_dir = os.path.join(os.path.dirname(__file__), "models")

for task in tasks:
    m     = joblib.load(os.path.join(model_dir, f"model_{task}.joblib"))
    feats = joblib.load(os.path.join(model_dir, f"features_{task}.joblib"))

    fi = dict(sorted(
        zip(feats, m.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True
    ))

    info = {
        "task": task,
        "model_type": type(m).__name__,
        "n_estimators": m.n_estimators,
        "max_depth": m.max_depth,
        "n_features": m.n_features_in_,
        "features": feats,
        "feature_importances": fi,
    }
    if hasattr(m, "classes_"):
        info["classes"] = m.classes_.tolist()

    path = os.path.join(out_dir, f"{task}.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Written: {path}")

print("Done. Open models/inspect/*.json in VSCode.")
