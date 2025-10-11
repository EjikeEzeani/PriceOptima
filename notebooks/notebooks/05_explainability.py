# notebooks/05_explainability.py
# %%
"""
Notebook 05: Explainability with SHAP for XGBoost
- Loads trained XGBoost model
- Computes SHAP values for test set features
- Saves summary plot
- With robust error handling
"""
# %%
import os, sys
from pathlib import Path
import yaml
import pandas as pd
import shap
import matplotlib.pyplot as plt

# optional xgboost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception as e:
    print("ERROR: XGBoost not available:", e)
    XGB_AVAILABLE = False

# === Helpers ===
def ensure_dir(pathlike):
    Path(pathlike).mkdir(parents=True, exist_ok=True)

# === Load config ===
CONFIG = Path("config.yaml")
if not CONFIG.exists():
    print("ERROR: config.yaml not found. Aborting.")
    sys.exit(1)

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

proc = Path(cfg["data"]["processed_dir"])
test_path = proc / "features_test.parquet"

if not test_path.exists():
    print(f"ERROR: Test parquet not found at {test_path}")
    sys.exit(1)

# === Load test data ===
test = pd.read_parquet(test_path)
print("Loaded test:", test.shape)

feat_cols = ['price_lag_1','price_lag_3','rolling_mean_3',
             'month_sin','month_cos','perishability']

missing_feats = [c for c in feat_cols if c not in test.columns]
if missing_feats:
    print(f"ERROR: Missing required features: {missing_feats}")
    sys.exit(1)

test = test.dropna(subset=feat_cols).reset_index(drop=True)
if test.empty:
    print("ERROR: No valid rows left in test set after dropping NA.")
    sys.exit(1)

X_sample = test[feat_cols].sample(min(1000, len(test)), random_state=42)
print("Sampled X_sample:", X_sample.shape)

# === Load trained model ===
ensure_dir("models")
model_path = Path("models/xgb_model.json")
if not model_path.exists():
    print(f"ERROR: Model file not found at {model_path}")
    sys.exit(1)

if not XGB_AVAILABLE:
    print("ERROR: XGBoost not installed, cannot run explainability.")
    sys.exit(1)

try:
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    print("Loaded XGBoost model.")
except Exception as e:
    print("ERROR: Could not load XGBoost model:", e)
    sys.exit(1)

# === SHAP explainability ===
ensure_dir("figures")
try:
    explainer = shap.Explainer(bst, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("figures/shap_summary.png", bbox_inches="tight")
    print("✅ Saved SHAP summary to figures/shap_summary.png")
except Exception as e:
    print("ERROR: SHAP explainability failed:", e)
    sys.exit(1)

