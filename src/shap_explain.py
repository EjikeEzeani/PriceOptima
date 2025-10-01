import os
from pathlib import Path
import yaml
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Optional

def run_shap(config_path: str = "config.yaml") -> Optional[str]:
    """
    Run SHAP explainability for XGBoost model. Returns path to SHAP summary plot if successful, else None.
    """
    # Try to import xgboost
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except Exception:
        XGB_AVAILABLE = False
    CONFIG = Path(config_path)
    if not CONFIG.exists():
        return None
    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["data"]["processed_dir"])
    test_path = proc / "features_test.parquet"
    if not test_path.exists():
        return None
    test = pd.read_parquet(test_path)
    feat_cols = ['price_lag_1','price_lag_3','rolling_mean_3','month_sin','month_cos','perishability']
    missing_feats = [c for c in feat_cols if c not in test.columns]
    if missing_feats:
        return None
    test = test.dropna(subset=feat_cols).reset_index(drop=True)
    if test.empty:
        return None
    X_sample = test[feat_cols].sample(min(1000, len(test)), random_state=42)
    model_path = Path("models/xgb_model.json")
    if not model_path.exists() or not XGB_AVAILABLE:
        return None
    try:
        bst = xgb.Booster()
        bst.load_model(str(model_path))
    except Exception:
        return None
    Path("figures").mkdir(parents=True, exist_ok=True)
    shap_plot_path = "figures/shap_summary.png"
    try:
        explainer = shap.Explainer(bst, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        return shap_plot_path
    except Exception:
        return None



