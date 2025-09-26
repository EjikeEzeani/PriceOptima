import os
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

def safe_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))

def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    rmse = safe_rmse(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)))) * 100)
    return dict(RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)

def ensure_dir(pathlike):
    Path(pathlike).mkdir(parents=True, exist_ok=True)

def run_ml(config_path: str = "config.yaml") -> Dict[str, Optional[dict]]:
    """
    Train supervised models (LR, RF, XGBoost if available), save models and metrics, return metrics dict.
    """
    CONFIG = Path(config_path)
    if not CONFIG.exists():
        raise FileNotFoundError("config.yaml not found.")
    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["data"]["processed_dir"])
    train_path = proc / "features_train.parquet"
    test_path  = proc / "features_test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Train/test parquet not found at {train_path} / {test_path}")
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)
    feat_cols = ['price_lag_1','price_lag_3','rolling_mean_3','month_sin','month_cos','perishability']
    target_col = 'price_per_kg_wins'
    for col in feat_cols + [target_col]:
        if col not in train.columns:
            raise ValueError(f"Missing column {col} in training data.")
    train = train.dropna(subset=feat_cols + [target_col]).reset_index(drop=True)
    test  = test.dropna(subset=feat_cols + [target_col]).reset_index(drop=True)
    if len(train) < 30:
        raise ValueError("Not enough training rows.")
    X_train = train[feat_cols]; y_train = train[target_col]
    X_test  = test[feat_cols];  y_test  = test[target_col]
    ensure_dir("models")
    ensure_dir("data/processed")
    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    y_lr = lr.predict(X_test)
    metrics_lr = eval_metrics(y_test, y_lr)
    joblib.dump(lr, "models/linear_regression.joblib")
    # Random Forest
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {'n_estimators': [100, 300], 'max_depth': [6, 12, None]}
    n_splits_req = 5
    max_splits_allowed = max(1, min(n_splits_req, len(X_train) - 1))
    if max_splits_allowed < 2:
        rf.fit(X_train, y_train)
        best_rf = rf
    else:
        try:
            tscv = TimeSeriesSplit(n_splits=max_splits_allowed)
            g = GridSearchCV(rf, param_grid, cv=tscv,
                             scoring="neg_mean_squared_error", n_jobs=-1)
            g.fit(X_train, y_train)
            best_rf = g.best_estimator_
        except Exception:
            rf.fit(X_train, y_train)
            best_rf = rf
    y_rf = best_rf.predict(X_test)
    metrics_rf = eval_metrics(y_test, y_rf)
    joblib.dump(best_rf, "models/rf_model.joblib")
    # XGBoost
    metrics_xgb = None
    if XGBOOST_AVAILABLE:
        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest  = xgb.DMatrix(X_test, label=y_test)
            params = {
                "objective": "reg:squarederror",
                "eta": float(cfg.get("xgboost", {}).get("eta", 0.1)),
                "max_depth": int(cfg.get("xgboost", {}).get("max_depth", 6)),
                "subsample": float(cfg.get("xgboost", {}).get("subsample", 1.0)),
            }
            num_round = int(cfg.get("xgboost", {}).get("nrounds", 100))
            early_stop = int(cfg.get("xgboost", {}).get("early_stopping_rounds", 10))
            bst = xgb.train(params, dtrain, num_boost_round=num_round,
                            evals=[(dtest, "test")],
                            early_stopping_rounds=early_stop,
                            verbose_eval=False)
            y_xgb = bst.predict(dtest)
            metrics_xgb = eval_metrics(y_test, y_xgb)
            bst.save_model("models/xgb_model.json")
        except Exception:
            metrics_xgb = None
    # Save comparison table
    comp_records = [
        ("LinearRegression", metrics_lr),
        ("RandomForest", metrics_rf),
    ]
    if metrics_xgb is not None:
        comp_records.append(("XGBoost", metrics_xgb))
    comp = pd.DataFrame({name: metrics for name, metrics in comp_records}).T
    comp.index.name = "model"
    comp_csv = "data/processed/model_comparison.csv"
    comp.to_csv(comp_csv)
    return {
        "metrics_lr": metrics_lr,
        "metrics_rf": metrics_rf,
        "metrics_xgb": metrics_xgb,
        "comparison_csv": comp_csv,
        "lr_model": "models/linear_regression.joblib",
        "rf_model": "models/rf_model.joblib",
        "xgb_model": "models/xgb_model.json" if metrics_xgb is not None else None
    }



