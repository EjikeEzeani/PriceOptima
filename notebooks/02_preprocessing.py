#!/usr/bin/env python3
"""
Notebook 02: Preprocessing (Clean Version with Logging)

Steps:
1. Load merged_input_dataset.parquet
2. Convert date and sort
3. Impute missing values (forward/backward fill)
4. Winsorize outliers
5. Create lag and rolling features
6. Add seasonal features (month, sin/cos)
7. Train/test split (last 12 months for test)
8. Save as Parquet
9. Log NaN checks after key steps
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

# === Utility: check missing values ===
def log_missing(df: pd.DataFrame, label: str, cols=None):
    """Log missing values in specified cols or all columns."""
    if cols is None:
        cols = df.columns
    na_counts = df[cols].isna().sum()
    total_na = int(na_counts.sum())
    if total_na == 0:
        print(f"[{label}] ✅ No missing values in {len(cols)} columns.")
    else:
        print(f"[{label}] ⚠️ Found {total_na} missing values:")
        print(na_counts[na_counts > 0].to_string())
    return total_na

# === Load config ===
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    print("ERROR: config.yaml not found.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

proc = Path(cfg["data"]["processed_dir"])
merged_path = proc / cfg["data"]["merged_filename"]

if not merged_path.exists():
    print(f"ERROR: merged parquet not found at {merged_path}")
    sys.exit(1)

df = pd.read_parquet(merged_path)
print("Loaded merged:", df.shape)

# === Date conversion ===
if "date" not in df.columns:
    print("ERROR: No 'date' column found in dataset.")
    sys.exit(1)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
log_missing(df, "After date conversion", ["date"])

# === Sort ===
df = df.sort_values(["market", "sku_description", "date"])

# === Imputation for price_per_kg ===
if "price_per_kg" not in df.columns:
    print("ERROR: Column 'price_per_kg' missing.")
    sys.exit(1)

before_na = df["price_per_kg"].isna().sum()
df["price_per_kg"] = df["price_per_kg"].ffill().bfill()
after_na = df["price_per_kg"].isna().sum()
print(f"Imputation: price_per_kg missing before={before_na}, after={after_na}")
log_missing(df, "After imputation", ["price_per_kg"])

# === Winsorization function ===
def winsorize(s: pd.Series, qlow=0.01, qhigh=0.99):
    low = s.quantile(qlow)
    high = s.quantile(qhigh)
    return s.clip(low, high)

df["price_per_kg_wins"] = (
    df.groupby(["market", "sku_description"])["price_per_kg"]
    .transform(winsorize)
)
log_missing(df, "After winsorization", ["price_per_kg_wins"])

# === Lag features ===
df = df.sort_values(["market", "sku_description", "date"])
for lag in [1, 3, 6]:
    colname = f"price_lag_{lag}"
    df[colname] = (
        df.groupby(["market", "sku_description"])["price_per_kg_wins"].shift(lag)
    )
    log_missing(df, f"After lag {lag}", [colname])

# === Rolling mean ===
df["rolling_mean_3"] = (
    df.groupby(["market", "sku_description"])["price_per_kg_wins"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
)
log_missing(df, "After rolling mean", ["rolling_mean_3"])

# === Seasonal features ===
df["month"] = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
log_missing(df, "After seasonal features", ["month", "month_sin", "month_cos"])

# === Train/Test Split ===
split_date = df["date"].max() - pd.DateOffset(months=12)
train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

train_path = proc / "features_train.parquet"
test_path = proc / "features_test.parquet"

train.to_parquet(train_path, index=False)
test.to_parquet(test_path, index=False)

print(f"\nTrain set: {train.shape}, Test set: {test.shape}")
print("Saved train/test:", train_path, test_path)
print("\nPreprocessing completed successfully ✅")

