# notebooks/03_eda.py
# %%
"""
Notebook 03: Exploratory Data Analysis (EDA)
- Descriptive stats
- Time-series plots, seasonal decomposition
- Correlations and ADF tests
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import yaml
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# === Load config ===
with open("config.yaml","r") as f:
    cfg = yaml.safe_load(f)

proc = Path(cfg["data"]["processed_dir"])

# ✅ Prefer processed features file if available
processed_file = proc / "features_train.parquet"
if processed_file.exists():
    df = pd.read_parquet(processed_file)
    print(f"Loaded {processed_file} with shape {df.shape}")
else:
    df = pd.read_parquet(proc / cfg["data"]["merged_filename"])
    print(f"Loaded merged file with shape {df.shape}")

# === Column fallback ===
target_col = "price_per_kg_wins" if "price_per_kg_wins" in df.columns else "price_per_kg"
print(f"Using target column: {target_col}")

# %%
# Descriptive stats for key commodities
keys = ['Tomatoes','Onions','Rice','Maize','Beans']
df_k = df[df['sku_description'].isin(keys)]
desc = df_k.groupby('sku_description')[target_col].describe().transpose()
print(desc)
desc.to_csv(proc / "eda_descriptive.csv")

# %%
# Time-series plot: Lagos Tomatoes
plt.figure(figsize=(10,5))
series = (
    df[(df.market=='Lagos') & (df.sku_description=='Tomatoes')]
    .groupby('date')[target_col].mean()
)
series.plot(title="Lagos Tomatoes Price (monthly avg)")
plt.ylabel("₦/kg")
plt.tight_layout()
plt.savefig("figures/tomatoes_lagos_timeseries.png")

# %%
# Seasonal decomposition (national avg)
series_all = (
    df[df['sku_description']=='Tomatoes']
    .groupby('date')[target_col].mean()
    .asfreq('MS')
    .ffill()   # ✅ modern replacement
)

if len(series_all) >= 24:   # ✅ only run if at least 24 observations
    res = seasonal_decompose(series_all, model='multiplicative', period=12)
    res.plot()
    plt.suptitle("Seasonal decomposition - Tomatoes (national avg)")
    plt.tight_layout()
    plt.savefig("figures/tomatoes_seasonal.png")
    print("Seasonal decomposition completed")
else:
    print(f"⚠️ Skipping seasonal decomposition: only {len(series_all)} observations (<24 required)")

# %%
# Correlation pivot by market for Tomatoes
pivot = df[df['sku_description']=='Tomatoes'].pivot_table(
    values=target_col, index='date', columns='market'
)
corr = pivot.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation between markets (Tomatoes)")
plt.tight_layout()
plt.savefig("figures/tomatoes_market_corr.png")
corr.to_csv(proc / "tomatoes_market_corr.csv")

# %%
# Stationarity test (ADF)
series_clean = series_all.dropna()
if len(series_clean) > 0:
    adf_res = adfuller(series_clean)
    print("ADF stat:", adf_res[0], "p-value:", adf_res[1])
    with open(proc / "adf_tomatoes.txt","w") as f:
        f.write(str(adf_res))
else:
    print("⚠️ Skipping ADF test: no valid data")

print("\nEDA completed successfully ✅")


