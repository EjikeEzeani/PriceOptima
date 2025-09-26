import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import yaml
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from typing import Dict

def run_eda(processed_file: str, config_path: str = "config.yaml") -> Dict[str, str]:
    """
    Run EDA on the given processed file. Save outputs and return their paths.
    Returns dict with keys: desc_csv, timeseries_png, seasonal_png, corr_csv, corr_png, adf_txt
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    proc = Path(cfg["data"]["processed_dir"])
    df = pd.read_parquet(processed_file)
    target_col = "price_per_kg_wins" if "price_per_kg_wins" in df.columns else "price_per_kg"
    keys = ['Tomatoes','Onions','Rice','Maize','Beans']
    df_k = df[df['sku_description'].isin(keys)]
    desc = df_k.groupby('sku_description')[target_col].describe().transpose()
    desc_csv = proc / "eda_descriptive.csv"
    desc.to_csv(desc_csv)
    # Time-series plot: Lagos Tomatoes
    timeseries_png = Path("figures/tomatoes_lagos_timeseries.png")
    timeseries_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,5))
    series = (
        df[(df.market=='Lagos') & (df.sku_description=='Tomatoes')]
        .groupby('date')[target_col].mean()
    )
    series.plot(title="Lagos Tomatoes Price (monthly avg)")
    plt.ylabel("₦/kg")
    plt.tight_layout()
    plt.savefig(timeseries_png)
    plt.close()
    # Seasonal decomposition (national avg)
    series_all = (
        df[df['sku_description']=='Tomatoes']
        .groupby('date')[target_col].mean()
        .asfreq('MS')
        .ffill()
    )
    seasonal_png = Path("figures/tomatoes_seasonal.png")
    if len(series_all) >= 24:
        res = seasonal_decompose(series_all, model='multiplicative', period=12)
        res.plot()
        plt.suptitle("Seasonal decomposition - Tomatoes (national avg)")
        plt.tight_layout()
        plt.savefig(seasonal_png)
        plt.close()
    else:
        seasonal_png = None
    # Correlation pivot by market for Tomatoes
    pivot = df[df['sku_description']=='Tomatoes'].pivot_table(
        values=target_col, index='date', columns='market'
    )
    corr = pivot.corr()
    corr_png = Path("figures/tomatoes_market_corr.png")
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation between markets (Tomatoes)")
    plt.tight_layout()
    plt.savefig(corr_png)
    plt.close()
    corr_csv = proc / "tomatoes_market_corr.csv"
    corr.to_csv(corr_csv)
    # Stationarity test (ADF)
    adf_txt = proc / "adf_tomatoes.txt"
    series_clean = series_all.dropna()
    if len(series_clean) > 0:
        adf_res = adfuller(series_clean)
        with open(adf_txt, "w") as f:
            f.write(str(adf_res))
    else:
        adf_txt = None
    return {
        "desc_csv": str(desc_csv),
        "timeseries_png": str(timeseries_png),
        "seasonal_png": str(seasonal_png) if seasonal_png else None,
        "corr_csv": str(corr_csv),
        "corr_png": str(corr_png),
        "adf_txt": str(adf_txt) if adf_txt else None
    }
