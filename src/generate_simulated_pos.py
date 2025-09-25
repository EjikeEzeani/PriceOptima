# src/generate_simulated_pos.py
"""
Generate or load WFP-like price data and synthesize simulated POS transactions.
Saves:
- data/processed/harmonized_prices.parquet
- data/processed/simulated_pos.parquet
- data/processed/merged_input_dataset.parquet
"""
import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import uuid

# Set deterministic seed default
def set_seeds(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def synthesize_wfp(config):
    # If real WFP file exists, load; else synthesize
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    wfp_path = raw_dir / "wfp_food_prices.csv"
    if wfp_path.exists():
        df = pd.read_csv(wfp_path, parse_dates=["date"])
        print("Loaded real WFP file.")
        return df
    # synthesize monthly price series for sample commodities and markets
    months = config["simulation"]["months"]
    start = pd.to_datetime(config["simulation"]["start_date"])
    dates = pd.date_range(start=start, periods=months, freq="MS")
    markets = config["simulation"]["markets"]
    commodities = ["Tomatoes","Onions","Rice","Beans","Maize","Fish"]
    rows = []
    base_prices = config["base_prices"]
    for market in markets:
        for commodity in commodities:
            base = base_prices.get(market, 200) * (1 + 0.2 * np.random.rand())
            season_amp = 0.25 if commodity in ["Tomatoes","Onions","Fish"] else 0.08
            for i, d in enumerate(dates):
                seasonal = 1 + season_amp * np.sin(2*np.pi*(d.month)/12 + 0.5*np.random.randn())
                price = max(20, base * seasonal * (1 + 0.02*np.random.randn()))
                rows.append({
                    "date": d,
                    "market": market,
                    "commodity": commodity,
                    "unit": "kg",
                    "price": round(price,2)
                })
    df = pd.DataFrame(rows)
    df.to_csv(wfp_path, index=False)
    return df

def harmonize_prices(df):
    df = df.copy()
    unit_map = {'kg':1.0, '100g':0.01, 'g':0.001, 'ltr':1.0, 'litre':1.0}
    df['unit_factor'] = df['unit'].map(unit_map).fillna(1.0)
    df['price_per_kg'] = df['price'] / df['unit_factor']
    return df

def generate_pos(df_prices, config):
    # Create sample stores per market and synthesize transactions
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    markets = config["simulation"]["markets"]
    stores = []
    for m in markets:
        for s in range(5):  # 5 stores per market (configurable)
            stores.append({"store_id": f"{m[:3].upper()}_S{s+1}", "market": m,
                           "refrigeration": np.random.choice([True, False], p=[0.6,0.4])})
    stores_df = pd.DataFrame(stores)
    # For each date, each store, sample transactions derived from price and elasticity
    elasticities = config["elasticities"]
    rows = []
    for _, row in df_prices.iterrows():
        d = row['date']
        market = row['market']
        commodity = row['commodity']
        price = row['price_per_kg']
        base_forecast = 30 if commodity in ["Tomatoes","Onions"] else 10
        elasticity = elasticities.get(commodity, 0.8)
        # demand generator per-day approximated for each store in market
        stores_in_market = stores_df[stores_df.market==market]
        for _, store in stores_in_market.iterrows():
            # Simulate 3–6 transactions per month for each SKU per store (sparse)
            n_tx = max(1, int(abs(np.random.poisson(3))))
            for _ in range(n_tx):
                unit_price = round(price * (1 + np.random.uniform(-0.2,0.1)),2)
                qty = int(max(1, np.random.poisson(lam=max(1, base_forecast * (unit_price/price)**(-elasticity)/10))))
                cost = round(unit_price * (0.55 + 0.2*np.random.rand()),2)
                txn_time = d + pd.Timedelta(days=np.random.randint(0,28), hours=np.random.randint(0,24))
                expiry_days = np.random.choice([3,5,7,14,30], p=[0.25,0.25,0.2,0.2,0.1])
                expiry_date = txn_time + pd.Timedelta(days=expiry_days)
                waste_units = int(max(0, np.random.binomial(qty, p=0.05*(1 if expiry_days<=5 else 0.02))))
                rows.append({
                    "transaction_id": str(uuid.uuid4()),
                    "date_time": txn_time,
                    "store_id": store['store_id'],
                    "market": market,
                    "sku": commodity[:6].upper() + "_SKU",
                    "sku_description": commodity,
                    "unit_price": unit_price,
                    "qty": qty,
                    "cost_price": cost,
                    "expiry_date": expiry_date.date(),
                    "inventory_open": max(0, int(qty*1.5)),
                    "inventory_close": max(0, int(qty*0.5)),
                    "waste_units": waste_units
                })
    pos_df = pd.DataFrame(rows)
    pos_path = processed_dir/"simulated_pos.parquet"
    pos_df.to_parquet(pos_path, index=False)
    return pos_df

def merge_and_save(price_df, pos_df, config):
    processed_dir = Path(config["data"]["processed_dir"])
    # Create harmonized price time series with lags and rolling features
    price_df = price_df.sort_values(['market','commodity','date']).copy()
    price_df = harmonize_prices(price_df)
    price_df = price_df.set_index('date')
    # Resample monthly per market commodity
    price_monthly = price_df.groupby(['market','commodity'])['price_per_kg'].resample('MS').mean().reset_index()
    # feature engineering
    price_monthly['month'] = price_monthly['date'].dt.month
    price_monthly['month_sin'] = np.sin(2*np.pi*price_monthly['month']/12)
    price_monthly['month_cos'] = np.cos(2*np.pi*price_monthly['month']/12)
    price_monthly = price_monthly.sort_values(['market','commodity','date'])
    for lag in [1,3,6]:
        price_monthly[f'price_lag_{lag}'] = price_monthly.groupby(['market','commodity'])['price_per_kg'].shift(lag)
    price_monthly['rolling_mean_3'] = price_monthly.groupby(['market','commodity'])['price_per_kg'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    perish_map = {'Tomatoes':1.0, 'Onions':0.9, 'Fish':1.1, 'Rice':0.2, 'Beans':0.3, 'Maize':0.4}
    price_monthly['perishability'] = price_monthly['commodity'].map(perish_map).fillna(0.5)
    # Save harmonized
    price_path = processed_dir/"harmonized_prices.parquet"
    price_monthly.to_parquet(price_path, index=False)
    # Merge a simplified version: for each pos row, attach nearest month price info
    pos_df['date'] = pd.to_datetime(pos_df['date_time']).dt.to_period('M').dt.to_timestamp()
    merged = pos_df.merge(price_monthly, left_on=['market','sku_description','date'], right_on=['market','commodity','date'], how='left')
    merged_path = processed_dir/Path(config["data"]["merged_filename"])
    merged.to_parquet(merged_path, index=False)
    print("Saved merged:", merged_path)
    return merged

def main(args):
    set_seeds(args.seed)
    config = load_config()
    price_df = synthesize_wfp(config)
    price_df = harmonize_prices(price_df)
    pos_df = generate_pos(price_df, config)
    merged = merge_and_save(price_df, pos_df, config)
    # Save small sample
    sample_path = Path(config["data"]["processed_dir"])/"sample_rows.csv"
    merged.head(5).to_csv(sample_path, index=False)
    print("Sample saved to", sample_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
