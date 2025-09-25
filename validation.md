# Validation & Expected Ranges

This file lists key checks to confirm the pipeline reproduced Chapter 4.

## Datasets
- `data/processed/merged_input_dataset.parquet` must exist with columns:
  date, market, commodity, price_per_kg, price_lag_1, rolling_mean_3, month_sin, month_cos, perishability, cpi, exchange_rate, rainfall, store_id, sku, qty, unit_price, cost_price, expiry_date

## ML model expectations (approx)
- XGBoost R2 (perishables): 0.70 – 0.90
- RMSE should be lower than Linear Regression and Random Forest improvements visible
- SHAP shows price_lag_1 and rolling_mean_3 are top features

## RL expectations (illustrative)
- DQN waste reduction vs baseline: 12% – 22% absolute reduction (depends on parameters)
- Profit improvement: 8% – 15%
- Paired t-test p-value for waste reduction < 0.05

## Reproducibility tips
- Ensure `config.yaml` matches your local environment
- Use seeds (config 'seed' or default 42)
- If DQN unstable, increase episodes and lower learning rate

