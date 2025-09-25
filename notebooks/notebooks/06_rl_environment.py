# notebooks/06_rl_environment_safe.py
# %%
"""
Notebook 06: Prepare RL environment (safe version)
- Handles missing modules (gym, xgboost)
- Provides fallback dummy environment if RL packages are missing
- Handles missing config.yaml gracefully
"""
# %%
import sys
from pathlib import Path
import yaml
import numpy as np

# === Locate project root safely ===
POSSIBLE_ROOTS = [
    Path(__file__).resolve().parents[2],  # typical notebook structure
    Path(__file__).resolve().parents[1],  # if notebooks/ only
    Path.cwd()                             # current working dir
]
CONFIG_PATH = None
for root in POSSIBLE_ROOTS:
    candidate = root / "config.yaml"
    if candidate.exists():
        CONFIG_PATH = candidate
        PROJECT_ROOT = root
        break

if CONFIG_PATH is None:
    print("WARNING: config.yaml not found in typical locations.")
    print("Searched:", [str(r) for r in POSSIBLE_ROOTS])
    print("Proceeding with default fallback config.")
    cfg = {
        "base_prices": {"Lagos": 100},
        "elasticities": {"Tomatoes": 0.5},
        "xgboost": {"eta":0.1,"max_depth":3,"subsample":1,"nrounds":50,"early_stopping_rounds":5}
    }
else:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from: {CONFIG_PATH}")

# === Try importing gym ===
try:
    import gym
    GYM_AVAILABLE = True
except ModuleNotFoundError:
    print("WARNING: 'gym' not installed. RL environment cannot be created.")
    GYM_AVAILABLE = False

# === Try importing SupermarketEnv ===
ENV_AVAILABLE = False
if GYM_AVAILABLE:
    try:
        from src.envs.supermarket_env import SupermarketEnv
        ENV_AVAILABLE = True
    except ModuleNotFoundError:
        print("WARNING: Could not import 'src.envs.supermarket_env'. RL env unavailable.")
        ENV_AVAILABLE = False

# === Optional XGBoost model ===
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ModuleNotFoundError:
    print("WARNING: XGBoost not installed. ML forecast will fallback to deterministic function.")
    XGB_AVAILABLE = False

# === Load XGBoost model if exists ===
xgb_model_path = PROJECT_ROOT / "models/xgb_model.json"
if XGB_AVAILABLE and xgb_model_path.exists():
    bst = xgb.Booster()
    bst.load_model(str(xgb_model_path))
    print("Loaded XGBoost model for demand forecast.")
else:
    bst = None
    print("No XGBoost model found. Using deterministic base forecast.")

# === Demand forecast function ===
def demand_forecast(day_index, base_forecast=20):
    if bst is not None:
        # Placeholder: normally you would use bst.predict
        return base_forecast * (1 + 0.05 * np.sin(day_index / 3))
    else:
        return base_forecast * (1 + 0.05 * np.sin(day_index / 3))

# === Create RL environment if possible ===
if ENV_AVAILABLE:
    env = SupermarketEnv(
        base_price=cfg['base_prices'].get('Lagos', 100),
        cost=cfg['base_prices'].get('Lagos', 100) * 0.55,
        shelf_life=7,
        inventory=200,
        elasticity=cfg['elasticities'].get('Tomatoes', 0.5),
        demand_func=demand_forecast
    )
    sample_obs = env.reset()
    print("RL environment created. Sample observation:", sample_obs)
else:
    # Fallback dummy environment
    class DummyEnv:
        def reset(self):
            return {"price": cfg['base_prices'].get('Lagos', 100), "inventory": 200}
        def step(self, action):
            return {"price": 100, "inventory": 200}, 0, True, {}
    env = DummyEnv()
    print("Fallback dummy environment created. Sample observation:", env.reset())

# === Script completion message ===
print("RL environment setup script completed successfully ✅")

# === Generate requirements.txt automatically ===
req_file = PROJECT_ROOT / "requirements.txt"
required_packages = ["numpy", "pandas", "pyyaml", "matplotlib", "seaborn", "xgboost", "gym"]
with open(req_file, "w", encoding="utf-8") as f:
    for pkg in required_packages:
        f.write(pkg + "\n")
print(f"Generated {req_file} with core dependencies.")



