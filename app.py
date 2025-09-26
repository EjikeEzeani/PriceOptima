"""
🚀 Docker Entry Point for MSc Project
- Runs full ML + RL pipeline automatically
- Launches Streamlit dashboard on port 8501
"""

import os
import yaml
import gymnasium as gym
import tensorflow as tf
from src.envs.supermarket_env import SupermarketEnv
from src.run_all_menu import evaluate  # reuse your functions
import subprocess

# ------------------------------
# Load Config
# ------------------------------
if os.path.exists("config.yaml"):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print("✅ Loaded config.yaml")
else:
    print("⚠️ config.yaml not found. Using defaults.")
    cfg = {}

# ------------------------------
# Initialize Env
# ------------------------------
try:
    env = SupermarketEnv(
        base_price=cfg.get("base_prices", {}).get("Lagos", 100),
        cost=cfg.get("base_prices", {}).get("Lagos", 55),
        shelf_life=7,
        inventory=200,
        elasticity=cfg.get("elasticities", {}).get("Tomatoes", 0.5),
        demand_func=lambda d: 20*(1+0.05*__import__("math").sin(d/3))
    )
    print("✅ Loaded real SupermarketEnv")
except Exception as e:
    print(f"⚠️ DummyEnv fallback: {e}")
    from src.run_all_menu import DummyEnv
    env = DummyEnv()

# ------------------------------
# Quick Evaluation (Baseline)
# ------------------------------
print("📊 Running quick evaluation...")
profits = evaluate(env)
print(f"Baseline Profit: {sum(profits)/len(profits):.2f}")

# ------------------------------
# Launch Streamlit Dashboard
# ------------------------------
print("🚀 Launching Streamlit dashboard...")
subprocess.run([
    "streamlit", "run", "dashboard.py",
    "--server.port", "8501",
    "--server.address", "0.0.0.0"
])

