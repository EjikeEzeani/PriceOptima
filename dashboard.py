# dashboard.py
"""
Streamlit Dashboard for Supermarket Pricing & RL Project
✅ Handles Gymnasium & Gym API differences safely
✅ Uses dummy data if dataset is missing
✅ Shows Price Elasticity, ML, and RL results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------
# Data Loading
# ------------------------------
DATA_PATH = "data/processed/merged_input_dataset.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.success(f"✅ Data loaded: {DATA_PATH}")
else:
    st.warning("⚠️ Dataset not found. Using dummy data instead.")
    os.makedirs("data/processed", exist_ok=True)
    df = pd.DataFrame({
        "day": np.arange(30),
        "price": np.random.uniform(80, 120, 30),
        "quantity": np.random.randint(50, 150, 30),
        "revenue": np.random.uniform(4000, 8000, 30)
    })
    df.to_csv(DATA_PATH, index=False)

# ------------------------------
# Page Title
# ------------------------------
st.title("📊 Supermarket Dynamic Pricing Dashboard")

# ------------------------------
# Price Elasticity
# ------------------------------
st.header("📉 Price Elasticity of Demand")
if "price" in df.columns and "quantity" in df.columns:
    elasticity = df["quantity"].pct_change().corr(df["price"].pct_change())
    st.write(f"**Estimated Elasticity:** {elasticity:.3f}")

    fig, ax = plt.subplots()
    ax.scatter(df["price"], df["quantity"], alpha=0.7)
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")
    ax.set_title("Price vs Quantity")
    st.pyplot(fig)
else:
    st.error("⚠️ Columns `price` and `quantity` not found for elasticity analysis.")

# ------------------------------
# Supervised ML
# ------------------------------
st.header("🤖 Supervised Machine Learning")
if "revenue" in df.columns:
    from sklearn.linear_model import LinearRegression
    X = df[["price", "quantity"]]
    y = df["revenue"]
    model = LinearRegression().fit(X, y)
    st.write("✅ ML model trained: Revenue ~ Price + Quantity")

    pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.plot(y.values, label="Actual", marker="o")
    ax.plot(pred, label="Predicted", marker="x")
    ax.set_title("Revenue Prediction")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("⚠️ No `revenue` column found for supervised ML.")

# ------------------------------
# RL Evaluation
# ------------------------------
st.header("🎮 Reinforcement Learning Evaluation")

try:
    import gymnasium as gym
    from src.envs.supermarket_env import SupermarketEnv
    RL_AVAILABLE = True
except Exception as e:
    st.warning(f"⚠️ RL not available: {e}")
    RL_AVAILABLE = False

def demand_forecast(day_index, base=20):
    return base * (1 + 0.05 * np.sin(day_index / 3))

if RL_AVAILABLE:
    try:
        env = SupermarketEnv(
            base_price=100,
            cost=55,
            shelf_life=7,
            inventory=200,
            elasticity=0.5,
            demand_func=demand_forecast
        )
        st.success("✅ RL Environment loaded")

        profits = []
        for _ in range(5):
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                s, _ = reset_out  # Gymnasium
            else:
                s = reset_out     # Older gym

            done, total_rev, total_waste = False, 0, 0
            while not done:
                a = np.random.randint(0, env.action_space.n)
                step_out = env.step(a)

                if len(step_out) == 5:  # Gymnasium
                    ns, r, terminated, truncated, info = step_out
                    done = terminated or truncated
                elif len(step_out) == 4:  # Older gym
                    ns, r, done, info = step_out
                else:
                    raise ValueError("Unexpected step() return format")

                total_rev += info.get("revenue", 0)
                total_waste += info.get("waste", 0)
                s = ns

            profits.append(total_rev - total_waste * env.cost)

        st.write(f"**Baseline Profit (Random Policy):** {np.mean(profits):.2f}")

        fig, ax = plt.subplots()
        ax.plot(profits, marker="o")
        ax.set_title("Baseline RL Profit per Run")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Profit")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"RL evaluation failed: {e}")
else:
    st.info("RL modules unavailable. Install `gymnasium` and ensure `src/envs/supermarket_env.py` exists.")
