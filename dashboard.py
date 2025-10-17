# dashboard.py
"""
📊 Streamlit Dashboard for MSc Project
- Handles Elasticity, Supervised ML, and RL Results
- Uses dummy data if real dataset is missing
- Compatible with latest Streamlit versions (no deprecated options)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# File Paths
# ------------------------------
DATA_PATH = Path("data/processed/merged_input_dataset.csv")
DUMMY_PATH = Path("data/processed/dummy_dashboard.csv")

# ------------------------------
# Load Data
# ------------------------------
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.success(f"✅ Loaded dataset: {DATA_PATH}")
else:
    # create dummy data if file not found
    st.warning("⚠️ Real dataset not found. Using dummy data.")
    DUMMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    dummy_data = {
        "date": pd.date_range("2022-01-01", periods=100),
        "price": np.random.uniform(50, 200, 100),
        "quantity": np.random.randint(10, 200, 100),
        "revenue": np.random.uniform(500, 5000, 100),
    }
    df = pd.DataFrame(dummy_data)
    df.to_csv(DUMMY_PATH, index=False)
    st.info(f"Dummy dataset created at {DUMMY_PATH}")

st.subheader("📄 Preview of Data")
st.dataframe(df.head())

# ------------------------------
# Elasticity Analysis
# ------------------------------
st.header("📉 Price Elasticity of Demand")

if {"price", "quantity"}.issubset(df.columns):
    X = df[["price"]].values
    y = df["quantity"].values
    model = LinearRegression().fit(X, y)
    elasticity = model.coef_[0] * (df["price"].mean() / df["quantity"].mean())
    st.write(f"*Estimated Elasticity:* {elasticity:.3f}")

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(df["price"], df["quantity"], alpha=0.6, label="Data")
    ax.plot(df["price"], model.predict(X), color="red", label="Fit")
    ax.set_xlabel("Price")
    ax.set_ylabel("Quantity")
    ax.set_title("Price vs Quantity")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("⚠️ Columns price and quantity not found for elasticity analysis.")

# ------------------------------
# Supervised Machine Learning
# ------------------------------
st.header("🤖 Supervised Machine Learning")

if "revenue" in df.columns and {"price", "quantity"}.issubset(df.columns):
    features = df[["price", "quantity"]]
    target = df["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    ml_model = LinearRegression().fit(X_train, y_train)
    y_pred = ml_model.predict(X_test)

    st.write(f"*R² Score:* {r2_score(y_test, y_pred):.4f}")
    st.write(f"*RMSE:* {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # Prediction vs Actual Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Predicted vs Actual Revenue")
    st.pyplot(fig)
else:
    st.error("⚠️ No revenue column found for supervised ML.")

# ------------------------------
# RL Placeholder
# ------------------------------
st.header("🎮 Reinforcement Learning Evaluation")
st.info("RL results will be integrated here (profits, waste reduction, policies).")
