# src/run_all_menu.py
"""
📊 End-to-End RL & ML Pipeline + Menu
- Keeps all safety fixes from run_all.py
- Adds interactive menu-driven mode for terminal use
- Streamlit dashboard still supported
"""

import sys
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# ------------------------------
# Config & project setup
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = BASE_DIR / "config.yaml"
SRC_PATH = BASE_DIR / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"✅ Loaded config from {CONFIG_FILE}")
else:
    print("⚠️ config.yaml not found. Using defaults.")
    cfg = {
        "simulation": {"seed": 42},
        "base_prices": {"Lagos": 100},
        "elasticities": {"Tomatoes": 0.5},
        "dqn": {"learning_rate": 0.0001, "replay_size": 500,
                "gamma": 0.99, "batch_size": 16,
                "eps_start": 1.0, "eps_min": 0.1,
                "eps_decay": 0.99, "episodes": 5}
    }

# ------------------------------
# Seed
# ------------------------------
np.random.seed(cfg["simulation"]["seed"])
random.seed(cfg["simulation"]["seed"])

# ------------------------------
# Demand forecast function
# ------------------------------
def demand_forecast(day_index, base=20):
    return base * (1 + 0.05*np.sin(day_index/3))

# ------------------------------
# RL Environment with safe fallback
# ------------------------------
try:
    import gymnasium as gym
    from src.envs.supermarket_env import SupermarketEnv
    env = SupermarketEnv(
        base_price=cfg["base_prices"].get("Lagos", 100),
        cost=cfg["base_prices"].get("Lagos", 55),
        shelf_life=7,
        inventory=200,
        elasticity=cfg["elasticities"].get("Tomatoes", 0.5),
        demand_func=demand_forecast
    )
    print("✅ Loaded real SupermarketEnv")
except Exception as e:
    print(f"⚠️ Using DummyEnv. Reason: {e}")
    class DummyEnv:
        def __init__(self):
            self.observation_space = type("obs", (), {"shape": (3,)})()
            self.action_space = type("act", (), {"n": 3})()
            self.cost = 50
        def reset(self, seed=None):
            return np.zeros(self.observation_space.shape), {}
        def step(self, action):
            return np.zeros(self.observation_space.shape), 0, True, False, {"revenue": 0, "waste": 0}
    env = DummyEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ------------------------------
# DQN Agent (safe)
# ------------------------------
try:
    import tensorflow as tf
    from collections import deque

    def build_qnet(input_dim, action_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(action_dim, activation="linear"),
        ])
        lr = float(cfg["dqn"]["learning_rate"])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return model

    class DQNAgent:
        def __init__(self):
            self.q = build_qnet(state_dim, action_dim)
            self.q_target = build_qnet(state_dim, action_dim)
            self.replay = deque(maxlen=cfg["dqn"]["replay_size"])
            self.gamma = cfg["dqn"]["gamma"]
            self.batch_size = cfg["dqn"]["batch_size"]
            self.epsilon = cfg["dqn"]["eps_start"]

        def act(self, s):
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, action_dim)
            qv = self.q.predict(s.reshape(1,-1), verbose=0)
            return int(np.argmax(qv[0]))

        def train_step(self):
            if len(self.replay) < self.batch_size: return
            batch = random.sample(self.replay, self.batch_size)
            states, actions, rewards, next_s, dones = map(np.array, zip(*batch))
            q_next = self.q_target.predict(next_s, verbose=0)
            q_target = self.q.predict(states, verbose=0)
            for i in range(self.batch_size):
                q_target[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i]) * (1 - dones[i])
            self.q.fit(states, q_target, epochs=1, verbose=0)
            self.epsilon = max(cfg["dqn"]["eps_min"], self.epsilon * cfg["dqn"]["eps_decay"])

    agent = DQNAgent()
except Exception as e:
    print(f"⚠️ DQN unavailable: {e}")
    agent = None

# ------------------------------
# Evaluation function
# ------------------------------
def evaluate(env, agent=None, runs=5):
    profits = []
    for _ in range(runs):
        s, _ = env.reset()
        done, total_rev, total_waste = False, 0, 0
        while not done:
            if agent is None:
                a = np.random.randint(0, env.action_space.n)
            else:
                qv = agent.q.predict(s.reshape(1,-1), verbose=0)[0]
                a = int(np.argmax(qv))
            ns, r, done, trunc, info = env.step(a)
            total_rev += info.get("revenue", 0)
            total_waste += info.get("waste", 0)
            s = ns
        profits.append(total_rev - total_waste*env.cost)
    return profits

# ------------------------------
# Menu-driven runner
# ------------------------------
def menu():
    while True:
        print("\n📌 Select Stage to Run:")
        print("1. Merge Data")
        print("2. Preprocess Data")
        print("3. Train ML / DQN")
        print("4. Evaluate Model")
        print("5. Launch Dashboard (Streamlit)")
        print("6. Run All")
        print("0. Exit")

        choice = input("👉 Enter choice: ").strip()

        if choice == "1":
            print("✅ Running Merge stage...")
            # TODO: Add your merge function call
        elif choice == "2":
            print("✅ Running Preprocess stage...")
            # TODO: Add your preprocess function call
        elif choice == "3":
            print("✅ Training DQN Agent...")
            if agent:
                rewards = []
                for ep in range(cfg["dqn"]["episodes"]):
                    s, _ = env.reset()
                    done, ep_r = False, 0
                    while not done:
                        a = agent.act(s)
                        ns, r, done, trunc, info = env.step(a)
                        agent.replay.append((s,a,r,ns,int(done)))
                        agent.train_step()
                        s, ep_r = ns, ep_r + r
                    rewards.append(ep_r)
                print("✅ Training finished. Rewards:", rewards)
            else:
                print("⚠️ Agent not available")
        elif choice == "4":
            print("✅ Evaluating...")
            base = evaluate(env)
            dqn = evaluate(env, agent) if agent else []
            print("Baseline Profit:", np.mean(base))
            if dqn:
                print("DQN Profit:", np.mean(dqn))
        elif choice == "5":
            print("🚀 Launching Streamlit dashboard...")
            os.system(f"streamlit run {os.path.abspath(__file__)}")
        elif choice == "6":
            print("✅ Running ALL stages...")
            # Simulate running all
            menu_choice = ["1", "2", "3", "4"]
            for c in menu_choice:
                choice = c
        elif choice == "0":
            print("👋 Exiting menu.")
            break
        else:
            print("❌ Invalid choice. Try again.")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    menu()
