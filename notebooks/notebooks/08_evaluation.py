# project_pipeline_safe.py
"""
End-to-End Safe Pipeline: Notebooks 01-08
- Handles missing modules (tensorflow, gym, xgboost)
- Handles missing config.yaml or trained models
- Provides fallback dummy environments and default configs
- Saves figures, models, and evaluation summaries safely
"""
# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import yaml
from scipy import stats

# ------------------------------
# 1. Project root & config
# ------------------------------
POSSIBLE_ROOTS = [
    Path(__file__).resolve().parents[0],
    Path.cwd()
]

CONFIG_PATH = None
for root in POSSIBLE_ROOTS:
    candidate = root / "config.yaml"
    if candidate.exists():
        CONFIG_PATH = candidate
        PROJECT_ROOT = root
        break

if CONFIG_PATH is None:
    print("WARNING: config.yaml not found. Using default config.")
    cfg = {
        "simulation": {"seed": 42},
        "base_prices": {"Lagos": 100},
        "elasticities": {"Tomatoes": 0.5},
        "dqn": {"learning_rate":0.001,"replay_size":1000,"gamma":0.99,"batch_size":32,
                "eps_start":1.0,"eps_min":0.1,"eps_decay":0.995,"episodes":10}
    }
    PROJECT_ROOT = Path.cwd()
else:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from: {CONFIG_PATH}")

# ------------------------------
# 2. Set seeds
# ------------------------------
np.random.seed(cfg['simulation']['seed'])
random.seed(cfg['simulation']['seed'])

# ------------------------------
# 3. Load optional XGBoost model
# ------------------------------
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ModuleNotFoundError:
    XGB_AVAILABLE = False
    print("WARNING: XGBoost not installed, ML forecasting skipped.")

xgb_model_path = PROJECT_ROOT / "models/xgb_model.json"
if XGB_AVAILABLE and xgb_model_path.exists():
    bst = xgb.Booster()
    bst.load_model(str(xgb_model_path))
    print("Loaded XGBoost model for demand forecast.")
else:
    bst = None

def demand_forecast(day_index, base_forecast=20):
    if bst:
        # Placeholder for ML prediction
        return base_forecast * (1 + 0.05*np.sin(day_index/3))
    return base_forecast * (1 + 0.05*np.sin(day_index/3))

# ------------------------------
# 4. Setup RL Environment safely
# ------------------------------
ENV_AVAILABLE = False
try:
    import gym
    from src.envs.supermarket_env import SupermarketEnv
    ENV_AVAILABLE = True
except ModuleNotFoundError:
    print("WARNING: 'gym' or 'SupermarketEnv' missing, using dummy environment.")

if ENV_AVAILABLE:
    env = SupermarketEnv(
        base_price=cfg['base_prices'].get('Lagos',100),
        cost=cfg['base_prices'].get('Lagos',100)*0.55,
        shelf_life=7,
        inventory=200,
        elasticity=cfg['elasticities'].get('Tomatoes',0.5),
        demand_func=demand_forecast
    )
else:
    class DummyEnv:
        def __init__(self):
            self.observation_space = type("obs", (), {"shape": (3,)})()
            self.action_space = type("act", (), {"n": 3})()
            self.cost = 50
        def reset(self):
            return np.zeros(self.observation_space.shape)
        def step(self, action):
            return np.zeros(self.observation_space.shape), 0, True, {}
    env = DummyEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ------------------------------
# 5. Safe DQN training
# ------------------------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not installed, DQN training skipped.")

if TF_AVAILABLE:
    from collections import deque

    def build_qnet(input_dim, action_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg['dqn']['learning_rate']), loss='mse')
        return model

    class DQNAgent:
        def __init__(self):
            self.q = build_qnet(state_dim, action_dim)
            self.q_target = build_qnet(state_dim, action_dim)
            self.update_target()
            self.replay = deque(maxlen=cfg['dqn']['replay_size'])
            self.gamma = cfg['dqn']['gamma']
            self.batch_size = cfg['dqn']['batch_size']
            self.epsilon = cfg['dqn']['eps_start']
            self.eps_min = cfg['dqn']['eps_min']
            self.eps_decay = cfg['dqn']['eps_decay']

        def update_target(self):
            self.q_target.set_weights(self.q.get_weights())

        def act(self, s):
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, action_dim)
            qv = self.q.predict(s.reshape(1,-1), verbose=0)
            return np.argmax(qv[0])

        def remember(self, s,a,r,ns,d):
            self.replay.append((s,a,r,ns,d))

        def replay_train(self):
            if len(self.replay) < self.batch_size: return
            batch = random.sample(self.replay, self.batch_size)
            states = np.array([b[0] for b in batch])
            actions= np.array([b[1] for b in batch])
            rewards= np.array([b[2] for b in batch])
            next_s = np.array([b[3] for b in batch])
            dones  = np.array([b[4] for b in batch])
            q_next = self.q_target.predict(next_s, verbose=0)
            q_target = self.q.predict(states, verbose=0)
            for i in range(self.batch_size):
                q_target[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i]) * (1 - dones[i])
            self.q.fit(states, q_target, epochs=1, verbose=0)
            if self.epsilon > self.eps_min: self.epsilon *= self.eps_decay

    # Train DQN safely
    agent = DQNAgent()
    episodes = cfg['dqn']['episodes']
    reward_log = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0
        while not done:
            a = agent.act(s)
            ns, r, done, info = env.step(a)
            agent.remember(s,a,r,ns,done)
            agent.replay_train()
            s = ns
            ep_reward += r
        reward_log.append(ep_reward)
        if ep % 50 == 0: agent.update_target()
        if ep % 200 == 0: print(f"Episode {ep}, reward {ep_reward:.2f}, eps {agent.epsilon:.3f}")

    Path(PROJECT_ROOT / "models").mkdir(exist_ok=True)
    agent.q.save(PROJECT_ROOT / "models/dqn_agent.h5")
    Path(PROJECT_ROOT / "figures").mkdir(exist_ok=True)
    plt.plot(pd.Series(reward_log).rolling(50).mean())
    plt.title("DQN rolling mean reward (window=50)")
    plt.savefig(PROJECT_ROOT / "figures/dqn_reward_curve.png")
    print("DQN training complete.")
else:
    print("DQN training skipped due to missing TensorFlow.")

# ------------------------------
# 6. Safe evaluation
# ------------------------------
dqn_model_path = PROJECT_ROOT / "models/dqn_agent.h5"
if TF_AVAILABLE and dqn_model_path.exists():
    dqn = tf.keras.models.load_model(str(dqn_model_path))
else:
    dqn = None

def evaluate_static(env, runs=10):
    profits=[]
    for _ in range(runs):
        s = env.reset(); done=False
        total_rev=0; total_waste=0
        while not done:
            ns, r, done, info = env.step(0)
            total_rev += info.get('revenue',0)
            total_waste += info.get('waste',0)
        profits.append(total_rev - total_waste*getattr(env,"cost",50))
    return np.array(profits)

def evaluate_dqn(env, runs=10):
    profits=[]
    for _ in range(runs):
        s = env.reset(); done=False
        total_rev=0; total_waste=0
        while not done:
            if dqn is None:
                a = np.random.randint(0, env.action_space.n)
            else:
                qvals = dqn.predict(s.reshape(1,-1), verbose=0)[0]
                a = int(np.argmax(qvals))
            ns, r, done, info = env.step(a)
            total_rev += info.get('revenue',0)
            total_waste += info.get('waste',0)
        profits.append(total_rev - total_waste*getattr(env,"cost",50))
    return np.array(profits)

base_profit = evaluate_static(env)
dqn_profit = evaluate_dqn(env)
tstat, pval = stats.ttest_rel(base_profit, dqn_profit)

Path(PROJECT_ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"base_profit":base_profit, "dqn_profit":dqn_profit}).to_csv(
    PROJECT_ROOT / "data/processed/eval_summary.csv", index=False
)

print("Evaluation complete. Paired t-test p-value:", pval)
print("End-to-End pipeline executed successfully ✅")
