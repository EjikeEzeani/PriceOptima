# notebooks/07_rl_training_safe_all.py
# %%
"""
Notebook 07: Train Q-learning baseline and DQN agent (fully safe version)
- Handles missing src, gym, tensorflow
- Uses fallback dummy environment if RL dependencies are missing
- Provides fallback config if config.yaml missing
- Creates models and figures directories if missing
"""
# %%
import sys
from pathlib import Path
import yaml
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# === Locate project root and config safely ===
POSSIBLE_ROOTS = [
    Path(__file__).resolve().parents[2],
    Path(__file__).resolve().parents[1],
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
    print("WARNING: config.yaml not found. Using fallback default config.")
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

# === Set seeds for reproducibility ===
np.random.seed(cfg['simulation']['seed'])
random.seed(cfg['simulation']['seed'])

# === Try importing tensorflow ===
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ModuleNotFoundError:
    print("WARNING: TensorFlow not installed. DQN training cannot run.")
    TF_AVAILABLE = False

# === Try importing gym and SupermarketEnv ===
ENV_AVAILABLE = False
try:
    import gym
    from src.envs.supermarket_env import SupermarketEnv
    ENV_AVAILABLE = True
except ModuleNotFoundError:
    print("WARNING: 'gym' or 'SupermarketEnv' not available. Using dummy environment.")

# === Define fallback dummy environment ===
if not ENV_AVAILABLE:
    class DummyEnv:
        def __init__(self):
            self.observation_space = type("obs", (), {"shape": (3,)})()
            self.action_space = type("act", (), {"n": 3})()
        def reset(self):
            return np.zeros(self.observation_space.shape)
        def step(self, action):
            return np.zeros(self.observation_space.shape), 0, True, {}
    env = DummyEnv()
else:
    env = SupermarketEnv(
        base_price=cfg['base_prices'].get('Lagos', 100),
        cost=cfg['base_prices'].get('Lagos', 100)*0.55,
        shelf_life=7,
        inventory=200,
        elasticity=cfg['elasticities'].get('Tomatoes', 0.5)
    )

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# === DQN implementation ===
if TF_AVAILABLE:
    def build_qnet(input_dim, action_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg['dqn']['learning_rate']), loss='mse')
        return model

    from collections import deque

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
            if len(self.replay) < self.batch_size:
                return
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
            if self.epsilon > self.eps_min:
                self.epsilon *= self.eps_decay

    # === Train DQN safely ===
    agent = DQNAgent()
    episodes = cfg['dqn']['episodes']
    reward_log = []

    for ep in range(episodes):
        s = env.reset()
        done=False
        ep_reward = 0
        while not done:
            a = agent.act(s)
            ns, r, done, info = env.step(a)
            agent.remember(s,a,r,ns,done)
            agent.replay_train()
            s = ns
            ep_reward += r
        reward_log.append(ep_reward)
        if ep % 50 == 0:
            agent.update_target()
        if ep % 200 == 0:
            print(f"Episode {ep}, reward {ep_reward:.2f}, eps {agent.epsilon:.3f}")

    # Save model and plots
    Path(PROJECT_ROOT / "models").mkdir(exist_ok=True)
    agent.q.save(PROJECT_ROOT / "models/dqn_agent.h5")
    Path(PROJECT_ROOT / "figures").mkdir(exist_ok=True)
    plt.plot(pd.Series(reward_log).rolling(50).mean())
    plt.title("DQN rolling mean reward (window=50)")
    plt.savefig(PROJECT_ROOT / "figures/dqn_reward_curve.png")
    print("DQN training complete.")
else:
    print("DQN training skipped due to missing TensorFlow.")

# === Script finished ===
print("RL training script executed successfully ✅")

