import sys
from pathlib import Path
import yaml
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def run_rl(config_path: str = "config.yaml") -> Optional[str]:
    """
    Train DQN agent in SupermarketEnv, save reward curve plot, and return plot path.
    Returns None if RL training cannot be run.
    """
    # === Load config ===
    CONFIG_PATH = Path(config_path)
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("simulation", {}).get("seed", 42)
    np.random.seed(seed)
    random.seed(seed)
    # === Try importing TensorFlow ===
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ModuleNotFoundError:
        TF_AVAILABLE = False
    # === Try gym/gymnasium ===
    ENV_AVAILABLE = False
    try:
        import gymnasium as gym
        ENV_AVAILABLE = True
        USE_GYMNASIUM = True
    except ImportError:
        try:
            import gym
            ENV_AVAILABLE = True
            USE_GYMNASIUM = False
        except ImportError:
            ENV_AVAILABLE = False
            USE_GYMNASIUM = False
    # === Try SupermarketEnv ===
    SupermarketEnv = None
    if ENV_AVAILABLE:
        try:
            from src.envs.supermarket_env import SupermarketEnv
        except Exception:
            SupermarketEnv = None
    # === Dummy environment as fallback ===
    if not ENV_AVAILABLE or SupermarketEnv is None:
        class DummyEnv:
            def __init__(self):
                self.observation_space = type("obs", (), {"shape": (3,)})()
                self.action_space = type("act", (), {"n": 3})()
            def reset(self):
                return np.zeros(self.observation_space.shape)
            def step(self, action):
                return np.zeros(self.observation_space.shape), 0.0, True, {}
        env = DummyEnv()
    else:
        env = SupermarketEnv(
            base_price=cfg["base_prices"].get("Lagos", 100),
            cost=cfg["base_prices"].get("Lagos", 100) * 0.55,
            shelf_life=7,
            inventory=200,
            elasticity=cfg["elasticities"].get("Tomatoes", 0.5)
        )
    # === Env helpers for gym vs gymnasium ===
    def reset_env(env):
        res = env.reset()
        if isinstance(res, tuple):
            return res[0]
        return res
    def step_env(env, action):
        res = env.step(action)
        if isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
            return obs, reward, done, info
        elif isinstance(res, tuple) and len(res) == 4:
            obs, reward, done, info = res
            return obs, reward, done, info
        else:
            return np.zeros(env.observation_space.shape), 0.0, True, {}
    state_dim = int(np.prod(getattr(env.observation_space, "shape", (3,))))
    action_dim = getattr(env.action_space, "n", 1)
    # === DQN implementation ===
    if not TF_AVAILABLE:
        return None
    lr_raw = cfg["dqn"].get("learning_rate", 1e-3)
    try:
        lr = float(lr_raw)
    except Exception:
        lr = 1e-3
    def build_qnet(input_dim, action_dim):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(action_dim, activation="linear")
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse"
        )
        return model
    from collections import deque
    class DQNAgent:
        def __init__(self):
            self.q = build_qnet(state_dim, action_dim)
            self.q_target = build_qnet(state_dim, action_dim)
            self.update_target()
            self.replay = deque(maxlen=int(cfg["dqn"].get("replay_size", 1000)))
            self.gamma = float(cfg["dqn"].get("gamma", 0.99))
            self.batch_size = int(cfg["dqn"].get("batch_size", 32))
            self.epsilon = float(cfg["dqn"].get("eps_start", 1.0))
            self.eps_min = float(cfg["dqn"].get("eps_min", 0.1))
            self.eps_decay = float(cfg["dqn"].get("eps_decay", 0.995))
        def update_target(self):
            self.q_target.set_weights(self.q.get_weights())
        def act(self, s):
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, action_dim)
            qv = self.q.predict(np.array(s).reshape(1, -1), verbose=0)
            return int(np.argmax(qv[0]))
        def remember(self, s, a, r, ns, d):
            self.replay.append((s, a, r, ns, d))
        def replay_train(self):
            if len(self.replay) < self.batch_size:
                return
            batch = random.sample(self.replay, self.batch_size)
            states = np.array([b[0] for b in batch])
            actions = np.array([b[1] for b in batch])
            rewards = np.array([b[2] for b in batch])
            next_s = np.array([b[3] for b in batch])
            dones = np.array([b[4] for b in batch])
            q_next = self.q_target.predict(next_s, verbose=0)
            q_target = self.q.predict(states, verbose=0)
            for i in range(self.batch_size):
                target = rewards[i]
                if not dones[i]:
                    target += self.gamma * np.max(q_next[i])
                q_target[i, actions[i]] = target
            self.q.fit(states, q_target, epochs=1, verbose=0)
            if self.epsilon > self.eps_min:
                self.epsilon *= self.eps_decay
    agent = DQNAgent()
    episodes = int(cfg["dqn"].get("episodes", 10))
    reward_log = []
    for ep in range(episodes):
        s = reset_env(env)
        done = False
        ep_reward = 0
        while not done:
            a = agent.act(s)
            ns, r, done, info = step_env(env, a)
            agent.remember(s, a, r, ns, done)
            agent.replay_train()
            s = ns
            ep_reward += r
        reward_log.append(ep_reward)
        if ep % 50 == 0:
            agent.update_target()
    # Save results
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)
    try:
        agent.q.save("models/dqn_agent.h5")
    except Exception:
        pass
    rl_plot_path = "figures/dqn_reward_curve.png"
    try:
        plt.plot(pd.Series(reward_log).rolling(50, min_periods=1).mean())
        plt.title("DQN Reward Curve")
        plt.savefig(rl_plot_path)
        plt.close()
    except Exception:
        return None
    return rl_plot_path



