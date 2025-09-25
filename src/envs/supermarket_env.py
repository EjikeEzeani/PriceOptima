# src/envs/supermarket_env.py
"""
OpenAI Gym-style environment for supermarket perishable pricing.
State: [inventory, days_left, current_price, base_price, day_index]
Actions: discrete price adjustments (e.g., -30%, -20%, -15%, -5%, 0%, +10%, +20%)
Reward: revenue - waste_cost
"""
import gym
from gym import spaces
import numpy as np

class SupermarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, base_price=100, cost=60, shelf_life=7, inventory=100, elasticity=0.8, demand_func=None):
        super(SupermarketEnv, self).__init__()
        self.base_price = base_price
        self.cost = cost
        self.shelf_life = shelf_life
        self.init_inventory = inventory
        self.elasticity = elasticity
        self.demand_func = demand_func or (lambda day: 20)
        self.action_space = spaces.Discrete(7)
        # observation: inventory, days_left, current_price, base_price, day_index
        self.observation_space = spaces.Box(low=0, high=1e6, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.day = 0
        self.inv = self.init_inventory
        self.current_price = self.base_price
        self.days_left = self.shelf_life
        self.cum_reward = 0.0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.inv, self.days_left, self.current_price, self.base_price, self.day], dtype=np.float32)

    def step(self, action):
        action_map = [-0.3,-0.2,-0.15,-0.05,0.0,0.1,0.2]
        pct = action_map[action]
        new_price = max(self.cost * 1.01, self.current_price * (1 + pct))
        base_forecast = self.demand_func(self.day)
        realized_demand = base_forecast * (new_price / self.base_price) ** (-self.elasticity)
        realized_demand *= np.random.normal(1, 0.08)
        sold = min(self.inv, max(0, int(round(realized_demand))))
        revenue = sold * new_price
        self.inv -= sold
        self.days_left -= 1
        waste = 0
        if self.days_left <= 0:
            waste = self.inv
            self.inv = 0
        waste_cost = waste * self.cost
        reward = revenue - waste_cost
        self.cum_reward += reward
        done = (self.day >= self.shelf_life - 1) or (self.inv <= 0)
        self.current_price = new_price
        self.day += 1
        info = {'sold': sold, 'waste': waste, 'revenue': revenue}
        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        print(f"Day {self.day}, Inv: {self.inv}, Price: {self.current_price:.2f}, Days_left: {self.days_left}")

