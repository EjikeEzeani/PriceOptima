#!/usr/bin/env python3
"""
🎮 Reinforcement Learning Analysis Script
RL-based pricing optimization and policy evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gym
from gym import spaces
import random
from collections import deque
import os
from datetime import datetime, timedelta

class PricingEnvironment(gym.Env):
    """
    Custom RL Environment for Dynamic Pricing
    """
    def __init__(self, data, max_steps=1000):
        super(PricingEnvironment, self).__init__()
        
        self.data = data
        self.max_steps = max_steps
        self.current_step = 0
        self.current_index = 0
        
        # Action space: price adjustment percentage (-50% to +50%)
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [price, quantity, revenue, day_of_week, month]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_index = random.randint(0, len(self.data) - 1)
        
        # Get initial state
        row = self.data.iloc[self.current_index]
        self.current_price = row['price']
        self.current_quantity = row['quantity']
        self.current_revenue = row['revenue']
        
        # Calculate state
        state = self._get_state()
        
        return state
    
    def _get_state(self):
        """Get current state observation"""
        row = self.data.iloc[self.current_index]
        
        # Normalize features
        price_norm = row['price'] / self.data['price'].max()
        quantity_norm = row['quantity'] / self.data['quantity'].max()
        revenue_norm = row['revenue'] / self.data['revenue'].max()
        
        # Time features
        if 'date' in self.data.columns:
            date = pd.to_datetime(row['date'])
            day_of_week = date.dayofweek / 7.0
            month = date.month / 12.0
        else:
            day_of_week = 0.5
            month = 0.5
        
        return np.array([price_norm, quantity_norm, revenue_norm, day_of_week, month], dtype=np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply price adjustment
        price_adjustment = action[0]
        new_price = self.current_price * (1 + price_adjustment)
        
        # Calculate new quantity based on price elasticity
        # Assume elasticity of -0.8 (quantity decreases as price increases)
        elasticity = -0.8
        quantity_change = elasticity * price_adjustment
        new_quantity = max(0, self.current_quantity * (1 + quantity_change))
        
        # Calculate new revenue
        new_revenue = new_price * new_quantity
        
        # Calculate reward (profit optimization)
        # Assume cost is 70% of original price
        cost = self.current_price * 0.7
        profit = new_revenue - (cost * new_quantity)
        
        # Reward function: maximize profit while maintaining reasonable sales volume
        volume_penalty = 0.1 if new_quantity < self.current_quantity * 0.5 else 0
        reward = profit - volume_penalty
        
        # Update state
        self.current_price = new_price
        self.current_quantity = new_quantity
        self.current_revenue = new_revenue
        
        # Move to next step
        self.current_step += 1
        self.current_index = (self.current_index + 1) % len(self.data)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        """Render the environment"""
        pass

class QLearningAgent:
    """
    Q-Learning Agent for Dynamic Pricing
    """
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Discretize action space
        self.action_bins = np.linspace(-0.5, 0.5, 21)  # 21 discrete actions
        
        # Q-table (simplified for demo)
        self.q_table = {}
    
    def get_state_key(self, state):
        """Convert continuous state to discrete key"""
        # Discretize state for Q-table
        state_key = tuple(np.round(state, 2))
        return state_key
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return np.array([random.uniform(-0.5, 0.5)])
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_bins))
        
        action_idx = np.argmax(self.q_table[state_key])
        return np.array([self.action_bins[action_idx]])
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_bins))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.action_bins))
        
        # Find closest action bin
        action_idx = np.argmin(np.abs(self.action_bins - action[0]))
        
        # Q-learning update
        current_q = self.q_table[state_key][action_idx]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action_idx] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def load_data():
    """Load data for RL training"""
    data_paths = [
        "DATASET/raw/wfp_food_prices_nga.csv",
        "DATASET/raw/wfp_food_prices.csv",
        "data/processed/merged_input_dataset.csv"
    ]
    
    for path in data_paths:
        try:
            df = pd.read_csv(path)
            st.success(f"✅ Loaded data from: {path}")
            return df
        except FileNotFoundError:
            continue
    
    # Create dummy data
    st.warning("⚠️ No data files found. Creating dummy data for RL training.")
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    base_price = 100
    base_quantity = 500
    
    prices = []
    quantities = []
    
    for i in range(n_samples):
        # Add some trend and seasonality
        trend = 0.01 * i
        seasonal = 0.1 * np.sin(2 * np.pi * i / 365)
        noise = np.random.normal(0, 0.05)
        
        price = base_price * (1 + trend + seasonal + noise)
        quantity = base_quantity * (1 - 0.5 * (price - base_price) / base_price + np.random.normal(0, 0.1))
        quantity = max(0, quantity)
        
        prices.append(price)
        quantities.append(quantity)
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'quantity': quantities,
        'revenue': np.array(prices) * np.array(quantities)
    })
    
    return df

def train_rl_agent(env, agent, episodes=100):
    """Train RL agent"""
    episode_rewards = []
    episode_profits = []
    episode_prices = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_profit = 0
        prices = []
        
        done = False
        step = 0
        
        while not done and step < 100:  # Limit steps per episode
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            total_profit += reward  # Assuming reward is profit
            prices.append(env.current_price)
            step += 1
        
        episode_rewards.append(total_reward)
        episode_profits.append(total_profit)
        episode_prices.append(np.mean(prices))
    
    return episode_rewards, episode_profits, episode_prices

def create_training_plots(episode_rewards, episode_profits, episode_prices):
    """Create training visualization plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Episode Profits', 'Average Prices', 'Reward Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(y=episode_rewards, mode='lines', name='Rewards'),
        row=1, col=1
    )
    
    # Episode profits
    fig.add_trace(
        go.Scatter(y=episode_profits, mode='lines', name='Profits'),
        row=1, col=2
    )
    
    # Average prices
    fig.add_trace(
        go.Scatter(y=episode_prices, mode='lines', name='Avg Prices'),
        row=2, col=1
    )
    
    # Reward distribution
    fig.add_trace(
        go.Histogram(x=episode_rewards, name='Reward Distribution'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    return fig

def evaluate_policy(env, agent, n_episodes=10):
    """Evaluate trained policy"""
    total_rewards = []
    total_profits = []
    price_changes = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_profit = 0
        initial_price = env.current_price
        final_price = initial_price
        
        done = False
        step = 0
        
        while not done and step < 50:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_profit += reward
            final_price = env.current_price
            step += 1
        
        total_rewards.append(episode_reward)
        total_profits.append(episode_profit)
        price_changes.append((final_price - initial_price) / initial_price * 100)
    
    return total_rewards, total_profits, price_changes

def main():
    st.set_page_config(
        page_title="Reinforcement Learning Analysis",
        page_icon="🎮",
        layout="wide"
    )
    
    st.title("🎮 Reinforcement Learning for Dynamic Pricing")
    st.markdown("---")
    
    # Load data
    st.header("📊 Data Loading")
    df = load_data()
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Price Range", f"₦{df['price'].min():.2f} - ₦{df['price'].max():.2f}")
    with col3:
        st.metric("Avg Revenue", f"₦{df['revenue'].mean():.2f}")
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # RL Configuration
    st.header("⚙️ RL Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        episodes = st.slider("Training Episodes", 10, 500, 100)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
        discount_factor = st.slider("Discount Factor", 0.8, 0.99, 0.95)
    
    with col2:
        epsilon = st.slider("Initial Epsilon", 0.1, 1.0, 1.0)
        max_steps = st.slider("Max Steps per Episode", 50, 200, 100)
        eval_episodes = st.slider("Evaluation Episodes", 5, 50, 10)
    
    # Training
    if st.button("🚀 Start RL Training", type="primary"):
        with st.spinner("Training RL agent..."):
            # Create environment
            env = PricingEnvironment(df, max_steps=max_steps)
            
            # Create agent
            agent = QLearningAgent(
                state_size=5,
                action_size=1,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon
            )
            
            # Train agent
            episode_rewards, episode_profits, episode_prices = train_rl_agent(env, agent, episodes)
        
        st.success("✅ RL training completed!")
        
        # Display training results
        st.subheader("📈 Training Results")
        
        # Training plots
        training_fig = create_training_plots(episode_rewards, episode_profits, episode_prices)
        st.plotly_chart(training_fig, use_container_width=True)
        
        # Training statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Reward", f"{episode_rewards[-1]:.2f}")
        with col2:
            st.metric("Best Reward", f"{max(episode_rewards):.2f}")
        with col3:
            st.metric("Avg Profit", f"{np.mean(episode_profits):.2f}")
        with col4:
            st.metric("Final Epsilon", f"{agent.epsilon:.3f}")
        
        # Policy evaluation
        st.subheader("🔍 Policy Evaluation")
        
        with st.spinner("Evaluating trained policy..."):
            eval_rewards, eval_profits, eval_price_changes = evaluate_policy(env, agent, eval_episodes)
        
        # Evaluation results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Evaluation Reward", f"{np.mean(eval_rewards):.2f}")
            st.metric("Avg Evaluation Profit", f"{np.mean(eval_profits):.2f}")
        
        with col2:
            st.metric("Avg Price Change", f"{np.mean(eval_price_changes):.2f}%")
            st.metric("Price Change Std", f"{np.std(eval_price_changes):.2f}%")
        
        # Evaluation plots
        eval_fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Evaluation Rewards', 'Evaluation Profits', 'Price Changes (%)')
        )
        
        fig.add_trace(
            go.Box(y=eval_rewards, name='Rewards'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(y=eval_profits, name='Profits'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Box(y=eval_price_changes, name='Price Changes'),
            row=1, col=3
        )
        
        st.plotly_chart(eval_fig, use_container_width=True)
        
        # Policy insights
        st.subheader("💡 Policy Insights")
        
        if np.mean(eval_price_changes) > 0:
            st.info("📈 **Price-Increasing Policy**: The agent learned to increase prices to maximize profit")
        else:
            st.info("📉 **Price-Decreasing Policy**: The agent learned to decrease prices to maximize volume")
        
        if np.std(eval_price_changes) < 5:
            st.success("✅ **Stable Policy**: Low price volatility indicates a stable pricing strategy")
        else:
            st.warning("⚠️ **Volatile Policy**: High price volatility may indicate aggressive pricing")
        
        # Business recommendations
        st.subheader("🎯 Business Recommendations")
        
        avg_price_change = np.mean(eval_price_changes)
        avg_profit = np.mean(eval_profits)
        
        if avg_price_change > 5 and avg_profit > 0:
            st.success("""
            **Recommended Strategy: Premium Pricing**
            - The RL agent suggests increasing prices
            - This strategy maximizes profit margins
            - Monitor customer satisfaction and market response
            """)
        elif avg_price_change < -5 and avg_profit > 0:
            st.success("""
            **Recommended Strategy: Volume Pricing**
            - The RL agent suggests decreasing prices
            - This strategy maximizes sales volume
            - Monitor profit margins and operational capacity
            """)
        else:
            st.info("""
            **Recommended Strategy: Balanced Pricing**
            - The RL agent suggests moderate price adjustments
            - This strategy balances profit and volume
            - Continue monitoring market conditions
            """)
        
        # Save model
        st.subheader("💾 Save RL Model")
        if st.button("Save Trained Agent"):
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"rl_agent_{timestamp}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(agent, f)
            
            st.success(f"✅ RL agent saved to: {model_path}")

if __name__ == "__main__":
    main()
