"""
train.py - Train a DQN agent on Atari Pong using Stable Baselines3.
"""

import time

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)   # registers the Atari environments with Gymnasium

ENV_ID = "ALE/Pong-v5"      # modern naming; old PongNoFrameskip-v4 was removed


def make_env(seed=42):
    """Standard Atari setup: preprocessing + 4-frame stacking so the
    agent can perceive motion (one frame can't show ball direction)."""
    env = make_atari_env(
        ENV_ID, n_envs=1, seed=seed,
        env_kwargs={"frameskip": 1, "repeat_action_probability": 0.0},
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def train_dqn(name, timesteps=500_000, seed=42):
    """Train one DQN agent with baseline hyperparameters and save the model."""
    env = make_env(seed)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=32,
        buffer_size=50_000,        # replay memory (kept modest for Kaggle RAM)
        learning_starts=10_000,    # random play before learning begins
        train_freq=4,
        target_update_interval=1_000,
        verbose=1,
        seed=seed,
    )

    start = time.time()
    model.learn(total_timesteps=timesteps, log_interval=25)
    minutes = (time.time() - start) / 60

    model.save(f"dqn_model_{name}")
    print(f">>> {name} done in {minutes:.1f} min")

    return model


if __name__ == "__main__":
    train_dqn(name="exp01_baseline")
