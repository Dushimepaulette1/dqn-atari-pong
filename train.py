"""
train.py - Train a DQN agent on Atari Pong using Stable Baselines3.

Can be run from the command line with configurable hyperparameters:
    python train.py --name exp01_baseline
    python train.py --name exp_high_lr --lr 0.001 --timesteps 150000
"""

import argparse
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


def train_dqn(
    name,
    lr=1e-4,
    gamma=0.99,
    batch_size=32,
    eps_start=1.0,
    eps_end=0.05,
    exploration_fraction=0.1,
    policy="CnnPolicy",
    timesteps=500_000,
    seed=42,
):
    """Train one DQN agent with the given hyperparameters and save the model."""
    env = make_env(seed)

    model = DQN(
        policy=policy,
        env=env,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        exploration_initial_eps=eps_start,
        exploration_final_eps=eps_end,
        exploration_fraction=exploration_fraction,
        buffer_size=50_000,        # replay memory (kept modest for Kaggle RAM)
        learning_starts=10_000,    # random play before learning begins
        train_freq=4,
        target_update_interval=1_000,
        verbose=1,
        seed=seed,
    )

    print(f"\n=== {name}: policy={policy}, lr={lr}, gamma={gamma}, "
          f"batch={batch_size}, eps={eps_start}->{eps_end} "
          f"over {exploration_fraction*100:.0f}% of training ===\n")

    start = time.time()
    model.learn(total_timesteps=timesteps, log_interval=25)
    minutes = (time.time() - start) / 60

    model.save(f"dqn_model_{name}")
    print(f"\n>>> {name} done in {minutes:.1f} min\n")

    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN agent on Atari Pong")
    p.add_argument("--name", required=True, help="experiment name, e.g. exp01_baseline")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--exploration_fraction", type=float, default=0.1)
    p.add_argument("--policy", default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    p.add_argument("--timesteps", type=int, default=500_000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(
        name=args.name,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        exploration_fraction=args.exploration_fraction,
        policy=args.policy,
        timesteps=args.timesteps,
    )
