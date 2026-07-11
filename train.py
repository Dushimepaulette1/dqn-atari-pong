"""
train.py - Train a DQN agent on Atari Pong using Stable Baselines3.

Can be used two ways:
  1. From the command line:
     python train.py --name exp01_baseline
     python train.py --name exp_high_lr --lr 0.001 --timesteps 150000
  2. Imported in a notebook:
     from train import train_dqn
     train_dqn(name="exp_high_lr", lr=1e-3, timesteps=150_000)

Each run saves the model as dqn_model_<name>.zip, logs to TensorBoard,
and appends a summary row (final evaluation reward, etc.) to results.csv.

Note on epsilon decay: SB3 expresses the decay speed as exploration_fraction,
the fraction of total training over which epsilon decays linearly from
eps_start to eps_end. Smaller = faster decay.
"""

import argparse
import csv
import os
import time

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)   # registers the Atari environments with Gymnasium

ENV_ID = "ALE/Pong-v5"      # modern naming; old PongNoFrameskip-v4 was removed
RESULTS_FILE = "results.csv"


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
    """Train one DQN agent with the given hyperparameters, save the model,
    evaluate it greedily, and log a summary row to results.csv."""
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
        tensorboard_log=f"./tb_logs/{name}",
        seed=seed,
    )

    print(f"\n{'='*60}")
    print(f"=== {name}: policy={policy}, lr={lr}, gamma={gamma}, "
          f"batch={batch_size}, eps={eps_start}->{eps_end} "
          f"over {exploration_fraction*100:.0f}% of training ===")
    print(f"{'='*60}\n")

    start = time.time()
    model.learn(total_timesteps=timesteps, log_interval=25)
    minutes = (time.time() - start) / 60

    model.save(f"dqn_model_{name}")

    # Greedy evaluation (deterministic=True = GreedyQPolicy): how good is
    # the agent when it stops exploring and just plays its best?
    eval_env = make_env(seed=123)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=3, deterministic=True
    )
    eval_env.close()
    env.close()

    print(f"\n>>> {name} done in {minutes:.1f} min | "
          f"greedy eval reward: {mean_reward:.1f} +/- {std_reward:.1f}\n")

    # Append summary row to results.csv -> this becomes your table
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "experiment", "policy", "lr", "gamma", "batch_size",
                "eps_start", "eps_end", "exploration_fraction",
                "timesteps", "train_minutes", "eval_reward_mean",
                "eval_reward_std",
            ])
        writer.writerow([
            name, policy, lr, gamma, batch_size, eps_start, eps_end,
            exploration_fraction, timesteps, round(minutes, 1),
            round(mean_reward, 2), round(std_reward, 2),
        ])

    return mean_reward


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
