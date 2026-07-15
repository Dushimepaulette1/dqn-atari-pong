"""
play.py - Load the trained DQN model and watch it play Pong.

Run this LOCALLY on your laptop. Put dqn_model.zip in the same folder.
"""

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)   # registers the Atari environments with Gymnasium

MODEL_PATH = "dqn_model.zip"
ENV_ID = "ALE/Pong-v5"
NUM_EPISODES = 3


def make_env():
    """Must match the training setup exactly: same env, same preprocessing,
    same 4-frame stacking, frameskip handled by the wrapper only."""
    env = make_atari_env(
        ENV_ID, n_envs=1, seed=0,
        env_kwargs={
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "render_mode": "human",
        },
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def main():
    env = make_env()
    model = DQN.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    for episode in range(1, NUM_EPISODES + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # deterministic=True = greedy Q policy: always pick the action
            # with the highest Q-value, no exploration (GreedyQPolicy).
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1

        print(f"Episode {episode}: reward = {total_reward}, length = {steps} steps")

    env.close()


if __name__ == "__main__":
    main()
