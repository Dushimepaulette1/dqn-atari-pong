"""
play.py - Load the trained DQN model and watch it play Pong.

Run this LOCALLY on your laptop. Put dqn_model.zip in the same folder.
Set RECORD_VIDEO = True to also save an mp4 into ./videos for the README.
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
RECORD_VIDEO = False # set True to also save mp4s into ./videos


def make_env(render_mode):
    """Must match the training setup exactly: same env, same preprocessing,
    same 4-frame stacking, frameskip handled by the wrapper only."""
    env = make_atari_env(
        ENV_ID, n_envs=1, seed=0,
        env_kwargs={
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "render_mode": render_mode,
        },
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def main():
    if RECORD_VIDEO:
        from stable_baselines3.common.vec_env import VecVideoRecorder
        env = make_env(render_mode="rgb_array")
        env = VecVideoRecorder(
            env, video_folder="./videos",
            record_video_trigger=lambda step: step == 0,
            video_length=20_000,
            name_prefix="dqn_pong",
        )
    else:
        # human mode opens a game window and renders every step automatically
        env = make_env(render_mode="human")

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
