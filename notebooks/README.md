# DQN Pong — Experiments k01–k10

This directory holds the work I've started on training a **Deep Q-Network (DQN)**
agent to play Atari **Pong** (`ALE/Pong-v5`).

## What this is

My share of the group project: rerunning ten assigned experiments (**k01–k10**)
on Pong so my results are directly comparable with the rest of the team's runs.
All experiments use the group's shared `train.py` unchanged, so every run differs
only by its hyperparameters — nothing else.

## What's here so far

- **`Kelvin_Task.ipynb`** — the notebook driving my k01–k10 experiments.
- **`formative-3.ipynb`** — earlier working notebook for the formative.
- **`DQN_Breakout_ColabPro_Production_(1) (1).ipynb`** — reference notebook from
  the earlier Breakout work.

## Setup

The notebook is built to run on Google Colab with a GPU runtime (T4 is enough).
It installs its own dependencies in the first cell:
