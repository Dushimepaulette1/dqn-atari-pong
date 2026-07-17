# DQN Atari Pong

Training a Deep Q-Network (DQN) agent to play Atari Pong using [Stable Baselines3](https://stable-baselines3.readthedocs.io/).

## Project structure

```
dqn-atari-pong/
├── train.py                  # Train a DQN agent, log results to CSV
├── play.py                   # Watch a trained agent play locally
├── requirements.txt
├── results/
│   └── results_pau.csv       # Summary of every experiment run
├── notebooks/
│   └── formative-3.ipynb     # Full experiment notebook
└── dqn_model.zip              # Best trained model checkpoint
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Run the default (best-config) training run:

```bash
python train.py --name final
```

Or a quick experiment with custom hyperparameters:

```bash
python train.py --name quick_test --lr 0.001 --timesteps 150000
```

Each run saves a model checkpoint (`dqn_model_<name>.zip`), TensorBoard logs, and appends a summary row to `results.csv`.

## Watching the trained agent play

```bash
python play.py
```

Loads `dqn_model.zip` and renders the agent playing Pong locally.

## Status
Hyperparameter sweep complete - see `results/results_pau.csv` for the full set of experiment results.
