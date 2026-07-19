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

## Results

Ten experiments varied one hyperparameter at a time against a CNN baseline, plus one MLP policy comparison:

| Experiment | Policy | lr | gamma | batch | eps decay | timesteps | Eval reward |
|---|---|---|---|---|---|---|---|
| exp01_baseline | CNN | 1e-4 | 0.99 | 32 | 10% | 500k | -14.33 |
| exp02_high_lr | CNN | 1e-3 | 0.99 | 32 | 10% | 500k | -21.0 |
| exp03_low_lr | CNN | 1e-5 | 0.99 | 32 | 10% | 500k | -21.0 |
| exp04_low_gamma | CNN | 1e-4 | 0.9 | 32 | 10% | 500k | -14.67 |
| exp05_high_gamma | CNN | 1e-4 | 0.999 | 32 | 10% | 500k | -16.0 |
| exp06_batch64 | CNN | 1e-4 | 0.99 | 64 | 10% | 500k | -8.0 |
| exp07_batch128 | CNN | 1e-4 | 0.99 | 128 | 10% | 500k | 9.0 |
| exp08_fast_decay | CNN | 1e-4 | 0.99 | 32 | 2% | 500k | -18.33 |
| exp09_slow_decay | CNN | 1e-4 | 0.99 | 32 | 30% | 500k | -12.0 |
| exp10_best | CNN | 1e-4 | 0.99 | 128 | 20% | 1M | **20.33** |
| exp_mlp | MLP | 1e-4 | 0.99 | 32 | 10% | 150k | -21.0 |

Full data in [`results/results_pau.csv`](results/results_pau.csv).

## Findings

- **Batch size mattered most**: moving from 32 to 128 turned a losing agent into a winning one (9.0 → 20.33 average reward).
- **Learning rate is sensitive**: both 10x higher and 10x lower than baseline collapsed to the worst possible score (-21, i.e. losing every point).
- **A slower epsilon decay helped**: giving the agent more random exploration before committing to greedy play (30% vs 10% of training) improved on the baseline.
- **CNN clearly beats MLP** on raw pixel input, as expected - the MLP policy never learned better than random play.
- The final configuration (`exp10_best`): lr=1e-4, gamma=0.99, batch_size=128, epsilon decayed over 20% of training, 1M timesteps - achieved a **+20.33** average greedy evaluation reward.

## License

Personal project for learning purposes.
