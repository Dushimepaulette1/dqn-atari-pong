# DQN Pong — Experiments k01–k10

This directory holds my share of the group project: training a **Deep Q-Network
(DQN)** agent to play Atari **Pong** (`ALE/Pong-v5`) across ten hyperparameter
experiments (**k01–k10**).

## What this is

I reran ten assigned experiments on Pong so my results are directly comparable
with the rest of the team's runs. Every experiment uses the group's shared
`train.py` **verbatim**, so runs differ only by their hyperparameters — nothing
else. The group reference points are Paulette's baseline of **−14.3** and her
best of **+9.0** (both at 500k steps).

## What's here

- **`dpqn_Pong.ipynb`** — the notebook that drives all ten k01–k10 experiments,
  writes `results.csv`, and produces the ranking below.
- **`results.csv`** — the summary table (one row per experiment). A copy also
  lives at `../results.csv`.

## Setup

The notebook runs on Google Colab with a GPU runtime (T4 is enough — an A100
just wastes units). The first cell installs its own dependencies:

```
stable-baselines3[extra]
gymnasium[atari]
ale-py
autorom[accept-rom-license]
```

The working directory is mounted on Google Drive, so `results.csv`, every model,
TensorBoard logs, and the run log are written straight to Drive as each run
happens. Each experiment writes a `DONE` flag on completion, so a session that
dies can be resumed by rerunning — finished experiments are skipped.

Each run trains for **500,000 timesteps** (matching the group's runs) and is
evaluated greedily over 3 episodes.

## Experiments

| Experiment | Hyperparameter change (vs. baseline) |
|---|---|
| k01_lr_5e4 | learning rate 5e-4 (5× higher) |
| k02_lr_5e5 | learning rate 5e-5 (2× lower) |
| k03_gamma_095 | gamma 0.95 (shorter reward horizon) |
| k04_gamma_098 | gamma 0.98 |
| k05_batch16 | batch size 16 |
| k06_batch256 | batch size 256 |
| k07_epsend_001 | final epsilon 0.01 |
| k08_epsend_015 | final epsilon 0.15 |
| k09_decay_5pct | epsilon decays over first 5% of training |
| k10_decay_50pct | epsilon decays over first 50% of training |

Baseline (unchanged) config: `CnnPolicy`, lr 1e-4, gamma 0.99, batch 32,
epsilon 1.0 → 0.05 over 10% of training.

## Results

Ranked by greedy evaluation reward (higher is better; Pong scores range −21 to +21):

| Rank | Experiment | Eval reward (mean ± std) | Train (min) | vs. baseline (−14.3) |
|---|---|---|---|---|
| 1 | k06_batch256 | **+19.67 ± 1.89** | 56.2 | +34.0 |
| 2 | k10_decay_50pct | −6.33 ± 0.47 | 27.7 | +8.0 |
| 3 | k02_lr_5e5 | −7.67 ± 0.94 | 31.1 | +6.6 |
| 4 | k03_gamma_095 | −9.33 ± 7.54 | 32.3 | +5.0 |
| 5 | k09_decay_5pct | −9.67 ± 0.94 | 30.4 | +4.6 |
| 6 | k08_epsend_015 | −15.00 ± 7.07 | 29.5 | −0.7 |
| 7 | k04_gamma_098 | −16.00 ± 2.83 | 32.3 | −1.7 |
| 8 | k07_epsend_001 | −17.33 ± 3.77 | 33.0 | −3.0 |
| 9 | k05_batch16 | −18.00 ± 0.00 | 29.9 | −3.7 |
| 10 | k01_lr_5e4 | −21.00 ± 0.00 | 30.9 | −6.7 |

## Key findings

- **Batch size 256 (k06) was the clear winner** — the only experiment to reach a
  positive score (+19.67), far above the group best of +9.0. Larger, smoother
  gradient batches gave much more stable learning, at the cost of nearly double
  the training time (56 min vs. ~30 min for the rest).
- **A larger learning rate hurt badly** — k01 (lr 5e-4) collapsed to the worst
  possible score (−21.0), i.e. losing every point; the updates were too large to
  learn a stable policy. Lowering the learning rate (k02) helped instead.
- **A slower epsilon decay helped** — k10 (decay over 50% of training) was the
  second best, suggesting the baseline explores too little, too early.
- **Small batches (k05) and a very low epsilon floor (k07) both underperformed**
  the baseline.
