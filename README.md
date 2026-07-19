# DQN on Atari Pong — Formative 2

Training a Deep Q-Network (DQN) agent to play Atari Pong using Stable Baselines3 and Gymnasium.

**Team:** Paulette Dushime · Kelvin Rwihimba · David Cyubahiro

**Final result: our best agent reaches a greedy evaluation reward of +20.3 out of a maximum +21** (it wins matches roughly 21–1). Trained for 1M timesteps with the best configuration found across 30 hyperparameter experiments (10 per member).

## Gameplay demo

[VIDEO EMBED GOES HERE — edit this README on GitHub and drag-and-drop videos/dqn_pong_gameplay.mp4 into the editor to make it playable inline]

The clip was produced by `play.py`, which loads the trained model and plays with a greedy policy (always the highest Q-value action, no exploration). Console output from the recorded run:

```
Episode 1: reward = 20.0, length = 1663 steps
Episode 2: reward = 20.0, length = 1732 steps
Episode 3: reward = 20.0, length = 1732 steps
```

## Environment

We use **ALE/Pong-v5** from the Arcade Learning Environment via Gymnasium. The agent controls the right paddle; the reward is +1 for winning a point and -1 for losing one, so a full game score ranges from -21 (lose every point) to +21 (win every point). A random agent scores about -21.

Preprocessing (applied by SB3's `make_atari_env`): grayscale conversion, resize to 84x84, frame skipping (4), reward clipping, and 4-frame stacking so the agent can perceive motion — a single frame cannot show which direction the ball is moving. We set `frameskip=1` and `repeat_action_probability=0.0` in the raw environment so that only the wrapper's frame skip applies and sticky actions are disabled, reproducing the classic deterministic Pong setup.

## Repository structure

```
├── train.py               # trains a DQN agent; hyperparameters via CLI or function args
├── play.py                # loads dqn_model.zip and plays with a greedy policy
├── dqn_model.zip          # final trained model (exp10_best, +20.3 eval reward)
├── requirements.txt
├── collaboration_tracker.pdf
├── results/
│   ├── results_pau.csv    # Paulette's 10 experiments + final model + MLP comparison
│   ├── results_kelvin.csv # Kelvin's 10 experiments
│   ├── results_david.csv  # David's 10 experiments
│   └── *_notes.md         # each member's written observations
├── videos/
│   └── dqn_pong_gameplay.mp4
└── notebooks/             # the Kaggle/Colab notebooks that ran the experiments
```

## How to run

Install dependencies (Python 3.10+):

```
pip install -r requirements.txt
```

**Train** (defaults are the best configuration found during tuning):

```
python train.py --name final
```

Any hyperparameter can be overridden, e.g. `python train.py --name test --lr 0.001 --batch 64 --timesteps 150000`. Each run saves `dqn_model_<name>.zip`, logs reward trends and episode lengths to the console and TensorBoard (`tb_logs/`), and appends a summary row (including a greedy evaluation reward) to `results.csv`.

**Play** (requires `dqn_model.zip` in the same folder):

```
python play.py
```

Opens a game window (`render_mode="human"`) and runs 3 episodes with the greedy policy. On headless machines (or WSL, which cannot open SDL windows), set `RECORD_VIDEO = True` inside play.py to save an mp4 to `videos/` instead.

Training was done on Kaggle (GPU T4 x2) and Colab; each 500k-step experiment takes roughly 30 minutes.

## MLP vs CNN policy comparison

We trained the same DQN setup with both policies. The input is raw pixels (84x84x4 stacked frames):

| Policy | Eval reward |
|---|---|
| MlpPolicy | **-21.0** (never learned; loses every point) |
| CnnPolicy | **+20.3** (near-perfect play, final model) |

The MLP flattens the frames into ~28k unrelated numbers, destroying the spatial relationships between the ball and the paddles, so it cannot discover visual patterns. The CNN's convolutional filters preserve spatial structure and learn to detect and track the ball and paddles, which is why all our experiments use **CnnPolicy**.

## Hyperparameter tuning

Each member ran 10 experiments (500k timesteps each unless noted) varying learning rate, gamma, batch size, and the epsilon schedule. SB3 expresses epsilon decay as `exploration_fraction`: the fraction of training over which epsilon decays linearly from `eps_start` to `eps_end` (smaller = faster decay).

### Headline finding: batch size

Across all three members' independent runs, evaluation reward improved monotonically with batch size:

| Batch size | Eval reward | Member |
|---|---|---|
| 16 | -18.0 | Kelvin |
| 32 | -14.3 | Paulette (baseline) |
| 64 | -8.0 | Paulette |
| 128 | +9.0 | Paulette |
| 256 | +19.7 | Kelvin |

Larger batches average the Bellman update over more replayed experiences, producing less noisy gradients — and DQN, already unstable due to its moving target, benefits enormously from that stability. The trade-off is compute: batch 256 took ~56 min vs ~30 min for batch 32 at the same number of timesteps.

### Member: Paulette Dushime

| Hyperparameter Set | Noted Behavior |
|---|---|
| lr=1e-4, gamma=0.99, batch=32, eps 1.0→0.05, decay over 10% (baseline) | Learned steadily but slowly; eval reward -14.3 ± 6.6 after 500k steps. Reference point for all other runs. |
| lr=1e-3 | Complete collapse: eval -21.0. Large Q-updates destabilized learning early; episode length shrank and never recovered — the agent locked into a degenerate policy. |
| lr=1e-5 | Eval -21.0 for the opposite reason: updates too small to make progress in 500k steps; behavior stayed near-random. |
| gamma=0.90 | Eval -14.7, essentially the baseline. Pong's rewards arrive quickly after the actions that cause them, so shortening the reward horizon barely matters. |
| gamma=0.999 | Eval -16.0, slightly slower learning than baseline; the very long horizon added noise without benefit for this game. |
| batch=64 | Clear improvement to -8.0 ± 7.1. Smoother gradient updates from averaging more experiences. |
| batch=128 | Big jump to +9.0 ± 1.4 — the best single-knob change. Confirmed batch size as the dominant lever. |
| eps decay over 2% of training | Eval -18.3. Exploration ended after ~10k steps, before the agent had seen enough varied situations; it committed to a weak strategy and barely recovered. |
| eps decay over 30% of training | Eval -12.0, better than baseline despite "wasting" 150k steps exploring. Early exploration is an investment. |
| **exp10 (best): lr=1e-4, gamma=0.99, batch=128, eps decay over 20%, 1M steps** | **Eval +20.3 ± 0.5 — the final model (dqn_model.zip).** Combined the batch-size winner with extended exploration and longer training; reward climbed -20 → -13 → -4 → +6 → +13 → +17 during training. |

### Member: Kelvin Rwihimba

I ran ten single-variable experiments (**k01–k10**), each changing exactly one hyperparameter from the shared baseline (`CnnPolicy`, lr=1e-4, gamma=0.99, batch=32, epsilon 1.0→0.05 decayed over 10% of training) so the effect of each knob is isolated. Every run used Paulette's `train.py` unchanged, trained for **500k timesteps**, and was scored by greedy evaluation over 3 episodes. I ran them on Colab (T4 GPU) from my own notebook using a resumable runner — each experiment writes a `DONE` flag so a dropped session skips finished runs and continues. Baseline for comparison is `exp01_baseline` at **-14.3**.

Full data: [`results/results_kelvin.csv`](results/results_kelvin.csv) · detailed per-experiment write-ups: [`results/kelvin_notes.md`](results/kelvin_notes.md).

| Experiment | Change vs baseline | Eval reward | Noted Behavior |
|---|---|---|---|
| k01_lr_5e4 | lr = 5e-4 (5×) | -21.0 ± 0.0 | Even 5× the baseline learning rate collapsed training completely, mirroring Paulette's lr=1e-3 run. The usable ceiling for the learning rate sits somewhere below 5e-4. |
| k02_lr_5e5 | lr = 5e-5 (½×) | -7.7 ± 0.9 | Clearly better than the -14.3 baseline. The contrast with Paulette's lr=1e-5 (stuck at -21) is telling: half the baseline rate is slow but still learns, a tenth is effectively frozen. |
| k03_gamma_095 | gamma = 0.95 | -9.3 ± 7.5 | Better than baseline, but the wide spread across evaluation episodes means I don't read too much into the average. |
| k04_gamma_098 | gamma = 0.98 | -16.0 ± 2.8 | Marginally below baseline. Taken with my gamma=0.95 run, gamma barely moves the needle in Pong, because its rewards land soon after the actions that earn them. |
| k05_batch16 | batch = 16 | -18.0 ± 0.0 | Small, noisy 16-sample gradient batches hurt stability, marking the bottom of the group's batch-size ladder. |
| k06_batch256 | batch = 256 | **+19.7 ± 1.9** | The best 500k-step result in the whole group. Very smooth gradient updates; the trade-off was compute, ~56 min vs ~30 min for the baseline at the same step count. |
| k07_epsend_001 | eps_end = 0.01 | -17.3 ± 3.8 | A 0.01 exploration floor means the agent almost stops exploring late in training, which underperformed the standard 0.05 floor over 500k steps. |
| k08_epsend_015 | eps_end = 0.15 | -15.0 ± 7.1 | A permanent 15% random-action rate kept play inconsistent — around baseline on average but with a wide spread. |
| k09_decay_5pct | eps decay over 5% | -9.7 ± 0.9 | Ahead of the baseline's 10% decay in this run: reaching mostly-greedy play early left more of training to exploit. |
| k10_decay_50pct | eps decay over 50% | -6.3 ± 0.5 | The best pure-epsilon result in the group. It backs up Paulette's finding that slower epsilon decay pays off: exploration is an investment. |

**My takeaways:**
- **Batch size is the biggest lever.** `k06_batch256` (+19.7) was the single best 500k-step run in the group and sits at the top of a batch-size ladder that improved monotonically across all three of us (16 → 32 → 64 → 128 → 256). Bigger batches average each update over more replayed experiences, so gradients are less noisy — and DQN, being inherently unstable, gains the most from that.
- **Learning rate has a narrow usable band.** My `k01` (5e-4) collapsed to -21 while `k02` (5e-5) worked (-7.7); combined with Paulette's 1e-3 and 1e-5 runs this maps both failure modes — too high overshoots and destroys the Q-values, too low never learns in the steps available.
- **Slower epsilon decay helps.** `k10` (decay over 50%) was my best epsilon-only run, reinforcing that early exploration pays back later.

### Member: David Cyubahiro

| Hyperparameter Set | Noted Behavior |
|---|---|
| lr=1e-3, batch=128 | Eval -21.0: a large batch could NOT rescue a bad learning rate — the collapse caused by oversized updates dominates no matter how smooth the gradients are. |
| lr=1e-5, eps decay over 30% | Eval -20.3: extra exploration cannot compensate for a learning rate too small to actually learn from the experience it gathers. |
| batch=128, gamma=0.95 | Eval -1.0: large-batch stability carried the run even with a shortened reward horizon, confirming gamma's weak effect under a different batch size. |
| eps decay over 2%, eps_end=0.01 | Eval -19.3: minimal exploration both early (fast decay) and late (low floor); the agent committed to a poor policy almost immediately and never escaped it. |
| batch=64, eps decay over 30% | Eval -2.7: two individually good changes (bigger batch + slower decay) stacked together beat either change alone. |
| lr=5e-4, batch=64 | Eval -21.0: confirms the learning-rate ceiling — batch 64 could not stabilize lr=5e-4 any more than batch 128 stabilized lr=1e-3 in my first experiment. |
| gamma=0.999, batch=128 | Eval +18.0 +/- 1.4: prioritizing long-term rewards with stable batch updates produced one of the strongest policies in the group. |
| eps_start=0.5 | Eval -6.7 +/- 9.5: starting with less exploration reduced state-space coverage, limiting the agent's ability to discover optimal behaviors; the very high variance shows the resulting policies were inconsistent. |
| gamma=0.90, eps decay over 2% | Eval -13.0: focusing on short-term rewards while cutting exploration quickly led to only marginal improvement over the -14.3 baseline. |
| lr=1e-4, batch=128, eps decay over 30%, eps_end=0.01 (combo) | Eval +19.0 +/- 2.8: balancing stable learning (batch 128), long-term reward optimization, and sustained exploration produced my best overall performance. |
*

### Discussion

**Learning rate** was the most fragile knob: 1e-4 worked, 5e-4 and above collapsed training entirely (confirmed independently by all three members, including in combination with large batches), and 1e-5 was too slow to learn — while 5e-5 remained functional. **Batch size** was the most powerful knob, improving results monotonically from 16 to 256 across all runs. **Epsilon schedule** told a consistent exploration-exploitation story: decaying too fast (2% of training) locked in weak policies, while slower decay (30-50%) consistently beat the baseline — early exploration is an investment that pays back later. **Gamma** mattered least: Pong's rewards follow their causes within seconds, so the effective horizon is short regardless of gamma; its variations stayed within run-to-run noise except when paired with a stabilizing large batch.

The final model combines these lessons: lr=1e-4, gamma=0.99, batch=128, epsilon 1.0→0.05 over 20% of training, trained for 1M timesteps → **+20.3**.

## Individual contributions

- **Paulette Dushime:** environment selection, shared training pipeline (train.py, play.py, Kaggle notebook used by all members), 10 hyperparameter experiments, MLP vs CNN comparison, final 1M-step model, gameplay video, repository assembly and README.
- **Kelvin Rwihimba:** 10 hyperparameter experiments (intermediate lr/gamma values, batch-size extremes, epsilon floors and decay extremes) run on Colab with a resumable experiment runner; written analysis of his results.
- **David Cyubahiro:** 10 hyperparameter experiments focused on hyperparameter combinations and interaction effects; written analysis of his results.

Each member's raw results are in `results/` and their notebooks in `notebooks/`; commit history reflects individual uploads.

## Notes

- The assignment names the decay parameter "epsilon_decay"; SB3 implements it as `exploration_fraction` (fraction of training over which epsilon decays linearly). The mapping is documented in train.py.
- `dqn_model.zip` at the repo root is the exp10_best model renamed as required by the assignment; per-experiment models were kept out of the repo for size reasons.
- The gameplay video was recorded via play.py's built-in `RECORD_VIDEO` mode on Windows; the live-window mode (`render_mode="human"`) works on any machine with a display.
