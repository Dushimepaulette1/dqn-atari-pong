# DQN on Atari Pong — Formative 2

Training a Deep Q-Network (DQN) agent to play Atari Pong using Stable Baselines3 and Gymnasium.

**Team:** Paulette Dushime · Kelvin [surname] · David Cyubahiro

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

### Member: Kelvin

<!-- KELVIN: Replace this comment block with your hyperparameter table and noted behaviors (see kelvin_readme_section.md sent to you). Commit as: "Add Kelvin's hyperparameter table and noted behaviors" -->
*Section to be added by Kelvin — his 10 experiment results are in `results/results_kelvin.csv`.*

### Member: David Cyubahiro

<!-- DAVID: Replace this comment block with your hyperparameter table and noted behaviors (see david_readme_section.md sent to you). Commit as: "Add David's hyperparameter table and noted behaviors" -->
*Section to be added by David — his 10 experiment results are in `results/results_david.csv`.*

### Discussion

**Learning rate** was the most fragile knob: 1e-4 worked, 5e-4 and above collapsed training entirely (confirmed independently by all three members, including in combination with large batches), and 1e-5 was too slow to learn — while 5e-5 remained functional. **Batch size** was the most powerful knob, improving results monotonically from 16 to 256 across all runs. **Epsilon schedule** told a consistent exploration-exploitation story: decaying too fast (2% of training) locked in weak policies, while slower decay (30-50%) consistently beat the baseline — early exploration is an investment that pays back later. **Gamma** mattered least: Pong's rewards follow their causes within seconds, so the effective horizon is short regardless of gamma; its variations stayed within run-to-run noise except when paired with a stabilizing large batch.

The final model combines these lessons: lr=1e-4, gamma=0.99, batch=128, epsilon 1.0→0.05 over 20% of training, trained for 1M timesteps → **+20.3**.

## Individual contributions

- **Paulette Dushime:** environment selection, shared training pipeline (train.py, play.py, Kaggle notebook used by all members), 10 hyperparameter experiments, MLP vs CNN comparison, final 1M-step model, gameplay video, repository assembly and README.
- **Kelvin [surname]:** 10 hyperparameter experiments (intermediate lr/gamma values, batch-size extremes, epsilon floors and decay extremes) run on Colab with a resumable experiment runner; written analysis of his results.
- **David Cyubahiro:** 10 hyperparameter experiments focused on hyperparameter combinations and interaction effects; written analysis of his results.

Each member's raw results are in `results/` and their notebooks in `notebooks/`; commit history reflects individual uploads.

## Notes

- The assignment names the decay parameter "epsilon_decay"; SB3 implements it as `exploration_fraction` (fraction of training over which epsilon decays linearly). The mapping is documented in train.py.
- `dqn_model.zip` at the repo root is the exp10_best model renamed as required by the assignment; per-experiment models were kept out of the repo for size reasons.
- The gameplay video was recorded via play.py's built-in `RECORD_VIDEO` mode on Windows; the live-window mode (`render_mode="human"`) works on any machine with a display.