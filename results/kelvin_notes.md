# Kelvin — Noted Behavior Write-ups (k01–k10)

Ten single-variable experiments on `ALE/Pong-v5`, each run for 500k timesteps
with the group's `train.py` unchanged, then evaluated greedily over 3 episodes.
Baseline for comparison is Paulette's `exp01_baseline` at **−14.3**. Pong scores
range from −21 (lose every point) to +21 (win every point).

Full numbers: [`results_kelvin.csv`](results_kelvin.csv).

---

### k01 — lr = 5e-4 (5× baseline) → **−21.0 ± 0.0**

Raising the learning rate to 5e-4 collapsed training completely: the agent lost
every point in every evaluation episode. This matches Paulette's lr=1e-3 result,
so both of our high-lr runs failed the same way. The reason is that large Q-value
updates overshoot their target and destabilise the network — once the Q-estimates
diverge, the greedy policy just follows a broken value function. The usable
ceiling for the learning rate therefore sits somewhere below 5e-4.

### k02 — lr = 5e-5 (½ baseline) → **−7.7 ± 0.9**

Halving the learning rate did the opposite: −7.7 is a clear improvement on the
−14.3 baseline, and the tight std shows the runs were consistent. The interesting
contrast is with Paulette's lr=1e-5, which stayed pinned at −21. So there's a
sharp cliff on the low side too — half the baseline rate is slow but still
learning, while a tenth of it is effectively frozen and never gets off the floor
within 500k steps.

### k03 — gamma = 0.95 → **−9.3 ± 7.5**

A shorter reward horizon beat the baseline on average, but the ±7.5 spread is the
widest of all my runs — some evaluation episodes were good, others poor. Because
the average sits on top of that much variance, I treat it as "roughly baseline or
a bit better" rather than a solid win.

### k04 — gamma = 0.98 → **−16.0 ± 2.8**

Slightly below baseline. Read together with k03, gamma barely moves results in
Pong. The reason is structural: in Pong the reward (a scored point) lands very
soon after the actions that earn it, so the agent doesn't need a long discount
horizon to credit the right moves. Changing gamma mostly just adds run-to-run
noise here.

### k05 — batch = 16 → **−18.0 ± 0.0**

Small 16-sample gradient batches hurt stability and finished near the bottom.
With so few transitions per update, each gradient is a noisy estimate of the true
one, so learning is erratic. This is the low end of the group's batch-size ladder
(16 → 32 → 64 → 128 → 256), which improved monotonically as the batch grew.

### k06 — batch = 256 → **+19.7 ± 1.9**  ⭐ best in group

The standout: +19.7 is the best 500k-step result any of the three of us produced,
and the only one of my runs to finish positive. Averaging each update over 256
replayed experiences makes the gradient far smoother, and DQN benefits especially
because it is already an unstable, off-policy learner. The trade-off was compute —
~56 minutes versus ~30 for the baseline at the same step count — but the payoff
was decisive.

### k07 — eps_end = 0.01 → **−17.3 ± 3.8**

Dropping the exploration floor to 0.01 means the agent almost stops trying new
actions late in training. Over only 500k steps that's premature: it commits to a
policy it hasn't finished refining, and it landed below the standard 0.05 floor.
More exploration would likely help at this training length.

### k08 — eps_end = 0.15 → **−15.0 ± 7.1**

A permanent 15% random-action rate keeps roughly one action in seven random even
at the end of training, so play stays inconsistent — about baseline on average but
with a wide ±7.1 spread. The forced randomness caps how sharp the greedy policy
can get.

### k09 — eps decay over 5% of training → **−9.7 ± 0.9**

Decaying epsilon over just the first 5% of training beat the baseline's 10% decay
in this run, with a tight std. The agent reaches mostly-greedy play early and then
spends the bulk of training exploiting and refining what works.

### k10 — eps decay over 50% of training → **−6.3 ± 0.5**

My best pure-epsilon result (batch and lr aside): stretching the decay across half
of training gave the agent much more time to explore before settling down. It
extends Paulette's finding that slower epsilon decay pays off — exploration is an
investment that keeps returning value well into the run.

---

## Two things I'd lead with in the Q&A

1. **batch=256 → +19.7** is the group's best 500k-step score and the top of a
   batch-size ladder that improved monotonically across all three of us
   (16 → 32 → 64 → 128 → 256). Bigger batches average each update over more
   replayed experiences, so the gradients are less noisy — and DQN, being
   inherently unstable, gains the most from that smoothing.

2. **The learning-rate contrast** — my 5e-4 collapsed to −21 while my 5e-5 worked
   (−7.7). Combined with Paulette's 1e-3 and 1e-5 runs, this maps the whole usable
   learning-rate range and shows both failure modes: too high, updates overshoot
   and wreck the Q-values; too low, updates are too small to learn anything in the
   steps available.
