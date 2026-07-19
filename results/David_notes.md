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
