# Hyperparameter Experiment Results

**Member Name:** Favor Ekep-Obasi Ayuk

**Environment:** ALE/BeamRider-v5
**Focus of my experiments:** exploration (epsilon start, end and decay) and mixed combinations
**Fixed settings (same for all runs):** 100,000 steps, replay buffer 30,000, learning starts 10,000, target update every 1,000 steps, Adam optimizer, Huber loss, train every 4 steps.  
**Epsilon decay** = number of steps for epsilon to go from *Epsilon start* to *Epsilon end* (linear).

**Score used to compare:** average reward of the last 20 episodes (`avg_reward_last_20` in each `summary.json`).
**Reference points:** a random agent scores about 260–330. The shared **baseline** (lr 0.0001, gamma 0.99, batch 32, epsilon 1.0 → 0.05 over 50,000 steps) is Delucie's experiment 1: **449.2**. Experiments 1–6, 8, 11, 12 and 15 change only one setting from this baseline, so they can be compared with it directly.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Avg reward (last 20) | Observed behaviour |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | .99 | 32 | 1.0 | 0.10 | 50000 | 363.0 | Rewards were about the same at the start and end; little sustained improvement. |
| 2 | 0.0001 | .99 | 32 | 1.0 | 0.01 | 50000 | 330.0 | Reward decreased from its first-20 average; ending epsilon at 0.01 did not improve this run. |
| 3 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 25000 | 381.0 | Modest improvement; one episode reached 756. |
| 4 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 90000 | 312.4 | Reward declined from the first 20 episodes; the long decay left exploration high for most of training. |
| 5 | 0.0001 | .99 | 32 | 0.5 | 0.05 | 50000 | 420.2 | Improved from an early average of 297.0; moderate final performance. |
| 6 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 10000 | 512.6 | Strong improvement from an early average of 294.8; last-20 episode length averaged about 1,789 steps. |
| 7 | 0.00025 | .98 | 32 | 1.0 | 0.02 | 40000 | 415.8 | Reward increased from an early average of 330.0; moderate improvement. |
| 8 | 0.0001 | .99 | 128 | 1.0 | 0.05 | 50000 | 514.0 | Strong improvement from 346.0 early; one episode reached 1,236. |
| 9 | 0.0005 | .95 | 64 | 1.0 | 0.10 | 25000 | **524.8 (best)** | Highest last-20 average; reward rose from 338.8 early and last-20 episode length averaged about 1,595 steps. |
| 10 | 0.00025 | .99 | 32 | 1.0 | 0.01 | 60000 | 391.6 | Improved from an early average of 319.0; moderate result. |
| 11 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 75000 | **297.0 (tied worst)** | Reward fell from 365.2 early; exploration continued through most of training. |
| 12 | 0.0001 | .99 | 32 | 0.8 | 0.05 | 50000 | 396.2 | Improved from 347.6 early; last-20 episode length averaged about 1,697 steps. |
| 13 | 0.0001 | .90 | 32 | 1.0 | 0.01 | 25000 | 445.8 | Modest improvement; one episode reached 804. |
| 14 | 0.0001 | .95 | 32 | 1.0 | 0.02 | 40000 | 400.4 | Recovered from an early average of 277.2 to 400.4 in the last 20 episodes. |
| 15 | 0.0001 | .99 | 32 | 1.0 | 0.20 | 50000 | **297.0 (tied worst)** | Reward fell from 376.2 early; the relatively high final epsilon kept exploration active. |

## Best configuration
**Experiment 9** (lr 0.0005, gamma 0.95, batch size 64, epsilon 1.0 → 0.10 over 25,000 steps), average reward **524.8**, about 1.7× a random agent. Experiments 8 (514.0) and 6 (512.6) were close behind.
Why it may have worked: it combined a faster epsilon decay (the agent started exploiting earlier) with a larger batch. A larger batch gives smoother updates, which can make a higher learning rate safe. With batch 32, the same learning rate 0.0005 was unstable in Delucie's experiment 2 (259.6), so the batch size probably made the difference.

**Best pure exploration setting: experiment 6** (decay over only 10,000 steps), **512.6** vs 449.2 for the baseline. Stopping exploration early gave the agent more time to use and improve what it learned within 100,000 steps.

## Worst configuration
**Experiments 11 and 15 (tie)**, both **297.0**, the same level as random play.
- Experiment 11 decayed epsilon over 75,000 steps, so the agent was still taking many random actions for most of training and had little time to exploit what it learned.
- Experiment 15 ended at epsilon 0.20, so even after the decay, 1 action in 5 was random. The agent kept wasting good opportunities, and the score never improved.

## What I learned about each hyperparameter
- **Epsilon decay (main finding):** faster decay was better in 100,000 steps. 10,000 steps → 512.6 (best), 25,000 → 381.0, 50,000 → 449.2 (baseline), 75,000 → 297.0, 90,000 → 312.4. Long decays (75k, 90k) were the worst because the agent explored for almost all of training.
- **Epsilon end:** 0.05 (baseline, 449.2) worked better than 0.10 (363.0), 0.01 (330.0) and 0.20 (297.0). Too high wastes moves on random actions; too low (0.01) gave a less consistent result in this run. A small but non-zero final epsilon was the best balance.
- **Epsilon start:** starting at 0.5 (420.2) or 0.8 (396.2) instead of 1.0 did not help; full exploration at the start (1.0) collects more varied experiences when the network knows nothing yet.
- **Batch size:** 128 (experiment 8) scored 514.0, better than the baseline 449.2. Larger batches gave smoother learning in my runs.
- **Learning rate and gamma (mixed experiments 7, 9, 10, 13, 14):** a higher learning rate (0.0005) worked only together with a larger batch (experiment 9). Lower gamma (0.90–0.95) improved results in experiments 13 and 14, which agrees with Delucie's finding that BeamRider's quick rewards favour a lower gamma.

## Final configuration
If I had to choose one exploration schedule: **epsilon 1.0 → 0.05 with a fast decay (10,000–25,000 steps)**, combined with the group's best settings (lr 0.0001, gamma 0.90, batch 32). For our final model, the group used Delucie's experiment 6 settings (best overall, 542.0) trained for 500,000 steps: average 993.6.

**Note:** each experiment was run once with the same seed, so differences of about 30–50 points may be random. The large differences (for example 512.6 vs 297.0) are clear.