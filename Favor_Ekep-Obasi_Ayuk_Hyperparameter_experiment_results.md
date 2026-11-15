# Hyperparameter Experiment Results

**Member Name:** Favor Ekep-Obasi Ayuk

**Environment:** ALE/BeamRider-v5
**Focus of my experiments:** exploration (epsilon start, end and decay) and mixed combinations  
**Fixed settings (same for all runs):** 100,000 steps, replay buffer 30,000, learning starts 10,000, target update every 1,000 steps, Adam optimizer, Huber loss, train every 4 steps.  
**Epsilon decay** = number of steps for epsilon to go from *Epsilon start* to *Epsilon end* (linear).

**Score used to compare:** average reward of the last 20 episodes (`avg_reward_last_20` in each `summary.json`).

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
**Experiment 9** (lr 0.0005, gamma 0.95, batch size 64, epsilon 1.0 → 0.10 over 25,000 steps), average reward **524.8**. Its last-20 average was the highest of these runs, and its average reward rose from 338.8 in the first 20 episodes. Experiments 8 (514.0) and 6 (512.6) were close behind. Since each configuration was run once and several settings differ, this result does not isolate which hyperparameter was responsible.

## Worst configuration
**Experiments 11 and 15 (tie)**, both average reward **297.0**. Experiment 11 decayed epsilon over 75,000 steps, leaving relatively little time at low exploration; Experiment 15 ended at epsilon 0.20, maintaining more exploration than the other runs. Both are consistent with insufficient time exploiting a learned policy, but single runs cannot establish the cause.

## What I learned about each hyperparameter
- **Learning rate:** No single effect is clear because the learning rate was changed alongside other settings in the mixed experiments. The highest score came from experiment 9 (0.0005), while other configurations with 0.0001 and 0.00025 also improved over their early episodes.
- **Gamma:** The best run used 0.95 (experiment 9), and the 0.90 run (experiment 13) also improved. Since these runs differed in other settings too, the results do not isolate gamma's effect.
- **Batch size:** Batch size 64 scored 524.8 in experiment 9, while batch size 128 scored 514.0 in experiment 8; batch size 32 ranged widely. These runs do not establish a best batch size independently of the other settings.
- **Epsilon (start / end / decay):** The strongest results used different schedules: experiment 9 decayed to 0.10 over 25,000 steps, experiment 8 to 0.05 over 50,000, and experiment 6 to 0.05 over 10,000. The 75,000-step decay (experiment 11) and higher final epsilon of 0.20 (experiment 15) tied for the lowest score. Overall, these results suggest a useful balance between exploration and exploitation, but do not identify one universally best schedule.

**Note:** Each experiment was run once with the same seed. Results may reflect run-to-run randomness, and several mixed experiments changed multiple hyperparameters, so comparisons cannot attribute score differences to one setting alone.
