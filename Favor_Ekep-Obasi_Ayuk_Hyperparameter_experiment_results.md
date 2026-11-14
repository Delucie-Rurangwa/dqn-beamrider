# Hyperparameter Experiment Results

**Member Name:** Favor Ekep-Obasi Ayuk

**Environment:** ALE/BeamRider-v5  
**Focus of my experiments:** exploration (epsilon start, end and decay) and mixed combinations  
**Fixed settings (same for all runs):** 100,000 steps, replay buffer 30,000, learning starts 10,000, target update every 1,000 steps, Adam optimizer, Huber loss, train every 4 steps.  
**Epsilon decay** = number of steps for epsilon to go from *Epsilon start* to *Epsilon end* (linear).

**Score used to compare:** average reward of the last 20 episodes (`avg_reward_last_20` in each `summary.json`).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Avg reward (last 20) | Observed behaviour |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | .99 | 32 | 1.0 | 0.10 | 50000 | _fill in_ | _fill in after training_ |
| 2 | 0.0001 | .99 | 32 | 1.0 | 0.01 | 50000 | _fill in_ | _fill in after training_ |
| 3 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 25000 | _fill in_ | _fill in after training_ |
| 4 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 90000 | _fill in_ | _fill in after training_ |
| 5 | 0.0001 | .99 | 32 | 0.5 | 0.05 | 50000 | _fill in_ | _fill in after training_ |
| 6 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 10000 | _fill in_ | _fill in after training_ |
| 7 | 0.00025 | .98 | 32 | 1.0 | 0.02 | 40000 | _fill in_ | _fill in after training_ |
| 8 | 0.0001 | .99 | 128 | 1.0 | 0.05 | 50000 | _fill in_ | _fill in after training_ |
| 9 | 0.0005 | .95 | 64 | 1.0 | 0.10 | 25000 | _fill in_ | _fill in after training_ |
| 10 | 0.00025 | .99 | 32 | 1.0 | 0.01 | 60000 | _fill in_ | _fill in after training_ |
| 11 | 0.0001 | .99 | 32 | 1.0 | 0.05 | 75000 | _fill in_ | _fill in after training_ |
| 12 | 0.0001 | .99 | 32 | 0.8 | 0.05 | 50000 | _fill in_ | _fill in after training_ |
| 13 | 0.0001 | .90 | 32 | 1.0 | 0.01 | 25000 | _fill in_ | _fill in after training_ |
| 14 | 0.0001 | .95 | 32 | 1.0 | 0.02 | 40000 | _fill in_ | _fill in after training_ |
| 15 | 0.0001 | .99 | 32 | 1.0 | 0.20 | 50000 | _fill in_ | _fill in after training_ |

## Best configuration
Experiment: _fill in_ — why it worked: _fill in_

## Worst configuration
Experiment: _fill in_ — why it failed: _fill in_

## What I learned about each hyperparameter
- **Learning rate:** _fill in_
- **Gamma:** _fill in_
- **Batch size:** _fill in_
- **Epsilon (start / end / decay):** _fill in_
