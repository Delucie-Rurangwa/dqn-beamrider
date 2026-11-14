# Hyperparameter Experiment Results

**Member Name:** Delucie Rurangwa

**Environment:** ALE/BeamRider-v5  
**Focus of my experiments:** learning rate, gamma and batch size. Experiments 11–15 test combinations around the best settings found in 1–10 (lr 0.0001, gamma 0.90).  
**Fixed settings (same for all runs):** 100,000 steps, replay buffer 30,000, learning starts 10,000, target update every 1,000 steps, Adam optimizer, Huber loss, train every 4 steps.  
**Epsilon decay** = number of steps for epsilon to go from *Epsilon start* to *Epsilon end* (linear).

**Score used to compare:** average reward of the last 20 episodes (`avg_reward_last_20` in each `summary.json`).
**Reference:** a random agent scores about 260–330 points per game in BeamRider.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Avg reward (last 20) | Observed behaviour |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50000 | 449.2 | Baseline. Clear improvement after epsilon reached 0.05; reward rose from ~320 to ~420–476. |
| 2 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 50000 | **259.6 (worst)** | Learning rate too high. Rewards went up and down and ended at random level; updates were too big and unstable. |
| 3 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 50000 | 336.6 | Even higher learning rate. Very unstable (jumped between 280 and 400), only a little above random. |
| 4 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 50000 | 380.6 | Learning rate too low. Stable but slow; learned less than the baseline in 100k steps. |
| 5 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 50000 | 488.4 | Lower gamma helped. Steady increase in the second half, reaching ~500. |
| 6 | 0.0001 | 0.90 | 32 | 1.0 | 0.05 | 50000 | **542.0 (best)** | Best run. Reward climbed to ~550 and episodes became longer (the agent survived more). |
| 7 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 50000 | 444.4 | Very high gamma. Similar to baseline but noisier in the middle; long-term targets are harder to learn. |
| 8 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 50000 | 394.0 | Bigger batch. Smooth but slower progress than batch 32. |
| 9 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 50000 | 354.2 | Small batch. Noisy updates; reward dropped after exploration ended. |
| 10 | 0.00025 | 0.99 | 64 | 1.0 | 0.01 | 75000 | 391.6 | Mixed setting. Explored longer, so it had less time to exploit; average result. |
| 11 | 0.0001 | 0.85 | 32 | 1.0 | 0.05 | 50000 | _fill in_ | _fill in after training_ |
| 12 | 0.0001 | 0.90 | 64 | 1.0 | 0.05 | 50000 | _fill in_ | _fill in after training_ |
| 13 | 0.00025 | 0.90 | 32 | 1.0 | 0.05 | 50000 | _fill in_ | _fill in after training_ |
| 14 | 0.0001 | 0.90 | 32 | 1.0 | 0.01 | 50000 | _fill in_ | _fill in after training_ |
| 15 | 0.00005 | 0.90 | 32 | 1.0 | 0.05 | 50000 | _fill in_ | _fill in after training_ |

## Best configuration
**Experiment 6** (lr 0.0001, gamma 0.90, batch 32, epsilon 1.0 → 0.05 over 50,000 steps), average reward **542.0**, about 1.7× a random agent.
Why it worked: in BeamRider, rewards come quickly (you shoot an enemy and get points right away). A lower gamma makes the agent focus on these near rewards, so the Q-value targets are smaller and less noisy, and the agent learns faster in a short training time. The learning rate 0.0001 is a stable middle value.

## Worst configuration
**Experiment 2** (lr 0.0005), average reward **259.6**, the same as random play.
Why it failed: the learning rate was too large. Each update changed the network too much, so the Q-values kept jumping and the agent never settled on a good strategy.

## What I learned about each hyperparameter
- **Learning rate:** the most sensitive setting. 0.0001 worked best. Higher values (0.0005, 0.001) made training unstable; a lower value (0.00005) was stable but too slow for 100k steps.
- **Gamma:** lower gamma (0.90, 0.95) gave the best results, because BeamRider rewards are immediate and short training favours short-term targets. Very high gamma (0.999) did not help.
- **Batch size:** 32 was the best balance. 16 gave noisy gradients; 64 was smoother but learned more slowly in the same number of steps.
- **Epsilon:** decaying over 50,000 steps (half of training) worked well. Decaying over 75,000 steps (exp 10) left less time to use what the agent learned.

**Note:** each experiment was run once with 100,000 steps, so results have some randomness. Differences of about 30–50 points may be noise; the big differences (exp 6 vs exp 2) are clear.
