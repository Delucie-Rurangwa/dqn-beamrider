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
| 11 | 0.0001 | 0.85 | 32 | 1.0 | 0.05 | 50000 | 376.2 | Gamma lower than 0.90. Worse than exp6: the agent became too short-sighted and ignored useful future rewards. |
| 12 | 0.0001 | 0.90 | 64 | 1.0 | 0.05 | 50000 | 429.0 | Best gamma with batch 64. Steady but slower than exp6, same pattern as exp8 (bigger batch learns more slowly). |
| 13 | 0.00025 | 0.90 | 32 | 1.0 | 0.05 | 50000 | 374.0 | Best gamma with a higher learning rate. Unstable (jumped between 280 and 470); confirms 0.0001 is safer. |
| 14 | 0.0001 | 0.90 | 32 | 1.0 | 0.01 | 50000 | 431.8 | Best settings but almost no exploration at the end. Rewards went up and down (308–481); less consistent than exp6. |
| 15 | 0.00005 | 0.90 | 32 | 1.0 | 0.05 | 50000 | 466.6 | Best gamma with a lower learning rate. Slow but steady, still rising at the end (~497); might improve with longer training. |

## Best configuration
**Experiment 6** (lr 0.0001, gamma 0.90, batch 32, epsilon 1.0 → 0.05 over 50,000 steps), average reward **542.0**, about 1.7× a random agent. Experiments 11–15 changed one setting around exp6 each time, and **all of them scored lower** (374–467), which confirms that exp6 is the best combination.
Why it worked: in BeamRider, rewards come quickly (you shoot an enemy and get points right away). A lower gamma makes the agent focus on these near rewards, so the Q-value targets are smaller and less noisy, and the agent learns faster in a short training time. The learning rate 0.0001 is a stable middle value.

## Worst configuration
**Experiment 2** (lr 0.0005), average reward **259.6**, the same as random play.
Why it failed: the learning rate was too large. Each update changed the network too much, so the Q-values kept jumping and the agent never settled on a good strategy.

## What I learned about each hyperparameter
- **Learning rate:** the most sensitive setting. 0.0001 worked best. Higher values (0.0005, 0.001) made training unstable; a lower value (0.00005) was stable but too slow for 100k steps.
- **Gamma:** lower gamma (0.90, 0.95) gave the best results, because BeamRider rewards are immediate and short training favours short-term targets. Very high gamma (0.999) did not help.
- **Batch size:** 32 was the best balance. 16 gave noisy gradients; 64 was smoother but learned more slowly in the same number of steps.
- **Epsilon:** decaying over 50,000 steps (half of training) worked well. Decaying over 75,000 steps (exp 10) left less time to use what the agent learned, and ending at 0.01 instead of 0.05 (exp 14) made results less consistent.
- **Combinations (exp 11–15):** gamma 0.85 was too low, and changing the learning rate or batch size away from exp6's values always made results worse. A good gamma only works well together with a stable learning rate (0.0001) and batch 32.

## Final configuration and model
I chose **exp6's settings** for the final model and trained them for **500,000 steps** (`models/Delucie_Rurangwa/best_long/`): average reward of the last 20 episodes **993.6**, best game **2,272**. In `play.py` (greedy) the agent scores between about 660 and 1,430, around 1,000 on average.

**Note:** each experiment was run once with 100,000 steps, so results have some randomness. Differences of about 30–50 points may be noise; the big differences (exp 6 vs exp 2) are clear.
