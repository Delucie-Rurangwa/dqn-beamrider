"""
train.py - Train a Deep Q-Network (DQN) agent on Atari BeamRider.

Example:
    uv run train.py --member Larissa --exp exp1 --lr 0.0001 --gamma 0.99 \
        --batch-size 32 --eps-start 1.0 --eps-end 0.05 --eps-decay 100000
"""

import argparse
import csv
import json
import os
import random

import ale_py
import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")  # save plots to files (no window needed)
import matplotlib.pyplot as plt

gym.register_envs(ale_py)  # makes the "ALE/..." Atari games available

ENV_ID = "ALE/BeamRider-v5"


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
def make_env(render_mode=None):
    """Create BeamRider with the standard DQN preprocessing.

    - grayscale, resized to 84x84
    - repeat each action for 4 frames (frame skip)
    - stack the last 4 frames so the agent can see how enemies move
    Final observation shape: (4, 84, 84)
    """
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


# ---------------------------------------------------------------------------
# 2. Q-Network (CNN built by us with torch.nn)
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Input: 4 stacked frames. Output: one Q-value for every action."""

    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)   # 84x84 -> 20x20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 20x20 -> 9x9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 9x9   -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255.0  # pixel values 0-255 -> 0-1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# 3. Experience Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Stores past experiences (s, a, r, s', done) and gives random batches."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0   # where the next experience will be written
        self.size = 0    # how many experiences are stored

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        # when the buffer is full, overwrite the oldest experience
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], device=device),
            torch.tensor(self.actions[idx], device=device),
            torch.tensor(self.rewards[idx], device=device),
            torch.tensor(self.next_states[idx], device=device),
            torch.tensor(self.dones[idx], device=device),
        )


# ---------------------------------------------------------------------------
# 4. Epsilon-greedy action selection
# ---------------------------------------------------------------------------
def get_epsilon(step, eps_start, eps_end, eps_decay):
    """Linear decay: epsilon goes from eps_start to eps_end over eps_decay steps."""
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(state, epsilon, q_net, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)            # explore: random action
    with torch.no_grad():
        state_t = torch.tensor(np.array(state), device=device).unsqueeze(0)
        q_values = q_net(state_t)
    return int(q_values.argmax(dim=1).item())         # exploit: best action


# ---------------------------------------------------------------------------
# 5. One training step (Bellman equation + loss + backpropagation)
# ---------------------------------------------------------------------------
def train_step(online_net, target_net, buffer, optimizer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)

    # Q(s, a) predicted by the online network for the actions we actually took
    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: r + gamma * max_a' Q_target(s', a')   (no future if done)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        targets = rewards + gamma * next_q * (1 - dones)

    # Huber loss is more stable than MSE for DQN
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()                                            # backpropagation
    nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)    # avoid huge updates
    optimizer.step()                                           # update online network
    return loss.item()


# ---------------------------------------------------------------------------
# 6. Saving results (log, plot, model)
# ---------------------------------------------------------------------------
def save_log(log, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)


def save_plot(log, path, title):
    episodes = [row["episode"] for row in log]
    rewards = [row["reward"] for row in log]
    lengths = [row["episode_length"] for row in log]
    window = 20
    moving_avg = [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    ax1.plot(episodes, rewards, alpha=0.4, label="Episode reward")
    ax1.plot(episodes, moving_avg, label=f"Moving average ({window} episodes)")
    ax1.set_ylabel("Reward")
    ax1.set_title(title)
    ax1.legend()
    ax2.plot(episodes, lengths, color="tab:green")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode length (steps)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# 7. Main training loop
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (4, 84, 84)

    # online network (learns) and target network (gives stable targets)
    online_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, obs_shape)

    out_dir = os.path.join("models", args.member, args.exp)
    os.makedirs(out_dir, exist_ok=True)

    state, _ = env.reset(seed=args.seed)
    log = []
    episode, episode_reward, episode_length = 0, 0.0, 0

    for step in range(1, args.total_steps + 1):
        # choose and play an action
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(state, epsilon, online_net, n_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # store the experience (reward clipped to -1, 0, +1 for stable learning)
        buffer.add(state, action, np.sign(reward), next_state, float(terminated))
        state = next_state
        episode_reward += reward
        episode_length += 1

        # learn from a random batch every 4 steps (after the buffer has some data)
        if step > args.learning_starts and step % 4 == 0:
            train_step(online_net, target_net, buffer, optimizer, args.batch_size, args.gamma, device)

        # copy online weights into the target network every N steps
        if step % args.target_update == 0:
            target_net.load_state_dict(online_net.state_dict())

        if done:
            episode += 1
            log.append({"episode": episode, "total_steps": step, "episode_length": episode_length,
                        "reward": episode_reward, "epsilon": round(epsilon, 3)})
            if episode % 10 == 0:
                avg = np.mean([row["reward"] for row in log[-10:]])
                print(f"Episode {episode} | Step {step} | Avg reward (last 10): {avg:.2f} | Epsilon: {epsilon:.3f}")
            state, _ = env.reset()
            episode_reward, episode_length = 0.0, 0

    env.close()

    # save everything for this experiment
    torch.save(online_net.state_dict(), os.path.join(out_dir, "dqn_model.pth"))
    save_log(log, os.path.join(out_dir, "training_log.csv"))
    save_plot(log, os.path.join(out_dir, "training_plot.png"), f"{args.member} - {args.exp}")
    final_avg = float(np.mean([row["reward"] for row in log[-20:]]))
    summary = {**vars(args), "episodes": episode, "avg_reward_last_20": round(final_avg, 2),
               "best_episode_reward": max(row["reward"] for row in log)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Avg reward (last 20 episodes): {final_avg:.2f}")
    print(f"Saved model, log and plot in: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari BeamRider")
    parser.add_argument("--member", type=str, default="member1", help="your name (folder name)")
    parser.add_argument("--exp", type=str, default="exp1", help="experiment name, e.g. exp1")
    # the 6 hyperparameters we tune
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=100_000, help="steps to go from eps-start to eps-end")
    # other settings (keep the same for all experiments)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
