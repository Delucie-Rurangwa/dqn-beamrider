"""
play.py - Load the trained DQN model and watch it play BeamRider.

Example:
    uv run play.py                                  # uses dqn_model.pth
    uv run play.py --model models/Larissa/exp3/dqn_model.pth --episodes 3
"""

import argparse

import numpy as np
import torch

from train import QNetwork, make_env  # same environment + same network as training

FIRE = 1  # FIRE starts the game (safe to press in any Atari game)


def play(model_path, episodes):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    env = make_env(render_mode="human")
    n_actions = env.action_space.n

    # rebuild the same network and load the trained weights
    q_net = QNetwork(n_actions).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()  

    for episode in range(1, episodes + 1):
        state, info = env.reset()
        state, _, _, _, info = env.step(FIRE)  # start the game
        lives = info["lives"]
        total_reward, done = 0.0, False

        while not done:
            with torch.no_grad():
                state_t = torch.tensor(np.array(state), device=device).unsqueeze(0)
                action = int(q_net(state_t).argmax(dim=1).item())

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # after losing a life, press FIRE so the game does not freeze
            if info["lives"] < lives and not done:
                lives = info["lives"]
                state, _, terminated, truncated, info = env.step(FIRE)
                done = terminated or truncated

        print(f"Episode {episode}: total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the trained DQN agent play BeamRider")
    parser.add_argument("--model", type=str, default="dqn_model.pth")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    play(args.model, args.episodes)
