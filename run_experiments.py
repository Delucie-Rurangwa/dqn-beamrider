"""
run_experiments.py - Run the 15 hyperparameter experiments of one member.
Works on Windows, Mac and Linux.

Example:
    uv run run_experiments.py --member delucie
    uv run run_experiments.py --member favor
    uv run run_experiments.py --member delucie --start 11   # only run exp11 to exp15
"""

import argparse
import subprocess
import sys

EXPERIMENTS = {

    "delucie": {
        "folder": "Delucie_Rurangwa",
        "configs": [
            (0.0001, 0.99, 32, 1.0, 0.05, 50000),
            (0.0005, 0.99, 32, 1.0, 0.05, 50000),
            (0.001, 0.99, 32, 1.0, 0.05, 50000),
            (0.00005, 0.99, 32, 1.0, 0.05, 50000),
            (0.0001, 0.95, 32, 1.0, 0.05, 50000),
            (0.0001, 0.90, 32, 1.0, 0.05, 50000),
            (0.0001, 0.999, 32, 1.0, 0.05, 50000),
            (0.0001, 0.99, 64, 1.0, 0.05, 50000),
            (0.0001, 0.99, 16, 1.0, 0.05, 50000),
            (0.00025, 0.99, 64, 1.0, 0.01, 75000),
            # extra experiments 11-15 (built on what worked best: lr 0.0001, gamma 0.90)
            (0.0001, 0.85, 32, 1.0, 0.05, 50000),
            (0.0001, 0.90, 64, 1.0, 0.05, 50000),
            (0.00025, 0.90, 32, 1.0, 0.05, 50000),
            (0.0001, 0.90, 32, 1.0, 0.01, 50000),
            (0.00005, 0.90, 32, 1.0, 0.05, 50000),
        ],
    },

    "favor": {
        "folder": "Favor_Ekep-Obasi_Ayuk",
        "configs": [
            (0.0001, 0.99, 32, 1.0, 0.10, 50000),
            (0.0001, 0.99, 32, 1.0, 0.01, 50000),
            (0.0001, 0.99, 32, 1.0, 0.05, 25000),
            (0.0001, 0.99, 32, 1.0, 0.05, 90000),
            (0.0001, 0.99, 32, 0.5, 0.05, 50000),
            (0.0001, 0.99, 32, 1.0, 0.05, 10000),
            (0.00025, 0.98, 32, 1.0, 0.02, 40000),
            (0.0001, 0.99, 128, 1.0, 0.05, 50000),
            (0.0005, 0.95, 64, 1.0, 0.10, 25000),
            (0.00025, 0.99, 32, 1.0, 0.01, 60000),
            # extra experiments 11-15
            (0.0001, 0.99, 32, 1.0, 0.05, 75000),
            (0.0001, 0.99, 32, 0.8, 0.05, 50000),
            (0.0001, 0.90, 32, 1.0, 0.01, 25000),
            (0.0001, 0.95, 32, 1.0, 0.02, 40000),
            (0.0001, 0.99, 32, 1.0, 0.20, 50000),
        ],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--member", choices=EXPERIMENTS.keys(), required=True)
parser.add_argument("--start", type=int, default=1, help="experiment number to start from")
parser.add_argument("--total-steps", type=int, default=100_000)
args = parser.parse_args()

member = EXPERIMENTS[args.member]
for number, (lr, gamma, batch, eps_start, eps_end, eps_decay) in enumerate(member["configs"], start=1):
    if number < args.start:
        continue
    print(f"\n===== {member['folder']} - exp{number} =====")
    subprocess.run([
        sys.executable, "train.py",
        "--member", member["folder"], "--exp", f"exp{number}",
        "--lr", str(lr), "--gamma", str(gamma), "--batch-size", str(batch),
        "--eps-start", str(eps_start), "--eps-end", str(eps_end), "--eps-decay", str(eps_decay),
        "--total-steps", str(args.total_steps),
    ], check=True)
