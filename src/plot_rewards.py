from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = Path("logs/dqn_highway_full")
OUTPUT_PATH = Path("media/reward_curve.png")
FIGURE_SIZE: Tuple[int, int] = (8, 5)
DPI = 150


def load_evaluation_data(log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load evaluation timesteps and mean rewards from Stable-Baselines3 logs.

    :param log_dir: Directory containing evaluations.npz
    :return: (timesteps, mean_rewards)
    """
    eval_file = log_dir / "evaluations.npz"

    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    data = np.load(eval_file)
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)

    return timesteps, mean_rewards


def plot_rewards() -> None:
    """
    Plot Reward vs Timesteps and save the figure for the README.
    """
    timesteps, rewards = load_evaluation_data(LOG_DIR)

    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(timesteps, rewards, linewidth=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Training Performance: Reward vs Timesteps")
    plt.grid(True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close()


def main() -> None:
    plot_rewards()


if __name__ == "__main__":
    main()
