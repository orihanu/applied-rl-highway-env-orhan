from typing import Optional
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from envs import make_highway_env


# =========================
# Hyperparameters
# =========================
LEARNING_RATE: float = 1e-4
BUFFER_SIZE: int = 100_000
LEARNING_STARTS: int = 1_000
BATCH_SIZE: int = 64
GAMMA: float = 0.99
TRAIN_FREQ: int = 4
TARGET_UPDATE_INTERVAL: int = 1_000


def make_monitored_env(render_mode: Optional[str] = None):
    """
    Create a monitored Highway environment.

    Args:
        render_mode (Optional[str]): Rendering mode for the environment.

    Returns:
        gym.Env: Wrapped environment with Monitor.
    """
    env = make_highway_env(render_mode=render_mode)
    return Monitor(env)


def create_untrained_model(
    model_path: str = "models/dqn_untrained",
) -> None:
    """
    Create and save an untrained DQN agent.

    This agent has randomly initialized weights and has not
    interacted with the environment via learning.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = make_monitored_env(render_mode=None)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        verbose=1,
    )

    # No training is performed for the untrained agent
    model.save(model_path)
    env.close()


def main() -> None:
    create_untrained_model()


if __name__ == "__main__":
    main()
