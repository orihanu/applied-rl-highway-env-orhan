from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from envs import make_highway_env


# =========================
# Training Configuration
# =========================
TOTAL_TIMESTEPS: int = 20_000
LOG_DIR = Path("logs/dqn_highway_half")
MODEL_PATH = Path("models/dqn_half_20k.zip")

LEARNING_RATE: float = 1e-4
BUFFER_SIZE: int = 100_000
LEARNING_STARTS: int = 1_000
BATCH_SIZE: int = 64
GAMMA: float = 0.99
TRAIN_FREQ: int = 4
TARGET_UPDATE_INTERVAL: int = 1_000

EVAL_FREQ: int = 5_000
N_EVAL_EPISODES: int = 5


def make_monitored_env(render_mode: Optional[str] = None):
    """
    Create a monitored Highway environment for training or evaluation.
    """
    env = make_highway_env(render_mode=render_mode)
    return Monitor(env)


def train_half_dqn(
    total_timesteps: int = TOTAL_TIMESTEPS,
) -> None:
    """
    Train a partially converged (half-trained) DQN agent.

    This agent is trained for a limited number of timesteps to
    demonstrate intermediate learning behavior.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = make_monitored_env()
    eval_env = make_monitored_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR / "best",
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

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

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    model.save(MODEL_PATH)

    env.close()
    eval_env.close()


def main() -> None:
    train_half_dqn()


if __name__ == "__main__":
    main()
