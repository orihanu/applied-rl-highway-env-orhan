from pathlib import Path
from typing import List

import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import DQN

from envs import make_highway_env


# =========================
# Constants
# =========================
MEDIA_DIR = Path("media")
FPS = 12
DEFAULT_STEPS = 200


def record_agent(
    model_path: Path,
    output_path: Path,
    steps: int = DEFAULT_STEPS,
) -> None:
    """
    Record an agent acting in the environment and save the result as a GIF.

    Args:
        model_path (Path): Path to the trained DQN model.
        output_path (Path): Output GIF file path.
        steps (int): Number of environment steps to record.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = make_highway_env(render_mode="rgb_array")
    model = DQN.load(model_path, env=env)

    obs, _ = env.reset()
    frames: List[np.ndarray] = []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

        if done or truncated:
            obs, _ = env.reset()

    env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=FPS, loop=0)


def main() -> None:
    record_agent(
        model_path=Path("models/dqn_untrained.zip"),
        output_path=MEDIA_DIR / "untrained.gif",
    )

    record_agent(
        model_path=Path("models/dqn_half_20k.zip"),
        output_path=MEDIA_DIR / "half_trained.gif",
    )

    record_agent(
        model_path=Path("models/dqn_final_60k.zip"),
        output_path=MEDIA_DIR / "fully_trained.gif",
    )


if __name__ == "__main__":
    main()
