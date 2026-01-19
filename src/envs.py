from typing import Dict, Any, Optional
import gymnasium as gym
import highway_env  # noqa: F401  # Registers the environment


# =========================
# Environment Configuration
# =========================
LANES_COUNT: int = 3
VEHICLES_COUNT: int = 30
EPISODE_DURATION: int = 40

COLLISION_REWARD: float = -5.0
HIGH_SPEED_REWARD: float = 0.5
RIGHT_LANE_REWARD: float = 0.1
NORMALIZE_REWARD: bool = True


def make_highway_env(render_mode: Optional[str] = None) -> gym.Env:
    """
    Create and configure the Highway-Fast environment.

    The agent controls a vehicle on a multi-lane highway and
    learns to drive safely while maintaining high speed and
    avoiding collisions.

    Args:
        render_mode (Optional[str]):
            - None: No rendering (training)
            - "human": Real-time visualization
            - "rgb_array": Frame-based rendering (video recording)

    Returns:
        gym.Env: Configured Gymnasium environment.
    """
    config: Dict[str, Any] = {
        "lanes_count": LANES_COUNT,
        "vehicles_count": VEHICLES_COUNT,
        "duration": EPISODE_DURATION,
        "collision_reward": COLLISION_REWARD,
        "high_speed_reward": HIGH_SPEED_REWARD,
        "right_lane_reward": RIGHT_LANE_REWARD,
        "normalize_reward": NORMALIZE_REWARD,
    }

    env: gym.Env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config=config,
    )

    return env
