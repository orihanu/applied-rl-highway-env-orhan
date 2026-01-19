from time import sleep
from typing import Optional

from envs import make_highway_env


def run_random_agent(render_mode: Optional[str] = "human") -> None:
    """
    Run a random (untrained) agent in the Highway environment.

    This script is used only for visualization and debugging purposes.
    The agent samples random actions from the action space and does not learn.
    """
    env = make_highway_env(render_mode=render_mode)

    obs, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)

    # Pause briefly so the final frame remains visible
    sleep(2)
    env.close()


def main() -> None:
    run_random_agent()


if __name__ == "__main__":
    main()
