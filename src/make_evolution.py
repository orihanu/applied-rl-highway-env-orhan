from pathlib import Path
from typing import List

import imageio.v2 as imageio


MEDIA_DIR = Path("media")
OUTPUT_GIF = MEDIA_DIR / "evolution.gif"
FPS = 15


def load_gif_frames(path: Path) -> List:
    """
    Load all frames from a GIF file.

    :param path: Path to the GIF file
    :return: List of image frames
    """
    if not path.exists():
        raise FileNotFoundError(f"GIF not found: {path}")

    return imageio.mimread(path)


def create_evolution_gif() -> None:
    """
    Concatenate untrained, half-trained, and fully-trained GIFs
    into a single evolution GIF.
    """
    gif_paths = [
        MEDIA_DIR / "untrained.gif",
        MEDIA_DIR / "half_trained.gif",
        MEDIA_DIR / "fully_trained.gif",
    ]

    all_frames: List = []

    for gif_path in gif_paths:
        frames = load_gif_frames(gif_path)
        all_frames.extend(frames)

    imageio.mimsave(OUTPUT_GIF, all_frames, fps=FPS)


def main() -> None:
    create_evolution_gif()


if __name__ == "__main__":
    main()
