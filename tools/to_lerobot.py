"""
Convert a recorded archive into a LeRobot dataset.

Run this on the laptop, under the LeRobot environment — LeRobot is not
installed in the sim image and does not need to be:

    ~/lerobot-env/bin/python tools/to_lerobot.py episodes.npz --repo-id local/vial_pick

Then train:

    lerobot-train --policy.type=act --dataset.repo_id=local/vial_pick \\
                  --policy.n_action_steps=20 --steps=15000 --batch_size=8

`n_action_steps` is worth setting explicitly. ACT's default predicts and then
executes a hundred actions before looking at the camera again, which at 10 Hz
is ten seconds flown blind; twenty gives it two seconds and then a fresh look.

The conversion asserts that actions differ from states before writing anything.
Training on a dataset where they match produces a policy that predicts where it
already is and freezes mid-task, and the failure looks like broken hardware
rather than a broken label.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

CAMERA_KEY = "observation.images.scene"


def parse_arguments() -> argparse.Namespace:
    """
    Read the command line.

    :return: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="The .npz downloaded from a recording run")
    parser.add_argument("--repo-id", default="local/vial_pick", help="Dataset id LeRobot will train against")
    parser.add_argument("--root", type=Path, default=None, help="Where to write it; defaults to LeRobot's cache")
    parser.add_argument("--robot-type", default="so101", help="Robot type recorded in the metadata")
    parser.add_argument("--images", default="video", choices=("video", "image"), help="Store frames as encoded video or as loose images")
    return parser.parse_args()


def read_archive(path: Path) -> dict:
    """
    Load a recorded archive without needing the sim project on the path.

    :param path: Archive path.
    :return: Arrays plus decoded frames.
    """

    from PIL import Image

    with np.load(path, allow_pickle=False) as data:
        blob = data["image_bytes"].tobytes()
        offsets = data["image_offsets"]
        images = [np.asarray(Image.open(io.BytesIO(blob[offsets[i] : offsets[i + 1]])).convert("RGB")) for i in range(len(offsets) - 1)]
        return {
            "state": data["state"],
            "action": data["action"],
            "images": images,
            "episode_index": data["episode_index"],
            "task": [str(value) for value in data["task"]],
            "fps": float(data["fps"]),
        }


def main() -> None:
    """
    Convert one archive and report what was written.

    :raises SystemExit: If LeRobot is missing or the labels are unusable.
    """

    arguments = parse_arguments()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise SystemExit("run this with the LeRobot environment, e.g. ~/lerobot-env/bin/python") from None

    recorded = read_archive(arguments.archive)
    state, action = recorded["state"], recorded["action"]
    gap = float(np.abs(action - state).max())
    if gap <= 1e-3:
        raise SystemExit(f"actions and states are identical (max gap {gap:.6f}); the recording labelled measured positions as actions")
    print(f"{len(state)} frames, {len(set(recorded['episode_index'].tolist()))} episodes, action/state gap {gap:.4f} rad")

    joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    height, width, channels = recorded["images"][0].shape
    features = {
        "observation.state": {"dtype": "float32", "shape": (state.shape[1],), "names": joints},
        "action": {"dtype": "float32", "shape": (action.shape[1],), "names": joints},
        CAMERA_KEY: {"dtype": arguments.images, "shape": (height, width, channels), "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=arguments.repo_id,
        fps=int(round(recorded["fps"])),
        features=features,
        root=arguments.root,
        robot_type=arguments.robot_type,
        use_videos=arguments.images == "video",
    )

    episodes = recorded["episode_index"]
    written = 0
    for episode in sorted(set(episodes.tolist())):
        rows = np.flatnonzero(episodes == episode)
        for row in rows:
            dataset.add_frame(
                {
                    "observation.state": state[row],
                    "action": action[row],
                    CAMERA_KEY: recorded["images"][row],
                    "task": recorded["task"][row],
                }
            )
        dataset.save_episode()
        written += 1
        print(f"  episode {episode}: {len(rows)} frames")

    root = getattr(dataset, "root", arguments.root)
    print(f"\nwrote {written} episodes to {root}")
    print(f"\nnow train:\n  lerobot-train --policy.type=act --dataset.repo_id={arguments.repo_id} \\\n                --policy.n_action_steps=20 --steps=15000 --batch_size=8")


if __name__ == "__main__":
    sys.exit(main())
