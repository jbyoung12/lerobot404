"""
Record demonstration episodes inside the simulator.

LeRobot is not installed in the sim image and does not need to be: this writes
one portable ``.npz`` per run, which converts into a LeRobot dataset on the
laptop where LeRobot already lives. Keeping the conversion off the GPU machine
also means the dataset format can be changed later without re-recording.

The action/state distinction is the whole point of the file:

* ``action`` is what the arm was *told* to do — the leader arm's pose, or the
  scripted expert's command. It is the label the policy learns to predict.
* ``state`` is where the arm actually *got*. It is the observation.

They differ, and recording the second as the first trains a policy to predict
where it already is, which at inference makes it freeze mid-task holding the
object. Every consumer of this file should assert they differ before training.

Images are stored as JPEG bytes end to end in one buffer with an offset index,
because a few hundred raw frames is gigabytes and a compressed ``.npz`` of raw
pixels barely helps.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

JPEG_QUALITY = 80


class EpisodeRecorder:
    """
    Accumulate the frames of one or more episodes, then write them out.
    """

    def __init__(self, task: str, fps: float) -> None:
        """
        Start an empty recording.

        :param task: Natural-language task, stored per frame for the policy to
            condition on, e.g. ``"move A3 to C5"``.
        :param fps: Control rate the episodes were recorded at.
        """

        self._task = task
        self._fps = float(fps)
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._images: list[bytes] = []
        self._episode_index: list[int] = []
        self._frame_index: list[int] = []
        self._tasks: list[str] = []
        self._episode_tasks: list[str] = []
        self._episode_success: list[bool] = []
        self._episode = -1
        self._frames_this_episode = 0

    def start_episode(self, task: str | None = None) -> None:
        """
        Begin a new episode.

        :param task: Task string for this episode, or ``None`` to reuse the
            recorder's default.
        """

        self._episode += 1
        self._frames_this_episode = 0
        self._episode_tasks.append(task or self._task)

    def add(self, state: np.ndarray, action: np.ndarray, image: np.ndarray) -> None:
        """
        Record one control tick.

        Call it *before* commanding the action, so the state is the observation
        the action was chosen from rather than its result.

        :param state: Measured joint angles, shape ``(6,)``.
        :param action: Commanded joint angles, shape ``(6,)``.
        :param image: Camera frame, shape ``(H, W, 3)`` uint8.
        :raises RuntimeError: If no episode has been started.
        """

        if self._episode < 0:
            raise RuntimeError("call start_episode() before add()")
        self._states.append(np.asarray(state, dtype=np.float32).reshape(-1))
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self._images.append(_encode_jpeg(image))
        self._episode_index.append(self._episode)
        self._frame_index.append(self._frames_this_episode)
        self._tasks.append(self._episode_tasks[self._episode])
        self._frames_this_episode += 1

    def end_episode(self, success: bool) -> None:
        """
        Close the current episode and record whether the scorer passed it.

        :param success: Whether the episode achieved its goal.
        """

        self._episode_success.append(bool(success))

    @property
    def episodes(self) -> int:
        """
        Count the episodes recorded so far.

        :return: Number of episodes started.
        """

        return self._episode + 1

    @property
    def frames(self) -> int:
        """
        Count the frames recorded so far.

        :return: Total frames across all episodes.
        """

        return len(self._states)

    def label_gap(self) -> float:
        """
        Measure how far actions differ from states.

        The single highest-value number in the file: if it is zero, the labels
        are wrong and nothing trained on them will work.

        :return: Largest absolute difference in radians, or 0.0 if empty.
        """

        if not self._states:
            return 0.0
        return float(np.abs(np.asarray(self._actions) - np.asarray(self._states)).max())

    def save(self, path: str | Path) -> Path:
        """
        Write every recorded episode to one compressed archive.

        :param path: Destination ``.npz`` path.
        :return: The path written.
        :raises RuntimeError: If nothing has been recorded.
        """

        if not self._states:
            raise RuntimeError("nothing recorded")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.zeros(len(self._images) + 1, dtype=np.int64)
        for index, blob in enumerate(self._images):
            offsets[index + 1] = offsets[index] + len(blob)
        np.savez_compressed(
            destination,
            state=np.asarray(self._states, dtype=np.float32),
            action=np.asarray(self._actions, dtype=np.float32),
            image_bytes=np.frombuffer(b"".join(self._images), dtype=np.uint8),
            image_offsets=offsets,
            episode_index=np.asarray(self._episode_index, dtype=np.int64),
            frame_index=np.asarray(self._frame_index, dtype=np.int64),
            task=np.asarray(self._tasks),
            episode_task=np.asarray(self._episode_tasks),
            episode_success=np.asarray(self._episode_success, dtype=bool),
            fps=np.asarray(self._fps, dtype=np.float64),
        )
        return destination


def _encode_jpeg(image: np.ndarray) -> bytes:
    """
    Compress one frame.

    :param image: ``(H, W, 3)`` uint8 RGB.
    :return: JPEG bytes.
    """

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def load(path: str | Path) -> dict[str, Any]:
    """
    Read an archive back, decoding the frames.

    :param path: Archive written by :meth:`EpisodeRecorder.save`.
    :return: Arrays plus ``images`` as a list of ``(H, W, 3)`` uint8 frames.
    """

    from PIL import Image

    with np.load(Path(path), allow_pickle=False) as data:
        blob = data["image_bytes"].tobytes()
        offsets = data["image_offsets"]
        images = [np.asarray(Image.open(io.BytesIO(blob[offsets[i] : offsets[i + 1]])).convert("RGB")) for i in range(len(offsets) - 1)]
        return {
            "state": data["state"],
            "action": data["action"],
            "images": images,
            "episode_index": data["episode_index"],
            "frame_index": data["frame_index"],
            "task": [str(value) for value in data["task"]],
            "episode_task": [str(value) for value in data["episode_task"]],
            "episode_success": data["episode_success"],
            "fps": float(data["fps"]),
        }
