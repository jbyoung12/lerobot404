"""
Receive leader-arm poses inside the simulator.

The leader arm is plugged into the operator's laptop; the simulator runs on the
GPU machine. Only the leader's six joint angles have to cross that gap — a few
dozen bytes at 30 Hz — while the frames, the physics and the recording all stay
on the machine, which is why the control loop lives there.

The laptop side is `tools/leader_bridge.py`. It posts to the port this module
listens on, published in `antioch.yaml`.

Deliberately a plain HTTP server: the sim image has no message broker, and the
one thing that must not happen is a teleop session dying because a dependency
was missing on the machine.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

TELEOP_PORT = 8765


class LeaderFeed:
    """
    Hold the most recent leader pose posted by the laptop.

    Only the latest pose matters. A queue would let the arm fall behind the
    operator's hand and then replay stale motion, which feels like the sim
    fighting you; dropping everything but the newest keeps it honest.
    """

    def __init__(self, port: int = TELEOP_PORT) -> None:
        """
        Start listening.

        :param port: TCP port to bind, matching the published port.
        """

        self._lock = threading.Lock()
        self._pose: list[float] | None = None
        self._recording = False
        self._episode = 0
        self._finished = False
        self._received = 0
        self._server = HTTPServer(("0.0.0.0", port), _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def submit(self, payload: dict[str, Any]) -> None:
        """
        Accept one posted frame from the bridge.

        :param payload: Decoded JSON body.
        """

        with self._lock:
            joints = payload.get("joints")
            if isinstance(joints, list) and len(joints) == 6:
                self._pose = [float(value) for value in joints]
                self._received += 1
            self._recording = bool(payload.get("recording", self._recording))
            self._episode = int(payload.get("episode", self._episode))
            self._finished = bool(payload.get("finished", self._finished))

    def snapshot(self) -> tuple[list[float] | None, bool, int, bool]:
        """
        Read the current pose and session flags.

        :return: Pose, whether an episode is being recorded, its index, and
            whether the operator has finished the session.
        """

        with self._lock:
            return (None if self._pose is None else list(self._pose)), self._recording, self._episode, self._finished

    @property
    def received(self) -> int:
        """
        Count posts accepted, for diagnosing a silent bridge.

        :return: Number of poses received.
        """

        with self._lock:
            return self._received

    def close(self) -> None:
        """
        Stop listening.
        """

        self._server.shutdown()
        self._server.server_close()


def _make_handler(feed: LeaderFeed) -> type[BaseHTTPRequestHandler]:
    """
    Build a request handler bound to one feed.

    :param feed: Feed to deliver posted poses to.
    :return: Handler class.
    """

    class Handler(BaseHTTPRequestHandler):
        """
        Accept posted poses and report the feed's state.
        """

        def do_POST(self) -> None:  # noqa: N802 - the base class names it
            """
            Take one pose from the bridge.
            """

            length = int(self.headers.get("Content-Length", 0))
            try:
                feed.submit(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, TypeError):
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - the base class names it
            """
            Report that the sim is listening, so the bridge can check first.
            """

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"received": feed.received}).encode())

        def log_message(self, *_args: Any) -> None:
            """
            Stay quiet: one line per pose at 30 Hz drowns the run's output.
            """

    return Handler
