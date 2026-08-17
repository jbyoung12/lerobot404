"""
Stream an SO-101 leader arm from this laptop into the simulator.

Run this on the machine the leader is plugged into, with the sim's teleop port
published (see `antioch.yaml`). It reads the leader over USB and posts its six
joint angles to the simulator, which is doing the physics, the camera and the
recording.

    ~/lerobot-env/bin/python tools/leader_bridge.py --port /dev/tty.usbmodem1101

Find the serial port with `lerobot-find-port`, and calibrate the arm once with
`lerobot-calibrate` before the numbers mean anything.

Press Enter to start an episode, Enter again to end it. The simulator records
only while an episode is open, so the fumbling between takes is not trained on.

This deliberately runs under the LeRobot environment, not the sim project's:
LeRobot is not installed in the sim image and does not need to be.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

LEADER_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def parse_arguments() -> argparse.Namespace:
    """
    Read the command line.

    :return: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial port the leader is on, from `lerobot-find-port`")
    parser.add_argument("--id", default="so101_leader", help="Calibration id used by `lerobot-calibrate`")
    parser.add_argument("--sim", default="http://localhost:8765", help="Published teleop port of the simulator")
    parser.add_argument("--hz", type=float, default=30.0, help="Rate to read and post at")
    parser.add_argument("--dry-run", action="store_true", help="Print poses instead of posting them")
    return parser.parse_args()


def connect_leader(port: str, identifier: str):
    """
    Open the leader arm.

    :param port: Serial port.
    :param identifier: Calibration id.
    :return: A connected teleoperator.
    :raises SystemExit: If LeRobot is missing or the arm will not open.
    """

    try:
        from lerobot.teleoperators.so_leader.so_leader import SOLeader, SOLeaderTeleopConfig
    except ImportError:
        raise SystemExit("run this with the LeRobot environment, e.g. ~/lerobot-env/bin/python") from None
    leader = SOLeader(SOLeaderTeleopConfig(port=port, id=identifier, use_degrees=True))
    try:
        leader.connect(calibrate=False)
    except Exception as error:  # noqa: BLE001 - the message is the useful part
        raise SystemExit(f"could not open the leader on {port}: {error}\nrun `lerobot-find-port`, then `lerobot-calibrate`") from None
    return leader


def read_pose(leader) -> list[float]:
    """
    Read the leader's six joints in its own units.

    Degrees for the body joints, 0-100 for the gripper. Converting to the
    simulator's radians happens on the simulator side, in `sim.leader_to_joints`,
    so the mapping lives next to the joint limits it has to respect.

    :param leader: Connected teleoperator.
    :return: Six values in leader units.
    """

    action = leader.get_action()
    return [float(action.get(f"{name}.pos", 0.0)) for name in LEADER_JOINTS]


def post(url: str, payload: dict) -> bool:
    """
    Send one frame to the simulator.

    :param url: Simulator teleop endpoint.
    :param payload: Body to send.
    :return: Whether it was accepted.
    """

    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=1.0):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


class Session:
    """
    Track which episode is open, driven by the operator pressing Enter.
    """

    def __init__(self) -> None:
        """
        Start with no episode open.
        """

        self.recording = False
        self.episode = 0
        self.finished = False
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        """
        Toggle recording every time the operator presses Enter.
        """

        while not self.finished:
            line = sys.stdin.readline()
            if line.strip().lower() in {"q", "quit"}:
                self.finished = True
                print("\nsession finished")
                return
            self.recording = not self.recording
            if self.recording:
                print(f"\n▶ recording episode {self.episode} — Enter to end it")
            else:
                self.episode += 1
                print(f"\n■ episode {self.episode - 1} closed — Enter to start the next, or q to finish")


def main() -> None:
    """
    Read the leader and stream it until the operator finishes.
    """

    arguments = parse_arguments()
    leader = connect_leader(arguments.port, arguments.id)
    print(f"leader open on {arguments.port}")
    if not arguments.dry_run and not post(f"{arguments.sim}/", {"joints": read_pose(leader)}):
        print(f"warning: nothing is listening on {arguments.sim} — start the teleop scenario first")
    print("Enter starts an episode, Enter again ends it, q finishes the session")

    session = Session()
    period = 1.0 / arguments.hz
    dropped = 0
    sent = 0
    try:
        while not session.finished:
            began = time.perf_counter()
            pose = read_pose(leader)
            payload = {"joints": pose, "recording": session.recording, "episode": session.episode, "finished": session.finished}
            if arguments.dry_run:
                print(" ".join(f"{value:7.2f}" for value in pose), end="\r")
            elif post(f"{arguments.sim}/", payload):
                sent += 1
            else:
                dropped += 1
            time.sleep(max(0.0, period - (time.perf_counter() - began)))
        post(f"{arguments.sim}/", {"joints": read_pose(leader), "recording": False, "episode": session.episode, "finished": True})
    finally:
        leader.disconnect()
        print(f"\nsent {sent} poses, dropped {dropped}")


if __name__ == "__main__":
    main()
