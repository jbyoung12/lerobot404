"""
The one adapter over the simulator. Everything else imports this.

Five people calling Isaac directly means five places to fix when an API turns
out to differ from what we assumed. This module is that one place.

Two backends, chosen automatically:

* ``isaac`` — the real simulator. Only importable inside an Antioch run, on
  the GPU machine.
* ``mock`` — hardcoded but *behaving* values, importable on a laptop with no
  GPU and no Isaac. The gripper still refuses to close on an object it is
  holding, so monitor and retry logic can be written and tested against it
  before the real scene exists.

Set ``VIAL_SIM=mock`` to force the mock even inside the simulator.

The contract, which is frozen:

    reset(scenario)   -> None
    frame()           -> (480, 640, 3) uint8, wrist camera
    joint_state()     -> (6,) float32 radians, MEASURED
    set_targets(q)    -> None, (6,) float32 radians, COMMANDED
    object_poses()    -> dict, ground truth, EVAL ORACLE ONLY
    score()           -> bool, did the scorer pass this episode

One deliberate difference from a real robot: ``set_targets`` also advances the
simulation by one control tick. On hardware time passes by itself; in a
simulator something has to step it, and the alternative was a seventh function
everybody would forget to call. The loop shape is unchanged::

    for action in chunk:
        sim.set_targets(action)

Perception comes from :func:`frame`. :func:`object_poses` is ground truth for
scoring only — a planner that reads object positions out of the simulator has
not solved perception, and a judge will ask.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random

import numpy as np

import so101

# ── Contract ──────────────────────────────────────────────────────────────────

CONTROL_HZ = 10.0
PHYSICS_HZ = 60.0
STEPS_PER_TICK = int(round(PHYSICS_HZ / CONTROL_HZ))
IMAGE_SHAPE = (480, 640, 3)
JOINT_COUNT = 6
# Joint index of the gripper, for the monitor's empty-hand rule
GRIPPER = so101.GRIPPER_INDEX
HOME = (0.0,) * JOINT_COUNT

# The vocabulary the whole system shares. The planner's prompt lists exactly
# these, the scorer accepts exactly these, so a name can never be agreed on in
# one module and misspelled in another.
OBJECT_NAMES = ("red vial", "blue vial", "green vial")
DISTRACTOR_NAMES = ("yellow cube", "blue sphere")
# Wells are addressed like a real rack: row letter, column number. "A" is the
# row nearest the arm, so "move A3 to C2" reaches over the front row into the
# back one, which is the situation the whole rearrangement idea rests on.
RACK_ROW_NAMES = ("A", "B", "C", "D", "E")
RACK_COLUMN_COUNT = 10
SLOT_NAMES = tuple(f"{row}{column + 1}" for row in RACK_ROW_NAMES for column in range(RACK_COLUMN_COUNT))
SCENARIOS = ("train", "eval", "ambiguous")

# ── Scene layout ──────────────────────────────────────────────────────────────

# A 2 mL HPLC vial, matching the amber blue-capped ones on the bench. Small
# enough that the grasp only works at the very tip of the fingers, which is
# why `so101` puts the tool centre there.
VIAL_DIAMETER = 0.012
VIAL_HEIGHT = 0.032
DISTRACTOR_SIZE = 0.024
OBJECT_MASS = 0.02
# Where objects may be spawned, as a grid over (reach, bearing) cells split
# into training and held-out halves. The split is by cell, so "held out" means
# a position the policy has never seen rather than a rounding of one it has.
# The outer reach is bounded by the arm's own top-down envelope: pointing the
# tool straight down costs height, and past about 0.24 m there is not enough
# left to lift a vial clear of the rack.
SPAWN_REACHES = (0.165, 0.185, 0.205, 0.225)
SPAWN_BEARINGS_DEG = (8.0, 24.0, 40.0, 56.0)
HELD_OUT_CELLS = ((0, 1), (1, 3), (2, 0), (3, 2))
SPAWN_JITTER_M = 0.008
SPAWN_JITTER_DEG = 3.0
# The gripper is roughly 80 mm across at the servo body, so two objects closer
# than this cannot both survive a grasp: descending onto one knocks the other,
# which poisons the demonstration it is supposed to be teaching.
MIN_OBJECT_SEPARATION = 0.085

# A 5 x 10 well rack, like the clear one on the bench. Only the front row is
# addressable: ten destinations is already more than a demo needs, and the far
# rows sit past the arm's top-down reach, so they are scenery.
# A real 5 x 10 rack at 20 mm pitch. The gripper fits only because it barely
# opens: the fingers already rest 15.8 mm apart and the vial is 12 mm, so the
# approach clearance is 3 mm rather than the 14 mm used for loose objects on a
# table. Opened wide, the jaw would swing into the neighbouring well.
RACK_BEARING_DEG = -25.0
RACK_REACH = 0.145
RACK_PLATE_THICKNESS = 0.006
ROW_PITCH = 0.020
COLUMN_PITCH = 0.020
# Approach gap beside a vial standing in the rack. Measured: at 3 mm the
# fingers clip the vial on the way down, because the arm's own settled
# tracking error is about 3 mm at the tool. Wells are therefore filled
# alternately, which buys the room to open to a clearance wider than the
# accuracy — the rack is still the full 5 x 10, just not densely packed, which
# is how a part-used rack looks anyway.
WELL_GRIP_CLEARANCE = 0.010
USABLE_ROWS = ("A", "C", "E")
USABLE_COLUMNS = (1, 3, 5, 7, 9)
WELL_POST = 0.004
# Shallow wells: a locating dimple, not a deep hole. A 14 mm well swallows
# half a 32 mm vial and the gripper has to drag it up a wall with 2 mm of
# side clearance — measured, and it slips every time. At 5 mm the vial stands
# proud and lifting it needs no clearance at all. Real racks are shallow for
# the same reason: you have to be able to get the vials out.
WELL_DEPTH = 0.005
SLOT_TOLERANCE = (COLUMN_PITCH - WELL_POST) / 2.0

# Grasp heights, measured from the floor. The tool centre is now the fingertip
# itself, so this is where the fingers close: two thirds of the way up a 32 mm
# vial, under the cap, above its centre of mass so it hangs straight.
# A vial in a well is gripped near its cap: the well swallows 14 mm of it, so
# the fingers must close above the rack walls, not inside them.
GRASP_HEIGHT = RACK_PLATE_THICKNESS + VIAL_HEIGHT - 0.008
LIFT_HEIGHT = 0.080
# Release height above the shelf. Measured at 10 mm the vial dropped far
# enough to bounce; a gripped vial hangs a few mm lower than the model says.
PLACE_CLEARANCE = 0.003
# A clear, already-vertical pose that every pick and place starts and ends
# from. Its bearing matters: unfolding from the load pose sweeps the arm
# through this whole vertical plane, so it has to be one nothing stands in.
# Objects spawn between 5 and 59 degrees and the rack spans -31 to -69, which
# leaves the gap below. The height is bounded by the arm's top-down ceiling,
# which the longer fingertip tool lowered to about 123 mm at this reach.
READY_BEARING_DEG = -15.0
READY_REACH = 0.170
READY_HEIGHT = 0.110

# A workspace camera anchored in world coordinates, not to the gripper. The
# wrist mount below renders, but renders the horizon: its orientation is
# expressed in the gripper link's own frame and comes out pointing sideways,
# which is invisible in a check that only asks whether frames arrived. A world
# pose can be aimed and verified directly, which is what a dataset that takes
# an hour to train on deserves.
SCENE_CAM_PRIM = "/World/workspace_cam"
SCENE_CAM_EYE = (0.60, 0.10, 0.32)
SCENE_CAM_TARGET = (0.14, 0.01, 0.02)

WRIST_CAM_PRIM = f"{so101.ARM_PRIM}/gripper/wrist_cam"
# Matched to the RealSense D435's colour stream: 640x480 at 69.4 degrees
# horizontal. Kit's default camera is 60 degrees, and a policy trained on a
# narrower view than the real camera delivers sees a different world at
# deployment than it did in training.
D435_HFOV_DEG = 69.4
D435_APERTURE = 20.955
D435_FOCAL_LENGTH = (D435_APERTURE / 2.0) / math.tan(math.radians(D435_HFOV_DEG) / 2.0)
# Measured against the gripper link's own frame: sits behind and above the
# fingers, looking forward and down at the grasp point 87 mm away.
WRIST_CAM_TRANSLATION = (0.046148, -0.000015, -0.008527)
WRIST_CAM_ORIENTATION = (-0.676512, 0.205743, -0.205736, 0.676516)

# LeRobot's SO-101 leader reports these joints, in this order, which happens to
# be our DOF order too. Signs and offsets do not carry over — they come from
# each leader's own calibration — so they live here as constants to be filled
# in by `calibrate_leader` rather than assumed to be identity.
LEADER_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LEADER_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
LEADER_OFFSETS_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_backend: _Backend | None = None


# ── Public API ────────────────────────────────────────────────────────────────


def reset(scenario: str = "train", episode: int | None = None) -> None:
    """
    Put the scene back to the start of an episode.

    :param scenario: One of :data:`SCENARIOS`. ``train`` and ``eval`` draw
        object positions from disjoint halves of the same grid; ``ambiguous``
        spawns two identical red vials for the handoff demo.
    :param episode: Episode index, which seeds the layout. ``None`` advances an
        internal counter, so repeated calls keep producing new positions.
    :raises ValueError: If the scenario is not one of :data:`SCENARIOS`.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'; expected one of {SCENARIOS}")
    _active().reset(scenario, episode)


def frame() -> np.ndarray:
    """
    Read the wrist camera.

    :return: ``(480, 640, 3)`` uint8 RGB.
    """

    return _active().frame()


def joint_state() -> np.ndarray:
    """
    Read where the arm actually is.

    This is the *state*: the observation a policy sees. It is not what was
    commanded, and recording it as the action is the label bug that makes a
    trained policy freeze mid-task.

    :return: ``(6,)`` float32 joint angles in radians.
    """

    return _active().joint_state()


def set_targets(joint_positions: np.ndarray) -> None:
    """
    Command the arm and advance one control tick.

    :param joint_positions: ``(6,)`` joint angles in radians, ordered as
        :data:`so101.JOINT_ORDER`. Values outside the arm's limits are clamped
        rather than refused, because a policy will emit them.
    """

    _active().set_targets(np.asarray(joint_positions, dtype=np.float32).reshape(JOINT_COUNT))


def object_poses() -> dict:
    """
    Read ground-truth object and slot poses.

    EVAL ORACLE ONLY. Perception belongs to :func:`frame`.

    :return: Name to ``{"position": [x, y, z], "upright": bool, "kind": str}``.
    """

    return _active().object_poses()


def score() -> bool:
    """
    Judge the current episode.

    :return: Whether the goal object is standing in the goal slot.
    """

    return _active().score()


def set_goal(obj: str, destination: str) -> None:
    """
    Tell the scorer what success means for this episode.

    Beyond the frozen six, because the goal arrives at run time from the
    planner and :func:`score` cannot be written without it.

    :param obj: Object name, from :data:`OBJECT_NAMES`.
    :param destination: Slot name, from :data:`SLOT_NAMES`.
    """

    _active().set_goal(obj, destination)


def vial_in_well(well: str, tolerance: float | None = None) -> str | None:
    """
    Name whichever vial is standing in a given well.

    The task is stated by location — "move A3 to C5" — so the mover is
    whatever is in A3, not a colour that a bench of amber vials does not have.

    :param well: Well name, from :data:`SLOT_NAMES`.
    :param tolerance: How close counts as in the well, in metres. Defaults to
        just under half the well spacing — anything larger reports one vial as
        standing in several wells at once, which is exactly what a
        rearrangement planner must never be told.
    :return: Vial name, or ``None`` if the well is empty.
    """

    limit = min(ROW_PITCH, COLUMN_PITCH) * 0.45 if tolerance is None else tolerance
    poses = object_poses()
    target = poses.get(well)
    if target is None:
        return None
    for name, record in poses.items():
        if record["kind"] == "slot":
            continue
        if math.dist(record["position"][:2], target["position"][:2]) <= limit:
            return name
    return None


def occupancy() -> dict:
    """
    Report which vial is standing in which well.

    This is the tracker the rearrangement planner keeps: after every move it
    knows where each vial went, so it can put them back.

    :return: Well name to vial name, for the wells that are occupied.
    """

    found = {}
    for well in SLOT_NAMES:
        vial = vial_in_well(well)
        if vial is not None:
            found[well] = vial
    return found


def goal() -> dict:
    """
    Read back the goal this episode is being scored against.

    :return: ``{"object": str, "destination": str}``.
    """

    return _active().goal()


def backend_name() -> str:
    """
    Report which backend is live, for logging and for the results table.

    :return: ``"isaac"`` or ``"mock"``.
    """

    return _active().name


def leader_to_joints(action: dict) -> np.ndarray:
    """
    Convert one LeRobot leader-arm reading into sim joint targets.

    The leader reports ``{"shoulder_pan.pos": ...}`` in degrees for the body
    joints and 0-100 for the gripper. Joint order matches ours; signs and zero
    offsets do not, and come from :data:`LEADER_SIGNS` and
    :data:`LEADER_OFFSETS_RAD`.

    :param action: Reading from ``SO101Leader.get_action()``.
    :return: ``(6,)`` float32 joint angles in radians.
    """

    low, high = so101.JOINT_LIMITS_DEG["Jaw"]
    values = []
    for index, name in enumerate(LEADER_JOINTS):
        raw = float(action.get(f"{name}.pos", action.get(name, 0.0)))
        if name == "gripper":
            # 0-100 percent open, mapped across the jaw's authored travel
            angle = math.radians(low + (high - low) * max(0.0, min(100.0, raw)) / 100.0)
        else:
            angle = math.radians(raw)
        values.append(LEADER_SIGNS[index] * angle + LEADER_OFFSETS_RAD[index])
    return np.asarray(so101.clamp_pose(tuple(values)), dtype=np.float32)


def calibrate_leader(action: dict, known_pose: tuple[float, ...] = HOME) -> list[float]:
    """
    Solve the leader's zero offsets by holding it in a known pose.

    Hold the leader where the sim arm's pose is known — its load pose is the
    easy one — call this once, and paste the result into
    :data:`LEADER_OFFSETS_RAD`. Signs still have to be checked by eye: move one
    leader joint and watch which way the sim arm goes.

    :param action: Reading from ``SO101Leader.get_action()`` at that pose.
    :param known_pose: The sim pose the leader is being held at, in radians.
    :return: Offsets to paste into :data:`LEADER_OFFSETS_RAD`.
    """

    saved = list(LEADER_OFFSETS_RAD)
    try:
        LEADER_OFFSETS_RAD[:] = [0.0] * JOINT_COUNT
        raw = leader_to_joints(action)
    finally:
        LEADER_OFFSETS_RAD[:] = saved
    return [round(float(target - measured), 6) for target, measured in zip(known_pose, raw, strict=True)]


def expert_actions(*, ticks_per_phase: int = 12) -> list[np.ndarray]:
    """
    Generate a scripted pick-and-place for the current goal.

    Coarse scripted motion is a legitimate action layer on its own and the
    named fallback if the learned policy does not converge. It also produces
    demonstration episodes without a human, which is the fastest way to find
    out whether the recording format is right.

    :param ticks_per_phase: Control ticks each waypoint segment occupies.
    :return: One ``(6,)`` float32 command per control tick.
    :raises RuntimeError: If the goal object or slot is out of reach.
    """

    return _active().expert_actions(ticks_per_phase)


# ── Backend selection ─────────────────────────────────────────────────────────


def _active() -> _Backend:
    """
    Resolve the backend once, on first use.

    :return: The live backend.
    """

    global _backend
    if _backend is None:
        _backend = _MockBackend() if _mock_requested() else _IsaacBackend()
    return _backend


def _mock_requested() -> bool:
    """
    Decide whether to run against the mock.

    :return: Whether the mock backend should be used.
    """

    choice = os.environ.get("VIAL_SIM", "").strip().lower()
    if choice in {"mock", "isaac"}:
        return choice == "mock"
    return importlib.util.find_spec("isaacsim") is None


def _cell_positions(scenario: str, episode: int, count: int) -> list[tuple[float, float]]:
    """
    Draw object positions from the training or held-out half of the grid.

    :param scenario: Scenario name.
    :param episode: Episode index, which seeds the draw.
    :param count: How many positions are needed.
    :return: World ``(x, y)`` positions.
    """

    cells = [(r, b) for r in range(len(SPAWN_REACHES)) for b in range(len(SPAWN_BEARINGS_DEG))]
    held_out = set(HELD_OUT_CELLS)
    usable = [cell for cell in cells if (cell in held_out) == (scenario == "eval")]
    rng = random.Random(f"{scenario}:{episode}")
    order = usable[:]
    rng.shuffle(order)
    positions: list[tuple[float, float]] = []
    for reach_index, bearing_index in order:
        if len(positions) == count:
            break
        reach = SPAWN_REACHES[reach_index] + rng.uniform(-SPAWN_JITTER_M, SPAWN_JITTER_M)
        bearing = math.radians(SPAWN_BEARINGS_DEG[bearing_index] + rng.uniform(-SPAWN_JITTER_DEG, SPAWN_JITTER_DEG))
        candidate = (so101.PAN_AXIS_XY[0] + reach * math.cos(bearing), so101.PAN_AXIS_XY[1] + reach * math.sin(bearing))
        # Greedy rejection rather than a sample: neighbouring cells are only a
        # few centimetres apart, and the gripper is wider than that
        if all(math.dist(candidate, taken) >= MIN_OBJECT_SEPARATION for taken in positions):
            positions.append(candidate)
    return positions


def _well_grid() -> dict[str, tuple[float, float, float]]:
    """
    Place every well of the rack in world coordinates.

    :return: Well name to world ``(x, y, z)`` of its floor.
    """

    bearing = math.radians(RACK_BEARING_DEG)
    radial = (math.cos(bearing), math.sin(bearing))
    tangent = (-math.sin(bearing), math.cos(bearing))
    wells = {}
    for row_index, row in enumerate(RACK_ROW_NAMES):
        for column in range(RACK_COLUMN_COUNT):
            out = RACK_REACH + row_index * ROW_PITCH
            along = (column - (RACK_COLUMN_COUNT - 1) / 2.0) * COLUMN_PITCH
            wells[f"{row}{column + 1}"] = (
                so101.PAN_AXIS_XY[0] + radial[0] * out + tangent[0] * along,
                so101.PAN_AXIS_XY[1] + radial[1] * out + tangent[1] * along,
                RACK_PLATE_THICKNESS,
            )
    return wells


def _slot_positions() -> dict[str, tuple[float, float, float]]:
    """
    Name every addressable destination.

    :return: Well name to world ``(x, y, z)`` of its floor.
    """

    return _well_grid()


def _episode_layout(scenario: str, episode: int) -> tuple[list[tuple[str, str, tuple[float, float, float], str]], str]:
    """
    Decide which vials exist this episode and which well each starts in.

    Vials start *in* the rack rather than loose on the table, because the task
    is rearrangement: reaching a vial in the back row means dealing with what
    is standing in front of it.

    :param scenario: Scenario name.
    :param episode: Episode index.
    :return: ``(name, kind, colour, well)`` entries, and the default goal vial.
    """

    red, blue, green = (0.85, 0.16, 0.16), (0.16, 0.36, 0.85), (0.18, 0.7, 0.28)
    usable = [f"{row}{column}" for row in USABLE_ROWS for column in USABLE_COLUMNS]
    rng = random.Random(f"{scenario}:{episode}:layout")
    # Held out by column, so an evaluation episode asks for a vial standing
    # where the policy has never picked one up
    held_out = {5, 9}
    candidates = [w for w in usable if not w.startswith("A") and (int(w[1:]) in held_out) == (scenario == "eval")]
    target_well = rng.choice(candidates or usable)
    spare = [w for w in usable if w != target_well]
    rng.shuffle(spare)
    if scenario == "ambiguous":
        wanted = [("red vial A", "vial", red, target_well), ("red vial B", "vial", red, spare[0]), ("blue vial", "vial", blue, spare[1])]
        default_goal = "red vial A"
    else:
        wanted = [("red vial", "vial", red, target_well), ("blue vial", "vial", blue, spare[0]), ("green vial", "vial", green, spare[1])]
        default_goal = "red vial"
    return wanted, default_goal


class _Backend:
    """
    Shared behaviour between the real simulator and the mock.
    """

    name = "base"

    def __init__(self) -> None:
        """
        Set up the episode bookkeeping both backends keep.
        """

        self._scenario = "train"
        self._episode = -1
        self._goal_object = OBJECT_NAMES[0]
        self._goal_slot = SLOT_NAMES[2]
        self._slots = _slot_positions()
        self._ticks = 0

    def set_goal(self, obj: str, destination: str) -> None:
        """
        Record what this episode is scored against.

        :param obj: Object name.
        :param destination: Slot name.
        :raises ValueError: If the destination is not a known slot.
        """

        if destination not in self._slots:
            raise ValueError(f"unknown destination '{destination}'; expected one of {tuple(self._slots)}")
        self._goal_object, self._goal_slot = obj, destination

    def goal(self) -> dict:
        """
        Read back the episode's goal.

        :return: The goal object and destination.
        """

        return {"object": self._goal_object, "destination": self._goal_slot}

    def _next_episode(self, episode: int | None) -> int:
        """
        Advance or accept the episode index.

        :param episode: Requested index, or ``None`` to advance.
        :return: The index to lay this episode out with.
        """

        self._episode = self._episode + 1 if episode is None else episode
        self._ticks = 0
        return self._episode

    def _scored(self, poses: dict) -> bool:
        """
        Apply the placement rule to a set of poses.

        :param poses: Output of :meth:`object_poses`.
        :return: Whether the goal object is standing in the goal slot.
        """

        held = poses.get(self._goal_object)
        slot = self._slots.get(self._goal_slot)
        if held is None or slot is None:
            return False
        x, y, z = held["position"]
        return bool(math.hypot(x - slot[0], y - slot[1]) <= SLOT_TOLERANCE and z <= slot[2] + 0.05 and held["upright"])

    def _waypoints(self, poses: dict) -> list[tuple[str, float, tuple[float, ...]]]:
        """
        Build the scripted pick-and-place for the current goal.

        :param poses: Ground-truth poses to plan against.
        :return: Labelled waypoints for :func:`so101.trajectory`.
        :raises RuntimeError: If either end of the move is out of reach.
        """

        held = poses.get(self._goal_object)
        if held is None:
            raise RuntimeError(f"goal object '{self._goal_object}' is not in the scene")
        slot = self._slots[self._goal_slot]
        open_jaw, shut_jaw = so101.grip_angles(VIAL_DIAMETER, clearance=WELL_GRIP_CLEARANCE)
        # Aim short of the vial's axis, not at it. Only the upper finger moves,
        # so the lower face is a fixed 7.9 mm from the tool axis and a 22 mm
        # vial centred on that axis is already 3 mm inside it — measured as the
        # arm knocking the vial over on approach before it ever closed.
        offset = so101.grip_offset(VIAL_DIAMETER)
        pick_x, pick_y = _pushed_radially(held["position"][0], held["position"][1], offset)
        # Placing uses where the vial ends up in the hand, not the wider
        # approach offset: the jaw has already pushed it against the fixed face
        # by then, so aiming the tool at the well would drop it beside one.
        place_x, place_y = _pushed_radially(slot[0], slot[1], -so101.held_offset(VIAL_DIAMETER))
        place_z = slot[2]

        def at(x: float, y: float, height: float, jaw: float, what: str) -> tuple[float, ...]:
            pose = so101.pose_at(x, y, height, jaw=jaw)
            if pose is None:
                raise RuntimeError(f"{what} at reach {so101.reach_of(x, y):.3f} m, height {height:.3f} m is outside the workspace")
            return pose

        # Pointing the tool straight down costs height, and the ceiling falls
        # off with reach, so the transit height is whatever the further of the
        # two ends can actually hold rather than one constant that works near
        # the base and silently fails at the edge of the grid
        lift = _transit_height(so101.reach_of(pick_x, pick_y), so101.reach_of(place_x, place_y))
        above_pick = at(pick_x, pick_y, lift, open_jaw, "pre-grasp")
        # Start from where the arm actually is. Without this the first waypoint
        # is the approach pose, the drives are handed it as a step change, and
        # the arm crosses the table at whatever speed it likes — measured as
        # the vial lying on its side by tick 11, knocked over before the
        # gripper had opened.
        start = tuple(float(value) for value in self.joint_state())
        # Get vertical *before* travelling. Blending joint-for-joint from the
        # load pose straight to the pre-grasp swings the gripper across the
        # table on a diagonal with the tool still half-turned, and its servo
        # body — 80 mm wide, well outside the fingers — sweeps whatever is
        # standing there. Measured: the vial on its side by tick 18, before the
        # jaw had finished opening. So the arm rises to a clear, already
        # top-down pose over empty floor, and every later move is either
        # horizontal at carry height or straight down.
        ready_bearing = math.radians(READY_BEARING_DEG)
        ready_xy = (so101.PAN_AXIS_XY[0] + READY_REACH * math.cos(ready_bearing), so101.PAN_AXIS_XY[1] + READY_REACH * math.sin(ready_bearing))
        ready = so101.pose_at(*ready_xy, READY_HEIGHT, jaw=open_jaw)
        if ready is None:
            raise RuntimeError("the top-down ready pose is outside the workspace")
        on_pick = at(pick_x, pick_y, GRASP_HEIGHT, open_jaw, "grasp")
        gripped = at(pick_x, pick_y, GRASP_HEIGHT, shut_jaw, "grip")
        lifted = at(pick_x, pick_y, lift, shut_jaw, "lift")
        place_height = place_z + GRASP_HEIGHT + PLACE_CLEARANCE
        above_place = at(place_x, place_y, lift, shut_jaw, "transit")
        on_place = at(place_x, place_y, place_height, shut_jaw, "place")
        released = at(place_x, place_y, place_height, open_jaw, "release")
        retreated = at(place_x, place_y, lift, open_jaw, "retreat")
        return [
            ("start", 0.3, start),
            ("rise", 1.0, ready),
            ("approach", 1.2, above_pick),
            ("descend", 1.0, on_pick),
            # A drive that is still catching up when the jaw shuts grips where
            # the arm was, not where it was sent, so the pose is held still for
            # a beat before anything closes
            ("steady", 0.4, on_pick),
            ("grip", 0.6, gripped),
            ("lift", 1.0, lifted),
            ("transit", 1.4, above_place),
            ("place", 1.0, on_place),
            ("settle", 0.3, on_place),
            ("release", 0.6, released),
            ("retreat", 1.0, retreated),
            ("home", 0.8, ready),
        ]

    def expert_actions(self, ticks_per_phase: int) -> list[np.ndarray]:
        """
        Turn the scripted waypoints into one command per control tick.

        :param ticks_per_phase: Control ticks each unit of share occupies.
        :return: Commands in radians.
        """

        waypoints = self._waypoints(self.object_poses())
        total = int(round(sum(share for _label, share, _pose in waypoints) * ticks_per_phase))
        schedule = so101.trajectory(waypoints, total)
        return [np.asarray(pose, dtype=np.float32) for _label, pose in schedule]

    def expert_labels(self, ticks_per_phase: int) -> list[str]:
        """
        Name the phase each expert command belongs to.

        :param ticks_per_phase: Control ticks each unit of share occupies.
        :return: One phase label per command.
        """

        waypoints = self._waypoints(self.object_poses())
        total = int(round(sum(share for _label, share, _pose in waypoints) * ticks_per_phase))
        return [label for label, _pose in so101.trajectory(waypoints, total)]

    def reset(self, scenario: str, episode: int | None) -> None:
        """
        Start an episode.

        :param scenario: Scenario name.
        :param episode: Episode index or ``None``.
        :raises NotImplementedError: Always, in the base class.
        """

        raise NotImplementedError

    def frame(self) -> np.ndarray:
        """
        Read the wrist camera.

        :return: RGB image.
        :raises NotImplementedError: Always, in the base class.
        """

        raise NotImplementedError

    def joint_state(self) -> np.ndarray:
        """
        Read measured joint angles.

        :return: Joint angles in radians.
        :raises NotImplementedError: Always, in the base class.
        """

        raise NotImplementedError

    def set_targets(self, joint_positions: np.ndarray) -> None:
        """
        Command the arm and advance one tick.

        :param joint_positions: Joint angles in radians.
        :raises NotImplementedError: Always, in the base class.
        """

        raise NotImplementedError

    def object_poses(self) -> dict:
        """
        Read ground-truth poses.

        :return: Name to pose record.
        :raises NotImplementedError: Always, in the base class.
        """

        raise NotImplementedError

    def score(self) -> bool:
        """
        Judge the episode.

        :return: Whether the goal was met.
        """

        return self._scored(self.object_poses())


class _IsaacBackend(_Backend):
    """
    Drive the real simulator, inside an Antioch run on the GPU machine.
    """

    name = "isaac"

    def __init__(self) -> None:
        """
        Note that nothing is built yet; the scene is built on first reset.
        """

        super().__init__()
        self._world = None
        self._arm = None
        self._order: list[int] = []
        self._sensor = None
        self._objects: dict[str, object] = {}
        self._kinds: dict[str, str] = {}
        self._commanded = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._native_frame_shape: tuple[int, ...] | None = None
        self._camera_pose: tuple | None = None

    def reset(self, scenario: str, episode: int | None) -> None:
        """
        Build the scene on first use, then lay this episode's objects out.

        :param scenario: Scenario name.
        :param episode: Episode index or ``None``.
        """

        index = self._next_episode(episode)
        rebuild = self._world is None or scenario != self._scenario
        self._scenario = scenario
        if self._world is None:
            self._build(scenario, index)
        else:
            self._lay_out(scenario, index, rebuild=rebuild)
        self._world.reset()
        self._commanded = np.zeros(JOINT_COUNT, dtype=np.float32)
        for _ in range(STEPS_PER_TICK * 3):
            self._world.step(render=False)

    def _build(self, scenario: str, episode: int) -> None:
        """
        Create the world, the arm, the rack, the camera and the objects.

        :param scenario: Scenario name.
        :param episode: Episode index.
        """

        import antioch
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.prims import create_prim

        world = antioch.world()
        world.scene.add_ground_plane(restitution=0.0)
        # A dome alone flattens the scene; the distant light casts the shadow
        # that tells a camera frame where an object is standing
        create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 250.0})
        create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 450.0})
        antioch.load_asset(so101.ASSET, prim_path=so101.ARM_PRIM, version=so101.ASSET_VERSION)
        self._arm = world.scene.add(Robot(prim_path=so101.articulation_root(so101.ARM_PRIM), name="so101"))

        bearing = math.radians(RACK_BEARING_DEG)
        radial = (math.cos(bearing), math.sin(bearing))
        tangent = (-math.sin(bearing), math.cos(bearing))
        rows_span = len(RACK_ROW_NAMES) * ROW_PITCH
        columns_span = RACK_COLUMN_COUNT * COLUMN_PITCH
        centre_radial = RACK_REACH + (len(RACK_ROW_NAMES) - 1) / 2.0 * ROW_PITCH

        def rack_point(out: float, along: float, height: float) -> np.ndarray:
            return np.array(
                [
                    so101.PAN_AXIS_XY[0] + radial[0] * out + tangent[0] * along,
                    so101.PAN_AXIS_XY[1] + radial[1] * out + tangent[1] * along,
                    height,
                ]
            )

        world.scene.add(
            FixedCuboid(
                prim_path="/World/rack",
                name="rack",
                position=rack_point(centre_radial, 0.0, RACK_PLATE_THICKNESS / 2.0),
                scale=np.array([rows_span, columns_span, RACK_PLATE_THICKNESS]),
                orientation=_yaw_quaternion(bearing),
                color=np.array([0.82, 0.84, 0.88]),
            )
        )
        # Walls, not posts: corner posts alone leave the sides open and a vial
        # slides out diagonally instead of dropping into a well.
        for index in range(len(RACK_ROW_NAMES) + 1):
            out = RACK_REACH + (index - 0.5) * ROW_PITCH
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/rack_row_{index}",
                    name=f"rack_row_{index}",
                    position=rack_point(out, 0.0, RACK_PLATE_THICKNESS + WELL_DEPTH / 2.0),
                    scale=np.array([WELL_POST, columns_span, WELL_DEPTH]),
                    orientation=_yaw_quaternion(bearing),
                    color=np.array([0.72, 0.75, 0.82]),
                )
            )
        for index in range(RACK_COLUMN_COUNT + 1):
            along = (index - RACK_COLUMN_COUNT / 2.0) * COLUMN_PITCH
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/rack_column_{index}",
                    name=f"rack_column_{index}",
                    position=rack_point(centre_radial, along, RACK_PLATE_THICKNESS + WELL_DEPTH / 2.0),
                    scale=np.array([rows_span, WELL_POST, WELL_DEPTH]),
                    orientation=_yaw_quaternion(bearing),
                    color=np.array([0.72, 0.75, 0.82]),
                )
            )
        self._lay_out(scenario, episode, rebuild=True)
        self._attach_camera()
        self._world = world
        world.reset()
        names = list(self._arm.dof_names)
        missing = [name for name in so101.JOINT_ORDER if name not in names]
        if missing:
            raise RuntimeError(f"the SO-101 articulation is missing {missing}; it reports {names}")
        self._order = [names.index(name) for name in so101.JOINT_ORDER]
        self._warm_renderer()

    def _lay_out(self, scenario: str, episode: int, *, rebuild: bool) -> None:
        """
        Place this episode's objects, creating them the first time.

        :param scenario: Scenario name.
        :param episode: Episode index.
        :param rebuild: Whether the object set itself has changed.
        """

        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicSphere

        entries, default_goal = _episode_layout(scenario, episode)
        if rebuild and self._goal_object not in {name for name, _kind, _colour, _position in entries}:
            self._goal_object = default_goal
        if not self._objects:
            import antioch

            # Glass on a printed finger is grippier than PhysX's default pair,
            # and a vial that slides out of a closed hand looks like a policy
            # failure when it is really a material one
            grip = PhysicsMaterial(prim_path="/World/physics/grip", name="grip", static_friction=0.9, dynamic_friction=0.85, restitution=0.0)
            world = antioch.world()
            wells = _well_grid()
            for index, (name, kind, colour, well) in enumerate(entries):
                x, y = wells[well][0], wells[well][1]
                shared = {"prim_path": f"/World/object_{index}", "name": name, "color": np.array(colour), "mass": OBJECT_MASS, "physics_material": grip}
                if kind == "vial":
                    obj = DynamicCylinder(position=np.array([x, y, RACK_PLATE_THICKNESS + VIAL_HEIGHT / 2.0]), radius=VIAL_DIAMETER / 2.0, height=VIAL_HEIGHT, **shared)
                elif kind == "sphere":
                    obj = DynamicSphere(position=np.array([x, y, DISTRACTOR_SIZE / 2.0]), radius=DISTRACTOR_SIZE / 2.0, **shared)
                else:
                    obj = DynamicCuboid(position=np.array([x, y, DISTRACTOR_SIZE / 2.0]), size=DISTRACTOR_SIZE, **shared)
                self._objects[name] = world.scene.add(obj)
                self._kinds[name] = kind
            return
        wells = _well_grid()
        for (_name, kind, _colour, well), obj in zip(entries, self._objects.values(), strict=False):
            x, y = wells[well][0], wells[well][1]
            obj.set_world_pose(position=np.array([x, y, RACK_PLATE_THICKNESS + _resting_height(kind)]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

    def _attach_camera(self) -> None:
        """
        Aim the viewport at the workspace.

        The experimental RTX camera sensor was tried first and abandoned: it
        re-authors its prim's transform, so the pose set on it never took, and
        every frame came back as a distant horizon. The platform's own viewport
        read-back is already proven in this project, and `set_camera_view` aims
        it in one line with no parent frame to get backwards.
        """

        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=list(SCENE_CAM_EYE), target=list(SCENE_CAM_TARGET), camera_prim_path="/OmniverseKit_Persp")
        self._sensor = None
        self._camera_pose = (tuple(SCENE_CAM_EYE), tuple(SCENE_CAM_TARGET))

    def _match_d435_optics(self) -> None:
        """
        Give the sim camera the real camera's field of view.

        Set on the USD prim rather than through the sensor, because the lens is
        a property of the camera and the sensor only reads it.
        """

        import antioch
        from pxr import UsdGeom

        prim = antioch.stage().GetPrimAtPath(SCENE_CAM_PRIM)
        if not prim.IsValid():
            return
        camera = UsdGeom.Camera(prim)
        camera.GetFocalLengthAttr().Set(float(D435_FOCAL_LENGTH))
        camera.GetHorizontalApertureAttr().Set(float(D435_APERTURE))
        camera.GetVerticalApertureAttr().Set(float(D435_APERTURE * IMAGE_SHAPE[0] / IMAGE_SHAPE[1]))

    def _warm_renderer(self) -> None:
        """
        Render until the materials have finished loading.

        Measured on this scene: RTX hands back a usable picture long before it
        hands back the right one, and the arm renders grey for the first few
        hundred steps. Spending them here keeps them out of every episode.
        """

        previous = None
        for _ in range(240):
            self._world.step(render=True)
            current = self._grab_rgb()
            if current is None:
                continue
            sample = current[::8, ::8].astype(np.float32)
            if previous is not None and float(np.abs(sample - previous).mean()) < 0.35:
                return
            previous = sample

    def frame(self) -> np.ndarray:
        """
        Read the wrist camera.

        :return: ``(480, 640, 3)`` uint8 RGB.
        """

        return _fit_image(self._grab_rgb())

    def _grab_rgb(self) -> np.ndarray | None:
        """
        Pull raw pixels from the wrist camera, or the viewport if it is absent.

        :return: RGB pixels, or ``None`` if nothing has rendered yet.
        """

        import antioch

        if self._sensor is not None:
            try:
                data, _info = self._sensor.get_data("rgb")
                pixels = np.asarray(data.numpy() if hasattr(data, "numpy") else data)
                if pixels.size:
                    pixels = pixels.reshape(pixels.shape[-3:]) if pixels.ndim > 3 else pixels
                    self._native_frame_shape = tuple(pixels.shape)
                    return pixels[:, :, :3]
            except Exception:  # noqa: BLE001 - fall through to the viewport
                pass
        captured = antioch.capture_viewport()
        if captured is None:
            return None
        pixels = np.asarray(captured)
        self._native_frame_shape = tuple(pixels.shape)
        return pixels[:, :, :3]

    def native_frame_shape(self) -> tuple[int, ...] | None:
        """
        Report the shape the camera actually returned, before fitting.

        The 6.0.1 docs disagree with themselves about whether a camera's
        resolution is ``(width, height)`` or ``(height, width)``, so the run
        reports what it really got instead of trusting either page.

        :return: Native pixel shape, or ``None`` if nothing has rendered.
        """

        return self._native_frame_shape

    def joint_state(self) -> np.ndarray:
        """
        Read measured joint angles.

        :return: ``(6,)`` float32 radians.
        """

        readings = self._arm.get_joint_positions()
        return np.asarray([float(readings[index]) for index in self._order], dtype=np.float32)

    def set_targets(self, joint_positions: np.ndarray) -> None:
        """
        Command the arm and advance one control tick.

        :param joint_positions: ``(6,)`` joint angles in radians.
        """

        from isaacsim.core.utils.types import ArticulationAction

        self._commanded = np.asarray(so101.clamp_pose(tuple(float(value) for value in joint_positions)), dtype=np.float32)
        action = ArticulationAction(joint_positions=self._commanded, joint_indices=np.array(self._order, dtype=np.int32))
        self._arm.apply_action(action)
        for step in range(STEPS_PER_TICK):
            self._world.step(render=step == STEPS_PER_TICK - 1)
        self._ticks += 1

    def commanded(self) -> np.ndarray:
        """
        Read back the last command, for the monitor's commanded-vs-measured rule.

        :return: ``(6,)`` float32 radians.
        """

        return self._commanded.copy()

    def object_poses(self) -> dict:
        """
        Read ground-truth object and slot poses.

        :return: Name to pose record.
        """

        poses = {}
        for name, obj in self._objects.items():
            position, orientation = obj.get_world_pose()
            poses[name] = {
                "position": [float(value) for value in position],
                "upright": _upright(np.asarray(orientation, dtype=np.float64)),
                "kind": self._kinds.get(name, "cube"),
            }
        for name, (x, y, z) in self._slots.items():
            poses[name] = {"position": [x, y, z], "upright": True, "kind": "slot"}
        return poses


class _MockBackend(_Backend):
    """
    Stand in for the simulator on a laptop, with enough behaviour to test against.

    The arm lags toward its target the way a real drive does, and the gripper
    refuses to close past the object it is holding — which is the one signal
    the monitor is built on, so retry logic can be finished before the real
    scene exists.
    """

    name = "mock"

    def __init__(self) -> None:
        """
        Start parked, holding nothing.
        """

        super().__init__()
        self._measured = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._commanded = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._positions: dict[str, list[float]] = {}
        self._kinds = {}
        self._held: str | None = None
        self._grasp_fails = False

    def mock_force_missed_grasp(self, failing: bool = True) -> None:
        """
        Make the next grasp close on air, to exercise the monitor.

        :param failing: Whether grasps should miss.
        """

        self._grasp_fails = failing

    def reset(self, scenario: str, episode: int | None) -> None:
        """
        Lay the scene out without a simulator.

        :param scenario: Scenario name.
        :param episode: Episode index or ``None``.
        """

        index = self._next_episode(episode)
        self._scenario = scenario
        entries, default_goal = _episode_layout(scenario, index)
        self._positions = {}
        self._kinds = {}
        wells = _well_grid()
        for name, kind, _colour, well in entries:
            self._positions[name] = [wells[well][0], wells[well][1], RACK_PLATE_THICKNESS + _resting_height(kind)]
            self._kinds[name] = kind
        if self._goal_object not in self._positions:
            self._goal_object = default_goal
        self._measured = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._commanded = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._held = None

    def frame(self) -> np.ndarray:
        """
        Draw a synthetic wrist view that changes as the arm moves.

        :return: ``(480, 640, 3)`` uint8 RGB.
        """

        image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        image[:, :, :] = 140
        image[: IMAGE_SHAPE[0] // 3, :, :] = 90
        tool = so101.chain_points(tuple(float(value) for value in self._measured))[-1]
        column = int(np.clip(320 + tool[1] * 1200, 40, IMAGE_SHAPE[1] - 80))
        row = int(np.clip(360 - tool[2] * 700, 40, IMAGE_SHAPE[0] - 80))
        image[row : row + 60, column : column + 60] = (210, 40, 40)
        return image

    def joint_state(self) -> np.ndarray:
        """
        Read the lagged joint state.

        :return: ``(6,)`` float32 radians.
        """

        return self._measured.copy()

    def commanded(self) -> np.ndarray:
        """
        Read back the last command.

        :return: ``(6,)`` float32 radians.
        """

        return self._commanded.copy()

    def set_targets(self, joint_positions: np.ndarray) -> None:
        """
        Move toward the target, and block the gripper when it is holding something.

        :param joint_positions: ``(6,)`` joint angles in radians.
        """

        self._commanded = np.asarray(so101.clamp_pose(tuple(float(value) for value in joint_positions)), dtype=np.float32)
        self._measured = self._measured + 0.35 * (self._commanded - self._measured)
        tool = so101.chain_points(tuple(float(value) for value in self._measured))[-1]
        contact = so101.jaw_for_gap(VIAL_DIAMETER)
        closing = float(self._commanded[GRIPPER]) < contact
        if self._held is None and closing and not self._grasp_fails:
            for name, position in self._positions.items():
                if math.dist(tool, position) < 0.05:
                    self._held = name
                    break
        if self._held is not None:
            # A held object blocks the jaw, which is the whole monitor signal
            self._measured[GRIPPER] = max(float(self._measured[GRIPPER]), contact)
            self._positions[self._held] = [tool[0], tool[1], max(VIAL_HEIGHT / 2.0, tool[2])]
            if float(self._commanded[GRIPPER]) > contact:
                slot = self._slots.get(self._goal_slot)
                resting = self._positions[self._held]
                self._positions[self._held] = [resting[0], resting[1], slot[2] if slot else VIAL_HEIGHT / 2.0]
                self._held = None
        self._ticks += 1

    def object_poses(self) -> dict:
        """
        Read the mock's bookkeeping as ground truth.

        :return: Name to pose record.
        """

        poses = {name: {"position": list(position), "upright": True, "kind": self._kinds.get(name, "cube")} for name, position in self._positions.items()}
        for name, (x, y, z) in self._slots.items():
            poses[name] = {"position": [x, y, z], "upright": True, "kind": "slot"}
        return poses


def _pushed_radially(x: float, y: float, offset: float) -> tuple[float, float]:
    """
    Move a world point in or out along its own bearing from the pan axis.

    With the tool pointing straight down the fingers open radially, so this is
    the axis a grasp offset has to be applied along.

    :param x: World x, in metres.
    :param y: World y, in metres.
    :param offset: Distance to move; negative is toward the arm.
    :return: The moved point.
    """

    bearing = so101.bearing_of(x, y)
    return x + offset * math.cos(bearing), y + offset * math.sin(bearing)


def _resting_height(kind: str) -> float:
    """
    Height an object's origin sits at when standing on the floor.

    :param kind: ``vial``, ``sphere`` or ``cube``.
    :return: Centre height in metres.
    """

    return {"vial": VIAL_HEIGHT / 2.0}.get(kind, DISTRACTOR_SIZE / 2.0)


def _transit_height(*reaches: float) -> float:
    """
    Choose a carry height both ends of the move can hold.

    :param reaches: Distances from the pan axis the arm has to reach, in metres.
    :return: Tool-centre height in metres, at or below :data:`LIFT_HEIGHT`.
    """

    ceiling = LIFT_HEIGHT
    for reach in reaches:
        while ceiling > GRASP_HEIGHT + 0.02 and so101.solve_reach(reach, ceiling, -90.0) is None:
            ceiling -= 0.005
    return ceiling


def _fit_image(pixels: np.ndarray | None) -> np.ndarray:
    """
    Force whatever the camera returned into the frozen frame shape.

    Nearest-neighbour, because the contract everyone codes against matters more
    than resampling quality, and a shape that varies by backend is a bug that
    surfaces at integration time.

    :param pixels: Raw RGB pixels, or ``None``.
    :return: ``(480, 640, 3)`` uint8 RGB.
    """

    height, width, _ = IMAGE_SHAPE
    if pixels is None or pixels.size == 0:
        return np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    if pixels.shape[0] == height and pixels.shape[1] == width:
        return np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)
    # Crop to the target aspect first. Resizing a 949x577 viewport straight to
    # 640x480 squashes it, and a policy trained on squashed geometry sees a
    # different world than the one it is asked to act in.
    aspect = width / height
    rows_available, columns_available = pixels.shape[0], pixels.shape[1]
    if columns_available / rows_available > aspect:
        keep = int(round(rows_available * aspect))
        start = (columns_available - keep) // 2
        pixels = pixels[:, start : start + keep]
    else:
        keep = int(round(columns_available / aspect))
        start = (rows_available - keep) // 2
        pixels = pixels[start : start + keep, :]
    rows = (np.arange(height) * pixels.shape[0] // height).clip(0, pixels.shape[0] - 1)
    columns = (np.arange(width) * pixels.shape[1] // width).clip(0, pixels.shape[1] - 1)
    return np.ascontiguousarray(pixels[rows][:, columns, :3], dtype=np.uint8)


def _upright(orientation: np.ndarray) -> bool:
    """
    Decide whether an object is still standing.

    :param orientation: Scalar-first quaternion.
    :return: Whether its own up axis is within 25 degrees of world up.
    """

    w, x, y, z = (float(value) for value in orientation[:4])
    return bool(1.0 - 2.0 * (x * x + y * y) > math.cos(math.radians(25.0)))


def _yaw_quaternion(angle: float) -> np.ndarray:
    """
    Build a scalar-first quaternion for a rotation about world up.

    :param angle: Yaw in radians.
    :return: ``(4,)`` quaternion, w first.
    """

    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])
