"""
Kinematics for the SO-101 arm, shared by the scenarios and the sim adapter.

Every number here was measured off asset `so101_antioch` 1.3.2 with the arm at
its zero pose, and is only true for that geometry — the version is pinned where
the asset is loaded for exactly that reason.

The arm is easier than a general 6-DOF chain: the pan turns about -Z, and
Pitch, Elbow and Wrist_Pitch all turn about +Y. So it is a planar three-link
chain that the pan sweeps around one vertical axis, tool height is a function
of three angles alone, and the reach below has a closed form.

Nothing here imports a simulator, so it can be tested on a laptop.
"""

from __future__ import annotations

import math

ASSET = "so101_antioch"
ASSET_VERSION = "1.3.2"
ARM_PRIM = "/World/SO101"

JOINT_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
GRIPPER_INDEX = JOINT_ORDER.index("Jaw")
JOINT_LIMITS_DEG = {"Rotation": (-110.0, 110.0), "Pitch": (-100.0, 100.0), "Elbow": (-100.0, 90.0), "Wrist_Pitch": (-95.0, 95.0), "Jaw": (-10.0, 100.0)}

PAN_AXIS_XY = (0.023075, 0.020791)
PAN_COLUMN_Z = 0.094882
SHOULDER_XYZ = (0.053474, 0.002513, 0.149082)
# One entry per pitch joint: segment length, its heading from +X toward +Z at
# the zero pose, and the sideways offset of its far end. Those offsets are
# constant because every one of these joints turns about +Y, which leaves y
# untouched — that is what makes this model exact rather than approximate.
SEGMENTS = ((0.116000, 1.327013, 0.002514), (0.135000, 0.038528, 0.002515), (0.158627, 0.000000, 0.020615))

# The tool centre is taken on the wrist-roll axis, which is also — measured,
# not arranged — the centre line between the two finger faces. A roll therefore
# never moves it, and commanding the tool centre onto an object puts the
# fingers either side of it.
#
# The finger faces converge toward the tip — 23.5 mm apart 15 mm back, 19.8 mm
# at mid-face, 15.8 mm at the tip — so where along them you grasp decides what
# you can hold. The tool centre is placed at the tip because that is the only
# part of the travel that closes on a 2 mL HPLC vial: 12 mm gripped mid-face
# needs the jaw at -7.2 degrees against a -10 degree limit, which leaves no
# squeeze, where at the tip it needs -2.8 and leaves seven degrees of it.
GRIP_REST_GAP = 0.0158
GRIP_LEVER = 0.0768
GRIP_MAX_GAP = GRIP_REST_GAP + GRIP_LEVER
# Only one finger moves. The fixed face sits this far from the tool axis and
# stays there whatever the jaw does, so an object wider than twice this cannot
# be centred on the tool axis — it would already be inside the fixed finger.
# Aiming the tool centre at an object's axis is therefore wrong for anything
# over about 16 mm, which is most of what is worth picking up.
# Confirmed against a measured carry: a gripped vial settles hard against this
# face, which is what `held_offset` below turns into a placement correction.
GRIP_FIXED_FACE = 0.0079
# Below this the fingers cannot close far enough to touch; above it the object
# is wider than the fingers open at their limit.
GRIP_MIN_OBJECT = 0.004
GRIP_MAX_OBJECT = 0.055


def jaw_for_gap(gap: float) -> float:
    """
    Solve the jaw angle that opens the fingers to a given gap.

    :param gap: Distance between the finger faces, in metres.
    :return: Jaw angle in radians, clamped to what the joint can do.
    """

    low, high = JOINT_LIMITS_DEG["Jaw"]
    ratio = max(-1.0, min(1.0, (gap - GRIP_REST_GAP) / GRIP_LEVER))
    return max(math.radians(low), min(math.radians(high), math.asin(ratio)))


def grip_angles(width: float, *, clearance: float = 0.014, squeeze: float = 0.12) -> tuple[float, float]:
    """
    Choose the open and closed jaw commands for one object width.

    The closed command is deliberately short of the angle where the fingers
    meet the object: a position drive answers the leftover error with force, so
    the squeeze is what holds the object, and asking for the hard limit instead
    would drive roughly forty newtons through a plastic cube.

    :param width: Object width across the grasp, in metres.
    :param clearance: Extra gap to leave when approaching, in metres.
    :param squeeze: How far past contact to command, in radians.
    :return: Open and closed jaw angles in radians.
    """

    # Never command into the hard stop. A 12 mm vial meets the fingers at
    # -2.8 degrees and the joint stops at -10, so a full squeeze from there
    # drives the jaw into its own limit: the drive has no error left to hold
    # with, and the vial is spat out instead of gripped. Measured exactly that.
    low = math.radians(JOINT_LIMITS_DEG["Jaw"][0]) + math.radians(2.0)
    contact = jaw_for_gap(width)
    return jaw_for_gap(width + clearance), max(contact - squeeze, low)


def grip_offset(width: float, *, clearance: float = 0.004) -> float:
    """
    Offset the tool centre so the fixed finger clears an object.

    Positive is toward the moving finger, which for a tool pointing straight
    down is radially outward from the arm's base. The result is normally
    negative: the tool aims *short* of the object's axis, and closing the jaw
    then pushes the object back against the fixed face.

    :param width: Object width across the grasp, in metres.
    :param clearance: Gap to leave beside the fixed face on approach.
    :return: Signed offset in metres.
    """

    return -(width / 2.0 - GRIP_FIXED_FACE + clearance)


def held_offset(width: float) -> float:
    """
    Where a gripped object ends up relative to the tool centre.

    Closing the jaw pushes the object against the fixed face and it stays
    there, so this is a fixed, predictable shift rather than noise — which
    makes it a correction worth applying when placing.

    :param width: Object width across the grasp, in metres.
    :return: Signed offset in metres, positive toward the moving finger.
    """

    return width / 2.0 - GRIP_FIXED_FACE


def chain_points(pose: tuple[float, ...]) -> list[tuple[float, float, float]]:
    """
    Place the arm's skeleton in world coordinates for one joint vector.

    :param pose: Joint angles in radians, ordered as :data:`JOINT_ORDER`.
    :return: Base, pan column, shoulder, elbow, wrist and tool-centre points.
    """

    pan_x, pan_y = PAN_AXIS_XY
    x, shoulder_y, z = SHOULDER_XYZ
    points = [(pan_x, pan_y, 0.0), (pan_x, pan_y, PAN_COLUMN_Z), (x, shoulder_y, z)]
    heading = 0.0
    for angle, (length, rest, lateral) in zip(pose[1:4], SEGMENTS, strict=True):
        # Turning about +Y subtracts from a heading measured from +X toward +Z,
        # and each joint carries everything outboard of it
        heading += angle
        x += length * math.cos(rest - heading)
        z += length * math.sin(rest - heading)
        points.append((x, lateral, z))
    turn = -pose[0]
    cosine, sine = math.cos(turn), math.sin(turn)
    return [(pan_x + (px - pan_x) * cosine - (py - pan_y) * sine, pan_y + (px - pan_x) * sine + (py - pan_y) * cosine, pz) for px, py, pz in points]


def tool_reach_height(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """
    Reduce a skeleton to the two numbers a pose is commanded in.

    :param points: Skeleton from :func:`chain_points`.
    :return: Tool distance from the pan axis, and tool height, in metres.
    """

    x, y, z = points[-1]
    return math.hypot(x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1]), z


def bearing_of(x: float, y: float) -> float:
    """
    Measure where a world point sits around the pan axis.

    :param x: World x, in metres.
    :param y: World y, in metres.
    :return: Bearing in radians, zero along +X.
    """

    return math.atan2(y - PAN_AXIS_XY[1], x - PAN_AXIS_XY[0])


def reach_of(x: float, y: float) -> float:
    """
    Measure how far a world point is from the pan axis.

    :param x: World x, in metres.
    :param y: World y, in metres.
    :return: Horizontal distance in metres.
    """

    return math.hypot(x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1])


def pan_for_bearing(bearing: float) -> float:
    """
    Convert a workspace bearing into a pan joint command.

    The pan turns about -Z, so a positive command sweeps toward -Y.

    :param bearing: Bearing in radians, zero along +X.
    :return: Pan joint angle in radians.
    """

    return -bearing


def within_limits(name: str, angle: float) -> bool:
    """
    Report whether one joint angle is inside its authored limits.

    :param name: Joint name from :data:`JOINT_ORDER`.
    :param angle: Candidate angle in radians.
    :return: Whether the angle is reachable.
    """

    low, high = JOINT_LIMITS_DEG[name]
    return math.radians(low) - 1e-9 <= angle <= math.radians(high) + 1e-9


def clamp_pose(pose: tuple[float, ...]) -> tuple[float, ...]:
    """
    Clamp a whole joint vector into the arm's limits.

    :param pose: Joint angles in radians, ordered as :data:`JOINT_ORDER`.
    :return: The same vector with every limited joint inside its range.
    """

    clamped = []
    for name, angle in zip(JOINT_ORDER, pose, strict=True):
        low, high = JOINT_LIMITS_DEG.get(name, (-180.0, 180.0))
        clamped.append(max(math.radians(low), min(math.radians(high), angle)))
    return tuple(clamped)


def solve_reach(reach: float, height: float, tool_pitch_deg: float | None = None) -> tuple[float, float, float] | None:
    """
    Solve the three pitch joints for one tool position.

    Three joints against two constraints leaves the tool's own pitch free, so
    the free angle is scanned and each candidate closed out exactly: the arm
    keeps a natural posture instead of whichever branch an iterative solver
    happened to fall into. Pinning that angle is what a grasp needs — the
    fingers close across the tool axis, so pointing the tool down is what puts
    them either side of something standing on the floor.

    :param reach: Tool distance from the pan axis, in metres.
    :param height: Tool height above the floor, in metres.
    :param tool_pitch_deg: Tool heading to hold, or ``None`` to choose one.
    :return: Pitch, elbow and wrist angles in radians, or ``None`` if the
        position is outside the workspace.
    """

    shoulder_x, _shoulder_y, shoulder_z = SHOULDER_XYZ
    (upper, rest_upper, _), (fore, rest_fore, _), (tool, rest_tool, _) = SEGMENTS
    target_x = PAN_AXIS_XY[0] + reach
    candidates = range(-90, 91) if tool_pitch_deg is None else (tool_pitch_deg,)
    best: tuple[float, tuple[float, float, float]] | None = None
    for tool_degrees in candidates:
        tool_pitch = math.radians(tool_degrees)
        wrist_x = target_x - tool * math.cos(tool_pitch)
        wrist_z = height - tool * math.sin(tool_pitch)
        span_x, span_z = wrist_x - shoulder_x, wrist_z - shoulder_z
        span = math.hypot(span_x, span_z)
        if not abs(upper - fore) + 1e-6 < span < upper + fore - 1e-6:
            continue
        interior = math.acos(max(-1.0, min(1.0, (span * span - upper * upper - fore * fore) / (2.0 * upper * fore))))
        for elbow_up in (1.0, -1.0):
            bend = elbow_up * interior
            heading_upper = math.atan2(span_z, span_x) + math.atan2(fore * math.sin(bend), upper + fore * math.cos(bend))
            pitch = rest_upper - heading_upper
            elbow = rest_fore - pitch - (heading_upper - bend)
            wrist = rest_tool - pitch - elbow - tool_pitch
            if not all(within_limits(name, angle) for name, angle in zip(("Pitch", "Elbow", "Wrist_Pitch"), (pitch, elbow, wrist), strict=True)):
                continue
            # Prefer the posture closest to the load pose with the tool nearest
            # level: both keep the arm away from its own limits, where a drive
            # that cannot quite reach its target reads as a tracking failure
            score = pitch * pitch + elbow * elbow + wrist * wrist + 0.35 * tool_pitch * tool_pitch
            if best is None or score < best[0]:
                best = (score, (pitch, elbow, wrist))
    return None if best is None else best[1]


def closest_reachable(reach: float, height: float, tool_pitch_deg: float | None = None) -> tuple[tuple[float, float, float], float] | None:
    """
    Solve for a tool position, or for the nearest height that can be held.

    :param reach: Tool distance from the pan axis, in metres.
    :param height: Requested tool height, in metres.
    :param tool_pitch_deg: Tool heading to hold, or ``None`` to choose one.
    :return: Joint solution and the height it actually reaches, or ``None``
        when no height at this reach is solvable.
    """

    solution = solve_reach(reach, height, tool_pitch_deg)
    if solution is not None:
        return solution, height
    for millimetres in range(1, 201):
        for candidate in (height - millimetres / 1000.0, height + millimetres / 1000.0):
            solution = solve_reach(reach, candidate, tool_pitch_deg)
            if candidate > 0.02 and solution is not None:
                return solution, candidate
    return None


def pose_at(x: float, y: float, height: float, *, jaw: float = 0.0, tool_pitch_deg: float | None = -90.0) -> tuple[float, ...] | None:
    """
    Build a whole joint vector that puts the tool centre over a world point.

    :param x: World x of the target, in metres.
    :param y: World y of the target, in metres.
    :param height: Tool-centre height above the floor, in metres.
    :param jaw: Jaw angle to hold, in radians.
    :param tool_pitch_deg: Tool heading to hold, or ``None`` to choose one.
    :return: Joint vector ordered as :data:`JOINT_ORDER`, or ``None`` if the
        point is outside the workspace.
    """

    solution = solve_reach(reach_of(x, y), height, tool_pitch_deg)
    if solution is None:
        return None
    return (pan_for_bearing(bearing_of(x, y)), *solution, 0.0, jaw)


def blend(start: tuple[float, ...], end: tuple[float, ...], fraction: float) -> tuple[float, ...]:
    """
    Ease between two joint vectors.

    A position drive handed a step change answers with an overshoot, so
    waypoints are blended with a smoothstep rather than a ramp.

    :param start: Joint vector to leave, in radians.
    :param end: Joint vector to arrive at, in radians.
    :param fraction: Progress from 0 to 1.
    :return: The blended joint vector.
    """

    eased = fraction * fraction * (3.0 - 2.0 * fraction)
    return tuple(first + (second - first) * eased for first, second in zip(start, end, strict=True))


def trajectory(waypoints: list[tuple[str, float, tuple[float, ...]]], steps: int) -> list[tuple[str, tuple[float, ...]]]:
    """
    Expand labelled waypoints into one commanded joint vector per step.

    :param waypoints: Ordered ``(label, share, pose)`` triples. The share is
        the fraction of the whole move that segment occupies.
    :param steps: Steps the whole move should occupy.
    :return: One ``(label, pose)`` pair per step.
    """

    total = sum(share for _label, share, _pose in waypoints)
    schedule: list[tuple[str, tuple[float, ...]]] = []
    previous = waypoints[0][2]
    elapsed = 0.0
    for label, share, pose in waypoints:
        elapsed += share
        span = max(1, round(steps * elapsed / total) - len(schedule))
        for index in range(span):
            schedule.append((label, blend(previous, pose, (index + 1) / span)))
        previous = pose
    return schedule


def articulation_root(prim_path: str) -> str:
    """
    Find the prim PhysX will treat as the articulation root.

    A reference lands the arm's own hierarchy *under* the prim it was made at,
    and the root API sits on the base link inside it. Pointing ``Robot`` at the
    reference prim instead leaves the articulation unresolved.

    :param prim_path: Prim path the asset was referenced at.
    :return: Path of the prim carrying the articulation root API.
    :raises RuntimeError: If no such prim exists beneath it.
    """

    import antioch
    from pxr import Usd

    for prim in Usd.PrimRange(antioch.stage().GetPrimAtPath(prim_path)):
        if "PhysicsArticulationRootAPI" in prim.GetAppliedSchemas():
            return str(prim.GetPath())
    raise RuntimeError(f"no articulation root found beneath '{prim_path}'")
