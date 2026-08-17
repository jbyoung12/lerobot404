"""
Example scenarios for Isaac Sim: one fast check, and one parameter sweep.

    antioch scenario run --scenario falling_cube        one run
    antioch suite run smoke                  the fast check
    antioch suite run sweep                  every case on one machine
    antioch suite run sweep --machines 4     opt into multi-machine fan-out

The sweep shows how one scenario declaration expands into independent cases.
Queue staging adds the submitted project source to the immutable sim image;
add a Dockerfile only when the project needs custom packages or another image
layer. Development watch rules are not part of queued runs.

`falling_cube` publishes pictures and 3D geometry; `cube_bounce` is
deliberately scalar-only, because six sweep cases want one comparable chart
rather than six sets of frames. `so101_move_to_pose` drives an articulated arm
and publishes all three: pictures, an animated skeleton, and the height it was
asked to hold.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import antioch

# The runner loads this file by path, so its directory is not necessarily on
# the import path. Put it there before reaching for the project's own modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import record  # noqa: E402 - needs the path above
import sim  # noqa: E402 - needs the path above
import teleop  # noqa: E402 - needs the path above
import so101  # noqa: E402 - needs the path above

logger = antioch.Logger("cube")
arm_logger = antioch.Logger("arm")
vial_logger = antioch.Logger("vial")


# `capture=False` turns off the automatic platform viewport, which points
# wherever Kit last left it. The pictures below are authored instead: aimed at
# the drop, and published only when the frame is worth reviewing.
@antioch.scenario(tags=["smoke"], capture=False)
def falling_cube(
    run: antioch.ScenarioRun,
    drop_height: float = antioch.param(2.0, ge=0.5, le=10.0, description="Initial cube height in meters"),
    steps: int = antioch.param(180, ge=1, description="Physics steps to simulate"),
) -> None:
    """
    Drop a dynamic cube and verify that it settles on the ground.
    """

    import numpy as np
    import rerun as rr
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view

    world = antioch.world()
    world.scene.add_ground_plane(restitution=0.0)
    # A dome alone flattens the scene; the distant light casts the shadow that
    # makes height readable in a still frame
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 200.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 400.0})
    cube = world.scene.add(
        DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.0, 0.0, drop_height]), size=0.5, color=np.array([0.2, 0.4, 0.9]))
    )
    world.reset()
    # Frame the whole drop, not a fixed vantage. The distance has to grow with
    # the height or a tall drop starts above the frame and lands below it —
    # measured at drop_height 10, where a fixed camera published no usable
    # picture at all and the run still passed.
    span = max(drop_height, 1.0)
    set_camera_view(eye=[1.4 * span, 1.4 * span, 0.75 * span], target=[0.0, 0.0, 0.45 * span], camera_prim_path="/OmniverseKit_Persp")
    # Boxes draw geometry; a transform alone would position an empty 3D pane
    logger.value("scene/ground", rr.Boxes3D(centers=[[0.0, 0.0, -0.025]], sizes=[[4.0, 4.0, 0.05]], colors=[[110, 110, 110]]))

    # The first capture of a run hands back a frame from before the scene was
    # drawn — measured as the bare Kit grid with no subject in it. It follows
    # the first call, not any particular step, so no cadence avoids it, and it
    # sits well inside the exposure band, so no check below can tell it from a
    # real picture. Spend it here and the review set never contains it.
    antioch.capture_viewport()

    attempted = 0
    review_frames = 0
    for step in range(steps):
        # Rendering is what gives `capture_viewport` something to return
        world.step(render=step % 3 == 0)
        position = cube.get_world_pose()[0]
        logger.scalar("height", float(position[2]))
        logger.value("scene/body", rr.Boxes3D(centers=[position.tolist()], sizes=[[0.5, 0.5, 0.5]], colors=[[51, 102, 230]]))
        # A 2 m drop is over in about 38 steps at 60 Hz, so every sixtieth step
        # photographed the cube twice and both times after it had landed
        if step % 10 == 0:
            attempted += 1
            frame = antioch.capture_viewport()
            if frame is not None:
                rgb = np.asarray(frame)[:, :, :3]
                # A black or blown-out picture is worse evidence than none
                if 10.0 <= float(rgb.mean()) <= 220.0:
                    logger.image("camera/rgb", rgb)
                    review_frames += 1

    final_z = float(cube.get_world_pose()[0][2])
    run.add_result("final_z", round(final_z, 4))
    # Both counts, so a run that published nothing says so instead of looking
    # like a run that never tried
    run.add_result("review_frames", review_frames)
    run.add_result("review_frames_attempted", attempted)
    run.check("the cube came to rest on the ground", final_z < 0.4, detail=f"cube centre rested at {final_z:.3f} m")
    run.check("the review camera published usable frames", review_frames > 0, detail=f"{review_frames} of {attempted} attempts passed the exposure gate")


@antioch.scenario(
    tags=["sweep"], capture=False, cases=[antioch.case(grid={"drop_height": [0.5, 2.0, 6.0], "restitution": [0.0, 0.7]}, id="h{drop_height}-e{restitution}")]
)
def cube_bounce(
    run: antioch.ScenarioRun,
    drop_height: float = antioch.param(2.0, ge=0.1, le=10.0, description="Initial cube height in meters"),
    restitution: float = antioch.param(0.0, ge=0.0, le=1.0, description="How bouncy the ground is"),
    steps: int = antioch.param(240, ge=1, description="Physics steps to simulate"),
) -> None:
    """
    Drop a cube onto ground of varying bounciness and measure the rebound.

    Six children from one declaration: three heights against two materials.
    """

    import numpy as np
    from isaacsim.core.api.objects import DynamicCuboid

    world = antioch.world()
    world.scene.add_ground_plane(restitution=restitution)
    cube = world.scene.add(
        DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.0, 0.0, drop_height]), size=0.3, color=np.array([0.9, 0.4, 0.2]))
    )
    world.reset()

    # The rebound is the highest point AFTER the cube first reaches the floor,
    # so the drop itself is never mistaken for a bounce
    landed = False
    rebound = 0.0
    for _ in range(steps):
        world.step(render=False)
        height = float(cube.get_world_pose()[0][2])
        logger.scalar("height", height)
        landed = landed or height < 0.2
        if landed:
            rebound = max(rebound, height)

    run.add_result("rebound_height", round(rebound, 4))
    # Both criteria are recorded even when the first one fails, so a run that
    # never reached the floor still reports what its rebound measured
    run.check("the cube reached the floor", landed, detail=f"within {steps} steps from {drop_height:.2f} m")
    run.check("the cube rebounded no higher than it was dropped", rebound < drop_height, detail=f"rebounded to {rebound:.3f} m from {drop_height:.2f} m")


# ── The move ─────────────────────────────────────────────────────────────────
#
# The kinematics live in `so101.py`, shared with the sim adapter: one copy of
# the geometry, so a number measured off the asset can never be right in one
# file and stale in another.


def _plan(goal: tuple[float, ...], park: tuple[float, ...], sweep: float, grip: float) -> list[tuple[str, float, tuple[float, ...]]]:
    """
    Lay out the move as named waypoints and the share of the run each takes.

    :param goal: Commanded joint vector at the target pose.
    :param park: Compact joint vector the arm passes through on the way.
    :param sweep: Pan angle swept either side of the goal, in radians.
    :param grip: Jaw opening used for the grip cycle, in radians.
    :return: Ordered ``(label, share, pose)`` waypoints.
    """

    def _with(pose: tuple[float, ...], **changes: float) -> tuple[float, ...]:
        edited = list(pose)
        for name, value in changes.items():
            edited[so101.JOINT_ORDER.index(name)] = value
        return tuple(edited)

    return [
        ("settle", 0.07, (0.0,) * 6),
        ("park", 0.13, park),
        ("reach", 0.20, goal),
        ("open jaw", 0.07, _with(goal, Jaw=grip)),
        ("close jaw", 0.07, goal),
        ("sweep left", 0.13, _with(goal, Rotation=sweep)),
        ("sweep right", 0.16, _with(goal, Rotation=-sweep)),
        ("recentre", 0.11, goal),
        ("hold", 0.06, goal),
    ]


# reviewing.
@antioch.scenario(tags=["smoke"], capture=False)
def so101_move_to_pose(
    run: antioch.ScenarioRun,
    target_height: float = antioch.param(0.30, ge=0.08, le=0.45, description="Target tool-centre height in meters"),
    target_reach: float = antioch.param(0.25, ge=0.12, le=0.32, description="Target tool-centre distance from the pan axis in meters"),
    steps: int = antioch.param(360, ge=120, description="Physics steps to simulate"),
) -> None:
    """
    Drive the SO-101 arm to a commanded tool pose and verify that it holds it.

    The arm parks, reaches the pose, works the jaw, sweeps the pose around the
    pan axis and returns to it. Every step publishes the measured skeleton and
    the measured tool height, so the recording is an animation of the move
    rather than a picture of a robot standing still.
    """

    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view

    world = antioch.world()
    world.scene.add_ground_plane(restitution=0.0)
    # A dome alone flattens the scene; the distant light casts the shadow that
    # separates the arm from the floor in a still frame
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 200.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 400.0})
    antioch.load_asset(so101.ASSET, prim_path=so101.ARM_PRIM, version=so101.ASSET_VERSION)
    arm = world.scene.add(Robot(prim_path=so101.articulation_root(so101.ARM_PRIM), name="so101"))

    reached = so101.closest_reachable(target_reach, target_height)
    if reached is None:
        run.fail(f"no tool height at reach {target_reach:.2f} m is inside the SO-101 workspace")
    solution, commanded_height = reached
    goal = (0.0, *solution, 0.0, 0.0)
    goal_xyz = so101.chain_points(goal)[-1]
    parked = so101.closest_reachable(0.17, 0.12)
    park = (0.0, *(parked[0] if parked is not None else solution), 0.0, 0.0)
    # A marker the review camera can see too, so a frame shows what the arm was
    # aiming at. Visual only: a collider here would be something to knock over.
    world.scene.add(VisualCuboid(prim_path="/World/goal", name="goal", position=np.array(goal_xyz), size=0.025, color=np.array([0.92, 0.28, 0.24])))
    world.reset()

    names = list(arm.dof_names)
    missing = [name for name in so101.JOINT_ORDER if name not in names]
    if missing:
        run.fail(f"the SO-101 articulation is missing {missing}; it reports {names}")
    order = [names.index(name) for name in so101.JOINT_ORDER]
    indices = np.array(order, dtype=np.int32)

    # Far enough back for the whole sweep to stay in frame, and off to one side
    # so the arm is read as an arm rather than as a line pointing at the lens
    set_camera_view(eye=[0.92, -0.78, 0.62], target=[0.10, 0.0, 0.20], camera_prim_path="/OmniverseKit_Persp")
    # The automatic layout roots its 3D view at "/", which pulls the camera
    # images into it and leaves the viewer showing "2D visualizers require a
    # pinhole ancestor". Rooting that view at the entities that really are 3D
    # is the fix. The time panel is opened deliberately: it is hidden by
    # default, and a hidden one leaves an animation parked on one frame.
    run.set_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/arm/world", name="Arm"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/arm/camera/rgb", name="Review camera"),
                    rrb.TimeSeriesView(origin="/arm/tool", name="Tool centre (m)"),
                    rrb.TimeSeriesView(origin="/arm/joints", name="Joint angles (deg)"),
                    row_shares=[4, 3, 3],
                ),
                column_shares=[3, 2],
            ),
            rrb.TimePanel(state=rrb.PanelState.Expanded),
        )
    )

    arm_logger.value("world/ground", rr.Boxes3D(centers=[[0.0, 0.0, -0.025]], sizes=[[2.0, 2.0, 0.05]], colors=[[110, 110, 110]]))
    arm_logger.value("world/goal", rr.Points3D([goal_xyz], colors=[[235, 70, 55]], radii=[0.018]))
    # The pan sweep traces this ring exactly, which makes the kinematics visible
    # rather than something the checks alone have to be trusted about
    ring = [
        (so101.PAN_AXIS_XY[0] + target_reach * math.cos(math.tau * turn / 96), so101.PAN_AXIS_XY[1] + target_reach * math.sin(math.tau * turn / 96), commanded_height)
        for turn in range(97)
    ]
    arm_logger.value("world/goal_ring", rr.LineStrips3D([ring], colors=[[235, 70, 55, 110]], radii=[0.0015]))

    # The first capture of a run hands back a frame from before the scene was
    # drawn — measured as the bare Kit grid with no arm in it. Spend it here and
    # the review set never contains it.
    antioch.capture_viewport()

    schedule = so101.trajectory(_plan(goal, park, math.radians(55.0), math.radians(55.0)), steps)
    trail: list[tuple[float, float, float]] = []
    measured: tuple[float, ...] = (0.0,) * 6
    phase = ""
    attempted = 0
    frames = 0
    travel = 0.0
    settled_error = 0.0
    for step, (label, commanded) in enumerate(schedule):
        if label != phase:
            phase = label
            arm_logger.info(f"step {step}: {label}")
        arm.apply_action(ArticulationAction(joint_positions=np.array(commanded, dtype=np.float32), joint_indices=indices))
        # Rendering is what gives `capture_viewport` something to return
        world.step(render=step % 3 == 0)

        readings = arm.get_joint_positions()
        measured = tuple(float(readings[index]) for index in order)
        points = so101.chain_points(measured)
        reach, height = so101.tool_reach_height(points)
        if trail:
            travel += math.dist(points[-1], trail[-1])
        trail.append(points[-1])
        if label == "hold":
            settled_error = max(settled_error, max(abs(read - told) for read, told in zip(measured, commanded, strict=True)))

        arm_logger.scalar("tool/height", height)
        arm_logger.scalar("tool/commanded_height", commanded_height)
        arm_logger.scalar("tool/reach", reach)
        for name, angle in zip(so101.JOINT_ORDER, measured, strict=True):
            arm_logger.scalar(f"joints/{name}", math.degrees(angle))
        arm_logger.value("world/links", rr.LineStrips3D([points], colors=[[236, 188, 66]], radii=[0.007]))
        arm_logger.value("world/joints", rr.Points3D(points, colors=[[48, 48, 62]], radii=[0.011]))
        arm_logger.value("world/tool", rr.Points3D([points[-1]], colors=[[64, 168, 255]], radii=[0.016]))
        arm_logger.value("world/trail", rr.LineStrips3D([trail], colors=[[64, 168, 255]], radii=[0.0025]))

        # Every sixth step is also a rendered one, so each capture is a fresh
        # frame rather than the same picture read back twice
        if step % 6 == 0:
            attempted += 1
            frame = antioch.capture_viewport()
            if frame is not None:
                rgb = np.asarray(frame)[:, :, :3]
                # A black or blown-out picture is worse evidence than none
                if 10.0 <= float(rgb.mean()) <= 220.0:
                    arm_logger.image("camera/rgb", rgb)
                    frames += 1

    final_reach, final_height = so101.tool_reach_height(so101.chain_points(measured))
    height_error = abs(final_height - commanded_height)
    run.add_result("commanded_height", round(commanded_height, 4))
    run.add_result("final_tool_height", round(final_height, 4))
    run.add_result("final_tool_reach", round(final_reach, 4))
    run.add_result("height_error", round(height_error, 4))
    run.add_result("tool_travel", round(travel, 3))
    run.add_result("settled_joint_error_deg", round(math.degrees(settled_error), 3))
    run.add_result("goal_joints_deg", [round(math.degrees(angle), 2) for angle in goal])
    run.add_result("steps_simulated", len(schedule))
    # Both counts, so a run that published nothing says so instead of looking
    # like a run that never tried
    run.add_result("review_frames", frames)
    run.add_result("review_frames_attempted", attempted)

    # Every criterion is recorded even when an earlier one fails, so a run that
    # missed the height still reports whether the arm moved at all
    run.check(
        "the requested pose is inside the arm's workspace",
        abs(commanded_height - target_height) < 1e-9,
        detail=f"asked for {target_height:.3f} m at {target_reach:.3f} m reach, held {commanded_height:.3f} m",
    )
    # The check this scenario exists for. Its predecessor loaded the arm,
    # commanded nothing, and passed on a recording of a robot standing still
    run.check("the arm moved rather than holding its load pose", travel > 0.2, detail=f"tool centre travelled {travel:.3f} m")
    run.check("the tool reached the commanded height", height_error <= 0.015, detail=f"settled {height_error * 1000:.1f} mm from {commanded_height:.3f} m")
    run.check("the drives held the pose they were given", settled_error <= math.radians(2.0), detail=f"{math.degrees(settled_error):.2f}° off at the hold")
    run.check("the review camera published usable frames", frames > 0, detail=f"{frames} of {attempted} attempts passed the exposure gate")


# The scenario that answers the only question the project cannot route around:
# can this gripper actually pick a vial up. Everything downstream — teleop,
# recording, the trained policy, retry — assumes it can.
@antioch.scenario(tags=["smoke"], capture=False)
def vial_pick_place(
    run: antioch.ScenarioRun,
    layout: str = antioch.param("train", description="Scene layout: train, eval or ambiguous"),
    episode: int = antioch.param(0, ge=0, description="Episode index, which seeds where the vials spawn"),
    source: str = antioch.param("", description="Well to lift from, e.g. A3. Empty picks whichever well the layout filled"),
    destination: str = antioch.param("C5", description="Well to place into, e.g. C5"),
    ticks_per_phase: int = antioch.param(12, ge=4, description="Control ticks per waypoint segment"),
) -> None:
    """
    Run the scripted expert through the sim adapter and grade the grasp.

    This is also the recording loop the teleop path uses, with the leader arm
    swapped in for the expert, so an episode recorded here has exactly the
    shape an episode recorded by hand will have.
    """

    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    sim.reset(layout, episode=episode)
    before = sim.occupancy()
    # The task is stated by location: whatever stands in the source well is
    # what moves. A bench of amber vials has no "red one" to ask for.
    goal_object = sim.vial_in_well(source) if source else ("red vial A" if layout == "ambiguous" else "red vial")
    if goal_object is None:
        run.fail(f"no vial is standing in {source}; the rack holds {before}")
    sim.set_goal(goal_object, destination)
    vial_logger.info(f"move {source or 'the ' + goal_object} to {destination}; rack holds {before}")
    backend = sim._active()

    run.set_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/vial/world", name="Scene"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/vial/camera/wrist", name="Wrist camera"),
                    rrb.TimeSeriesView(origin="/vial/gripper", name="Gripper: commanded vs measured (deg)"),
                    rrb.TimeSeriesView(origin="/vial/object", name="Vial height (m)"),
                    row_shares=[4, 3, 3],
                ),
                column_shares=[3, 2],
            ),
            rrb.TimePanel(state=rrb.PanelState.Expanded),
        )
    )
    vial_logger.value("world/ground", rr.Boxes3D(centers=[[0.0, 0.0, -0.025]], sizes=[[1.6, 1.6, 0.05]], colors=[[110, 110, 110]]))
    for name, record in sim.object_poses().items():
        if record["kind"] == "slot":
            vial_logger.value(f"world/slots/{name.replace(' ', '_')}", rr.Points3D([record["position"]], colors=[[90, 200, 120]], radii=[0.012]))

    contact = so101.jaw_for_gap(sim.VIAL_DIAMETER)
    start_height = sim.object_poses()[goal_object]["position"][2]
    actions = sim.expert_actions(ticks_per_phase=ticks_per_phase)
    labels = backend.expert_labels(ticks_per_phase)

    states, commands = [], []
    peak_height = start_height
    grip_measured: list[float] = []
    phase = ""
    frames = 0
    for tick, (label, action) in enumerate(zip(labels, actions, strict=True)):
        if label != phase:
            phase = label
            vial_logger.info(f"tick {tick}: {label}")
        # Read BEFORE commanding: this ordering is what makes the state the
        # observation the action was chosen from, and the reverse is the label
        # bug that trains a policy to predict where it already is
        state = sim.joint_state()
        picture = sim.frame()
        sim.set_targets(action)
        states.append(state)
        commands.append(np.asarray(action, dtype=np.float32))

        measured = sim.joint_state()
        poses = sim.object_poses()
        held = poses[goal_object]["position"]
        peak_height = max(peak_height, held[2])
        if label in {"lift", "transit"}:
            # The gap between what the jaw was told and where it stopped is the
            # whole monitor signal: a jaw that reaches its own command closed on
            # nothing. Compared against a modelled contact angle instead, this
            # read as a failure on a grasp that was in fact holding the vial.
            grip_measured.append(float(measured[sim.GRIPPER]) - float(action[sim.GRIPPER]))

        vial_logger.scalar("gripper/commanded", math.degrees(float(action[sim.GRIPPER])))
        vial_logger.scalar("gripper/measured", math.degrees(float(measured[sim.GRIPPER])))
        vial_logger.scalar("gripper/contact_angle", math.degrees(contact))
        vial_logger.scalar("object/height", held[2])
        points = so101.chain_points(tuple(float(value) for value in measured))
        vial_logger.value("world/links", rr.LineStrips3D([points], colors=[[236, 188, 66]], radii=[0.006]))
        vial_logger.value("world/tool", rr.Points3D([points[-1]], colors=[[64, 168, 255]], radii=[0.012]))
        vial_logger.value(
            "world/objects",
            rr.Points3D(
                [record["position"] for name, record in poses.items() if record["kind"] != "slot"],
                colors=[[235, 60, 50] if "red" in name else [70, 110, 235] if "blue" in name else [70, 200, 90] for name in poses if poses[name]["kind"] != "slot"],
                radii=[0.012],
            ),
        )
        if tick % 2 == 0:
            vial_logger.image("camera/wrist", picture)
            frames += 1

    actions_array, states_array = np.array(commands), np.array(states)
    label_gap = float(np.abs(actions_array - states_array).max())
    settled = sim.object_poses()[goal_object]["position"]
    slot = sim.object_poses()[destination]["position"]
    miss = math.dist(settled[:2], slot[:2])
    held_blocked = min(grip_measured) if grip_measured else float("nan")

    run.add_result("backend", sim.backend_name())
    run.add_result("frame_shape", list(sim.frame().shape))
    run.add_result("native_frame_shape", list(backend.native_frame_shape() or []))
    run.add_result("ticks", len(actions))
    run.add_result("start_height", round(start_height, 4))
    run.add_result("peak_object_height", round(peak_height, 4))
    run.add_result("final_object_height", round(settled[2], 4))
    run.add_result("slot_miss_m", round(miss, 4))
    run.add_result("contact_angle_deg", round(math.degrees(contact), 3))
    run.add_result("jaw_blocked_by_vial_deg", round(math.degrees(held_blocked), 3))
    # The number Person D's monitor thresholds on. A jaw holding a vial stops
    # short of its command by this much; a jaw that closed on air reaches it.
    run.add_result("suggested_closed_threshold_rad", round(held_blocked / 2.0, 4))
    run.add_result("action_state_max_gap_rad", round(label_gap, 4))
    run.add_result("wrist_frames", frames)
    run.add_result("rack_before", {well: name for well, name in before.items()})
    run.add_result("rack_after", {well: name for well, name in sim.occupancy().items()})

    run.check("actions differ from measured states", label_gap > 0.01, detail=f"largest gap {label_gap:.4f} rad")
    run.check("the wrist camera published frames", frames > 0, detail=f"{frames} frames at {list(sim.IMAGE_SHAPE)}")
    run.check(
        "the jaw closed onto the vial rather than onto air",
        held_blocked > math.radians(0.5),
        detail=f"the vial blocked the jaw {math.degrees(held_blocked):.2f}° short of its command",
    )
    run.check("the vial left the table", peak_height > start_height + 0.03, detail=f"rose from {start_height:.3f} m to {peak_height:.3f} m")
    run.check(f"the vial ended in {destination}", sim.score(), detail=f"settled {miss * 1000:.0f} mm from the slot centre, {settled[2]:.3f} m up")


# The scenario that produces training data. The action source is deliberately
# swappable: the scripted expert today, the leader arm once its bridge is up.
# Both write the same file, so the format is proven before a human spends
# twenty minutes driving.
@antioch.scenario(tags=["record"], capture=False)
def record_demos(
    run: antioch.ScenarioRun,
    episodes: int = antioch.param(25, ge=1, le=200, description="Episodes to record"),
    layout: str = antioch.param("train", description="Scene layout: train, eval or ambiguous"),
    destination: str = antioch.param("C5", description="Well to place into, e.g. C5"),
    ticks_per_phase: int = antioch.param(12, ge=4, description="Control ticks per waypoint segment"),
) -> None:
    """
    Record demonstration episodes into one portable archive.

    Every episode re-randomises where the vials stand, which is what lets a
    policy trained on this handle positions it never saw. The archive converts
    into a LeRobot dataset on the laptop; LeRobot is deliberately not a
    dependency of the sim image.
    """

    import numpy as np

    recorder = record.EpisodeRecorder(task=f"put the vial in {destination}", fps=sim.CONTROL_HZ)
    scored = 0
    lifted = 0
    for episode in range(episodes):
        sim.reset(layout, episode=episode)
        goal_object = "red vial A" if layout == "ambiguous" else "red vial"
        source = next((well for well, name in sim.occupancy().items() if name == goal_object), None)
        sim.set_goal(goal_object, destination)
        recorder.start_episode(task=f"move {source} to {destination}" if source else f"put the vial in {destination}")

        start_height = sim.object_poses()[goal_object]["position"][2]
        peak = start_height
        try:
            actions = sim.expert_actions(ticks_per_phase=ticks_per_phase)
        except RuntimeError as error:
            vial_logger.warning(f"episode {episode} skipped: {error}")
            recorder.end_episode(False)
            continue
        for action in actions:
            # Read before commanding: this ordering is what makes the state the
            # observation the action was chosen from
            recorder.add(sim.joint_state(), action, sim.frame())
            sim.set_targets(action)
            peak = max(peak, sim.object_poses()[goal_object]["position"][2])
        success = sim.score()
        recorder.end_episode(success)
        scored += int(success)
        lifted += int(peak > start_height + 0.02)
        vial_logger.info(f"episode {episode}: {source} -> {destination}, lifted {peak - start_height:.3f} m, scored {success}")

    archive = recorder.save(f"/tmp/{run.name}_{layout}.npz")
    size_mb = archive.stat().st_size / 1e6
    run.add_artifact(archive, name="episodes.npz")

    gap = recorder.label_gap()
    run.add_result("episodes_recorded", recorder.episodes)
    run.add_result("frames_recorded", recorder.frames)
    run.add_result("archive_mb", round(size_mb, 1))
    run.add_result("episodes_lifted", lifted)
    run.add_result("episodes_scored", scored)
    run.add_result("action_state_max_gap_rad", round(gap, 4))
    run.add_result("fps", sim.CONTROL_HZ)

    # The assert that matters most: labels equal to observations train a policy
    # to predict where it already is, and it freezes mid-task at inference
    run.check("actions differ from states", gap > 0.01, detail=f"largest gap {gap:.4f} rad")
    run.check("every episode recorded frames", recorder.frames >= recorder.episodes * 10, detail=f"{recorder.frames} frames over {recorder.episodes} episodes")
    run.check("the arm lifted the vial in most episodes", lifted >= max(1, int(0.6 * episodes)), detail=f"{lifted} of {episodes} lifted clear")
    run.check("the archive is small enough to download", size_mb < 900.0, detail=f"{size_mb:.1f} MB")


# The teleop counterpart of `record_demos`: same recorder, same archive, but the
# actions come from a human's hand instead of the scripted expert. Recording
# both through one path is what makes the two mixable in a single dataset.
@antioch.scenario(tags=["teleop"], capture=False)
def teleop_record(
    run: antioch.ScenarioRun,
    layout: str = antioch.param("train", description="Scene layout: train, eval or ambiguous"),
    destination: str = antioch.param("C5", description="Well the operator is placing into"),
    wait_s: float = antioch.param(120.0, ge=5.0, description="Seconds to wait for the bridge before giving up"),
    session_s: float = antioch.param(1200.0, ge=10.0, description="Longest session to hold the machine for"),
) -> None:
    """
    Drive the arm from a leader arm on the operator's laptop and record it.

    Start this first, then run `tools/leader_bridge.py` on the laptop. The
    scenario holds the machine open, following the leader and recording only
    while the operator has an episode open.
    """

    import time

    import numpy as np

    sim.reset(layout, episode=0)
    feed = teleop.LeaderFeed()
    recorder = record.EpisodeRecorder(task=f"put the vial in {destination}", fps=sim.CONTROL_HZ)
    vial_logger.info(f"listening for the leader on port {teleop.TELEOP_PORT}; start tools/leader_bridge.py now")

    deadline = time.monotonic() + wait_s
    while feed.received == 0 and time.monotonic() < deadline:
        time.sleep(0.25)
    if feed.received == 0:
        feed.close()
        run.fail(f"no leader poses arrived within {wait_s:.0f}s; is the bridge running and the port published?")
    vial_logger.info("leader connected")

    episodes_seen: set[int] = set()
    open_episode: int | None = None
    ends = time.monotonic() + session_s
    frames = 0
    try:
        while time.monotonic() < ends:
            pose, recording, episode, finished = feed.snapshot()
            if finished:
                break
            if pose is None:
                time.sleep(0.01)
                continue
            action = sim.leader_to_joints(dict(zip(sim.LEADER_JOINTS, pose, strict=True)))
            if recording and open_episode != episode:
                # A fresh episode starts from a fresh scene, so each recording
                # is a whole task rather than a fragment of the previous one
                sim.reset(layout, episode=episode)
                sim.set_goal("red vial", destination)
                recorder.start_episode(task=f"put the vial in {destination}")
                open_episode = episode
                episodes_seen.add(episode)
                vial_logger.info(f"recording episode {episode}")
            if recording and open_episode == episode:
                recorder.add(sim.joint_state(), action, sim.frame())
                frames += 1
            elif open_episode is not None and not recording:
                success = sim.score()
                recorder.end_episode(success)
                vial_logger.info(f"episode {open_episode} closed, scored {success}")
                open_episode = None
            sim.set_targets(action)
    finally:
        if open_episode is not None:
            recorder.end_episode(sim.score())
        feed.close()

    run.add_result("poses_received", feed.received)
    run.add_result("episodes_recorded", recorder.episodes)
    run.add_result("frames_recorded", frames)
    if recorder.frames:
        archive = recorder.save(f"/tmp/{run.name}_{layout}.npz")
        run.add_artifact(archive, name="episodes.npz")
        run.add_result("archive_mb", round(archive.stat().st_size / 1e6, 1))
        gap = recorder.label_gap()
        run.add_result("action_state_max_gap_rad", round(gap, 4))
        run.check("actions differ from states", gap > 0.01, detail=f"largest gap {gap:.4f} rad")
    run.check("the leader arm was connected", feed.received > 0, detail=f"{feed.received} poses received")
    run.check("at least one episode was recorded", recorder.episodes > 0, detail=f"{recorder.episodes} episodes, {frames} frames")
