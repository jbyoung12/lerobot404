# lerobot404

Language-tasked pick-and-place for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm, simulated in Isaac Sim, with a data pipeline that feeds [LeRobot](https://github.com/huggingface/lerobot).

Say *"move A3 to C5"*. The arm lifts the vial out of one well of a rack and puts it in another. When it fumbles it notices, retreats and retries; if it is still stuck, it shows you the camera frame and asks.

## What's here

| file | does |
|---|---|
| `src/so101.py` | Arm kinematics: closed-form IK, joint limits, gripper geometry. Imports no simulator, so it runs on a laptop. |
| `src/sim.py` | **The one adapter.** Everything else imports this. Real backend inside Isaac Sim, mock backend on a laptop. |
| `src/record.py` | Records episodes to a portable archive: state, action, camera frames, task. |
| `src/teleop.py` | Receives leader-arm poses inside the simulator. |
| `src/scenarios.py` | Antioch scenarios: the move, the graded pick-and-place, and both recorders. |
| `tools/leader_bridge.py` | Laptop side: reads an SO-101 leader over USB and streams it to the sim. |
| `tools/to_lerobot.py` | Archive → LeRobot dataset, ready for `lerobot-train`. |

## The adapter contract

Frozen, so five people can build against it at once:

```python
reset(scenario)   -> None                     # "train" | "eval" | "ambiguous"
frame()           -> (480, 640, 3) uint8      # workspace camera
joint_state()     -> (6,) float32 radians     # MEASURED
set_targets(q)    -> None                     # COMMANDED, and steps one control tick
object_poses()    -> dict                     # ground truth — EVAL ORACLE ONLY
score()           -> bool
```

`set_targets` also advances the simulation by one control tick, because in a
simulator something has to step time. Perception comes from `frame()`;
`object_poses()` is for scoring, not for planning.

Run it on a laptop with no GPU and no Isaac — the mock backend answers, and its
gripper still refuses to close on an object it is holding, so monitor and retry
logic can be written and tested before the real scene exists:

```python
import sim
sim.reset("train"); sim.set_goal("red vial", "C5")
for action in sim.expert_actions():
    sim.set_targets(action)
print(sim.score(), sim.occupancy())
```

## Actions are not states

The one bug that matters:

```python
action = leader_joint_positions()    # where you asked it to go   ← the label
state  = follower_joint_positions()  # where it actually got      ← the observation
```

They differ because the arm lags, sags, and **stops when it hits something**.
Record the second as the first and the policy learns to predict where it
already is — at inference it reaches, grasps, lifts, then freezes. It looks
exactly like a hardware fault. Both recorders assert the gap is non-zero before
writing, and so does the converter.

## Recording and training

```bash
# scripted expert, unattended
antioch scenario run --scenario record_demos --set episodes=25

# or teleop: start the sim listening, then stream your leader into it
antioch scenario run --scenario teleop_record
~/lerobot-env/bin/python tools/leader_bridge.py --port $(lerobot-find-port)

# pull it off the machine and convert
antioch scenario download <run-id> --artifact episodes.npz -o episodes.npz
~/lerobot-env/bin/python tools/to_lerobot.py episodes.npz --repo-id local/vial_pick

lerobot-train --policy.type=act --dataset.repo_id=local/vial_pick \
              --policy.n_action_steps=20 --steps=15000 --batch_size=8
```

Set `n_action_steps`. ACT defaults to executing all 100 predicted actions before
looking at the camera again — ten seconds blind at 10 Hz.

## Numbers measured off the hardware model

Everything in `so101.py` was read out of the arm asset rather than assumed:

- Finger faces converge toward the tip: **23.5 mm** apart 15 mm back, **19.8 mm**
  at mid-face, **15.8 mm** at the tip. Where along them you grasp decides what
  you can hold.
- Only one finger moves, and the fixed face sits **7.9 mm** off the tool axis,
  so aiming the tool centre at an object's axis is wrong for anything wider
  than ~16 mm.
- A vial in the hand blocks the jaw **~2.5°** short of its command; an empty
  grasp reaches the command exactly. That gap is the whole failure monitor.
- Top-down reach ceiling falls from **123 mm** at 0.17 m to **85 mm** at 0.25 m.

## Known limits

- **The scripted expert does not reliably lift a 12 mm vial out of a well.** The
  approach clearance available at a 20 mm well pitch is smaller than the arm's
  own ~3 mm tracking error. Teleop does not have this problem, because a human
  aims by eye. Wider well spacing also fixes it.
- Leader sign and offset constants in `sim.py` are placeholders until calibrated
  against a physical arm.
- Placement lands within ~24 mm of a target well; a 17 mm well needs ~8 mm.
