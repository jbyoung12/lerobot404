# Runbook: leader arm → dataset → trained policy

Follow top to bottom. Two gates are marked **STOP** — do not pass them until
they read the way they should, because everything after costs real time.

Two Pythons are in play and they are not interchangeable:

- `~/lerobot-env/bin/python` — talks to the leader arm, converts datasets, trains
- `.venv/bin/antioch` — talks to the simulator

---

## 1. Plug in the leader

USB into the laptop, **and the servo board's power supply into the wall.** The
servos cannot report positions on USB power alone; this is the most common
"it isn't detected".

## 2. Find the serial port

```bash
~/lerobot-env/bin/lerobot-find-port
```

Interactive: it lists ports, asks you to unplug the USB, you press Enter, and it
names the one that disappeared. Plug it back in. Expect something like
`/dev/tty.usbmodem1101`. Use it everywhere below as `$PORT`.

## 3. Calibrate the arm, once

```bash
~/lerobot-env/bin/lerobot-calibrate \
  --teleop.type=so101_leader --teleop.port=$PORT --teleop.id=so101_leader
```

Two prompts, and both matter:

1. **"Move to the middle of its range of motion"** — a neutral posture, roughly
   halfway on every joint. This sets where zero is.
2. **"Move all joints through their entire ranges"** — sweep each joint from one
   hard stop to the other, one at a time, including squeezing the trigger fully
   closed and fully open.

**Check the table it prints.** Servo ticks run 0–4095 for a full turn, about 11
per degree, so each joint's MIN..MAX spread should be **hundreds to a couple of
thousand** ticks:

```
NAME            |    MIN |    POS |    MAX
shoulder_pan    |   1973 |   1974 |   2051     ← 78 ticks ≈ 7°. Far too small.
gripper         |   2039 |   2041 |   2056     ← 17 ticks. Barely moved.
```

A spread that small gets stretched across the joint's whole output range, so a
centimetre of hand movement becomes a wild sim swing and anything outside the
window clamps. If the numbers look like the above, run it again and sweep
harder.

Calibration is **per physical arm** — offsets differ between units, so one
arm's file on another reads roughly 90° out and pins the sim arm against its
limits.

## 4. STOP — confirm the arm reads

```bash
~/lerobot-env/bin/python tools/leader_bridge.py --port $PORT --dry-run
```

Six live numbers on one refreshing line: five in degrees, then 0–100 for the
trigger. **Move the leader and watch them change.**

If they do not move, stop here. The problem is the arm, its power, or the
calibration — nothing downstream can work, and going further wastes the time
you have left.

## 5. Open the port tunnel

```bash
.venv/bin/antioch services up
```

Its own step, and easy to miss. A scenario dispatch starts the services but
does **not** open the local port; only `services up` does. It prints the
tunnel it established:

```
TUNNEL          TARGET
localhost:8765  sim:8765 (teleop)
```

Without this the bridge reports `nothing is listening on http://localhost:8765`
and nothing you do in the other terminals will fix it.

## 6. Drive the simulator

Two terminals.

```bash
# terminal one — the sim comes up and waits for you
.venv/bin/antioch scenario run --scenario teleop_record --set "destination=C5"
```

Wait for `listening for the leader on port 8765`, then:

```bash
# terminal two
~/lerobot-env/bin/python tools/leader_bridge.py --port $PORT
```

Move the leader. The sim arm follows. Check two things before recording:

- Every joint moves the **same direction** you do. A mirrored joint is a sign
  flip in `LEADER_SIGNS` in `src/sim.py`.
- **Squeezing the trigger closes** the gripper. If it opens, same fix, last entry.

**Controls:** `Enter` starts recording an episode, `Enter` again ends it, `q`
finishes the session and writes the archive.

## 7. STOP — record ONE episode and verify it

One pick-and-place. Enter, drive, Enter, `q`. Then:

```bash
.venv/bin/antioch scenario list                       # grab the run id
.venv/bin/antioch scenario download <run-id> --artifact episodes.npz -o episodes.npz
~/lerobot-env/bin/python tools/to_lerobot.py episodes.npz --repo-id local/vial_pick
```

It must print a **non-zero action/state gap**. That means the labels are right.
A zero gap means measured positions were recorded as actions, and a policy
trained on it will reach, grasp, lift, and then freeze mid-task — a failure that
looks like broken hardware for hours before anyone suspects the labels.

Do not record twenty-five episodes until this one converts clean.

## 8. Record the rest

Same session, `Enter`/`Enter` per episode, about 25 in total.

**Vary where you pick from every single time.** That variation is the only
reason the policy will handle positions it has not seen, and it is what makes
retry possible — a retry begins from a pose that is not the standard one.

Throw away your first three or four. Teleop feels clumsy until your hand adapts
to the lag; those episodes teach the policy your flailing.

## 9. Convert and start training

```bash
.venv/bin/antioch scenario download <run-id> --artifact episodes.npz -o episodes.npz
~/lerobot-env/bin/python tools/to_lerobot.py episodes.npz --repo-id local/vial_pick

lerobot-train --policy.type=act --dataset.repo_id=local/vial_pick \
              --policy.n_action_steps=20 --steps=15000 --batch_size=8
```

**About an hour, unattended.** Start it and walk away — everything else waits on
it, so the earlier it starts the better.

Set `n_action_steps` explicitly. ACT's default executes all 100 predicted
actions before looking at the camera again; at 10 Hz that is ten seconds flown
blind. Twenty gives it two seconds, then a fresh look.

## 10. While it trains — build the other three layers

None of these need a GPU or the simulator. `import sim` on a laptop picks the
mock backend, which lags like a real drive and refuses to close its gripper on
something it is holding, so the monitor can be written and tested for real.

**Monitor** — commanded gripper closed *and* measured gripper reached that
command means it closed on air. Measured on this arm: a held vial blocks the jaw
about **2.5°** short of its command; an empty grasp reaches it exactly.

**Retry** — on failure, retreat to home and try again. Twice, then hand off.

**Handoff** — a page showing the current frame, a question, and buttons.

**Planner and tracker** — `sim.occupancy()` returns which vial is in which well,
so "move A3 to C5" becomes a sequence of the one trained skill.

## 11. Integrate

Move the checkpoint to the GPU machine, then swap one call:

```python
chunk = policy(sim.frame(), sim.joint_state())    # instead of sim.expert_actions()
for action in chunk[:20]:
    sim.set_targets(action)
```

Nothing else changes. Scene, camera, scoring and the monitor signal already work.

## 12. Evaluate

About 20 episodes on `sim.reset("eval", ...)` — positions the policy has never
seen — three times: policy alone, plus retry, plus handoff. `sim.score()` counts
them. Three rows, measured, reported honestly including how often a human
stepped in.

---

## If something breaks

| symptom | cause |
|---|---|
| Leader not detected | Power supply not plugged in, or wrong port |
| Numbers do not move in `--dry-run` | Not calibrated, or the arm is on a different port |
| Sim arm mirrors your hand | Sign flip in `LEADER_SIGNS`, `src/sim.py` |
| Trigger opens instead of closing | Same, last entry |
| Bridge says nothing is listening | Run `antioch services up` — the tunnel is its own step |
| Sim arm pinned, cannot reach the table | Calibration is from a different arm, or the sweep was too small |
| Sim arm twitches wildly | Calibration ranges too narrow; sweep each joint to both hard stops |
| Policy reaches, grasps, then freezes | Actions were recorded as measured states. Re-record. |
| Policy ignores the vial's position | Trained without the camera, or on episodes all from one position |
