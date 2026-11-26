# Player lane/pose and obstacle occupancy: validation milestone

This implementation is intentionally conservative. Before operator calibration,
the existing player template tracker can locate an upper body but **cannot**
certify a lane, running, jumping, rolling, feet on track, or a safe action.
Unreviewed detector rectangles can suggest occupied lanes but do not certify
which track is blocked. Nothing in this change enables automatic keypresses.

## Obstacle lane occupancy

`Geometry.assign` now compares the **entire box width**, at the box-bottom
depth, against the three perspective-calibrated lane intervals. A narrow
obstacle may get one candidate; a long train or cross-lane barrier may have
multiple. All unreviewed boxes remain `UNKNOWN`, and clipped boxes or objects
outside observed calibration depth remain unlocated. Boxes may contain
decorations, open gaps or occluded feet: overlap is **not a collision mask**.

To review a specific frame's occupancy, add a provenance-bearing item in
the calibration JSON with exactly matching pixel bounding box:

```json
{
  "occupied_lanes": {
    "frame_000105.jpg": [
      {
        "bbox": [240, 220, 610, 660],
        "lanes": [0, 1],
        "provenance": "human review of visible blocking footprint on source frame"
      }
    ]
  }
}
```

The numbers above are illustrative, **not measurements**; replace with actual
detector boxes and visibly blocked lanes. Only a correctly scoped, reviewed
calibration without `provisional: true` sets `KNOWN`. Known multi-lane
objects now occupy each corresponding lane slot in the **137-value v3**
observation. Existing v1/v2 logs must retain their original schema.

## Calibrating the player visually

The optional `player_visual_calibration` learns horizontal upper-body
references for each lane, and, where distinguishable, approximate vertical
displacements for RUNNING/JUMPING/ROLLING. It **does not measure foot contact**,
is character/camera/theme-specific, and cannot reliably distinguish train-roof
states. Any ambiguous or missing observation stays UNKNOWN.

First capture original gameplay at the **same crop, avatar, zoom and theme**
while deliberately running in each lane and performing jumps and rolls:

```bash
.venv/bin/python scripts/gameplay.py
```

Then annotate spread-out frames (not dozens of adjacent copies):

```bash
.venv/bin/python scripts/calibrate_player.py data/raw/session_<ID> --every 12
```

Each clear green player box: press `1` for left, `2` for center, `3` for
right, then `r` for running, `j` for jumping, or `o` for rolling.
Space skips; `q` ends. Incorrectly recognized players must be **skipped**.
You need at least three positions per lane and two RUNNING observations per
lane; at least three recognizable JUMPING and ROLLING frames with distinct
vertical motion are needed even to **attempt** pose estimation. More
independent examples are strongly recommended. If the upper-body tracker
loses the character during a roll, pose cannot be calibrated from that frame.

The tool creates `config/player_calibration_session_<ID>.json` with
`validated: false` to prevent draft predictions from being used as known
game states. It refuses to overwrite an existing output.

Test **on a separate game recording** with the same viewport and avatar:

```bash
.venv/bin/python scripts/observe_live.py --calibration config/player_calibration_session_<ID>.json --duration 60 --headless
```

The annotated `observation.mp4` reports **DRAFT / NOT VALIDATED**
estimates; `states.jsonl` contains `player.localization.visual_calibration`
with `lane`, `pose`, `stable`, `reason`, and `validated`.
`player.lane` and `player.movement` remain UNKNOWN during draft validation.

Compare at least 20 visually verified frames per movement and lane when
possible, including transitions, misdetections, occlusions and different
obstacles. Count **wrong confident estimates** and **unknowns** separately.
Report class/pose confusion, lane accuracy on non-UNKNOWN estimates, coverage,
and latency. Do not mark a calibration `validated: true` based only on its
training frames or the prerecorded control-check's *expected* lane sequence.
Doing so promotes its scene-specific visual estimates into GameState; it does
not independently establish ground contact. The rule policy still requires
calibrated TTC, movement state, confirmed obstacle occupancy and safe target
lane information before recommending action.

## Verify before testing live

```bash
.venv/bin/python -m unittest discover -v
.venv/bin/python scripts/replay_gameplay.py data/raw/session_<ID> --start 480 --limit 60 --video
```

Replay should display lane candidates beside detections. Check whether they
overclaim adjacent lanes, especially wide trains, overhead obstacles,
near-screen clipped objects and perspective changes. Nothing here changes
YOLO weights or datasets.

**Known limitations:** no independent held-out accuracy has been measured for
this feature, no automatic game control, no train-roof recognition, no robust
full-body tracking, no collision timing. If the localizer fails to recognize
rolling poses, a separate pose detector or animation classifier will be needed.
