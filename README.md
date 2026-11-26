# Subway Surfers vision

Read-only gameplay observation: YOLO obstacle detection, distance filtering,
tracking, player localization, provisional lane estimates, overlays and video.
Normal observation does not press keys or train an RL agent. An explicit
control-validation mode can send a bounded sequence of test arrow taps.

The standing requirements are in the [implementation task](docs/live_state_engine_task.md).
See the [engine report](docs/game_state_engine.md),
[player tracking results](docs/player_tracking.md),
[distance-filter evaluation](docs/proximity_results.md), and
[code audit](docs/code_audit.md) when resuming work.

## Observe live

Open Subway Surfers in Chrome, then run:

```bash
.venv/bin/python scripts/observe_live.py
```

Click two opposite corners of the gameplay area and press **Enter**. Return to
Chrome during the **15-second countdown**. Keep gameplay visible and move the
observer window outside the selected area so it does not capture itself.

The terminal prints **CAPTURING**, **OBSERVING**, and **VIDEO ON** with the output
path. A normal run lasts about 30 seconds. Q/Esc, closing the observer, or Ctrl+C
stops it. B/L/P/H/S toggle boxes, lanes, player, hazards and status overlays.

Video saves by default. Each new `models/evaluation/engine_live_<run ID>/` contains:

- `observation.mp4` and `observation.json`: annotated 15 FPS review copy and timing metadata.
- `states.jsonl`: detections, tracks, player/lane estimates and fixed 137-value v2 observations.
- `observation_schema.json` and `benchmark.json`: field ordering and measured performance.

Video has no audio, is capped at 960px width without stretching, and holds frames
across capture gaps. Encoding adds work. Use `--no-video` to disable recording,
`--duration 60` to run longer, or `--start-delay 20` to change the post-selection delay.
To record without the preview window's display overhead:

```bash
.venv/bin/python scripts/observe_live.py --headless
```

## Check lane estimates and keyboard controls

Start a run in Chrome in the **center lane**, then use:

```bash
.venv/bin/python scripts/observe_live.py --validate-controls
```

Select the area as usual and return to the game during the countdown. Keep
Chrome focused and do not steer manually. This mode saves video without opening
a preview window. Relative to the first observed frame, it taps left at 5s,
right at 8s, right at 11s, left at 14s, up at 17s and down at 20s. Switch to
another app to prevent subsequent taps; the next focus check aborts the sequence.
It never activates Chrome or follows policy recommendations.

macOS needs Accessibility permission for the launching Terminal/VS Code and may
request Automation access to System Events. A failed permission/focus check stops
the test; missed deadlines are not replayed in a burst.

The usual output folder also contains `controls.jsonl`: command results and
later player/lane observations, aligned with video timestamps. Expected lanes
assume a center start and successful moves; **they are not ground truth or lane
accuracy**. Review the video for movement, collisions, respawns and candidate
lanes. Commands do not overwrite the estimator or turn UNKNOWN into confirmed
lanes. This prepares a validation recording; actual keyboard delivery and lane
correctness must be checked on that recording before autonomous gameplay.

## Record or replay screenshots

For screenshots for labeling/testing, open `scripts/gameplay.py` in VS Code and
press **F5 → Record gameplay**, or run:

```bash
.venv/bin/python scripts/gameplay.py
```

Bring Chrome forward during the first countdown. Select two corners, press Enter,
then return to Chrome during the 10-second countdown. R redoes selection; Esc
cancels. Recording stops after 60 seconds; Ctrl+C stops early. Every run creates a
new `data/raw/session_<ID>/` with original JPEGs and metadata. Target FPS is 10;
metadata reports actual rate, dimensions and completion status. Recorder settings
are at the top of `scripts/gameplay.py`; it has no CLI options.

Replay existing footage without Chrome:

```bash
.venv/bin/python scripts/replay_gameplay.py data/raw/session_<ID> --start 1 --limit 20
```

Replay uses the same engine. `--headless` suppresses display, `--video` enables
video (opt-in for replay), and `--preview-every 5` saves separate overlay PNGs.
Existing recordings have approximate source timestamps derived from measured FPS;
exact timestamp files can be supplied. Run scripts with `--help` for other options.

## Models, proximity and limits

- `models/baseline/weights/best.pt`: original detector.
- `models/barriers_640/weights/best.pt`: retained three-class checkpoint for comparison.
- `models/training/weights/best.pt`: current six-class checkpoint (default).
- `models/evaluation/`: saved test results and observation runs.

The distance filter is code, not another trained model. Shared settings live in
`agent/detection.py`: confidence **50%**, plus both box-position and width rules:

| Object | Box bottom, fraction of image height | Minimum fraction of image width |
| --- | --- | --- |
| Ground/overhead barrier | 55% down from top | 6% |
| Train | 65% down from top | 12% |

These are proximity estimates, not meters or collision deadlines. Live/replay
hazards use nearby tracks. Overlays/video show all retained detections above the
confidence cutoff: distant tracks are muted and marked UPCOMING; nearby tracks
are bright cyan and marked NEARBY. All tracks remain logged for lane lookahead.
`nearby_track_ids` identifies the subset and the overlay shows its count.
Use an engine settings JSON with `distance_filter: false` to disable proximity
filtering for hazard checks too. UPCOMING is not a collision-time estimate.

The green box follows the player's upper body using the original color-template
matcher and optical flow in `agent/player_vision.py` (unchanged). A lightweight
postprocessor in `agent/player.py` now estimates LEFT/CENTER/RIGHT from those same
boxes when the player is near the reviewed screen-column anchors, with two
consecutive matching observations required for a stable provisional estimate.
The preview prints **ESTIMATED**, and `states.jsonl` records
`player.estimated_lane` and `player.localization.screen_lane`. The anchors are
in `config/player_templates.json`; they were approximately inspected in
`session_1795724241854738000` and apply only to comparable camera framing.
Mid-lane transitions, missing matches, gaps or incompatible viewport sizes
return unknown instead of recycling an old lane.

**Estimated does not mean confirmed**: `player.lane` remains null unless
independent reviewed ground-contact evidence exists. The 137-value v2
observation mask therefore stays unchanged. Running/jumping/rolling/train-roof
states remain UNKNOWN, and no automatic actions are enabled. Compare the
**estimated lane** shown in the original observer video with the actual lane
during deliberate lane changes before using it for control. Templates and rail
heuristics are appearance-specific; weak matches and unsupported themes yield
UNKNOWN. Boxes do not directly identify occupied obstacle lanes.

After training, evaluate the new six-class checkpoint printed by the training script.
For example (the run directory may have a numeric suffix):

```bash
.venv/bin/python scripts/evaluate.py models/training/weights/best.pt data/training/dataset.yaml val
```

The dataset now includes bush, small-obstacle and central-pillar classes. See
[data counts and annotation rules](data/README.md).
The default observer uses the six-class `models/training/weights/best.pt`.
For a legacy three-class comparison, specify
`--checkpoint models/barriers_640/weights/best.pt`.
Class IDs are validated at model load. The 137-value v2 observation schema is
not interchangeable with older 113-value v1 logs. Neither checkpoint establishes
confirmed lane occupancy or calibrated collision timing.
Earlier distance-filter AP50 reports use the older frozen dataset and must not be
compared directly with six-class results. `scripts/distance_map.py` deliberately
rejects changed annotation revisions.
The optional `scripts/train.py` keeps one conservative training configuration and
creates a separate numbered `models/training/` run; no training is needed to observe.

## Setup and checks

Use Python 3.12, select `.venv/bin/python` in VS Code, and install requirements:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
```

On this Mac `.venv` already links to a local environment under
`~/Library/Application Support/subway-surfers-rl/venv`, outside Desktop syncing.
For capture permission, enable Terminal or VS Code in **System Settings → Privacy
& Security → Screen & System Audio Recording**. If project reads hang, choose
**Keep Downloaded** in Finder. Runtime benchmarks detect actual hardware and
versions; current measurements do not assume a future Mac's performance.

## Data and development rules

- Preserve source-image bytes, dimensions and aspect ratios. Keep overlays/crops separate.
- `data/training/` holds the curated originals, labels and explicit 607/322 train/validation lists. `data/raw/` is for new recordings. See `data/README.md` for the cleanup record and folder layout.
- Label only recognizable visible extents; preserve correct labels, record corrections/provenance, and exclude ambiguous frames. Predictions are review candidates, not ground truth.
- The original recording retains its chronological train ≤780 / validation ≥850 split and gap. New clips have fixed recording-level assignments in the manifest. Never train on validation data or treat previously inspected clips as an untouched independent test.
- Use real screenshots. Compare one training change at a time with matched seeds/budgets; avoid significant rotation, perspective warp and vertical flips. Never paste objects permanently into datasets.
- Empty-label negatives need visual confirmation. Paused, obscured or ambiguous frames are not verified negatives.
- Preserve earlier checkpoints and compare partial/small-object recall, false positives and latency alongside AP. Never claim gains from relabeling alone or training accuracy.
- Keep resource use bounded and code simple. Preserve UNKNOWN rather than inventing lanes, confidence or seconds. No automatic controls or RL training in this milestone.
- Prefer ordinary loops, descriptive names and one statement per line. Avoid comprehensions and anonymous functions in production code. Comment important timing, geometry, file-writing and control behavior. Keep necessary NumPy/OpenCV operations instead of reimplementing image processing.

## Next milestone

Validate player ground contact/animation and lane estimates through manual lane
changes, then obstacle footprints and collision/action timing on a fresh recording.
Automatic controls, a Gymnasium environment and DQN training follow that validation.
