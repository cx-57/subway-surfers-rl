# Read-only game-state engine

Player-estimation update: see [visual tracking and lane candidates](player_tracking.md)
for the newer default template/flow localizer and frame-local rail extraction.
The manual-only player description and benchmarks below describe the initial
engine milestone; automatic position/candidates now augment it. Animation and
confirmed ground contact remain unresolved.

Implemented against the [full task](live_state_engine_task.md). This is a working
capture/replay -> detection -> tracking -> state -> overlay/logging pipeline.
Geometry and policy preserve uncertainty; this is not a validated gameplay agent.
No YOLO training, RL training or keyboard execution is implemented or performed.

## Inspection and sequence

At inspection, the retained checkpoint existed at
`models/barriers_640/weights/best.pt` (SHA256
`3408b9b9bb1515b5396ff8ad3a0e471d96e31bafd80f9d81e9dc431df7a00ad6`).
The original detector, recordings, labels and prior evaluation artifacts remain.
Existing recording, evaluation and training scripts are unchanged by this engine.
The only unfinished planner files at the start were superseded by this engine.

Reused: the recorder's two-click selection, MSS/OpenCV capture conventions,
retained YOLO11 weights, the evaluation confidence cutoff (50%), original images
and metadata, and the existing incomplete-NMS guard.

Missing evidence: measured lane geometry, automatic player localization, reliable
train footprints, per-frame timestamps for old recordings, calibrated collision
boundaries/action durations and independent lane/tracking/action ground truth.
Those are reported as missing rather than inferred from rectangular overlap.

Implementation order: shared state contracts and detection, scoped geometry,
timestamp tracking and provisional player observations, hazards and policy,
fixed-size observations, offline replay and rendering, then live capture and
benchmarks. Each component has deterministic tests; practical tests use originals.

## Run locally

From the repository root, offline replay with a display:

```bash
.venv/bin/python scripts/replay_gameplay.py data/raw/session_1795440601129365000 --start 880 --limit 20
```

Headless replay with separate overlay images:

```bash
.venv/bin/python scripts/replay_gameplay.py data/raw/session_1795440601129365000 --start 880 --limit 20 --headless --preview-every 5
```

Live observation (open Subway Surfers, switch to Chrome during the countdown,
select the gameplay region, then return to Chrome):

```bash
.venv/bin/python scripts/observe_live.py --duration 30
```

After Enter confirms the selection, a 15-second countdown runs before capture.
Use `--start-delay SECONDS` to change it. The terminal reports CAPTURING at startup
and OBSERVING after the first processed frame; preview timestamps/frame counts
confirm ongoing observation. Detector initialization can take several seconds.

The shared live/replay engine now applies the distance ROI by default: barriers
need box bottom >=55% and width >=6%, trains >=65% and >=12%, with the retained
50% confidence cutoff. Overlays/video show distant tracks as muted UPCOMING boxes
and nearby tracks as bright NEARBY boxes. Only nearby tracks enter current hazard
checks; all tracks remain logged for lane lookahead, and
`nearby_track_ids` identifies the filtered subset. This is not measured distance
or collision timing. `settings.distance_filter=false` disables hazard proximity filtering too.

The observer window does not need focus for capture, but nothing verifies Chrome's
identity or that the game remains visible. If another page/window covers the
region, its pixels are captured instead. Keep the game visible and select only
gameplay. The engine cannot interpret unrelated content as valid player state.
Q/Esc or closing the viewer stops; Ctrl+C also stops. Overlay toggles are B boxes,
L lanes, P player, H hazards and S status. These are viewer inputs, not game keys.

Explicit-region capture smoke test, without selection or display:

```bash
.venv/bin/python scripts/observe_live.py --region 62 40 1284 793 --duration 5 --headless
```

Those coordinates come from the existing recording metadata, not a universal
game window position. Adjust them for the current screen. Screen Recording
permission failures and blank captures report errors. Capture resources close on
their worker thread; live capture has a single latest-frame slot, never a FIFO.

Use `--device cpu` to override MPS, `--calibration FILE` for scoped geometry/player
data, `--settings config/engine.example.json` for explicit engineering limits,
and `--output NEW_FOLDER` to choose a new output directory. Existing outputs are
never overwritten. Run `--help` for replay selection, pacing and timestamps.

Live observation saves `observation.mp4` plus `observation.json` metadata by default;
use `--no-video` to disable it. Replay uses optional `--video`. Use `--headless`
to record live overlays without the preview window. This review
copy is 15 FPS, capped at 960px width, resampled against source timestamps with
previous-frame holds across gaps. It has no audio; encoding cost is logged as
`video_ms`. A real 20-frame replay produced a readable MP4; the full suite now
passes 94 tests. Video encoding added about 4 ms per processed frame after warmup
in that short replay, not a guaranteed live cost.

Every run writes `states.jsonl`, `observation_schema.json` and `benchmark.json`.
Preview images are optional and always separate. Old recordings have only
aggregate FPS; their timestamps are explicitly approximate. `--timestamps FILE`
accepts a JSON map from frame filenames to measured seconds when available.
`--speed 1` paces source time; waiting is excluded from processing age. Replay
age measures this processing run, not unknown original capture latency.

## Components and contracts

| Module | Input -> output |
| --- | --- |
| `perception.py` | BGR frame + timestamp -> original-pixel detections; one loaded YOLO instance, stage timings, MPS/CPU fallback |
| `capture.py`, `sources.py` | Original image files or MSS region -> timestamped frames; live latest-slot worker |
| `geometry.py` | Scoped lane rows + visible footprint -> lane candidates/known/ambiguous/unknown, normalized image depth |
| `tracking.py` | Detections + actual timestamps -> stable candidate IDs, bounded history, pixel velocity, missing/ambiguous status |
| `player.py` | Exact reviewed frame observation -> provisional player state; otherwise UNKNOWN |
| `hazard.py` | Tracks + player + freshness -> path conflict and relative approach urgency; TTC remains null |
| `rule_policy.py` | GameState -> one read-only recommendation, reason, timing and actionable/repeated flags |
| `state.py`, `pipeline.py` | Component outputs -> owned GameState snapshot, fixed-size observation and streaming JSONL |
| `visualization.py`, `runtime.py` | Frame + state -> immutable overlay, optional display, logging and bounded timing metrics |

Live uses monotonic capture timestamps. Tracking uses source timestamps, never
assumed 30/60 FPS. History is bounded (12 observations by default), expiration is
in seconds (0.6 by default), and active tracks are capped at 128. Missing tracks
are retained briefly without pretending they were observed or extrapolating
indefinitely. Ambiguous matches are flagged instead of certifying an ID switch.
An earlier GameState does not change when the tracker processes a later frame.

### Tracker choice

Evaluated installed ByteTrack and BoT-SORT on the 92 saved sparse frames and on
20 freshly inferred contiguous frames. On the contiguous sequence, all three
trackers produced two IDs. The existing implementations advance motion and
expiration by update count, not elapsed frame timestamps. A small timestamp-based
association implementation was retained for dropped/irregular input. This is an
engineering choice, not proof it outperforms the existing trackers.

Diagnostics are in `models/evaluation/tracker_comparison.json` and
`engine_contiguous_tracker_comparison.json`. No identity ground truth exists, so
ID-switch accuracy is null. Counts of lost events/ambiguity are diagnostics only.
The comparison import temporarily auto-installed optional `lap`; it was removed
afterward. Runtime disables Ultralytics auto-installation. No dependencies were
added to requirements.

## Calibration and player limits

Create a small configuration by clicking **left, center, right lane centers** at
each of at least two ground-plane rows on an original image. Use more rows for a
curve. Do not place points on hidden lanes or extrapolate beyond reviewed depth.

```bash
.venv/bin/python scripts/calibrate_lanes.py data/raw/session_1795440601129365000/frame_000420.jpg --output config/measured.json
```

The tool saves actual clicked coordinates, original-image SHA256, aspect ratio,
and exact source-path scope. R resets and Esc cancels. No measured coordinates
are bundled: `config/geometry.example.json` is deliberately invalid. The mouse UI
was not exercised on the user's desktop; conversion/cancellation/rendering have
headless tests. A source path, frame/session scope, aspect or depth mismatch
explicitly invalidates assignment.

An explicit `--session live --first-frame 1 --last-frame 900` calibration-tool
override can create a bounded live profile; live frames are named
`live/frame_000001.jpg`, etc. Range reuse is marked provisional, never measured
for the whole camera sequence. Camera motion/curves need revalidation; there is
no automatic camera-calibration tracker in this milestone.

Ground/overhead bottom centers supply only provisional lane candidates, not
confirmed occupancy. A train rectangle supplies no lane. Optional reviewed
footprint points are tied to the exact frame and detection box; they do not
silently propagate or establish hidden geometry. This conservative implementation
does not yet automatically extract train fronts/supports or visible ground
footprints. Calibration alone therefore does not make every lane known.

`PlayerEstimator` is an isolated provisional interface, not a player detector.
The recorder supplies no keyboard-event stream, so action-based state propagation
is unavailable. A fixed screen position cannot resolve manual lane changes and
camera movement. A character template would need validation across jump, roll,
roof and occlusion views; no robust template result is claimed or enabled.
It accepts reviewed per-frame observations with provenance and otherwise returns
UNKNOWN immediately. The player-only example config marks original frame 885 as
center-lane/on-train from AI-assisted visual review; confidence remains null.
It is not human-verified ground truth or a running-state estimate. On the 20-frame
demo it applies to exactly one frame, not to subsequent manual lane changes.

Without known running/transition state and path occupancy, hazards remain unknown.
Consistent positive image motion can support relative urgency for reviewed path
conflicts; static depth alone does not create a collision deadline. TTC is always
null on real demos: no collision-boundary or action-time calibration is available.
This avoids falsely certifying that an action fits a deadline. Policy behavior
with supplied deadlines/provisional durations is unit-tested, not game-validated.
Adjacent lanes also require explicit visibility and no conflicting/unknown tracks.
Recommendations are deduplicated by encounter ID; none sends input.

## Observation schema v1

`GameState.to_vector()` returns **113 float32 values** in the exact field order
written by `observation_schema.json` (`agent.state.OBSERVATION_FIELDS`). It includes:

- Player lane and movement one-hots with known masks; transition and visibility masks.
- Calibration, freshness, normalized age, source timing quality and state validity.
- Unknown-lane/missing-track counts and available-action masks.
- Two nearest tracks per lane, ordered by image depth then stable ID: presence,
  observed/missing, lane-known, three class flags, confidence, depth + mask,
  TTC + mask, relative urgency + mask, association-valid mask.

Normalized image depth is not meters. Age clips at 1 second, TTC at 10 seconds,
and count channels at 10. Unknown TTC encodes value 0 **and mask 0**; an actual
supplied zero-second deadline has mask 1. No-track and temporarily missing-track
slots differ through presence/observed masks. Global unknown-lane counts and
calibration/visibility masks must be consulted before treating empty lane slots
as free. Detailed histories and reasons remain in JSONL. No Gymnasium/DQN training
is added; this is a perception interface, not a complete Markov state guarantee.

## Measured verification

Hardware: Apple M1, 8 cores, 8 GiB RAM, macOS 15.6.1; Python 3.12.3,
PyTorch 2.15.0, Ultralytics 8.4.216. MPS used without fallback.

| Measurement | Result and scope |
| --- | --- |
| Detector MPS, 17 timed frames after 3 warmups | Mean 19.00 ms, median 17.94 ms; 52.63 detector FPS |
| Detector CPU, same originals | Mean 43.15 ms, median 41.82 ms; 23.17 detector FPS |
| Offline integrated final demo, 20 frames 880–899 | Completed; 17 post-warmup frames at 30.19 processed FPS, mean processing age 21.63 ms; overlay PNG every fifth frame |
| Offline cold-inclusive throughput | 3.94 FPS; first-call setup dominates this short sequence; model loading separately 2.29 s |
| Final live capture smoke, current Chrome content | 141 processed frames; 31.33 post-warmup FPS; mean post-warmup age 45.06 ms |
| Final live capture/drop diagnostics | 337 captured, 195 overwritten latest-slot frames, 0 pre-inference stale skips; includes model loading/cold startup |
| Peak process RSS | About 449 MiB final replay, 512 MiB final capture smoke; allocator/process peak, not total system or GPU memory |

Final live smoke cold-inclusive throughput was 16.75 FPS over 8.42 seconds
(model loading separately 2.56 seconds). Live smoke content was **not active Subway Surfers**; it proves screen capture,
bounded delivery and engine execution, not gameplay detection/accuracy. Headless
benchmarks include logging and optional saved overlays, not interactive display
composition. Cold-start, waiting on cloud-backed original files and PNG writes
are distinct bottlenecks; reducing detector resolution was not justified.
The GUI event loop's toggle, quit and window-close branches have mocked-display
integration tests; actual rendered PNGs were visually inspected.

The final replay contains two candidate track IDs, two lost-detection events,
zero expirations and zero ambiguity events. The first cold frame is stale; it
cannot produce an actionable suggestion. The source has aggregate timing only,
so all source timestamps are flagged approximate. There is no tracking accuracy
or collision-timing score.

Practical examples/failures: frame 885 detects the overhead barrier at about 84%
confidence and the train at about 90%; the train rectangle includes adjacent track
background and the player is on its roof. Unknown occupancy/TTC is appropriate.
Earlier heavily hidden overhead views still fail detection; this task did not
change weights. Ground barriers clipped at the bottom and curved-camera views
remain unvalidated for automatic footprint assignment. Synthetic geometry/policy
tests establish contracts, not real-game accuracy.

Artifacts: `models/evaluation/engine_replay_final/` contains actual overlays, states,
schema and benchmark; `engine_device_benchmark.json` includes before/after hashes
of all 20 originals and the checkpoint; `engine_live_final/` retains final live logs and metrics; the earlier
`engine_live_capture_smoke/` retained preliminary metrics (unrelated-page preview
images were removed).

## Next blockers before automatic gameplay or RL

Measure/validate player localization through manual lane changes, automatic
footprint assignment and camera updates, then action/collision timings and
encounter-level ground truth on a fresh recording. Connect input controls only
as a separate authorized milestone. The fixed observation/log format is ready
for experimentation, but reliable state estimation and reward/termination logic
remain blockers to a useful Gymnasium environment or DQN agent.
