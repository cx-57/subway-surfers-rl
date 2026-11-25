# Visual player tracking and rail-based lane candidates

The observer now processes pixels to locate the character and re-estimates track
geometry on every frame. It no longer depends exclusively on handwritten player
observations. It still does not certify ground contact or classify animation.
No YOLO weights, source screenshots or existing labels were changed; no training
or keyboard control was performed.

## Implementation

- `agent/player_vision.py`: four upper-body color templates from inspected older
  recording frames 300/500/600/700, normalized matching at 320px working width,
  competing-match rejection and bounded Lucas–Kanade optical flow between refreshes.
- `config/player_templates.json` and four separate PNG crops: template provenance,
  source/asset SHA256, scale/search settings and appearance-specific thresholds.
  These are reference-image derivatives, not modified screenshots or YOLO data.
- `agent/track_geometry.py`: current-frame longitudinal rail segments, six distinct
  rail groups, consistent perspective convergence/gauges/spacing and shared observed
  depth. No fixed screen thirds and no retained camera geometry.
- `agent/geometry.py`: upper-body image-column lane candidates and a stricter
  ground-foot mapping API for reviewed/trusted running/full-body observations.
- `agent/player.py`, `pipeline.py`, `state.py`, `visualization.py`: integrated live
  and replay tracking, green player boxes, candidate lanes, timing and JSON logging.

Default NCC refresh interval is 0.15 seconds; flow requires coherent features,
forward/backward agreement and bounded motion. A frame gap above 0.2 seconds,
dimension change, blank/occluded frame, ambiguous identity or a 0.4-second absence
of strong matching discards the track. Flow uses actual image features, not guessed
coordinates propagated from a keypress. Optical flow is not a guarantee of identity.

NCC scores are correlations, not accuracy probabilities. Flow observations have
`match_score=null`; feature count, tracking error and identity age are logged.
`player.confidence` stays null rather than manufacturing a probability.

Rail thresholds are specific to the inspected pink-highlight track theme. Every
frame must independently supply the six-rail pattern; trains, clipping and curves
often prevent assignment. Automatic geometry is explicitly labeled a visual
heuristic, not measured calibration. An operator's valid scoped calibration still
takes precedence. Row/depth/aspect/source checks remain enforced.

## What the lane and movement outputs mean

The visible upper-body box is **not a foot-contact measurement**. Its horizontal
span is projected over supported ground depths to produce `player.lane_candidates`.
One candidate can be useful for review but does not certify a grounded player.
The source/reason states that foot contact is unknown; `player.lane` remains null.
The policy therefore cannot turn these candidates into confident action commands.

`Geometry.map_player` supplies confirmed geometric mapping only when a caller
provides a trustworthy full-body footpoint, a known running state, valid current
geometry and supported depth. Jumping, rolling, train-roof and clipped/untrusted
observations remain unknown. This API does not independently establish those facts.

**Running/jumping/rolling/on-train classification is still UNKNOWN for automatic
observations.** A running-looking torso can occur in the air or on a train roof.
No reliable animation classifier was established in this implementation; templates
are not used to invent one. Scoped reviewed observations remain available for
explicitly labeled offline cases and never propagate to later frames.

The stable 113-value numerical observation keeps these unconfirmed player values
masked unknown. New upper-body boxes, candidate lists and diagnostics are in JSONL;
the vector format/order is unchanged. This is not ready for autonomous actions.

## Actual validation and failures

On original newest-recording frames **480–520**, the integrated pipeline localized
the character in **41/41 frames**, produced current-frame rail geometry and center
lane candidates in **21/41**, and reported **zero confirmed ground lanes**. All
automatic movement states and recommendations remained UNKNOWN. The easy clip was
inspected during development, so these counts are availability, not independent
accuracy or generalization results.

In eight more varied inspected examples, the initial NCC-only diagnostic accepted
two clear actors and rejected six weak/ambiguous/occluded cases. Strong matching
alone does not establish pose. General-purpose pretrained YOLO11n-pose produced no
person detections at 25% confidence in three selected cartoon frames; it was not
adopted and its downloaded candidate weights were removed. No extra neural model
or dependency is required at runtime.

The final one-degree Hough implementation supplied geometry in **6/45** sampled
frames across three recordings. These examples informed development; there are no
independent rail/lane labels. A reviewed older frame 600 demonstrates the intended
perspective correction: the actor lies in the middle image third (x≈0.549), but
the rails and a reviewed ground box place it in the **right** lane. Newest frame
500 gives a center candidate. Occluded frames 300/900 cannot identify three lanes;
roof/airborne examples must not project body boxes into ground lanes.

Measured standalone flow/NCC tracking: **4.08 ms mean, 1.59 ms median** over the
41-frame clip; the earlier NCC-only benchmark was about 13.5 ms. The optimized rail
diagnostic averaged **7.91 ms** over the 45 images. End-to-end headless replay with
YOLO, video and every-tenth-frame PNG saving achieved **15.18 FPS after the first
three frames**; player cost averaged 3.95 ms, rail cost 12.97 ms and processing age
44.87 ms. These are different measurements and include no interactive display cost.
Adding perception has a CPU cost; this is not a promised speed improvement.

Artifacts:

- `models/evaluation/player_tracking_final/`: actual MP4, original-size overlays,
  JSONL states, player counts and benchmark.
- `models/evaluation/player_flow_clip.json`: feature tracking diagnostics and hashes.
- `models/evaluation/player_geometry/optimized_rail_review.json`: final sampled
  availability and latency; earlier engineering diagnostics are retained separately.
- `models/evaluation/player_localizer_examples.json`: varied appearance failures.

Verification: **122 unittest tests pass**; `pip check` and `git diff --check` are
clean. Source/template and retained-checkpoint hashes were rechecked after the
integrated demo. Live GUI use of these new estimates has not been tested on an
active game; the practical evidence above is original-file replay.

## Run

Automatic tracking/candidates are enabled by default:

```bash
.venv/bin/python scripts/observe_live.py --video
```

Use `--headless` to avoid the live preview's display cost. To replay the actual demo:

```bash
.venv/bin/python scripts/replay_gameplay.py data/raw/session_1795524944131662000 --start 480 --limit 41 --video
```

Set `visual_player` or `automatic_geometry` to false in an engine settings JSON to
disable either component for benchmarking or unsupported avatars/themes. Neither
setting changes the detector or distance filter. Reliable ground contact/animation,
occluded lane occupancy and calibrated collision timing remain the next blockers.
