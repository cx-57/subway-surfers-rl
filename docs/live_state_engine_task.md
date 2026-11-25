# CODEX IMPLEMENTATION TASK — Subway Surfers Live Game-State Engine

## 0. Mission

Repository: `https://github.com/cx-57/subway-surfers-rl`

Build the next major milestone of my Subway Surfers AI project: a **real-time, perspective-aware game-state engine** that converts live gameplay screenshots and YOLO detections into structured observations, tracks obstacles, estimates collision risks, and recommends actions.

The long-term objective is a reinforcement-learning agent trained with DQN or another appropriate RL algorithm.

**This task is NOT to train the RL agent or make it play autonomously.** It is to build and validate the perception and state-estimation system that those later stages will depend on.

### Target pipeline

```text
Subway Surfers (Chrome)
        ↓
Live screen capture
        ↓
Existing YOLO11 detector
        ↓
Perspective-aware lane geometry
        ↓
Temporal obstacle tracking
        ↓
Player-state + collision-risk estimation
        ↓
Structured GameState
        ↓
Shadow-mode action recommendations
        ↓
Live visualization + observation logging
```

This must be a functioning, integrated Python system, not a collection of disconnected scripts or a conceptual architecture.

### Hardware

Current machine:

- Apple M1 MacBook
- Approximately 8 GB RAM
- macOS
- Python virtual environment

I may upgrade to a newer MacBook with 16–32 GB RAM shortly.

Detect and benchmark the actual hardware. Do not unnecessarily hardcode memory or processor assumptions.

Prefer a small, efficient implementation that runs on both machines.

---

# 1. Inspect the Existing Project First

Before changing anything, inspect the current repository and local working tree.

Pay particular attention to:

- `scripts/gameplay.py`
- `scripts/evaluate.py`
- `scripts/train.py`
- Existing YOLO checkpoints
- Existing annotations and dataset manifests
- Previously recorded gameplay sessions
- Existing tests and documentation
- Current Git status and uncommitted changes

The previously recommended detector was:

`models/barriers_640/weights/best.pt`

Verify that it remains the correct checkpoint and exists locally. Do not assume that the latest repository state is identical to a previous commit.

The current project already supports screen recording, object detection, proximity filtering, and preliminary action recommendations.

**Reuse functioning components. Do not rebuild established functionality without a technical reason.**

Known limitations include:

- Detection failures on heavily occluded overhead barriers.
- Bounding boxes do not directly represent lane occupancy.
- The existing detector does not identify the player.
- The current proximity filter is not a calibrated collision predictor.
- Existing gameplay sessions contain temporally correlated frames.

Treat these as known engineering constraints, not problems to conceal.

### Initial work

Briefly document:

1. What exists and works.
2. What can be reused.
3. What is missing.
4. What assumptions require calibration.
5. The implementation sequence and critical dependencies.

Then proceed with implementation. Do not stop after planning.

Preserve user changes. Do not perform destructive Git operations.

---

# 2. Implementation Strategy

Use incremental, testable stages.

Implement the system in the following order:

1. Reusable perception pipeline.
2. Perspective-aware lane geometry.
3. Obstacle tracking.
4. Player-state estimation.
5. Collision-risk and timing estimation.
6. Structured game-state representation.
7. Read-only shadow policy.
8. Offline/live visualization.
9. Integration testing and performance benchmarking.

Do not implement everything in one enormous function.

Each stage should have:

- A clear input/output contract.
- Deterministic unit tests where possible.
- Error handling.
- Documented assumptions.
- Integration with the preceding stage.

**Working, testable simplicity is preferable to sophisticated but unreliable complexity.**

Do not introduce large dependencies without justification.

Do not train or modify YOLO checkpoints during this task.

---

# 3. Build Reusable Real-Time Perception

Implement one reusable detection interface supporting both live capture and offline replay.

Required functionality:

- Load the retained YOLO model once.
- Reuse existing screen-region selection where possible.
- Capture the Chrome gameplay region efficiently.
- Perform inference on incoming frames.
- Return class, confidence, bounding box, and timestamp.
- Record actual image dimensions and processing latency.
- Avoid unnecessary disk writes.
- Preserve source-image aspect ratios.
- Handle model-loading failures gracefully.

Use monotonic timestamps for live elapsed-time calculations.

Separate capture, inference, and visualization logic so they can be tested independently.

### Performance requirements

Do not accumulate a backlog of old frames.

Use a bounded queue or equivalent latest-frame mechanism so the agent processes recent observations rather than outdated screenshots.

Benchmark:

- Capture time
- Preprocessing time
- Inference time
- Postprocessing time
- End-to-end observation age
- Actual processed FPS

Do not equate YOLO-only inference speed with total pipeline speed.

Use Apple MPS when supported and beneficial, with a CPU fallback.

Preserve small/distant obstacle detection quality when considering changes to inference resolution.

### Acceptance criterion

The same perception interface must process an existing gameplay recording and live screen frames without duplicating detector logic.

---

# 4. Implement Perspective-Aware Lane Geometry

**This is a critical subsystem.**

Subway Surfers has three lanes, but their image-space widths and boundaries change with depth.

Do not divide screenshots into three fixed vertical thirds.

Build a geometry module that can estimate left, center, and right lane regions as functions of image position.

Investigate the available footage before choosing a geometric model.

Possible approaches:

- Calibrated lane-boundary curves
- Interpolated lane-center trajectories
- Perspective-aware track geometry
- Lightweight scene-specific calibration

Choose the simplest method that can be meaningfully validated.

Do not automatically introduce a second neural network if a calibrated geometric approach is sufficient.

### Calibration

Provide a small, documented calibration configuration.

If useful, implement an interactive tool allowing a user to mark a limited number of track/lane reference points on representative images.

This is acceptable. It is not a request to manually annotate the full dataset.

Do not invent calibration coordinates and present them as measured values.

If calibration is invalid, the system should explicitly report that condition.

### Object-to-lane mapping

For each obstacle, estimate:

- Candidate lane or lanes
- Confidence or uncertainty
- Image-space position relative to calibrated track geometry
- Approximate approach/depth position
- Relevant visible footprint

Important cases:

- Long foreground trains
- Trains partially outside the frame
- Small distant barriers
- Overhead structures with visible supports
- Ground barriers
- Objects obscured by trains
- Camera changes and curved track sections

**Do not determine occupied lanes solely from horizontal bounding-box overlap.**

Bounding boxes often contain background or adjacent track regions.

Use object-specific visible geometry and calibrated lane relationships where supported.

If reliable assignment cannot be established, return an explicit unknown state or candidate-lane set.

Do not invent precise lane confidence values without a meaningful estimation or calibration method.

### Acceptance criterion

Test lane assignment on representative original gameplay frames, including difficult perspective and occlusion cases.

Document both correct assignments and failures.

---

# 5. Implement Temporal Obstacle Tracking

Object detection occurs independently on each frame.

The next system must identify which detections belong to the same obstacle over time.

Evaluate Ultralytics ByteTrack or BoT-SORT before implementing a custom association algorithm.

Use an existing tracker if it provides reliable results on actual footage.

Track:

- Stable obstacle ID
- Class
- Bounding-box history
- Lane estimates
- First and last observation times
- Estimated image-space motion
- Detection confidence
- Occlusion/missing-detection status

### Required behavior

Handle:

- Occasional missed detections
- Temporarily obscured obstacles
- New obstacles entering the image
- Obstacles exiting the image
- Irregular frame intervals
- Stale track expiration
- Multiple obstacles in one lane
- ID switches

Use actual timestamps when computing motion.

Do not assume 30 or 60 FPS.

Avoid treating repeated views of the same barrier as separate obstacle encounters.

Do not extrapolate tracks indefinitely after detections disappear.

### Acceptance criterion

Replay real gameplay sequences and inspect object IDs across frames.

Report ID switches, lost tracks, and ambiguous associations.

---

# 6. Estimate the Player's State

The existing YOLO model detects obstacles, not the player.

Do not pretend that obstacle detection alone supplies the player's lane.

Implement a player-state estimator with explicit confidence and unknown-state handling.

It should eventually support:

- Current lane
- Running, jumping, or rolling state
- Lane-transition status
- Last known movement
- State-estimation confidence

Investigate feasible approaches:

- Visual localization of the character
- Lightweight image/template tracking
- Calibrated player position
- State propagation from observed keyboard events, if reliably available

Do not assume that user keypresses are observable unless the system actually captures them.

An initial lane assignment is insufficient if the player subsequently changes lanes manually.

If robust player localization is not feasible with the existing data, implement a clearly labeled provisional interface and preserve `UNKNOWN` rather than producing a fictitious position.

Do not retrain YOLO or require full manual player annotation for this milestone.

### Acceptance criterion

The module must distinguish a confidently known player state from an uncertain or unavailable one.

Other components must respect that distinction.

---

# 7. Implement Collision-Risk and Timing Estimation

Build a hazard estimator that combines:

- Player lane/state
- Obstacle tracks
- Lane occupancy
- Apparent approach motion
- Camera geometry
- Processing latency
- Available movement time

The objective is to answer:

**Does this obstacle threaten the player, and how much time remains to respond?**

Do not treat every nearby object as dangerous.

### Time-to-collision

Investigate a calibrated collision boundary and temporal approach model.

If projected position and velocity have meaningful calibrated units, estimate:

`TTC = remaining_projected_distance / approach_rate`

Otherwise, return a relative urgency estimate, not fabricated seconds.

Use temporal smoothing conservatively.

Account for:

- Noisy detections
- Unstable apparent velocity
- Camera movement
- Occlusion
- Changing gameplay speed
- Insufficient observation history
- Stale frames
- Invalid or negative TTC values

### Action feasibility

Represent estimated:

- Lane-change duration
- Jump duration
- Roll duration
- Movement cooldown
- Player transition state

Measure timing when possible.

Where unavailable, use explicitly named, configurable provisional parameters rather than pretending they are experimentally verified.

Do not assert that an action is feasible without considering its execution time and estimated collision deadline.

### Acceptance criterion

Evaluate known gameplay sequences containing approaching obstacles.

The system should identify hazards and uncertainty without producing unstable or physically impossible timing claims.

---

# 8. Create a Stable GameState Interface

Implement a reusable typed representation of the entire estimated game state.

Prefer Python dataclasses or similarly lightweight structures.

Conceptually:

```python
GameState(
    timestamp=...,
    player=...,
    lanes=...,
    tracked_obstacles=...,
    hazards=...,
    available_actions=...,
    confidence=...,
    valid=...
)
```

Design the structure for both:

1. A rule-based controller.
2. A future Gymnasium/DQN agent.

It must support:

- Player lane and movement state
- Tracked obstacles
- Lane occupancy
- Estimated collision urgency
- Observation freshness
- Confidence/uncertainty
- Available actions
- Recent action/state history

### RL compatibility

Provide a deterministic conversion to a fixed-dimensional numerical observation vector.

For example, encode the nearest few obstacles per lane using class, normalized position, timing estimate, and confidence.

Use masks to distinguish:

- No obstacle
- Unknown obstacle state
- Missing track
- Invalid measurement

Do not silently replace unknown TTC with zero, since zero could indicate an immediate collision.

Keep observation ordering and normalization stable.

Document the schema.

**Do not implement Gymnasium or DQN training yet.**

The goal is to make future integration straightforward without prematurely constraining the entire RL architecture.

---

# 9. Implement a Read-Only Shadow Policy

Create a simple deterministic policy that consumes `GameState` and proposes one action:

```text
NO_OP
LEFT
RIGHT
JUMP
ROLL
UNKNOWN
```

`UNKNOWN` represents insufficient information for a defensible recommendation, not an executable game action.

The shadow policy must never send keyboard commands.

### Decision requirements

Consider:

- Current player lane
- Nearby and approaching hazards
- Alternative lanes
- Estimated time to collision
- Action feasibility
- Simultaneous obstacles
- Current player animation
- Uncertainty

Do not implement naive rules such as:

"Ground barrier detected → jump immediately."

Only recommend an action when the relevant hazard threatens the player's path.

Avoid recommending a lane change into an occupied or uncertain lane without accounting for the risk.

Prevent repeated recommendations for the same tracked obstacle from being misinterpreted as separate required actions.

Keep the policy simple and interpretable.

### Acceptance criterion

Given a game state, the policy must return:

- Proposed action
- Reason
- Threatening obstacle ID, if applicable
- Relevant timing estimate
- Confidence/uncertainty
- Whether the recommendation is actionable

No automatic keypresses.

---

# 10. Build Offline Replay and Live Visualization

Provide two executable modes using the same underlying pipeline.

## Mode A — Offline replay

Process previously recorded gameplay.

Requirements:

- Read original image sequences.
- Preserve original image dimensions.
- Use genuine frame timestamps when available.
- If only approximate timing exists, explicitly identify it as approximate.
- Optionally replay at a controlled speed.
- Produce structured game states.
- Display visual overlays.
- Export results for inspection.

The replay system should not depend on Chrome being open.

This must be implemented and tested first.

## Mode B — Live observation

Capture Subway Surfers in Chrome and run the same state-estimation pipeline continuously.

Requirements:

- Display the latest annotated gameplay frame.
- Show capture/inference FPS.
- Show end-to-end latency.
- Support clean start/stop behavior.
- Handle window focus or screen-permission failures.
- Avoid uncontrolled memory growth.
- Never send keypresses.

### Visualization

Display:

- YOLO bounding boxes
- Classes and confidence
- Track IDs
- Calibrated lane boundaries
- Player-lane estimate
- Obstacle-lane assignments
- Hazard status
- Available TTC estimates
- Suggested action and explanation
- Unknown/uncertain states
- System latency

Distinguish direct observations from inferred states visually.

Do not cover important portions of the game unnecessarily.

Use toggles to show/hide overlay categories.

Do not permanently modify original screenshots.

---

# 11. Testing Requirements

Implement meaningful tests for the new modules.

Prioritize deterministic tests over superficial code-coverage targets.

Required categories:

**Perception**

- Correct output schema
- Invalid model path
- Missing frame
- Image dimensions
- Timestamp propagation

**Geometry**

- Lane assignment
- Perspective changes
- Edge-clipped objects
- Large train rectangles
- Ambiguous footprints
- Invalid calibration

**Tracking**

- Stable IDs
- Missing detections
- Track expiration
- Irregular timestamps
- Multiple similar obstacles

**Player state**

- Known vs. unknown lane
- Missing observations
- State transitions

**Hazard estimation**

- Approaching obstacle
- Non-threatening neighboring lane
- Unknown TTC
- Noisy motion
- Insufficient reaction time
- Simultaneous threats

**GameState**

- Valid fixed-dimensional encoding
- Missing-value masks
- Deterministic ordering
- Stable normalized values

**Shadow policy**

- Safe candidate actions
- Unsafe neighboring lane
- Unknown player state
- Conflicting hazards
- Repeated recommendations
- No keyboard execution

**Integration**

- Offline replay
- State logging
- Clean shutdown
- Bounded queues
- Backward compatibility

Use existing footage for practical integration tests.

Do not claim that synthetic test success proves real-game accuracy.

Do not invent ground-truth lane or timing labels.

Where labeled evaluation data is insufficient, provide qualitative examples and explicitly state the limitation.

---

# 12. Benchmark and Optimize

Use the available Mac to benchmark actual pipeline performance.

Report:

- Machine and processor
- RAM
- Python/PyTorch/Ultralytics versions
- Device used (MPS or CPU)
- Average and median inference latency
- End-to-end observation latency
- Processed FPS
- Frame drops
- Memory use
- Major bottlenecks

Optimize only where measurements justify it.

Potential optimizations:

- Model reuse
- Bounded frame queues
- Reducing image copies
- Avoiding unnecessary disk writes
- Skipping stale frames
- Separating display from inference
- Reusing computed geometry
- Efficient object-history storage

Do not sacrifice detection reliability solely to increase FPS.

If the live Chrome environment is inaccessible, do not fabricate live benchmarks. Test offline and provide an exact local smoke-test command.

The potential future 16–32 GB Mac upgrade should not require architectural rewrites.

---

# 13. Repository Structure

Prefer a maintainable layout resembling:

```text
agent/
    perception.py
    geometry.py
    tracking.py
    player.py
    hazard.py
    state.py
    rule_policy.py
    visualization.py

scripts/
    gameplay.py
    train.py
    evaluate.py
    observe_live.py
    replay_gameplay.py

tests/
    test_perception.py
    test_geometry.py
    test_tracking.py
    test_player.py
    test_hazard.py
    test_state.py
    test_rule_policy.py
    test_pipeline.py
```

Use this as guidance, not an inflexible requirement.

Prefer fewer cohesive modules over unnecessary abstraction.

Preserve existing script behavior and compatible interfaces unless a change is justified.

Do not introduce unrelated functionality.

---

# 14. Engineering Quality Requirements

Follow these rules throughout:

1. Reuse existing code and installed libraries wherever practical.
2. Keep original screenshots unchanged.
3. Preserve existing labels, checkpoints, and experimental results.
4. Do not retrain YOLO.
5. Do not implement RL training.
6. Do not send keyboard input.
7. Do not assume fixed FPS.
8. Do not assume bounding-box overlap implies lane occupancy.
9. Do not silently substitute guesses for missing observations.
10. Do not claim calibrated TTC without calibration.
11. Do not fabricate accuracy or performance improvements.
12. Do not use unrealistically clean synthetic examples as the only evidence.
13. Avoid new dependencies unless necessary.
14. Make the pipeline testable offline before connecting live capture.
15. Keep logging efficient and avoid unbounded memory usage.
16. Preserve uncertainty throughout the pipeline rather than discarding it.
17. Do not perform destructive Git operations or overwrite user changes.

When existing footage is insufficient to validate a component, implement a conservative provisional version, document the uncertainty, and avoid declaring it production-ready.

---

# 15. Execution and Completion Criteria

Work through the implementation rather than only drafting plans.

Use this order:

### Stage 1 — Repository inspection and architecture

Inspect existing code, establish reusable interfaces, identify missing data and calibration needs.

### Stage 2 — Offline core implementation

Build perception, geometry, tracking, player-state, hazard estimation, and GameState conversion.

Run tests and debug against existing recordings.

### Stage 3 — Shadow policy and visualization

Integrate state estimation into offline replay and render actual detections, hazards, lane estimates, and proposed actions.

### Stage 4 — Live integration

Connect the same pipeline to live screen capture, preserving read-only behavior.

### Stage 5 — Benchmark and documentation

Run tests, benchmark what is accessible, document failures and remaining limitations.

Do not proceed to a more complex stage merely because the previous module imports successfully. Verify its key behavior.

Do not stall the entire project because one difficult feature, such as player localization, cannot be solved robustly. Implement a clearly isolated fallback that reports uncertainty, then continue with independently testable components.

Prioritize functioning integration over polishing secondary features.

---

# FINAL DELIVERABLES

At completion, provide a concise technical report.

Include:

**1. Implementation**

- Files created/modified.
- Architecture and component interfaces.
- What actually works.

**2. Demonstration**

- Offline replay command.
- Live observation command.
- Screenshots or example outputs showing real detections and game-state overlays.

**3. Verification**

- Tests executed.
- Pass/fail results.
- Actual benchmarks.
- Representative successes and failures.

**4. Limitations**

- Unreliable lane assignments.
- Missing player-state information.
- Occluded overhead-barrier failures.
- Uncalibrated collision timing.
- Missing independent evaluation evidence.

**5. Future RL readiness**

- Whether GameState is suitable for a future Gymnasium environment.
- Which components remain blockers.
- What must be implemented next for automatic gameplay.

Provide exact commands for running the code locally.

Do not label untested functionality as complete.

## FINAL DEFINITION OF DONE

I should be able to launch an offline replay script or a live observation script and see a continuous, timestamped representation of the game.

The system should, where supported by actual observations:

1. Detect obstacles using the retained YOLO model.
2. Estimate obstacle lanes using perspective-aware geometry.
3. Track obstacles over time.
4. Estimate the player's lane or explicitly report uncertainty.
5. Determine which obstacles may threaten the player.
6. Estimate collision urgency or TTC when calibrated.
7. Recommend a movement action without executing it.
8. Display all of this in an understandable visualization.
9. Save structured observations for future RL integration.

**The critical outcome is a reliable bridge between computer vision and sequential decision-making.**

This is the foundation for the next milestone:

**Automatic controls → Rule-based gameplay → Gymnasium environment → DQN training.**

Implement the working system, validate what can actually be tested, and distinguish completed functionality from provisional estimates. Do not claim the game-state engine is reliable solely because it produces outputs.



