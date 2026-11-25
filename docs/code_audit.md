# Code audit

## Simplification follow-up

Reviewed every project Python file, including tests. Replaced list, dictionary
and set comprehensions with ordinary loops, expanded packed one-line statements,
and replaced production anonymous functions with named operations. Added comments
for capture timing, proximity filtering, tracking, player matching, immutable
state snapshots and the fixed observation schema.

`scripts/observe_live.py` now contains just the live workflow. Its argument
handling lives alongside the shared runtime options, and the existing capture
class supplies the bounded frame iterator instead of a nested function.
Replay filenames use straightforward string parsing instead of regular
expressions. The six pinned third-party packages are still needed for screen
capture, image processing and YOLO; no application dependency was added.

Ordinary loops and comments can take more lines than compressed expressions.
This pass prioritizes readable steps while retaining capture, video, detection,
UPCOMING boxes, proximity checks, evaluation integrity and opt-in controls.

Verification includes the full unittest suite, a real 20-frame replay with video,
before/after checks of 100 observation vectors, 30 tracker updates and five engine
frames, plus a separate review comparing 5,000 randomized vector/policy cases.
No training or live keyboard actions were performed. The current commands in
the README still apply.

## Earlier audit

Reviewed the production modules in `agent/`, all seven scripts, tests, configuration
and entrypoint documentation. This audit refactors the existing read-only engine;
it does not train a model or enable keyboard controls.

## Cleanup

- Moved shared confidence/proximity settings, box overlap and the NMS completeness
  guard to `agent/detection.py`. Live inference no longer imports training or
  evaluation scripts. Existing script imports remain compatible.
- Removed duplicate overlap/boundary calculations, redundant policy checks,
  unused imports and dead assignments. Simplified pipeline threat collection.
- Consolidated repeated window naming and rewrote the README around current
  commands, output locations and limitations.
- Kept public geometry/state interfaces and provenance fields needed by the
  implementation task, even when automatic observation cannot populate them yet.

## Correctness fixes

- An incomplete NMS pass now retries once without incorrectly disabling MPS.
  Repeated incomplete results fail explicitly; genuine MPS inference failures
  retain the CPU fallback.
- Resource cleanup attempts both display and video shutdown, preserves the
  original processing exception and records cleanup failures. Reports no longer
  claim a video exists when none was encoded.
- Video close is terminal and idempotent; subsequent writes cannot reopen and
  truncate the recording. Existing timing sidecars are protected.
- Reviewed player observations must match their source frame. Per-frame
  diagnostics reset, and optical-flow refresh uses the configured ambiguity
  margin rather than a hidden constant.
- Invalid tracker/localizer settings, malformed geometry inputs and nonfinite
  frame timing fail explicitly or remain UNKNOWN, as appropriate.
- An empty nearby subset cannot advertise a safe NO_OP when distant lane
  occupancy is uncertain.
- Evaluation no longer matches the wrong class at a zero IoU threshold. Dataset
  integrity checks remain active under optimized Python (`python -O`).

## Verification

All 142 tests pass. The full unittest suite covers these regressions as well as capture, recording,
geometry, tracking, policy, observation schema and evaluation behavior. The audit
also checks dependency consistency, imports and local documentation links.

A real ten-frame replay of session `1795524944131662000`, starting at frame 480,
completed using the retained detector on MPS. Its outputs are in
`models/evaluation/code_audit_replay/`, including `observation.mp4`, `states.jsonl`
and `benchmark.json`. This checks actual inference and video output; it is not an
accuracy evaluation or a representative long-run performance measurement.

Recordings, source images, labels, checkpoints, existing videos and the 113-value
observation layout are preserved. Recorder behavior, live video defaults, the
50% confidence cutoff and proximity thresholds are unchanged. Automatic player
ground contact and confirmed lanes remain unvalidated; UNKNOWN is intentional.
