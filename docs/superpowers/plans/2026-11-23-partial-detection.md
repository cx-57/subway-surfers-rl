# Partial-obstacle detection implementation plan

> **For agentic workers:** Use executing-plans for implementation. Independent visual audits are delegated with disjoint frame ranges.

**Goal:** Correct real partial-obstacle annotations and measure conservative YOLO11 training changes without altering source images or tuning against the regression recording.

**Architecture:** Keep the chronological image lists and small YOLO11n detector. Add strict dataset checks, separate visual review artifacts, and a compact experiment/evaluation script; use Ultralytics training-time transforms only.

**Tech stack:** Python 3.12, Ultralytics 8.4.216, Torch/MPS, OpenCV, unittest.

**Spec:** User attachment `b5f5cf58-8229-4ec7-91a9-127518a8ef4b/Pasted text.txt`.

## Constraints

- Original image bytes, dimensions and aspect ratios stay unchanged.
- Labels use confidently visible extents only. Ambiguous frames are excluded from quantitative evaluation; preserve their review metadata.
- Train source frames <=780, validation >=850, gap untouched. Newer recording is reserved regression data because prior diagnosis already inspected it.
- Preserve all previous checkpoints. The rejected pasted-hood run is never a starting checkpoint.
- Reevaluate frozen baseline on revised labels. Train matched controls from the same frozen checkpoint with identical seeds, resolution, batch, learning rate, epoch cap and stopping rule.
- Evaluate natural validation before opening candidate regression scores. No claim of unseen generalization without a new recording.

## Review focus

- Empty labels mean background only after positive visual verification.
- Tiny ambiguous targets must not become negative training/evaluation targets.
- Boxes covering an occluder may still tightly enclose disconnected visible parts.
- Same-size label edits invalidate Ultralytics scan caches.
- Training-time transformed previews must preserve source integrity and label geometry.

## Tasks

- [x] Visually audit 1,035 old-session frames and examine annotation candidates at native resolution. Save proposed before/after corrections and uncertainty reasons before applying them.
- [x] Apply visually justified corrections and update manifest, audit trail and separate overlays. Verify every image hash and label boundary.
- [x] Extend dataset-preparation tests: finite valid class/coordinates, empty labels, uncertainty exclusion, stable temporal splits and unchanged sources. Implement minimal corresponding checks.
- [x] Make fixed-seed configurations for matched control, translation/scale and conservative Mosaic, each capped at 12 epochs with patience 5, batch 2 and 640px input. Inspect built-in transformed training samples; no pasted-hood generator.
- [x] Re-score frozen baseline on revised natural validation. Run control, translation/scale and Mosaic sequentially on MPS with epoch-end memory cleanup. Record completed budgets and errors. Investigate MixUp/CutMix feasibility; defer if unsupported or unjustified.
- [x] Select on natural validation; evaluate completed candidates once on the locked regression labels. Report class metrics, preliminary partial/small/edge counts, verified-negative false positives, near-player misses and latency with limitations.
- [x] Run tests, integrity checks and independent code review. Document actual completed results and reproduction commands in a concise repository report.

## Execution record

- Completed the full annotation audit, five bounded training runs and final evaluation/integrity checks. Results: `docs/partial_detection_results.md`.
- Ruling: implemented in the existing shared repository to preserve the supplied local data, virtual environment and active project context. Source screenshots and previous checkpoints were protected by hashes.
- Ruling: added isolated MixUp0.05 after the higher-priority runs; no combinations were justified.
- Ruling: added one data-coverage control retaining all eight reviewed early partial-barrier views (six additional screenshots), before candidate regression scoring. It changes no pixels and keeps chronological separation.
- Review: fixed the actual file-loader batching/NMS-truncation issue, guarded both AP validation and prediction, and verified source-image fingerprints before/after evaluation. Regression tests pass.
- Final model choice: preserve the frozen existing detector. No candidate fixed the five newer hood-only misses; generalization remains unverified.
