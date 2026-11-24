# Independent test session

All 920 frames in session_1795440601129365000 remain held out from training.
Every tenth frame (92 images) was selected before predictions. The original
visual labels were created without detector pseudo-labels and were scored as
v1. They are AI-assisted drafts, not verified human ground truth.

## Diagnostic label revision v2

A later prediction-error audit found two visible ground barriers missing from
frames 50/60. Their visible caps/supports were independently bounded on original
native pixel grids, with approximately ±2px uncertainty. Two rows were added;
no other object coordinates changed. This repair happened AFTER predictions and
is recorded with exact before/after rows in review_audit.json.

Original v1 metrics remain in models/barriers_640/test_session/comparison.json.
Both fixed checkpoints were re-scored on identical v2 labels; exact metrics are
in models/barriers_640/diagnosis/test_v2_rescore.json. No training or checkpoint
selection used the test labels. Future reports must identify which label version
they use.

The current 92-image sample contains 245 trains, 59 ground barriers, 8 overhead
barriers and two object-free frames 910/920. All eight overhead boxes are repeated
views of one obstacle, mostly hidden behind a train, so they are not eight
independent encounters. Quantitative scores describe the sampled frames, not all
920 frames. Every raw frame has predictions saved separately.

Open review/index.html to inspect current v2 labels. Original pixels remain
unchanged. dataset.yaml is evaluation-only; its train key points to the OLD
training list only to satisfy the evaluation library schema.
