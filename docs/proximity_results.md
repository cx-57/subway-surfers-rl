# Expanded proximity check

## Distance-filtered AP50 rerun

Fresh inference on the same 236 frames with the retained checkpoint, confidence
floor 0.001, input size 640, NMS IoU 0.7 and one-to-one matching IoU >=0.5 gave:

| Class | Combined AP50 | Nearby labeled appearances |
| --- | --- | --- |
| Train | 96.0% | 333 |
| Ground barrier | 96.2% | 31 |
| Overhead barrier | 92.5% | 13 |
| Mean | 94.9% | 377 |

Older 144-frame validation mAP50: 93.9%; newer 92-frame regression mAP50: 97.5%.
The newer cohort has just two nearby overhead appearances (AP50 99.5%), so that
score is not strong generalization evidence. Combined overhead appearances still
cover only two encounters. AP sweeps confidence; it is not recall at the live
50% threshold. These scores evaluate an easier, explicitly selected nearby subset,
not changed model weights or calibrated physical collision distances.

Strict ROI protocol: apply the existing class-specific bottom/width conditions
to both predictions and labels, independent of confidence. Unmatched retained
predictions count as false positives, including geometry-boundary mismatches.
AP uses Ultralytics' 101-point integration. This AP protocol differs from the
earlier fixed-cutoff table below, where matching a true object outside the nearby
label subset was not counted as a false positive.

Reproduce with `.venv/bin/python scripts/distance_map.py`. Predictions, hashes,
protocol and summaries are saved in
`models/evaluation/distance_map50_1795539557960037000/`.
No training or live-filter integration was performed by this evaluation.

## Earlier fixed-confidence check

Replayed frozen-checkpoint predictions on 236 labeled frames at 50% confidence:
144 held-out validation frames from the older recording, plus 92 previously
inspected regression frames from the newer recording. No training, fresh model
inference, source-image edits, or threshold tuning took place. The checkpoint
hash and annotation revisions agree across the saved evaluations and manifests.
Both recordings have been inspected; neither is a new untouched test recording.

The unchanged gate uses box bottom and width as fractions of the cropped gameplay
image: barriers need bottom >=55% and width >=6%; trains need bottom >=65% and
width >=12%. A true object is counted as nearby when its label meets those same
geometry conditions. Kept predictions are matched one-to-one against all labels
at IoU >=0.5; this avoids calling a correctly detected object a false positive
merely because its predicted and labeled boxes fall on opposite sides of the zone.
Nearby recall requires a matching prediction that also passes the gate.

| Cohort | Class | Nearby objects found | Recall | Kept false positives |
| --- | --- | --- | --- | --- |
| Older validation | Train | 222/240 | 92.5% | 1 |
| Older validation | Ground barrier | 16/20 | 80.0% | 0 |
| Older validation | Overhead barrier | 10/11 | 90.9% | 0 |
| Newer regression | Train | 82/93 | 88.2% | 0 |
| Newer regression | Ground barrier | 9/11 | 81.8% | 0 |
| Newer regression | Overhead barrier | 2/2 | 100% | 0 |
| Combined | Train | 304/333 | 91.3% | 1 |
| Combined | Ground barrier | 25/31 | 80.6% | 0 |
| Combined | Overhead barrier | 12/13 | 92.3% | 0 |

These are frame-level object appearances, not independent encounters. The nearby
overhead cases comprise older frames 1025–1035 and newer frames 890/900: only two
barrier encounters. The two verified empty newer frames are included; no additional
verified empty held-out frames were available in the older validation snapshot.

The older frame 1025 overhead miss is caused by the gate, not recognition. The
detector scored it 96.8%, but its predicted bottom was y=439.7, above the gate's
y=447.7 boundary (55% of height 814). The label bottom was y=452.7. It is admitted
on later frames. The other nearby misses were already missed at the 50% confidence
cutoff before applying the distance gate. This demonstrates boundary sensitivity;
the gate has not been proven to allow sufficient reaction time. It also cannot
determine whether a nearby train actually blocks the player's lane.

The gate improves precision by restricting which detections are considered nearby;
it does not improve the underlying model or overall mAP. These measurements do not
establish gameplay success or a safe collision deadline.

Per-frame predictions, nearby selections, missed boxes and summaries are saved in
`models/evaluation/proximity_expanded/`. The input snapshots are
`models/evaluation/frozen_validation/` and `frozen_regression/`.
