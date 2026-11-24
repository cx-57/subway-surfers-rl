# Inference-only confidence baseline

No training epochs ran for this check. The retained checkpoint was
`models/barriers_640/weights/best.pt`, SHA256
`3408b9b9bb1515b5396ff8ad3a0e471d96e31bafd80f9d81e9dc431df7a00ad6`.
The checkpoint and ten original screenshots passed before/after hash checks.
No source image, label, weight or default threshold was changed.

The sample contains newer-recording frames 830–870 (five hidden overhead
barriers), 880–900 (three exposed barriers), and 910–920 (two verified empty
frames). These frames were already inspected, so this is a regression diagnostic,
not independent evidence of generalization or a set for selecting a default.

| Overhead score cutoff | Hidden barriers detected | Overhead false positives across ten frames |
| --- | --- | --- |
| 25% | 0/5 | 0 |
| 5% | 0/5 | 0 |
| 1% | 1/5 | 2 |
| 0.5% | 1/5 | 9 |
| 0.1% | 1/5 | 44 |
| 0.01% | 2/5 | 287 |
| 0.001% | 3/5 | 1,165 |

All three exposed barriers were detected at each cutoff. False positives count
unmatched overhead detections under class-correct, one-to-one IoU >= 0.5 matching;
they can include inaccurate boxes and duplicates. The two errors at 1% were both
on frame 840, not the empty frames. Train and ground detections were not evaluated
in this overhead-only check.

One inference pass used 640-pixel inputs, batch 2, no augmentation, an overhead
class filter, a 0.00001 confidence floor, NMS IoU 0.7 and a 1,000-detection cap
(never reached). Higher cutoffs were replayed over those cached predictions.
Verified labels came from the frozen regression snapshot, checked against its
checkpoint hash, annotation revision and the manifest's label hashes.

Lowering the cutoff admits some hidden barriers but quickly increases errors.
This does not fit a probability calibration function, improve recognition, or
justify increasing displayed confidence scores. The existing default is retained.

Reproduce with `.venv/bin/python scripts/calibrate.py`. Detailed results and
predictions are under `models/partial_experiments/confidence_baseline/`; repeat
runs use numbered folders.
