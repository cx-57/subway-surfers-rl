# Gameplay dataset

The curated dataset contains **783 images: 510 training and 273 validation**.
It combines the previous 344 labeled images with 439 reviewed frames from all
four new recordings. All **4,649 new source screenshots remain unchanged** in
`raw/`; unselected frames are preserved but are not silently treated as negatives.

## Folders

- `training/images/`: session-prefixed images prevent filename collisions.
- `training/labels/`: matching YOLO labels, separate from the image pixels.
- `training/train.txt`, `val.txt`, `dataset.yaml`: current training inputs.
- `training/manifest.json`: source hashes, annotation provenance and fixed splits.
- `training/dataset_report.json`: per-class box/frame counts.
- `raw/session_*/review.json`: integrity checks, selected frames and exclusions.
- `cleanup.json`: historical cleanup record from before these four recordings.

## Classes and labeled examples

Counts are **bounding boxes**, including repeated views of an obstacle; they are
not counts of distinct physical obstacles. Labels are AI-assisted visual drafts,
not independently human-verified ground truth.

| ID | Class | Training boxes | Validation boxes |
| --- | --- | ---: | ---: |
| 0 | train | 991 | 622 |
| 1 | ground_barrier | 258 | 116 |
| 2 | overhead_barrier | 140 | 148 |
| 3 | bush | 46 | 10 |
| 4 | small_obstacle | 13 | 2 |
| 5 | pillar | 192 | 74 |

The small-obstacle class contains 12 orange warning-box views (all training)
and three domed-bin views (one training, two validation). There are currently
no orange warning-box views in validation.

`bush` means free-standing foliage in a playable lane. `small_obstacle` includes
orange warning boxes and low domed bins that must be jumped. A warning box amid
a bush can have separate overlapping component labels. `pillar` includes solid
central stone columns and the vertical blocking stems of T-shaped dividers.
Decorative sidewalk shrubs/columns, overhead crossbeams, coins and powerups are
excluded. Attached ramps may be part of a visible train; unresolved standalone
ramps are excluded rather than assigned a made-up class.

All 344 older retained images were reviewed for the new classes. No additional
bush, small-obstacle or blocking-pillar examples were identified there.

## Four-recording selection

| Recording suffix | Split | Reviewed images added | Ambiguous selected images excluded |
| --- | --- | ---: | ---: |
| 1795570315962077000 | train | 132 | 0 |
| 1795570539890370000 | train | 108 | 10 |
| 1795570682988894000 | validation | 129 | 3 |
| 1795570876212146000 | train | 70 | 0 |

Broad scene samples retain different train fronts, lanes and distances. Rare
obstacles receive denser views. Every selected frame was either annotated after
visual review or explicitly excluded. This does **not** mean every raw frame
has been annotated. Pause/countdown/editor/death-overlay frames are kept out of
curated lists. Empty labels require a specific visual background review.

The third clip stays entirely in validation. These clips appear to continue one
ongoing game, so the holdout is not an independent unseen gameplay run. Only two
validation boxes represent small obstacles; that class needs more varied examples.

## Rebuild and train

Reviewed native-pixel annotation JSONs live in
`models/evaluation/obstacle_labels/session_*/annotations.json`. Rebuild using:

```bash
.venv/bin/python scripts/prepare_obstacles.py
```

The importer checks hashes, bounds, class order, reviewed negatives and split
leaks before publishing labels. It preserves explicit selections without sampling
them again. A removed/excluded annotation is removed from the active curated
split while its raw screenshot remains intact.

No training was run during preparation. The retained `barriers_640` checkpoint
still has the original three classes. Train a new six-class checkpoint before
expecting the observer to detect bushes, small obstacles or pillars:

```bash
.venv/bin/python scripts/train.py
```

Existing checkpoints remain intact. The script prints the new checkpoint path.
Earlier mAP reports refer to older labels/class sets and are historical results.

Record additional clips with `.venv/bin/python scripts/gameplay.py`: 20 FPS target,
60 seconds, with 15 seconds after selection to return to the game. Actual FPS is
stored in session metadata. Review overlays/crops are separate derivatives under
`models/evaluation/`; no source image was painted, stretched or annotated in place.
