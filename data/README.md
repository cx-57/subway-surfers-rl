# Gameplay dataset

The curated dataset contains **929 images: 607 training and 322 validation**.
It combines the previous 783 labeled images with **146 reviewed images** from
the four latest recordings. All **7,657 source screenshots across eight clips**
remain unchanged in `raw/`; unselected frames are preserved and are not silently
treated as negatives. The empty recording folder is not a training example.

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
| 0 | train | 1230 | 720 |
| 1 | ground_barrier | 323 | 139 |
| 2 | overhead_barrier | 170 | 177 |
| 3 | bush | 75 | 38 |
| 4 | small_obstacle | 26 | 11 |
| 5 | pillar | 213 | 74 |

The small-obstacle class now contains **25 orange warning-box boxes in training
and nine in validation**, plus three domed-bin boxes (one training, two validation).
Individual visible bushes flanking a warning box are labeled separately from
each other and from the box; coins and the pursuing guard are excluded.

`bush` means free-standing foliage in a playable lane. `small_obstacle` includes
orange warning boxes and low domed bins that must be jumped. A warning box amid
a bush can have separate overlapping component labels. `pillar` includes solid
central stone columns and the vertical blocking stems of T-shaped dividers.
Decorative sidewalk shrubs/columns, overhead crossbeams, coins and powerups are
excluded. Attached ramps may be part of a visible train; unresolved standalone
ramps are excluded rather than assigned a made-up class.

All 344 older retained images were reviewed for the new classes. No additional
bush, small-obstacle or blocking-pillar examples were identified there.

## First four recordings

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
ongoing game, so the holdout is not an independent unseen gameplay run.

## Four latest recordings

| Recording suffix | Split | Reviewed images added |
| --- | --- | ---: |
| 1795666372799776000 | validation | 49 |
| 1795666505136434000 | train | 53 |
| 1795666809314546000 | train | 11 |
| 1795666972776391000 | train | 33 |

These four clips contain 3,008 original screenshots. Selection preserves scene
changes and multiple distances for orange warning boxes, individual low bushes
and blocking pillar stems, while avoiding dense runs of nearly identical frames.
The illuminated T divider contributes stem labels; its overhead crossbeam is
excluded from the pillar box. Golden central blocking columns are also pillars;
decorative sidewalk columns and potted shrubs are not.

Collision/revive/countdown screens, reward screens, advertisements and editor
captures are excluded from the curated splits. The warning-box clip held out for
validation is different from the new training clips. Entire recordings stay in
one split, but recurring layouts and continued games still limit independence.
These are visually reviewed AI-assisted drafts, including one tiny heavily
occluded barrier marked medium confidence in its annotation notes.

The source images are never painted or resized. `data/raw/` remains ignored,
with zero raw files tracked by Git; the selected originals, separate labels,
portable lists and annotation provenance are Git-visible. The current checkpoint
has not been retrained on these 146 additions.

## Rebuild and train

GitHub includes the curated images, labels, portable split files, annotation JSONs
and best checkpoints as ordinary files (no Git LFS pointers). Full raw clips,
environments and generated videos remain local. A checkout can train/evaluate
from the curated originals without those raw clips. Image hashes still verify
the curated bytes; raw-source hashes are also checked whenever raw files exist.

Split paths beginning `./` are relative to their split file. The dataset YAML
uses its own directory as the root, so it works outside the original Mac path.

Reviewed native-pixel annotation JSONs live in
`models/evaluation/obstacle_labels/session_*/annotations.json`. Rebuild using:

```bash
.venv/bin/python scripts/prepare_obstacles.py
```

The importer checks hashes, bounds, class order, reviewed negatives and split
leaks before publishing labels. It preserves explicit selections without sampling
them again. A removed/excluded annotation is removed from the active curated
split while its raw screenshot remains intact.

The retained `barriers_640` checkpoint still has the original three classes.
A subsequent 12-epoch run produced `models/training/weights/best.pt` with all six
classes (saved-checkpoint validation mAP50 0.6386, mAP50-95 0.40699). Use the new
checkpoint explicitly for live observation:

```bash
.venv/bin/python scripts/observe_live.py --checkpoint models/training/weights/best.pt
```

Start another training run with:

```bash
.venv/bin/python scripts/train.py
```

Existing checkpoints remain intact. The script prints the new checkpoint path.
Earlier mAP reports refer to older labels/class sets and are historical results.

Record additional clips with `.venv/bin/python scripts/gameplay.py`: 10 FPS target,
60 seconds, with 10 seconds after selection to return to the game. Actual FPS is
stored in session metadata. Review overlays/crops are separate derivatives under
`models/evaluation/`; no source image was painted, stretched or annotated in place.
