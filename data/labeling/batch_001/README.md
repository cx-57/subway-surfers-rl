# Gameplay labeling batch 001

All **1,035 images** from `session_1795365206207213000` now have YOLO label files.
The batch expanded from 100 selected images to all 1,035 recorded frames.
A later ground-barrier revision also corrected 15 of the original selected images.
Every image received AI-assisted visual review. Neighboring reviewed frames were
used to assist temporal labeling where recorded in `review_audit.json`; this is
not verified human ground truth. Ambiguous cases remain explicit in that report.

Open `review/index.html` to view any frame with its boxes. Use Previous/Next or
arrow keys and the frame selector. The gallery reads original images directly,
avoiding 935 additional full-size JPEG copies. The 52 contact sheets in `review/`
show every frame. The redundant standalone preview copies were removed; the gallery draws boxes
over the unchanged originals.

Validated: 1,035 readable images, 1,035 label files, 3,679 boxes,
finite normalized coordinates within image bounds, and unchanged source SHA256
fingerprints for all 1,035 images. `manifest.json` maps each label to its source.

**Training status:** 876 frames were originally eligible and 159 excluded for
pause/countdown, obscured camera, or individual review reasons. The partial-object
revision leaves 810 usable and 225 excluded after combining eligibility and
uncertainty flags. Excluded frames still have labels and remain in the gallery.
Use `training_eligible` and require `annotation_uncertain` to be false when preparing data. The original
pause/countdown range 1–104 and camera-obscured range 635–665 remain excluded.
Empty label files only mean no confidently identifiable target object was labeled;
excluded or uncertain frames must not be treated as clear gameplay negatives.

## Partial-obstacle revision (2026-11-23)

Reexamined all 1,035 original frames on aspect-preserving contact sheets, with
native crops for correction candidates. Added 60 overhead barriers and 27 train
fragments; adjusted 17 train boxes, three overhead boxes and eight ground boxes.
The 115 row changes affect 103 images. Sources remain unchanged, and the labels
remain AI-assisted drafts rather than verified human ground truth.

`partial_audit.json` records every old/new row, visible pixel bounds, reason,
before/after frame labels, uncertainty notes and label hashes. It flags 118 frames
as uncertain, including previously excluded frames. Of the original 876 eligible
frames, 810 remain usable. The fixed temporal/sample split now contains 200 training
frames and 144 validation frames; 37 former validation frames are excluded because
their object extents remain ambiguous. Both baseline and candidates must use the
same revised labels and exclusions.

Open `review/partial_before_after.jpg` for six before/after examples, or
`review/index.html` for every frame. Contact sheets have been rebuilt from the
current labels. No original screenshot was warped, pasted into or overwritten.
No usable background-only image was verified in the old session; its empty-label
frames show pause menus or obscured camera views and were not added as negatives.

`review_audit.json` records actual review methods, uncertainties and provenance for
each frame, plus the previous 100-image correction audit. `annotation_notes.json`
contains the initial selected-frame notes. All source recordings remain untouched.

## Ground-barrier revision (2026-11-23)

Reviewed all 565 existing ground-barrier boxes across 475 images at original-pixel
resolution. Corrected 206 images and added one clearly visible missed barrier;
there are now 566 ground-barrier boxes. All other classes were preserved.
Corrections follow visible caps, supports and bases, using the original images
rather than model predictions. Before/after rows and review methods are recorded
in `review_audit.json` under `ground_barrier_revision`. Labels remain AI-assisted
drafts; heavily occluded fragments are flagged explicitly.

Validation labels changed in 38 images; 12 more remain flagged for human review.
Re-score both baseline and candidate on the same revised validation labels. The
old 66.7% ground-barrier mAP50 used earlier labels and cannot serve as the direct
baseline for a model improvement claim. No new recording was added.

## Classes and annotation rules

YOLO text files use: class_id center_x center_y width height, normalized to [0,1].

- 0: train — visible contiguous train/cart body, including roof and an attached
  boarding ramp. The theme's blue/white and red/yellow lantern-like rail vehicles
  count as trains. Separate trains on separate lanes get separate boxes.
- 1: ground_barrier — short horizontal red/white beam obstacles.
- 2: overhead_barrier — tall raised red/white downward-chevron frames.

These are provisional visual categories, not action labels: some horizontal
barriers can apparently be jumped over OR rolled under. Review class definitions
against gameplay before training or mapping detections to actions.

Draw boxes around visible object extents, clipping at image boundaries. Include
visible supports for a barrier. Do not invent occluded train geometry, label
background buildings, coins, power-ups, player, policeman, HUD, or game UI.
A contiguous train with unclear car boundaries gets one enclosing visible box;
do not split it arbitrarily into extra instances.

Labels are visually produced AI-assisted DRAFTS, not verified human ground truth.
Review the overlaid images before training, especially small distant obstacles
and partially occluded trains. No detector or unverified class assumptions should
be used to create automatic labels. An empty label file means the image was
reviewed and contains no confidently identifiable objects in these three classes.

All images are from one session. Do not randomly split them into training and
validation sets; collect another session for validation before training.

## Perspective and collision review

Detection boxes are NOT collision zones. A tight rectangle enclosing an angled
train naturally includes some empty track. Do not shrink a correct object box
just to leave an apparent lane clear, and never block a lane merely because a
rectangle overlaps it. Three clearly loose foreground-train boxes were tightened;
the exact changes are recorded under `perspective_audit` in annotation_notes.json.
Broad train boxes have a `review_flags` entry in manifest.json; that is a review
reminder, not proof that a label is wrong.

Before adding game controls:

- Calibrate the track/lane geometry near the player, including curves and camera
  movement. Image thirds and the center/bottom-center of a long train box are not
  reliable lane assignments on their own.
- Estimate the object's footprint at the relevant depth, using lane geometry and
  visible boundaries, segmentation, or another validated method. Test empty lanes
  beside long trains and trains extending beyond the image before blocking actions.
- Treat barriers separately: a lane with an overhead barrier may permit rolling;
  some short barriers permit more than one action. Class names are not an action policy.
- Keep occlusion/uncertain labels explicit. Use measured frame timing for later
  tracking and collision timing, not the requested 60 FPS as if it were achieved.

No lane-occupancy or collision model has been implemented or validated yet.
