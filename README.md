# Gameplay screenshots

Open Subway Surfers in Chrome, then open `scripts/gameplay.py` in VS Code.
Press **F5** and choose **Record gameplay**.

1. Bring Chrome forward during the first countdown.
2. Click two opposite corners of the game, then press **Enter**. Press **R** to redo or **Esc** to cancel.
3. Return to Chrome during the 15-second countdown and play.
4. Recording stops after 30 seconds. Screenshots are saved in `data/raw/session_.../`.

Press **Ctrl+C** in the terminal to stop early. Each run creates a new folder.
Change `FPS`, `DURATION`, `START_DELAY`, `SELECTION_DELAY`, and `MONITOR` at the top
of `scripts/gameplay.py` when needed. The target is 60 screenshots per second (about 1,800 images in 30 seconds).
The final summary reports the actual rate; large crops or a busy Mac can be slower.
Each session also saves a small `metadata.json` with settings, crop, timestamps,
frame count, actual FPS and completion/interruption status. Images and metadata
are published atomically, and Ctrl+C finishes counting the current image before
stopping. Use a local Mac folder with hard-link support, such as APFS.
There are no command-line options; settings stay at the top of the script.

To run from the terminal:

```bash
.venv/bin/python scripts/gameplay.py
```

If setup is needed, use Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

In VS Code, select `.venv/bin/python` as the Python interpreter. For screen
permission, enable VS Code or Terminal in **System Settings → Privacy & Security →
Screen & System Audio Recording**. If project files hang, choose **Keep Downloaded**
in Finder.

On this Mac, `.venv` links to a local Python environment under
`~/Library/Application Support/subway-surfers-rl/venv`. This keeps installed
packages outside Desktop syncing; the commands above still work.

Run tests with `.venv/bin/python -m unittest discover -v`.

For future detection/control work: bounding boxes are object labels, not blocked
lane maps. Validate perspective-aware lane occupancy near the player before using
detections to restrict actions. See `data/labeling/batch_001/README.md` for the
annotation review rules and known edge cases.

## Train the obstacle detector

Install the requirements, then run:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/train.py
```

Settings are at the top of `scripts/train.py`. The default run fine-tunes a small
YOLO11 model for up to 12 epochs, with validation-based early stopping, using the Mac GPU when available. It uses one
in three eligible early frames for training, holds out later frames for validation,
and leaves a gap between them. Excluded frames never enter either set. Images and
labels stay in their original locations; only small image lists are generated.
Training uses real gameplay screenshots, without pasted objects.

The current settings fine-tune `models/barriers_640/weights/best.pt` at 640 pixels
and save a separate run in `models/partial_control/`. Later runs get a new numbered
folder. Change `MODEL` to `models/yolo11n.pt` to start again from generic YOLO
weights. The detector recognizes trains and two barrier types; it does not control
the game. Labels are AI-assisted drafts, and validation is from the same recording.
Collect a separate session, including clear scenes with no obstacles, before
treating these scores as evidence that the detector works on new gameplay.

## Partial-obstacle detection requirements

- Never change source screenshots: preserve their bytes, dimensions and aspect ratios. Store labels and overlays separately.
- Label every confidently recognizable visible target, including clipped objects, supports and occluded fragments. Enclose visible extents only; never invent hidden geometry.
- Preserve correct labels. Record each correction's old/new row, class, visual evidence and AI-assisted provenance. Flag ambiguous fragments and exclude their frames from quantitative evaluation and training negatives.
- Inspect original pixels; model predictions are review candidates, never unquestioned ground truth.
- Split chronologically before augmentation: training <=780, validation >=850, with the gap preserved. Never train on validation or the newer recording.
- The newer recording already informed diagnosis, so it is a regression set. Genuine unseen generalization needs another untouched recording.
- Compare frozen checkpoints and matched training controls on identical revised labels. Keep seeds, training budgets and all settings reproducible; select checkpoints on validation only.
- Test built-in translation/scaling, conservative Mosaic and low MixUp individually. Use no significant rotation, perspective warp or vertical flip. Never permanently paste synthetic objects into the dataset.
- Add empty-label hard negatives only after visually confirming no target objects. Ambiguous, paused and camera-obscured frames are not negatives.
- Keep the architecture small for the 8 GB Mac, preserve earlier checkpoints, and judge class/partial/small-object recall, false positives and latency alongside mAP. Training accuracy alone is insufficient.
- Report actual completed experiments and uncertainty. Reject changes that harm reliability; do not repeatedly tune against the regression set or claim gains from label corrections alone.

Controlled runs use `.venv/bin/python scripts/train.py control`,
`translate_scale`, `mosaic`, or `mixup` (one name at a time). Only that training transform
changes; validation images remain unaugmented. `partial_views` is a separate data-selection
control retaining six additional reviewed early frames. The implementation plan is in
`docs/superpowers/plans/2026-11-23-partial-detection.md`.

Current recommended checkpoint: **`models/barriers_640/weights/best.pt`**.
Individual detections now use a **50% confidence cutoff** in `scripts/evaluate.py`.
Historical reports retain their original cutoffs. The mAP calculation still sweeps
confidence scores. There is no live gameplay controller yet.

Evaluation now saves `nearby_prediction_indexes` separately from all detections.
Candidates need 50% confidence plus both a minimum screen position and width:
barrier boxes must reach 55% down the image and span 6% of its width; train boxes
must reach 65% down and span 12%. Change `NEARBY_LIMITS` in `scripts/evaluate.py`
to adjust these starting estimates. They use the cropped gameplay image dimensions.
The gate does not change detection metrics or labels, estimate meters, determine
lane occupancy, or prove there is enough time to react. Large side-view train boxes
can pass it; a future controller must check their actual footprint before avoiding
a lane. Distant detections remain available for tracking but are excluded from
the nearby list. This is recorded-output filtering, not automatic gameplay.
See the [236-frame proximity check](docs/proximity_results.md) for per-class results
and a case where the gate delayed a correct overhead detection.
Five new training variants were completed. None fixed the five hood-only misses
in the newer recording, so the existing checkpoint was retained. The standard
split now uses 200 training and 144 validation frames, with uncertain frames excluded.
See [measured results and reproduction commands](docs/partial_detection_results.md)
and [annotation examples](data/labeling/batch_001/review/partial_before_after.jpg).

For an inference-only confidence baseline, run `.venv/bin/python scripts/calibrate.py`.
It checks ten previously inspected frames without training or changing the default
cutoff. Lowering the overhead cutoff from 25% to 1% recovered one of five hidden
barriers but added two false-positive detections. See
[confidence baseline details](docs/confidence_baseline.md).
