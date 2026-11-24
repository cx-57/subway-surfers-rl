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

## Kept models

- `models/baseline/weights/best.pt`: original detector.
- `models/barriers_640/weights/best.pt`: retained detector used with the distance filter.
- `models/evaluation/`: retained validation, regression and proximity-check results.

The distance filter is code in `scripts/evaluate.py`, not a second set of weights.
Experimental training runs and calibration diagnostics were removed. The original
training utility remains available in `scripts/train.py` with one configuration;
running it creates a separate `models/training/` run (numbered on repeat runs).
No training is required to use the retained detector.

To evaluate the retained detector on the newer labeled recording:

```bash
.venv/bin/python scripts/evaluate.py models/barriers_640/weights/best.pt data/labeling/test_001/dataset.yaml test
```

The three scripts are `gameplay.py` (record), `evaluate.py` (detect and report),
and `train.py` (optional training). Detection output includes nearby objects and
conditional key recommendations; it does not press keys.

## Partial-obstacle detection requirements

- Never change source screenshots: preserve their bytes, dimensions and aspect ratios. Store labels and overlays separately.
- Label every confidently recognizable visible target, including clipped objects, supports and occluded fragments. Enclose visible extents only; never invent hidden geometry.
- Preserve correct labels. Record each correction's old/new row, class, visual evidence and AI-assisted provenance. Flag ambiguous fragments and exclude their frames from quantitative evaluation and training negatives.
- Inspect original pixels; model predictions are review candidates, never unquestioned ground truth.
- Split chronologically before augmentation: training <=780, validation >=850, with the gap preserved. Never train on validation or the newer recording.
- The newer recording already informed diagnosis, so it is a regression set. Genuine unseen generalization needs another untouched recording.
- Compare frozen checkpoints and matched training controls on identical revised labels. Keep seeds, training budgets and all settings reproducible; select checkpoints on validation only.
- If training again, compare changes individually. Use no significant rotation, perspective warp or vertical flip. Never permanently paste synthetic objects into the dataset.
- Add empty-label hard negatives only after visually confirming no target objects. Ambiguous, paused and camera-obscured frames are not negatives.
- Keep the architecture small for the 8 GB Mac, preserve earlier checkpoints, and judge class/partial/small-object recall, false positives and latency alongside mAP. Training accuracy alone is insufficient.
- Report actual completed experiments and uncertainty. Reject changes that harm reliability; do not repeatedly tune against the regression set or claim gains from label corrections alone.

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
Each nearby barrier also gets a `barrier_strokes` recommendation: ground barrier
-> `up` (jump), overhead barrier -> `down` (roll), linked to its prediction index.
Trains require lane planning and get no fixed key. Recommendations are conditional
on the barrier being in the player's path; they do not send keys, choose between
conflicting barriers, or decide action timing. Consecutive frames can repeat a
recommendation, so a future controller must also prevent repeated keypresses for
the same encounter.
See the [236-frame proximity check](docs/proximity_results.md) for per-class results
and a case where the gate delayed a correct overhead detection.
Rejected training variants were removed because they did not fix the hidden-barrier
misses. Source recordings, reviewed labels and annotation audits are preserved.
See [annotation examples](data/labeling/batch_001/review/partial_before_after.jpg).
