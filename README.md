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
YOLO11 model for about 12 minutes using the Mac GPU when available. It uses one
in three eligible early frames for training, holds out later frames for validation,
and leaves a gap between them. Excluded frames never enter either set. Images and
labels stay in their original locations; only small image lists are generated.

The current settings fine-tune `models/baseline/weights/best.pt` at 640 pixels
and save a separate run in `models/barriers_640/`. Later runs get a new numbered
folder. Change `MODEL` to `models/yolo11n.pt` to start again from generic YOLO
weights. The detector recognizes trains and two barrier types; it does not control
the game. Labels are AI-assisted drafts, and validation is from the same recording.
Collect a separate session, including clear scenes with no obstacles, before
treating these scores as evidence that the detector works on new gameplay.
