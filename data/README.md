# Gameplay data

- `training/images/`: 344 unchanged, labeled originals (200 train + 144 validation).
- `training/labels/`: matching YOLO annotations.
- `training/train.txt` and `val.txt`: explicit, separate image lists.
- `training/manifest.json`: hashes, source provenance, review status and split membership.
- `training/partial_audit.json`: historical annotation audit; includes notes on removed frames.
- `raw/`: incoming recordings. The recorder creates a new session folder for every clip.

Record with `.venv/bin/python scripts/gameplay.py`: 20 FPS target, 60 seconds,
15 seconds after area selection to return to the game. Actual FPS is in each
recording's metadata. Keep the third new clip separate for validation.

The retained model originally used 209 training and 181 validation images.
Later uncertainty exclusions left the usable 200/144 split retained here.
Removed recordings, previews, labels and the obsolete recorder are recoverable
from the macOS Trash dataset folder named in `cleanup.json`. Model weights and
saved observation videos are unchanged. The player template PNGs remain in
`config/player_templates/`; their original source-path entries are historical.
