# Subway Surfers AI

A Mac-based Subway Surfers perception project: gameplay capture → six-class
YOLO detection → tracking → player and obstacle lane estimates → saved
observations. **Autonomous gameplay and reinforcement learning are not implemented.**

Everything is on **main**. For ordinary use, run one command from the repository root.

## Quick start

```bash
cd ~/Desktop/subway-surfers-rl
git switch main
git pull --ff-only origin main
.venv/bin/python run.py
```

The default command opens the live observer. Select the two opposite corners
of the Subway Surfers gameplay area, press Enter, then return to Chrome during
the countdown. The observer records a 30-second session by default.

Recordings are saved in unique `models/evaluation/engine_live_<ID>/` folders:

- `observation.mp4`: annotated gameplay video
- `states.jsonl`: per-frame detections, tracks and estimates
- `observation_schema.json`: fields for the fixed 137-float v3 observations
- `benchmark.json`: measured runtime and processing latency

**No keypresses are sent by default.** Press Q/Esc to stop the live preview.

## All commands

Run with `.venv/bin/python run.py [command]`:

| Command | Purpose |
| --- | --- |
| `run.py` | **Live observation**, 30 seconds (default) |
| `run.py observe --duration 60` | Longer live observation |
| `run.py record` | Record new raw gameplay frames |
| `run.py calibrate data/raw/session_<ID> --every 12` | Label player lane and movement in a recording |
| `run.py observe --calibration config/player_calibration_session_<ID>.json` | View draft player lane/pose estimates from a calibration |
| `run.py replay data/raw/session_<ID> --video` | Analyze saved gameplay without playing |
| `run.py controls` | **Explicitly opted-in** fixed six-key control check |
| `run.py test` | Run all unit tests |
| `run.py evaluate models/training/weights/best.pt data/training/dataset.yaml val` | Evaluate detector |
| `run.py train` | Train a detector with the existing training script |

Scripts under `scripts/` remain available directly, but normal usage goes
through `run.py`; no separate launch or workflow scripts need memorizing.

For the control check, start the game in the center lane and keep Chrome in
front. Mac Accessibility/Automation permissions may be required. Expected
movement is **not verified** player position.

## The two active perception tasks

**Player:** Existing color-template/optical-flow tracking locates the upper
body. An optional **reviewed, scene-specific** lane and movement estimator is
now inside `agent/player.py`. It needs reference frames labeled by a human.
Until independently validated, output remains a draft hypothesis: it does not
claim a known lane, running, jumping, or rolling state.

To build a draft calibration, record gameplay deliberately demonstrating
left/center/right lanes, jumping and rolling:

```bash
.venv/bin/python run.py record
.venv/bin/python run.py calibrate data/raw/session_<ID> --every 12
```

When labeling frames: `1/2/3` selects left/center/right; then `r/j/o`
selects running/jumping/rolling. Space skips; Q finishes.
Use a **separate recording** for checking predictions.

**Obstacles:** Six classes (train, ground barrier, overhead barrier, bush,
small obstacle, pillar) now map to **possible** occupied lane sets based on
perspective-aware box overlap. One obstacle may overlap two lanes. An
unreviewed box never certifies a physical blocking footprint. Scoped
human-reviewed footprints and occupied lanes can be entered via a calibration
file; uncertain objects stay UNKNOWN.

Instructions and validation requirements:
[Player and obstacle lane testing](docs/player_and_obstacle_lanes.md).

## Models, environment and safety

Default detector: `models/training/weights/best.pt` (six classes).
Legacy checkpoints remain in `models/barriers_640/` and `models/baseline/`.
Raw images, curated labels and checkpoints are kept unchanged by observation.
To install dependencies in a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py test
```

On macOS, allow Screen & System Audio Recording for the app launching Python.
The program refuses to claim reliable lane occupancy or safe actions from
detections alone. **Collision deadlines, verified foot contact, and autonomous
controls are still pending.** The new code does not train an RL policy.

Prior results and architecture details (some predate schema v3):
[engine](docs/game_state_engine.md),
[player tracking](docs/player_tracking.md),
[proximity evaluation](docs/proximity_results.md),
[code audit](docs/code_audit.md).
