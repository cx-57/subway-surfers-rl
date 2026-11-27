# Minimal live reinforcement learning

The RL subsystem reuses `ScreenSource`, `Detector`, `Engine`, player localization,
and `ObstacleTracker`. It does not require lane geometry, player pose, or collision
timing. The normal `observe_live.py` command stays read-only. Only `scripts/rl.py`
in check/train/evaluate mode sends input.

## Install and configure (Windows PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe scripts/rl.py setup
```

Setup is interactive and sends no automated input:

1. Show Chrome and select the complete gameplay crop using two clicks and Enter.
2. Play until the final game-over screen appears. Dismiss revive/continue offers.
   Return to the terminal and follow its countdown. In the captured game image,
   select a small, distinctive static game-over label with two clicks and Enter.
   Exclude score numbers and animated backgrounds.
3. Reload Chrome yourself and leave the game on its initial start menu. Follow the
   next countdown, then click the START button position in the screenshot and Enter.
4. Select a small static start-menu-only label, such as PRESS TO PLAY.
5. Start playing and follow the next countdown. In the gameplay screenshot, select
   a static gameplay-only HUD icon such as PAUSE, excluding changing score digits.

Setup writes ignored, machine-specific `config/rl.local.json` and
`config/rl.local.game_over.png`, plus start/gameplay UI template PNGs.
Keep the Chrome window position, crop, zoom,
theme, and game UI unchanged. Use `--config config/rl.local.other.json` for a new
configuration. macOS uses the same commands with `.venv/bin/python`; screen
recording and accessibility permissions are required.

The generated reset sequence reloads the focused tab, waits five seconds, clicks
the configured start position, and waits half a second. Edit `reset_commands` if
the particular game host needs extra waits/clicks/keys (e.g. dismissal of a menu).
Supported commands:

```json
[
  {"type": "hotkey", "keys": ["ctrl", "r"]},
  {"type": "wait", "seconds": 5.0},
  {"type": "click", "x": 0.5, "y": 0.8},
  {"type": "press", "key": "space"},
  {"type": "wait", "seconds": 0.5}
]
```

Click coordinates are fractions of the gameplay crop, not desktop pixels. On
macOS the reload modifier is `command`. No OCR, website DOM, or host-specific
restart button is assumed. Revive/ad screens are host-specific: configure their
handling or use the final stable death UI as the template.

Reset waits for the calibrated start-menu marker before the first configured
click. It then waits for a persistent gameplay-HUD match with visible motion
before returning the initial episode observation. Animated ads alone cannot pass
this positive UI check. Ad/loading frames do not enter the perception engine,
replay buffer, survival reward, or episode timer. The default total reset timeout
is 90 seconds. Ads that require manual dismissal can still cause a timeout; the
agent does not guess ad close buttons or click ad content.

To upgrade an existing configuration without redoing its gameplay crop/death patch:

```powershell
.\.venv\Scripts\python.exe scripts/rl.py calibrate-ui
```

This asks for the initial start menu and an active gameplay screenshot, updates
the first start-click position and UI templates, and extends the reset timeout to
at least 90 seconds. Use the same `--config` argument as your training command.
Older configurations without these UI markers now require this calibration.

## Check, train, evaluate

```powershell
.\.venv\Scripts\python.exe scripts/rl.py check --episode-seconds 60
.\.venv\Scripts\python.exe scripts/rl.py train --steps 10000 --output models/rl/first_run
.\.venv\Scripts\python.exe scripts/rl.py evaluate --model models/rl/first_run/dqn_final.zip --episodes 5
```

Every command prints a countdown. Focus the intended Chrome game tab during it
and leave it visible. The check command plays **two complete no-op episodes**,
checking reset and game-over recognition; a timeout raises an error rather than
claiming detection worked. Watch this check before a longer training run. All live
modes are headless so a preview cannot steal keyboard focus. Switch away from
Chrome or press Ctrl+C to stop. PyAutoGUI's mouse-corner failsafe remains enabled.

`--action-seconds 0.12` sets the minimum action-to-capture delay; inference,
rendering in Chrome, and DQN updates add real time. `--episode-seconds 120` bounds
each episode. At the limit the environment returns `truncated=True`, then reloads
for the next episode. Game over returns `terminated=True`. The terminal patch
must match twice over at least 0.1 seconds. Missing player boxes never mean death.
After five seconds without visible motion, the watchdog waits up to eight more
seconds for the configured game-over patch. This allows a five-second revive/ad
offer to expire. No keys are sent during that grace period; no ad is accepted.
If game over is confirmed, the last visible-motion timestamp becomes an inferred
episode end, and earlier provisional reward for the stationary interval is
retracted. It is a visual timing heuristic, not a calibrated death timestamp.
If no death UI appears, the run still stops as frozen. Change
`terminal_grace_seconds` in the local config for a longer offer; existing configs
default to eight seconds. This is not a substitute for configuring the death patch.

Training uses SB3 `DQN("MlpPolicy")`, a 64/64 MLP, 20,000-transition replay buffer,
32-sample batches, learning after 500 transitions, one gradient update per four
steps, and a target-network update every 500 steps. Epsilon decreases from 1.0
to 0.05 over 30% of the requested run. Defaults use CPU for both the existing
detector and the small DQN. Useful flags include `--learning-starts`,
`--buffer-size`, `--checkpoint-every`, `--checkpoint` (YOLO weights), and `--seed`.

Resume weights, optimizer, counters and replay using:

```powershell
.\.venv\Scripts\python.exe scripts/rl.py train --model models/rl/first_run/dqn_final.zip --replay models/rl/first_run/dqn_final.replay.pkl --steps 10000
```

Resume starts a new browser episode, not the exact previous game state. Without
`--replay`, replay starts empty; prefer supplying the buffer for continued training.
Exploration decay follows SB3's combined timestep horizon for the resumed run.
Only load trusted local model/replay files (SB3 replay uses pickle).

## Observation and timing contract

RL schema v1 is **151 float32 values**, bounded to [-1, 1]. It is separate from
the observer's 141-value v3 lane-centric schema:

- 15 globals: player presence, normalized xyxy box, normalized screen-center
  velocities and their availability mask, observation age, sample interval, and
  previous action as five one-hot values.
- Eight obstacle slots with 17 values each: presence, observed/missed status,
  normalized xyxy, confidence, normalized bottom-edge approach velocity and its
  mask, time since seen, ambiguous association flag, and six-class one-hot label.

Slots prefer observed tracks, then largest screen-bottom coordinate, then track
ID. This is a screen-proximity ordering, not distance or TTC. Velocities use image
widths/heights per second and clip at +/-1. Ages and sample intervals clip at one
second. Missing data has zero values and explicit presence/velocity masks. Player
velocities require consecutive localized samples and reset each episode. Track
IDs are used only to break sorting ties; they are not numerical policy inputs.
The complete ordered schema is saved with every model and checked on load.

Actions are `0=no-op, 1=left, 2=right, 3=jump/up, 4=roll/down`. A tap is immediately
released. All five actions remain available even with unknown lanes/hazards.

Capture timestamps and action issue/completion times use `time.monotonic()`.
The backend captures a new frame only after the action completes plus the action
delay. There is no pre-action capture queue. Each transition records both action
timestamps, observation timestamp/age, sample interval, reward, and termination
reason. Observations older than 0.5s, source observations stale before a tap, or
transitions over two seconds abort instead of entering replay, except for a
confirmed terminal transition whose logged grace wait is excluded from that limit.
Exception handling
marks the environment as requiring reset and saves the completed replay data.

Reward is elapsed observed survival seconds minus **1.0** on game over. This
accounts for variable step duration; DQN still uses its usual fixed per-transition
discount (0.99), so this is an initial approximation to a real-time control task.
No reward depends on guessed lanes, action success, or uncalibrated TTC.

## Outputs and evaluation limits

Each run has a new directory under `models/rl/` (or a new `--output` directory):

- `dqn_<step>.zip`, `.replay.pkl`, `.schema.json`: periodic snapshots.
- `dqn_final.*`: final weights/replay/schema, also saved on interrupted learning.
- `progress.csv`: SB3 losses, exploration rate, episode rewards and lengths.
- `episodes.monitor.csv`: per-episode Monitor metrics.
- `transitions.jsonl`: timestamped actions, rewards and timing/lifecycle metrics.
- `training.json`, `config.json`, `observation_schema.json`: run metadata.

Periodic callbacks run before SB3 inserts the current transition, so a periodic
replay snapshot can be one transition behind its filename's step. Final snapshots
contain every transition successfully inserted by the learner.

Evaluation alternates the order of random and deterministic DQN episodes on the
same instance. It writes individual rows to `evaluation_episodes.jsonl` and means,
standard deviations, truncation counts, and the DQN-minus-random survival
difference to `evaluation.json`. Both policies use identical timing/reward limits.
Random actions are uniform over all five actions. Python seeds do not seed the
external Chrome game, so these are repeated live trials, not matched seeded game
states. An inaccurate death template can inflate survival for either policy.
Review the check run and truncation counts before interpreting improvements.

`SubwayEnv` depends on a small backend interface (`reset`, `act`, `observe`,
`close`), with no global perception state. Each backend owns its detector/capture,
engine, lifecycle detector and controls. The desktop implementation allows one
active backend per process and binds input to the selected Chrome window on
Windows (Chrome process on macOS). Separate game tabs do not count as isolated
instances. Future browser-isolated backends can implement the same interface and
be supplied to vectorized environments without changing observations or rewards.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rl -v
```

Tests check Gymnasium/SB3 compatibility, unknown-geometry feature preservation,
terminal recognition, focus and timestamp failures, and actual DQN gradients,
replay, checkpoint round-trip and random comparison against a deterministic
test fixture. These are infrastructure checks, not a game-playing benchmark.

API references: [SB3 custom environment contract](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
and [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html).
