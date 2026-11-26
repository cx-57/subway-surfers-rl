"""Single command entrypoint for the Subway Surfers observation project.

Examples:
    .venv/bin/python run.py
    .venv/bin/python run.py record
    .venv/bin/python run.py calibrate data/raw/session_<ID> --every 12
    .venv/bin/python run.py observe --duration 60
    .venv/bin/python run.py controls
    .venv/bin/python run.py test
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "observe": ["scripts/observe_live.py"],
    "record": ["scripts/gameplay.py"],
    "calibrate": ["scripts/calibrate_player.py"],
    "replay": ["scripts/replay_gameplay.py"],
    "controls": ["scripts/observe_live.py", "--validate-controls"],
    "evaluate": ["scripts/evaluate.py"],
    "train": ["scripts/train.py"],
    "test": ["-m", "unittest", "discover", "-v"],
    "player-eval": ["scripts/evaluate_player_session.py"],
}

USAGE = """Usage: python run.py [command] [options]

  (none)       Observe live gameplay (default)
  observe      Observe live gameplay and save annotated video
  record       Capture original gameplay frames
  calibrate    Label player lane/pose in an existing recording
  replay       Run the observer on existing recording frames
  controls     Opt-in six predefined keyboard control checks
  evaluate     Evaluate a detector checkpoint
  train        Train a detector
  test         Run the Python unit tests
  player-eval  Evaluate reviewed draft player calibration on recorded gameplay

Examples:
  python run.py
  python run.py observe --duration 60
  python run.py calibrate data/raw/session_<ID> --every 12
  python run.py replay data/raw/session_<ID> --video
"""


def build_command(args):
    """Return a child command or raise for unsupported workflows."""
    args = list(args)
    if not args:
        args = ["observe"]
    if args[0] in ("help", "-h", "--help"):
        return None
    if args[0].startswith("-"):
        args.insert(0, "observe")
    name = args[0]
    if name not in COMMANDS:
        raise ValueError(f"Unknown command: {name}")
    return [sys.executable, "-u", *COMMANDS[name], *args[1:]]


def main(args=None):
    args = sys.argv[1:] if args is None else args
    try:
        command = build_command(args)
    except ValueError as error:
        print(error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if command is None:
        print(USAGE)
        return 0
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    except OSError as error:
        print(f"Could not start command: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
