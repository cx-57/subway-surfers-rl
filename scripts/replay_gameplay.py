"""Replay originals through the same engine as live observation. Never presses keys."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.runtime import add_common_options, run
from agent.sources import ReplaySource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--fps", type=float, help="Explicit approximate source FPS if metadata is absent"
    )
    parser.add_argument(
        "--timestamps", type=Path, help="JSON mapping frame filenames to measured seconds"
    )
    parser.add_argument(
        "--speed", type=float, default=0.0, help="0: process freely; 1: source-time pacing"
    )
    add_common_options(parser)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed < 0:
        parser.error("Speed cannot be negative")
    timestamps = json.loads(args.timestamps.read_text()) if args.timestamps else None
    frames = ReplaySource(args.recording, args.start, args.limit, args.step, args.fps, timestamps)
    run(frames, args, "replay")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Replay failed: {error}")
