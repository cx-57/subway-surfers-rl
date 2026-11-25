"""Observe gameplay; read-only unless --validate-controls explicitly enables test taps."""

from pathlib import Path
import sys
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.capture import LatestFrame
from agent.runtime import live_arguments, run
from agent.sources import ScreenSource, select_region
from scripts.gameplay import countdown


def main():
    args = live_arguments()
    if args.validate_controls:
        from agent.control_validation import enable_control_check

        enable_control_check(args)
    # Choose an explicit rectangle or ask for two corners on the screen.
    if args.region:
        left, top, width, height = args.region
        region = dict(left=left, top=top, width=width, height=height)
    else:
        print(
            f"Switch to Chrome; selecting gameplay area in {args.countdown:g} seconds.", flush=True
        )
        sleep(args.countdown)
        region = select_region(args.monitor)
    if region is None:
        print("Selection cancelled.")
        return
    print("Region selected. Return to Chrome; capture starts in:", flush=True)
    countdown(args.start_delay)
    print(
        "CAPTURING — screen capture is starting. Loading the detector; the live preview will follow.",
        flush=True,
    )
    source = ScreenSource(region, args.capture_fps)
    # Capture runs in the background; the engine always gets the newest frame.
    with LatestFrame(source) as mailbox:
        run(mailbox.frames(args.duration), args, "live", mailbox)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Observation failed: {error}")
