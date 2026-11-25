"""Opt-in, bounded Chrome arrow-key check. Commands are never lane ground truth."""

import json
import subprocess
from time import monotonic

SCHEDULE = (
    (5.0, "LEFT", 0),
    (8.0, "RIGHT", 1),
    (11.0, "RIGHT", 2),
    (14.0, "LEFT", 1),
    (17.0, "JUMP", 1),
    (20.0, "ROLL", 1),
)
KEY_CODES = {"LEFT": 123, "RIGHT": 124, "JUMP": 126, "ROLL": 125}


def enable_control_check(args):
    """Only this explicit mode requests keyboard access; normal runs stay read-only."""
    check_access()
    args.headless = True  # Opening a preview could take focus away from Chrome.
    args.video = True
    print(
        "CONTROL CHECK ON: start in CENTER lane. Do not steer manually. "
        "Six arrow taps begin 5 seconds after the first observed frame. "
        "Switch away from Chrome to stop further taps.",
        flush=True,
    )


def check_access():
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to return UI elements enabled'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode or result.stdout.strip() != "true":
        raise RuntimeError(
            "Control check needs Accessibility permission for Terminal/VS Code "
            "and permission to automate System Events. " + result.stderr.strip()
        )


def send_arrow(action):
    # Never activate Chrome: switching to another app stops the check.
    script = f"""tell application "System Events"
        if name of first application process whose frontmost is true is not "Google Chrome" then
            error "Chrome is not focused; control check stopped"
        end if
        key code {KEY_CODES[action]}
    end tell"""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())


class ControlValidation:
    def __init__(self, output, sender=None):
        self.path = output / "controls.jsonl"
        self.sender = sender or send_arrow
        self.start = None
        self.index = 0
        self.pending = None
        self.aborted = False

    def _write(self, event):
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def update(self, state, frame_id):
        if self.start is None:
            self.start = state.timestamp
            self._write(
                dict(
                    event="start",
                    timestamp=self.start,
                    initial_lane_assumption=1,
                    ground_truth=False,
                )
            )
        elapsed = state.timestamp - self.start
        if self.pending and state.timestamp >= self.pending["sample_after"]:
            self._write(
                dict(
                    event="observation",
                    timestamp=state.timestamp,
                    frame_id=str(frame_id),
                    action=self.pending["action"],
                    command_expected_lane=self.pending["lane"],
                    lane_candidates=list(state.player.lane_candidates),
                    confirmed_lane=state.player.lane,
                    lane_reason=state.player.lane_reason,
                    player_bbox=state.player.bbox,
                    ground_truth=False,
                )
            )
            self.pending = None
        if self.aborted or self.index == len(SCHEDULE):
            return
        due, action, lane = SCHEDULE[self.index]
        if elapsed < due:
            return
        self.index += 1
        error = None
        sent = False
        try:
            if elapsed - due > 1.0:
                raise RuntimeError("Missed command deadline; check stopped without catching up")
            self.sender(action)
            sent = True
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
            self.aborted = True
        self._write(
            dict(
                event="command",
                timestamp=state.timestamp,
                issued_at=monotonic(),
                action=action,
                sent=sent,
                error=error,
                command_expected_lane=lane,
                ground_truth=False,
            )
        )
        print(f'CONTROL CHECK: {action}: {"sent" if sent else error}', flush=True)
        if sent:
            # Wait for a later captured frame, not the pre-command observation.
            self.pending = dict(
                action=action, lane=lane, sample_after=state.timestamp + state.observation_age + 0.6
            )
