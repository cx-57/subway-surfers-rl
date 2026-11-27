"""One focused Chrome window, direct fresh capture, configurable UI reset sequence."""

import ctypes
from ctypes import wintypes
import json
import math
from pathlib import Path
import platform
import subprocess
from threading import Lock
from time import monotonic, sleep
import cv2
import numpy as np
from agent.pipeline import Engine
from agent.perception import Detector
from agent.sources import ScreenSource
from .env import Sample
from .lifecycle import GameOverDetector

DESKTOP_LOCK = Lock()
KEYS = (None, "left", "right", "up", "down")


class ChromeController:
    """Check focus before every event; never activate or switch windows automatically."""

    def __init__(self):
        import pyautogui

        self.keyboard = pyautogui
        self.keyboard.PAUSE = 0.02
        self.keyboard.FAILSAFE = True
        self.target = self.foreground()

    def foreground(self):
        if platform.system() == "Windows":
            import psutil

            user32 = ctypes.windll.user32
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            window = user32.GetForegroundWindow()
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
            if not pid.value or psutil.Process(pid.value).name().lower() != "chrome.exe":
                raise RuntimeError("Chrome must be focused; keyboard control stopped")
            return int(window)
        if platform.system() == "Darwin":
            script = ('tell application "System Events" to get {name, unix id} of '
                      'first application process whose frontmost is true')
            result = subprocess.run(["osascript", "-e", script], capture_output=True,
                                    text=True, check=True, timeout=3).stdout.strip()
            if not result.startswith("Google Chrome,"):
                raise RuntimeError("Chrome must be focused; keyboard control stopped")
            return result
        raise RuntimeError("Live RL controls currently support Windows and macOS")

    def check(self):
        if self.foreground() != self.target:
            raise RuntimeError("The selected Chrome instance lost focus; control stopped")

    def wait(self, seconds):
        end = monotonic() + seconds
        while monotonic() < end:
            self.check()
            sleep(min(0.05, max(0.0, end - monotonic())))

    def act(self, action):
        self.check()
        issued = monotonic()
        if KEYS[action] is not None:
            self.keyboard.press(KEYS[action])
        return issued, monotonic()

    def reset_game(self, commands, region):
        for command in commands:
            self.check()
            if command["type"] == "wait":
                self.wait(command["seconds"])
            elif command["type"] == "press":
                self.keyboard.press(command["key"])
            elif command["type"] == "hotkey":
                self.keyboard.hotkey(*command["keys"])
            elif command["type"] == "click":
                # Click locations are normalized to the configured gameplay crop.
                x = region["left"] + round(command["x"] * region["width"])
                y = region["top"] + round(command["y"] * region["height"])
                self.keyboard.click(x, y)

    def close(self):
        # press()/hotkey() release immediately; no held keys survive a transition.
        pass


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    region = config["region"]
    for name in ("left", "top", "width", "height"):
        if type(region[name]) is not int:
            raise ValueError("Capture region must use integer screen coordinates")
    if min(region["width"], region["height"]) <= 0:
        raise ValueError("Capture dimensions must be positive")
    commands = config["reset_commands"]
    if not isinstance(commands, list) or not 1 <= len(commands) <= 30:
        raise ValueError("Provide 1 to 30 reset commands")
    for command in commands:
        kind = command["type"]
        if kind == "wait":
            if not math.isfinite(command["seconds"]) or not 0 < command["seconds"] <= 60:
                raise ValueError("Reset waits must be in (0, 60] seconds")
        elif kind == "click":
            if not 0 <= command["x"] <= 1 or not 0 <= command["y"] <= 1:
                raise ValueError("Reset clicks must be crop-relative coordinates in [0, 1]")
        elif kind == "press":
            if not isinstance(command["key"], str) or not command["key"]:
                raise ValueError("Reset key must be nonempty")
        elif kind == "hotkey":
            if (not isinstance(command["keys"], list) or not command["keys"]
                    or not all(isinstance(key, str) and key for key in command["keys"])):
                raise ValueError("Reset hotkey requires a list of keys")
        else:
            raise ValueError(f"Unknown reset command: {kind}")
    config["template_path"] = str(path.parent / config["game_over"]["template"])
    for name in ("start_ui", "gameplay_ui"):
        if name not in config:
            raise ValueError("UI calibration required: run scripts/rl.py calibrate-ui --config " + str(path))
        config[name]["template_path"] = str(path.parent / config[name]["template"])
    for name, default in (("reset_timeout", 15.0), ("freeze_seconds", 5.0),
                          ("motion_threshold", 0.005), ("terminal_grace_seconds", 8.0)):
        value = config.get(name, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    return config


class ChromeBackend:
    def __init__(self, config, checkpoint="models/training/weights/best.pt"):
        self.config = config
        terminal = config["game_over"]
        template = cv2.imread(config["template_path"])
        if template is None:
            raise ValueError("Cannot read configured game-over template")
        self.terminal = GameOverDetector(template, terminal["roi"],
                                         terminal.get("threshold", 0.94))
        self.start_ui = self._ui_detector(config["start_ui"])
        self.gameplay_ui = self._ui_detector(config["gameplay_ui"])
        self.detector = Detector(checkpoint)
        # Warm inference before the user focuses Chrome and before reset starts.
        self.detector.detect(np.zeros((360, 640, 3), np.uint8), 0.0)
        self.source = ScreenSource(config["region"], fps=60)
        self.controller = None
        self.locked = False
        self.closed = False
        self.engine = None
        self.last_image = None
        self.last_motion = None
        self.last_state_timestamp = None

    @staticmethod
    def _ui_detector(spec):
        template = cv2.imread(spec["template_path"])
        if template is None:
            raise ValueError("Cannot read calibrated UI template: " + spec["template_path"])
        return GameOverDetector(template, spec["roi"], spec.get("threshold", 0.94))

    def _open(self):
        if self.closed:
            raise RuntimeError("Backend is closed")
        if not self.locked:
            if not DESKTOP_LOCK.acquire(blocking=False):
                raise RuntimeError("Only one desktop Chrome backend may run at once")
            self.locked = True
            try:
                self.controller = ChromeController()
            except Exception:
                self.close()
                raise

    def reset(self):
        self._open()
        deadline = monotonic() + self.config.get("reset_timeout", 90.0)
        guarded_start = False
        for command in self.config["reset_commands"]:
            # Reload may show an interstitial before the configured start click.
            # Never deliver that click until the actual start menu is recognized.
            if command["type"] == "click" and not guarded_start:
                self._wait_for_ui(self.start_ui, "start menu", deadline)
                guarded_start = True
            self.controller.reset_game([command], self.config["region"])
        frame = self._wait_for_ui(self.gameplay_ui, "gameplay HUD", deadline, require_motion=True)
        # A new engine clears tracks, player flow, and all episode histories.
        # Ad frames were used only for UI matching, never perception or replay.
        self.engine = Engine(detector=self.detector,
                             settings=dict(automatic_geometry=False, distance_filter=False))
        self.terminal.reset()
        self.last_image = self._small_image(frame.image)
        self.last_motion = frame.timestamp
        state = self.engine.process(frame)
        self.controller.check()
        terminal, score = self.terminal.update(frame.image, frame.timestamp)
        if score >= self.terminal.threshold:
            raise RuntimeError("Gameplay HUD marker also matches the game-over UI; recalibrate-ui")
        self.last_state_timestamp = state.timestamp
        return Sample(state, terminal, score)

    @staticmethod
    def _small_image(image):
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (80, 45))

    def _wait_for_ui(self, detector, label, deadline, require_motion=False):
        print(f"RESET: waiting for {label}; ads/loading are excluded from the episode.", flush=True)
        detector.reset()
        previous = None
        while monotonic() < deadline:
            self.controller.check()
            frame = self.source()
            matched, _ = detector.update(frame.image, frame.timestamp)
            small = self._small_image(frame.image)
            moving = False
            if previous is not None:
                delta = np.abs(small.astype(np.float32) - previous.astype(np.float32)).mean() / 255
                moving = delta > self.config.get("motion_threshold", 0.005)
            previous = small
            if matched and (moving or not require_motion):
                return frame
            self.controller.wait(0.1)
        raise RuntimeError(f"Reset timed out waiting for {label}; ad may need dismissal or UI calibration")

    def act(self, action):
        self.controller.check()
        if monotonic() - self.last_state_timestamp > 0.5:
            raise RuntimeError("Source observation is stale before action; control stopped")
        return self.controller.act(action)

    def _sample(self):
        self.controller.check()
        frame = self.source()
        terminal, score = self.terminal.update(frame.image, frame.timestamp)
        small = self._small_image(frame.image)
        moving = False
        if self.last_image is not None:
            delta = np.abs(small.astype(np.float32) - self.last_image.astype(np.float32)).mean() / 255
            moving = bool(delta > self.config.get("motion_threshold", 0.005))
        if moving:
            self.last_motion = frame.timestamp
        self.last_image = small
        state = self.engine.process(frame)
        self.controller.check()
        return Sample(state, terminal, score), moving

    def observe(self, after):
        self.controller.wait(max(0.0, after - monotonic()))
        sample, _ = self._sample()
        if sample.state.timestamp < after:
            raise RuntimeError("Capture timestamp precedes action boundary")
        if not sample.terminal and sample.state.timestamp - self.last_motion > self.config.get("freeze_seconds", 5.0):
            # Revive/ad offers may hold the scene before the final death UI appears.
            # Send no keys while waiting, and require the configured death evidence.
            wait_started = monotonic()
            last_motion = self.last_motion
            deadline = wait_started + self.config.get("terminal_grace_seconds", 8.0)
            print("Gameplay stopped moving; waiting for the revive offer/game-over screen.", flush=True)
            while monotonic() < deadline:
                self.controller.wait(0.1)
                sample, _ = self._sample()
                if sample.terminal:
                    sample.terminal_at = last_motion
                    sample.observation_wait_seconds = max(0.0, sample.state.timestamp - wait_started)
                    break
            else:
                raise RuntimeError("Gameplay appears frozen; no game-over detected after the revive grace period")
        self.last_state_timestamp = sample.state.timestamp
        return sample

    def close(self):
        self.closed = True
        try:
            self.source.close()
            if self.controller is not None:
                self.controller.close()
        finally:
            if self.locked:
                DESKTOP_LOCK.release()
                self.locked = False
