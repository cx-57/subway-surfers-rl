"""Original-file replay and thread-owned screen capture; neither writes images."""

import json
import math
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np
from .capture import Frame


def frame_number(path):
    """Read the number in frame_000123.jpg; reject other filename formats."""
    number = path.name.removeprefix("frame_").removesuffix(".jpg")
    if not path.name.startswith("frame_") or path.suffix != ".jpg" or not number.isdecimal():
        raise ValueError("Replay filenames must follow frame_NUMBER.jpg")
    return int(number)


class ReplaySource:
    def __init__(self, folder, start=1, limit=None, step=1, fps=None, timestamps=None):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Recording folder not found: {folder}")
        if start < 1 or step < 1 or (limit is not None and limit < 1):
            raise ValueError("Replay start, step and limit must be positive")
        paths = list(self.folder.glob("frame_*.jpg"))
        paths.sort(key=frame_number)
        self.paths = []
        for path in paths:
            if frame_number(path) >= start:
                self.paths.append(path)
        self.paths = self.paths[::step]
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError("No replay frames selected")
        metadata = self.folder / "metadata.json"
        self.metadata = json.loads(metadata.read_text()) if metadata.exists() else {}
        self.fps = fps if fps is not None else self.metadata.get("actual_fps")
        self.timestamps = timestamps or self.metadata.get("frame_timestamps", {})

    def __iter__(self):
        previous = -math.inf
        for path in self.paths:
            start = monotonic()
            if not path.is_file():
                raise FileNotFoundError(f"Missing replay frame: {path}")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode replay frame: {path}")
            number = frame_number(path)
            if path.name in self.timestamps:
                timestamp = float(self.timestamps[path.name])
                quality = "exact"
            else:
                if self.fps is None or not math.isfinite(float(self.fps)) or float(self.fps) <= 0:
                    raise ValueError(
                        "No frame timestamps or actual recording FPS; supply --fps explicitly"
                    )
                timestamp = (number - 1) / float(self.fps)
                quality = "approximate"
            if not math.isfinite(timestamp) or timestamp <= previous:
                raise ValueError("Replay timestamps must be finite and strictly increasing")
            previous = timestamp
            end = monotonic()
            yield Frame(image, timestamp, end, (end - start) * 1000, str(path), quality)


def select_region(monitor_number=1):
    """Reuse the recorder's two-click selector and Retina coordinate conversion."""
    import mss
    from scripts.gameplay import select_area

    with mss.MSS() as screen:
        if monitor_number < 1 or monitor_number >= len(screen.monitors):
            raise ValueError("Monitor number is unavailable")
        monitor = screen.monitors[monitor_number]
        image = cv2.cvtColor(np.asarray(screen.grab(monitor)), cv2.COLOR_BGRA2BGR)
        height, width = image.shape[:2]
        scale = min(1.0, 1200 / width, 750 / height)
        preview = cv2.resize(image, (round(width * scale), round(height * scale)))
        selection = select_area(preview)
        if selection is None:
            return None
        x, y, w, h = selection
        sx, sy = monitor["width"] / preview.shape[1], monitor["height"] / preview.shape[0]
        return dict(
            left=monitor["left"] + round(x * sx),
            top=monitor["top"] + round(y * sy),
            width=max(1, round(w * sx)),
            height=max(1, round(h * sy)),
        )


class ScreenSource:
    """Created on the main thread, but MSS is opened and closed on its worker."""

    def __init__(self, region, fps=60.0):
        if fps <= 0 or not math.isfinite(fps):
            raise ValueError("Capture FPS must be positive")
        self.region = dict(region)
        if self.region.get("width", 0) <= 0 or self.region.get("height", 0) <= 0:
            raise ValueError("Capture region must have positive dimensions")
        self.interval = 1 / fps
        self.screen = None
        self.number = 0

    def __call__(self):
        import mss

        if self.screen is None:
            self.screen = mss.MSS()
        start = monotonic()
        try:
            image = cv2.cvtColor(np.asarray(self.screen.grab(self.region)), cv2.COLOR_BGRA2BGR)
            if not image.size or not np.any(image):
                raise RuntimeError(
                    "Blank screen capture; check Screen Recording permission and region"
                )
        except Exception as exc:
            raise RuntimeError(
                f"Screen capture failed; check macOS Screen Recording permission: {exc}"
            ) from exc
        end = monotonic()
        self.number += 1
        frame = Frame(
            image,
            start,
            end,
            (end - start) * 1000,
            f"live/frame_{self.number:06d}.jpg",
            "monotonic",
        )
        sleep(max(0.0, self.interval - (monotonic() - start)))
        return frame

    def close(self):
        if self.screen is not None:
            self.screen.close()
