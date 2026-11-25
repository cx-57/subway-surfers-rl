"""Optional small MP4 preview, resampled against actual frame timestamps."""

import json
import math
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, path, fps=15):
        self.path = Path(path)
        if self.path.exists() or self.path.with_suffix(".json").exists():
            raise FileExistsError(self.path)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be positive")
        self.fps = fps
        self.writer = None
        self.first = self.last = None
        self.previous = None
        self.count = 0
        self.input_shape = None
        self.closed = False

    def write(self, image, timestamp):
        if self.closed:
            raise RuntimeError("Video recorder is closed")
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or not image.size
            or image.dtype != np.uint8
        ):
            raise ValueError("Expected nonempty uint8 BGR video frame")
        if not math.isfinite(timestamp) or (self.last is not None and timestamp < self.last):
            raise ValueError("Video timestamps must be finite and nondecreasing")
        if self.input_shape is not None and image.shape[:2] != self.input_shape:
            raise ValueError("Capture dimensions changed during video recording")
        h, w = image.shape[:2]
        scale = min(1.0, 960 / w)
        frame = cv2.resize(image, (max(2, round(w * scale)), max(2, round(h * scale))))
        # MP4 codecs require even dimensions. Pad rather than stretch the image.
        frame = cv2.copyMakeBorder(
            frame, 0, frame.shape[0] % 2, 0, frame.shape[1] % 2, cv2.BORDER_CONSTANT
        )
        if self.writer is None:
            self.input_shape = (h, w)
            self.first = timestamp
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise OSError(f"Cannot open video encoder: {self.path}")
        target = int(math.floor((timestamp - self.first) * self.fps + 1e-7))
        # Hold the preceding frame across capture gaps; never speed up the footage.
        while self.count < target:
            self.writer.write(self.previous)
            self.count += 1
        if self.count == target:
            self.writer.write(frame)
            self.count += 1
        self.previous = frame
        self.last = timestamp

    def close(self):
        self.closed = True
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.path.with_suffix(".json").write_text(
                json.dumps(
                    dict(
                        fps=self.fps,
                        frames_written=self.count,
                        input_shape=self.input_shape,
                        source_duration_seconds=self.last - self.first,
                        playback_duration_seconds=self.count / self.fps,
                        note="Annotated preview, no audio. Width capped at 960 pixels; aspect preserved with even-dimension padding. Actual source timestamps drive constant-FPS resampling; gaps hold the preceding frame. Original screenshots are unchanged.",
                    ),
                    indent=2,
                )
                + "\n"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Video cleanup also failed: {cleanup_error}")
