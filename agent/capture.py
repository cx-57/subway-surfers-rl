"""Bounded latest-frame mailbox; source owns its capture resources."""

from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import monotonic
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    captured_at: float
    capture_ms: float = 0.0
    frame_id: str | int = 0
    timestamp_quality: str = "monotonic"


class LatestFrame:
    def __init__(self, source):
        self.source = source
        self.dropped = 0
        self.captured = 0
        self._slot = None
        self._error = None
        self._condition = Condition()
        self._stop = Event()
        self._thread = None

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Capture cannot be restarted")
        self.started_at = monotonic()
        self._thread = Thread(target=self._run, daemon=True, name="latest-frame")
        self._thread.start()
        return self

    def _run(self):
        try:
            while not self._stop.is_set():
                start = monotonic()
                value = self.source()
                end = monotonic()
                if value is None:
                    break
                frame = (
                    value
                    if isinstance(value, Frame)
                    else Frame(value, start, end, (end - start) * 1000, self.captured)
                )
                with self._condition:
                    self.captured += 1
                    self.dropped += self._slot is not None
                    self._slot = frame
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            # Capture handles such as MSS must close on their creating thread.
            try:
                if hasattr(self.source, "close"):
                    self.source.close()
            except Exception as exc:
                with self._condition:
                    if self._error is None:
                        self._error = exc
            self._stop.set()
            with self._condition:
                self._condition.notify_all()

    def _ready(self):
        # Wake the reader for a frame, an error, or the end of capture.
        return self._slot is not None or self._error is not None or self._stop.is_set()

    def get(self, timeout=None):
        with self._condition:
            ready = self._condition.wait_for(self._ready, timeout)
            if not ready:
                raise TimeoutError("Timed out waiting for capture")
            if self._error is not None:
                raise RuntimeError(f"Frame capture failed: {self._error}") from self._error
            if self._slot is None:
                raise EOFError("Capture stopped")
            frame, self._slot = self._slot, None
            return frame

    def frames(self, duration):
        """Yield the latest frames until the run ends or capture stops."""
        start = monotonic()
        while monotonic() - start < duration:
            try:
                yield self.get(timeout=2.0)
            except EOFError:
                return

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.alive:
            raise RuntimeError("Capture source did not stop within two seconds")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Capture cleanup also failed: {cleanup_error}")
