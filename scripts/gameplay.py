"""Select the game and save screenshots while you play. Run with F5."""

from contextlib import contextmanager, suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import cv2
import mss
import numpy as np

# Change these settings when needed.
FPS = 10
DURATION = 60
START_DELAY = 10
SELECTION_DELAY = 5
MONITOR = 1
OUTPUT = Path(__file__).resolve().parents[1] / "data/raw"


def countdown(seconds):
    for remaining in range(seconds, 0, -1):
        print(f"{remaining}...", flush=True)
        time.sleep(1)


def select_area(image):
    """Click two corners instead of holding down the mouse to drag."""
    title = "Click 2 corners | Enter: confirm | R: redo | Esc: cancel"
    corners = []
    cursor = [0, 0]

    def mouse(event, x, y, flags, data):
        cursor[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) == 2:
                corners.clear()
            corners.append((x, y))

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            preview = image.copy()
            if corners:
                end = corners[1] if len(corners) == 2 else tuple(cursor)
                cv2.rectangle(preview, corners[0], end, (0, 255, 0), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("c")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord("r"):
                corners.clear()
            if key in (13, 10, 32) and len(corners) == 2:
                left, right = sorted((corners[0][0], corners[1][0]))
                top, bottom = sorted((corners[0][1], corners[1][1]))
                if right > left and bottom > top:
                    return left, top, right - left, bottom - top
    finally:
        cv2.destroyAllWindows()


def save_image(filename, frame):
    """Publish a complete JPEG without overwriting an existing image."""
    temporary = filename.with_name("." + filename.name)
    try:
        if (
            not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            or temporary.stat().st_size == 0
        ):
            raise OSError(f"Could not save {filename}")
        # A hard link publishes the finished file and refuses to replace a frame.
        os.link(temporary, filename)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def save_metadata(folder, metadata):
    """Replace metadata only after the new JSON has been written completely."""
    temporary = folder / ".metadata.tmp"
    try:
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(folder / "metadata.json")
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


@contextmanager
def finish_frame_before_stopping():
    """Defer Ctrl+C until the saved frame has also been counted."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def main():
    folder = None
    metadata = None
    start = None
    count = 0
    try:
        with mss.MSS() as screen:
            monitor = screen.monitors[MONITOR]
            print("Bring Chrome forward. Taking the selection screenshot in:")
            countdown(SELECTION_DELAY)
            image = cv2.cvtColor(np.asarray(screen.grab(monitor)), cv2.COLOR_BGRA2BGR)
            height, width = image.shape[:2]
            scale = min(
                1,
                min(1200, monitor["width"] * 0.85) / width,
                min(750, monitor["height"] * 0.85) / height,
            )
            preview = cv2.resize(
                image, (max(1, round(width * scale)), max(1, round(height * scale)))
            )

            selection = select_area(preview)
            if selection is None or selection[2] == 0 or selection[3] == 0:
                print("Selection cancelled.")
                return
            x, y, w, h = selection

            # Convert preview pixels to MSS desktop coordinates (also on Retina).
            sx = monitor["width"] / preview.shape[1]
            sy = monitor["height"] / preview.shape[0]
            region = dict(
                left=monitor["left"] + round(x * sx),
                top=monitor["top"] + round(y * sy),
                width=max(1, round(w * sx)),
                height=max(1, round(h * sy)),
            )
            folder = OUTPUT / f"session_{time.time_ns()}"
            folder.mkdir(parents=True)
            metadata = dict(
                session_id=folder.name,
                start_timestamp=None,
                end_timestamp=None,
                requested_fps=FPS,
                requested_duration=DURATION,
                countdown=START_DELAY,
                monitor=MONITOR,
                region=region,
                jpeg_quality=95,
                frames_saved=0,
                frame_width=None,
                frame_height=None,
                actual_duration=0,
                actual_fps=0,
                status="countdown",
                reason=None,
            )
            save_metadata(folder, metadata)
            print(f"Saving to: {folder}")
            print("Switch back to the game. Recording starts in:")
            countdown(START_DELAY)
            print(f"Recording at up to {FPS} FPS. Press Ctrl+C in the terminal to stop early.")
            start = time.perf_counter()
            metadata.update(
                status="recording", start_timestamp=datetime.now(timezone.utc).isoformat()
            )
            save_metadata(folder, metadata)
            while time.perf_counter() - start < DURATION:
                before = time.perf_counter()
                frame = cv2.cvtColor(np.asarray(screen.grab(region)), cv2.COLOR_BGRA2BGR)
                filename = folder / f"frame_{count + 1:06d}.jpg"
                with finish_frame_before_stopping():
                    save_image(filename, frame)
                    count += 1
                    metadata.update(
                        frames_saved=count, frame_width=frame.shape[1], frame_height=frame.shape[0]
                    )
                # Account for capture time instead of adding it to every interval.
                time.sleep(
                    max(
                        0,
                        min(
                            1 / FPS - (time.perf_counter() - before),
                            DURATION - (time.perf_counter() - start),
                        ),
                    )
                )
            metadata["status"] = "completed"
    except KeyboardInterrupt:
        if metadata is not None:
            metadata.update(status="interrupted", reason="Ctrl+C")
        print("\nStopped.")
    except Exception as error:
        if metadata is not None:
            metadata.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        if metadata is not None:
            elapsed = time.perf_counter() - start if start is not None else 0
            metadata.update(
                end_timestamp=datetime.now(timezone.utc).isoformat(),
                actual_duration=elapsed,
                actual_fps=count / elapsed if elapsed else 0,
            )
            save_metadata(folder, metadata)
            print(f'Saved {count} images at {metadata["actual_fps"]:.1f} FPS to {folder}')


if __name__ == "__main__":
    main()
