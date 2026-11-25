"""Shared observer: streaming logs, display, metrics and opt-in live control check."""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
from time import monotonic, sleep, time_ns

import cv2
from .pipeline import Engine, StateLog, write_schema
from .perception import Detector
from .visualization import render, toggle_key
from .video import VideoRecorder

ROOT = Path(__file__).resolve().parents[1]
WINDOW_TITLE = "Subway shadow observer | Q: stop | B/L/P/H/S: overlays"


def hardware_info():
    import torch
    import ultralytics

    info = dict(
        system=platform.platform(),
        processor=platform.processor(),
        python=platform.python_version(),
        torch=torch.__version__,
        ultralytics=ultralytics.__version__,
        cpu_count=os.cpu_count(),
    )
    if platform.system() == "Darwin":
        for key, command in [
            ("ram_bytes", "hw.memsize"),
            ("processor", "machdep.cpu.brand_string"),
        ]:
            value = subprocess.run(
                ["sysctl", "-n", command], capture_output=True, text=True, check=True
            ).stdout.strip()
            info[key] = int(value) if key == "ram_bytes" else value
    else:
        try:
            info["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            info["ram_bytes"] = None
    return info


class Metrics:
    """Keep totals and a small sample window instead of storing every frame."""

    def __init__(self):
        self.started = monotonic()
        self.count = 0
        self.total = {}
        self.samples = {}
        self.warm_started = None
        self.warm_total = {}

    def add(self, values):
        self.count += 1
        for key, value in values.items():
            if key.endswith("_ms") and isinstance(value, (int, float)):
                self.total[key] = self.total.get(key, 0.0) + value
                self.samples.setdefault(key, deque(maxlen=512)).append(value)
                if self.count > 3:
                    self.warm_total[key] = self.warm_total.get(key, 0.0) + value
        if self.count == 3:
            self.warm_started = monotonic()

    def summary(self):
        elapsed = monotonic() - self.started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        means = {}
        medians = {}
        warm_means = {}
        for name, total in self.total.items():
            means[name] = total / self.count
        for name, samples in self.samples.items():
            medians[name] = statistics.median(samples)
        for name, total in self.warm_total.items():
            warm_means[name] = total / (self.count - 3)
        return dict(
            processed_frames=self.count,
            elapsed_seconds=elapsed,
            processed_fps=self.count / elapsed if elapsed else 0.0,
            peak_process_rss_bytes=rss if platform.system() == "Darwin" else rss * 1024,
            mean_ms=means,
            median_last_512_ms=medians,
            steady_frames=max(0, self.count - 3),
            steady_processed_fps=(
                (self.count - 3) / (monotonic() - self.warm_started)
                if self.warm_started and self.count > 3
                else None
            ),
            steady_mean_ms=warm_means,
            note="Processed FPS includes reading/capture waits, state estimation, rendering and logging; model loading is reported separately. Peak RSS includes Python, model and allocator retention.",
        )


def add_common_options(parser):
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/barriers_640/weights/best.pt"
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--settings", type=Path, help="Optional provisional engine/timing settings JSON"
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--video",
        action="store_true",
        help="Save annotated observation.mp4 in the run folder (adds encoding work)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--preview-every",
        type=int,
        default=0,
        help="Save separate overlay PNG every N processed frames",
    )


def live_arguments():
    """Parse live options here so the entrypoint only contains capture steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--capture-fps", type=float, default=60.0)
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Bounded session duration in seconds"
    )
    parser.add_argument(
        "--countdown", type=float, default=5.0, help="Time to switch to Chrome before selection"
    )
    parser.add_argument(
        "--start-delay",
        type=int,
        default=15,
        help="Countdown after selection, before capture starts",
    )
    parser.add_argument(
        "--validate-controls",
        action="store_true",
        help="Opt-in Chrome arrow-key/lane check; start in center lane and keep Chrome focused",
    )
    add_common_options(parser)
    parser.set_defaults(video=True)
    parser.add_argument(
        "--no-video",
        dest="video",
        action="store_false",
        help="Disable the default annotated MP4 recording",
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Duration must be positive and finite")
    if not math.isfinite(args.countdown) or args.countdown < 0 or args.start_delay < 0:
        parser.error("Countdowns must be finite and nonnegative")
    return args


def run(frames, args, mode, mailbox=None):
    control_check = getattr(args, "validate_controls", False)
    if control_check and mode != "live":
        raise ValueError("Keyboard validation is only supported in live mode")
    if args.preview_every < 0:
        raise ValueError("Preview interval cannot be negative")
    output = args.output or ROOT / f"models/evaluation/engine_{mode}_{time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    settings = json.loads(args.settings.read_text()) if args.settings else None
    started = monotonic()
    detector = Detector(args.checkpoint, device=args.device)
    model_load_ms = (monotonic() - started) * 1000
    engine = Engine(detector, calibration, settings)
    validator = None
    if control_check:
        from .control_validation import ControlValidation

        validator = ControlValidation(output)
    write_schema(output / "observation_schema.json")
    metrics = Metrics()
    toggles = {}
    status, error = "completed", None
    wall_start = monotonic()
    first_timestamp = None
    stale_dropped = 0
    video = VideoRecorder(output / "observation.mp4") if getattr(args, "video", False) else None
    print(f"{mode}: {detector.device}, logs -> {output}", flush=True)
    if video is not None:
        print(f'VIDEO ON — saving {output / "observation.mp4"}', flush=True)
    try:
        # Every frame follows the same path in replay and live observation.
        with StateLog(output / "states.jsonl") as log:
            for frame in frames:
                before = monotonic()
                if (
                    mode == "live"
                    and before - frame.captured_at + frame.capture_ms / 1000 > engine.max_age
                ):
                    # Skip old captures instead of showing an ever-growing delay.
                    stale_dropped += 1
                    continue
                if first_timestamp is None:
                    first_timestamp = frame.timestamp
                speed = getattr(args, "speed", 0.0)
                if mode == "replay" and speed > 0:
                    waiting = monotonic()
                    sleep(
                        max(
                            0.0,
                            wall_start + (frame.timestamp - first_timestamp) / speed - monotonic(),
                        )
                    )
                    # A replay delivery wait is not historical camera/compute latency.
                    frame.captured_at += monotonic() - waiting
                    before = monotonic()
                state = engine.process(frame)
                if mode == "live" and metrics.count == 0:
                    print(
                        f"OBSERVING — first frame processed ({frame.image.shape[1]}×{frame.image.shape[0]}). "
                        "The preview timestamps and terminal frame count show progress.",
                        flush=True,
                    )
                state.timings["processed_fps"] = metrics.count / max(
                    1e-9, monotonic() - metrics.started
                )
                if mailbox is not None:
                    state.timings["capture_fps"] = mailbox.captured / max(
                        1e-9, monotonic() - mailbox.started_at
                    )
                show = not args.headless
                save = args.preview_every and metrics.count % args.preview_every == 0
                if show or save or video is not None:
                    # Render on a copy; source screenshots are never annotated in place.
                    drawn = render(frame.image, state, engine.geometry, toggles)
                    if video is not None:
                        video_start = monotonic()
                        video.write(drawn, frame.timestamp)
                        state.timings["video_ms"] = (monotonic() - video_start) * 1000
                    if save:
                        destination = (
                            output
                            / f"overlay_{metrics.count:06d}_{Path(str(frame.frame_id)).stem}.png"
                        )
                        if not cv2.imwrite(str(destination), drawn):
                            raise OSError(f"Cannot save overlay: {destination}")
                    if show:
                        cv2.imshow(WINDOW_TITLE, drawn)
                        key = cv2.waitKey(1) & 0xFF
                        try:
                            closed = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
                        except cv2.error:
                            closed = True
                        if key in (27, ord("q")) or closed:
                            status = "stopped"
                            break
                        toggle_key(toggles, key)
                state.timings["display_age_ms"] = (
                    monotonic() - frame.captured_at
                ) * 1000 + frame.capture_ms
                log.write(state, frame.frame_id)
                if validator is not None:
                    validator.update(state, frame.frame_id)
                metrics.add(dict(state.timings, loop_ms=(monotonic() - before) * 1000))
                if metrics.count % 25 == 0:
                    print(
                        f"{metrics.count} frames | {metrics.count/(monotonic()-metrics.started):.1f} FPS | "
                        f"{state.recommendation.action}: {state.recommendation.reason}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        primary_failed = error is not None
        cleanup_error = None
        try:
            if video is not None:
                video.close()
        except Exception as exc:
            cleanup_error = exc
        try:
            if not args.headless:
                cv2.destroyAllWindows()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None and error is None:
            status, error = "failed", f"{type(cleanup_error).__name__}: {cleanup_error}"
        report = dict(
            status=status,
            error=error,
            mode=mode,
            hardware=hardware_info(),
            device=detector.device,
            fallback_reason=detector.fallback_reason,
            model_load_ms=model_load_ms,
            timings=metrics.summary(),
            tracker=engine.tracker.metrics,
            stale_frames_skipped=stale_dropped,
            captured_frames=mailbox.captured if mailbox else None,
            capture_fps=(
                mailbox.captured / max(1e-9, monotonic() - mailbox.started_at) if mailbox else None
            ),
            frame_drops=mailbox.dropped if mailbox else 0,
            video=(
                str(video.path)
                if video is not None and video.count and video.path.is_file()
                else None
            ),
            cleanup_error=str(cleanup_error) if cleanup_error is not None else None,
            control_validation=bool(validator),
            caveat=(
                "Opt-in scheduled keyboard validation; see controls.jsonl. "
                if validator
                else "No keyboard input. "
            )
            + "Replay timestamp quality is logged per row. Unknown player/geometry/TTC are not measured accuracy. Detector cold-start is included in first-frame loop timing.",
        )
        (output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if cleanup_error is not None and not primary_failed:
            raise cleanup_error
    return output
