"""Runner logging, GUI controls and error shutdown using controlled input frames."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from agent.capture import Frame
from agent.runtime import Metrics, live_arguments, run


class EmptyDetector:
    last_timings = {"preprocess_ms": 1.0, "inference_ms": 2.0, "postprocess_ms": 1.0}
    device = "cpu"
    fallback_reason = None

    def detect(self, image, timestamp):
        return []


class RuntimeTests(unittest.TestCase):
    def test_live_defaults_keep_video_on_and_controls_off(self):
        with patch("sys.argv", ["observe_live.py"]):
            args = live_arguments()
        self.assertTrue(args.video)
        self.assertFalse(args.validate_controls)
        self.assertEqual(args.start_delay, 15)

    def test_invalid_live_duration_fails_before_requesting_keyboard_access(self):
        with patch(
            "sys.argv", ["observe_live.py", "--duration", "nan", "--validate-controls"]
        ), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                live_arguments()
        self.assertEqual(error.exception.code, 2)

    def test_opt_in_live_check_logs_commands_and_lane_observations(self):
        def frames():
            for timestamp in (0.0, 5.0, 5.8):
                yield Frame(
                    np.zeros((60, 100, 3), np.uint8),
                    timestamp,
                    monotonic(),
                    frame_id=str(timestamp),
                )

        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), patch("agent.control_validation.send_arrow"), redirect_stdout(io.StringIO()):
            args = self.args(folder)
            args.validate_controls = True
            output = run(frames(), args, "live")
            events = []
            for x in (output / "controls.jsonl").read_text().splitlines():
                events.append(json.loads(x))
            event_types = []
            for event in events:
                event_types.append(event["event"])
            self.assertEqual(event_types, ["start", "command", "observation"])
            self.assertEqual(events[1]["action"], "LEFT")
            self.assertTrue(events[1]["sent"])
            self.assertFalse(events[2]["ground_truth"])
            self.assertTrue(
                json.loads((output / "benchmark.json").read_text())["control_validation"]
            )

    def test_closing_window_stops_without_a_key(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), patch("agent.runtime.cv2.imshow"), patch(
            "agent.runtime.cv2.waitKey", return_value=-1
        ), patch(
            "agent.runtime.cv2.getWindowProperty", return_value=0
        ), patch(
            "agent.runtime.cv2.destroyAllWindows"
        ), redirect_stdout(
            io.StringIO()
        ):
            output = run(self.frames(), self.args(folder, False), "replay")
            self.assertEqual(
                json.loads((output / "benchmark.json").read_text())["status"], "stopped"
            )

    def test_replay_pacing_does_not_make_fresh_observations_stale(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), redirect_stdout(io.StringIO()):
            args = self.args(folder)
            args.speed = 2.0
            output = run(self.frames(), args, "replay")
            rows = []
            for x in (output / "states.jsonl").read_text().splitlines():
                rows.append(json.loads(x))
            self.assertLess(rows[-1]["state"]["observation_age"], 0.1)

    def args(self, folder, headless=True):
        return SimpleNamespace(
            output=Path(folder) / "result",
            calibration=None,
            settings=None,
            checkpoint="unused",
            device="cpu",
            headless=headless,
            preview_every=0,
            speed=0.0,
        )

    def frames(self):
        for n in range(2):
            yield Frame(np.zeros((60, 100, 3), np.uint8), float(n), monotonic(), frame_id=str(n))

    def test_headless_logs_real_states_and_writes_benchmark(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), redirect_stdout(io.StringIO()):
            output = run(self.frames(), self.args(folder), "replay")
            rows = []
            for line in (output / "states.jsonl").read_text().splitlines():
                rows.append(json.loads(line))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["state"]["recommendation"]["action"], "UNKNOWN")
            benchmark = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(benchmark["timings"]["processed_frames"], 2)
            self.assertEqual(benchmark["status"], "completed")

    def test_gui_toggle_then_quit_closes_display_and_keeps_partial_log(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), patch("agent.runtime.cv2.imshow"), patch(
            "agent.runtime.cv2.waitKey", side_effect=[ord("b"), ord("q")]
        ), patch(
            "agent.runtime.cv2.getWindowProperty", return_value=1
        ), patch(
            "agent.runtime.cv2.destroyAllWindows"
        ), redirect_stdout(
            io.StringIO()
        ):
            output = run(self.frames(), self.args(folder, False), "replay")
            benchmark = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(benchmark["status"], "stopped")
            self.assertEqual(len((output / "states.jsonl").read_text().splitlines()), 1)

    def test_source_failure_records_failed_status(self):
        def bad_frames():
            yield next(self.frames())
            raise OSError("Recording unavailable")

        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "Recording unavailable"):
                run(bad_frames(), self.args(folder), "replay")
            result = json.loads((Path(folder) / "result/benchmark.json").read_text())
            self.assertEqual(result["status"], "failed")

    def test_metrics_separate_fps_from_latency_and_exclude_initial_three(self):
        m = Metrics()
        for value in [100.0, 100.0, 100.0, 10.0, 10.0]:
            m.add(dict(inference_ms=value, processed_fps=25.0))
        report = m.summary()
        self.assertNotIn("processed_fps", report["mean_ms"])
        self.assertEqual(report["steady_mean_ms"]["inference_ms"], 10.0)
        self.assertEqual(report["steady_frames"], 2)

    def test_video_cleanup_failure_preserves_primary_error_and_report(self):
        def bad_frames():
            raise ValueError("primary source failure")
            yield

        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), patch("agent.runtime.VideoRecorder") as video, redirect_stdout(io.StringIO()):
            args = self.args(folder)
            args.video = True
            video.return_value.close.side_effect = OSError("encoder cleanup")
            video.return_value.count = 0
            with self.assertRaisesRegex(ValueError, "primary source failure"):
                run(bad_frames(), args, "replay")
            report = json.loads((Path(folder) / "result/benchmark.json").read_text())
            self.assertIn("primary source failure", report["error"])
            self.assertIn("encoder cleanup", report["cleanup_error"])
            self.assertIsNone(report["video"])

    def test_cleanup_failure_alone_is_reported_and_raised(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "agent.runtime.Detector", return_value=EmptyDetector()
        ), patch("agent.runtime.VideoRecorder") as video, redirect_stdout(io.StringIO()):
            args = self.args(folder)
            args.video = True
            video.return_value.close.side_effect = OSError("encoder cleanup")
            video.return_value.count = 0
            with self.assertRaisesRegex(OSError, "encoder cleanup"):
                run(iter(()), args, "replay")
            report = json.loads((Path(folder) / "result/benchmark.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIsNone(report["video"])
