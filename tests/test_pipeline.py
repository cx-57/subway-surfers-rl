"""Real pipeline boundaries and replay timing; no screen or model download needed."""

import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

from agent.capture import Frame
from agent.state import Detection
from agent.pipeline import Engine, StateLog
from agent.sources import ReplaySource


class ExampleDetector:
    last_timings = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}
    device = "cpu"

    def detect(self, image, timestamp):
        return [Detection(1, 0.8, (20, 20, 40, 40), timestamp)]


class PipelineTests(unittest.TestCase):
    def test_uncertain_distant_occupancy_does_not_advertise_safe_no_op(self):
        class Detector(ExampleDetector):
            def detect(self, image, timestamp):
                return [Detection(0, 0.9, (40, 10, 60, 30), timestamp)]

        config = dict(
            lane_rows=[[0.2, 0.4, 0.5, 0.6], [0.9, 0.15, 0.5, 0.85]],
            provenance="synthetic test",
            valid_frames=["f.jpg"],
            player_observations={
                "f.jpg": dict(
                    lane=1, movement="RUNNING", transitioning=False, provenance="synthetic test"
                )
            },
        )
        engine = Engine(detector=Detector(), calibration=config)
        state = engine.process(
            Frame(np.zeros((100, 100, 3), np.uint8), 1.0, time.monotonic(), frame_id="f.jpg")
        )
        self.assertEqual(state.recommendation.action, "UNKNOWN")
        self.assertNotIn("NO_OP", state.available_actions)

    def test_invalid_switch_and_capture_clock_are_rejected(self):
        for name in ("distance_filter", "automatic_geometry", "visual_player"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Engine(detector=ExampleDetector(), settings={name: "false"})
        engine = Engine(detector=ExampleDetector())
        with self.assertRaises(ValueError):
            engine.process(Frame(np.zeros((60, 100, 3), np.uint8), 1.0, float("nan")))

    def test_distance_gate_limits_hazards_but_preserves_lookahead_tracks(self):
        class Detector(ExampleDetector):
            def detect(self, image, timestamp):
                return [
                    Detection(1, 0.9, (40, 60, 60, 90), timestamp),
                    Detection(1, 0.9, (45, 10, 55, 20), timestamp),
                ]

        state = Engine(detector=Detector()).process(
            Frame(np.zeros((100, 100, 3), np.uint8), 1.0, time.monotonic())
        )
        self.assertEqual(len(state.tracks), 2)
        self.assertEqual(state.nearby_track_ids, [state.tracks[0].id])
        hazard_ids = []
        for hazard in state.hazards:
            hazard_ids.append(hazard.track_id)
        self.assertEqual(hazard_ids, state.nearby_track_ids)

    def test_freshness_limit_is_shared_with_hazards_and_policy(self):
        engine = Engine(detector=ExampleDetector(), settings={"max_observation_age": 1.0})
        self.assertEqual(engine.hazards.max_age, 1.0)
        self.assertEqual(engine.policy.max_age, 1.0)

    def test_unknown_player_prevents_action_and_log_contains_vector(self):
        engine = Engine(detector=ExampleDetector())
        frame = Frame(
            np.zeros((60, 100, 3), np.uint8), 1.0, time.monotonic(), frame_id="frame_000001.jpg"
        )
        state = engine.process(frame)
        self.assertEqual(state.recommendation.action, "UNKNOWN")
        self.assertIsNone(state.player.lane)
        self.assertEqual(state.image_shape, (60, 100))
        self.assertEqual(len(state.tracks), 1)
        self.assertFalse(state.valid)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "states.jsonl"
            with StateLog(path) as log:
                log.write(state, frame.frame_id)
            row = json.loads(path.read_text())
            self.assertEqual(row["frame_id"], frame.frame_id)
            self.assertEqual(row["state"]["timestamp"], 1.0)
            np.testing.assert_array_equal(row["observation"], state.to_vector())

    def test_tracking_continues_but_stale_observation_is_not_actionable(self):
        engine = Engine(detector=ExampleDetector())
        image = np.zeros((60, 100, 3), np.uint8)
        first = engine.process(Frame(image, 1.0, time.monotonic()))
        second = engine.process(Frame(image, 1.1, time.monotonic() - 1))
        self.assertEqual(first.tracks[0].id, second.tracks[0].id)
        self.assertGreater(second.observation_age, 0.5)
        self.assertFalse(second.valid)
        self.assertFalse(second.recommendation.actionable)

    def test_replay_keeps_dimensions_and_marks_approximate_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            for n in [2, 4]:
                cv2.imwrite(str(p / f"frame_{n:06d}.jpg"), np.zeros((12, 20, 3), np.uint8))
            (p / "metadata.json").write_text(json.dumps({"actual_fps": 20}))
            frames = list(ReplaySource(p))
            timestamps = []
            for frame in frames:
                timestamps.append(frame.timestamp)
            self.assertEqual(timestamps, [0.05, 0.15])
            self.assertEqual(frames[0].image.shape, (12, 20, 3))
            self.assertEqual(frames[0].timestamp_quality, "approximate")

    def test_exact_timestamps_and_missing_frame_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            image = p / "frame_000001.jpg"
            cv2.imwrite(str(image), np.zeros((12, 20, 3), np.uint8))
            source = ReplaySource(p, timestamps={image.name: 12.5})
            self.assertEqual(next(iter(source)).timestamp_quality, "exact")
            image.unlink()
            with self.assertRaises(FileNotFoundError):
                list(source)

    def test_no_assumed_fps_and_no_overwriting_existing_log(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            cv2.imwrite(str(p / "frame_000001.jpg"), np.zeros((12, 20, 3), np.uint8))
            with self.assertRaises(ValueError):
                list(ReplaySource(p))
            log = p / "states.jsonl"
            log.write_text("keep")
            with self.assertRaises(FileExistsError):
                with StateLog(log):
                    pass
            self.assertEqual(log.read_text(), "keep")
