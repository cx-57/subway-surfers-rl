import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent.state import GameState, Track, LaneEstimate, Hazard
from agent.visualization import render, toggle_key
from scripts.calibrate_lanes import calibration_config, collect_points


class VisualizationTests(unittest.TestCase):
    def test_distant_tracks_are_visible_before_becoming_nearby(self):
        image = np.zeros((100, 100, 3), np.uint8)
        state = GameState(
            0.0,
            (100, 100),
            tracks=[Track(1, 1, (20, 20, 40, 40), 0.9, 0.0, 0.0)],
            nearby_track_ids=[],
        )
        drawn = render(
            image, state, toggles=dict(status=False, player=False, lanes=False, hazards=False)
        )
        self.assertTrue(np.any(drawn != image))
        self.assertEqual(state.nearby_track_ids, [])
        self.assertEqual(state.hazards, [])
        # Crossing the proximity threshold changes the annotation, not visibility.
        state.nearby_track_ids = [1]
        nearby = render(
            image, state, toggles=dict(status=False, player=False, lanes=False, hazards=False)
        )
        self.assertTrue(np.any(nearby != image))
        self.assertTrue(np.any(nearby != drawn))

    def setUp(self):
        self.image = np.full((300, 517, 3), 60, dtype=np.uint8)
        self.state = GameState(1.25, (300, 517), calibration_valid=True)
        self.state.tracks = [
            Track(4, 1, (150, 130, 230, 205), 0.8, 1, 1.25, lane=LaneEstimate((1,), "KNOWN"))
        ]
        self.state.hazards = [Hazard(4, "UNKNOWN")]

    def test_preserves_source_and_dimensions(self):
        before = self.image.copy()
        result = render(
            self.image, self.state, {"lane_rows": [[0.2, 0.4, 0.5, 0.6], [0.9, 0.2, 0.5, 0.8]]}
        )
        self.assertEqual(result.shape, self.image.shape)
        self.assertFalse(np.shares_memory(result, self.image))
        np.testing.assert_array_equal(self.image, before)
        self.assertTrue(np.any(result != before))

    def test_all_overlays_disabled_is_exact_copy(self):
        result = render(
            self.image,
            self.state,
            toggles=dict.fromkeys(["boxes", "lanes", "player", "hazards", "status"], False),
        )
        np.testing.assert_array_equal(result, self.image)

    def test_each_toggle_has_visible_effect(self):
        config = {"lane_rows": [[0.2, 0.4, 0.5, 0.6], [0.9, 0.2, 0.5, 0.8]]}
        for category in ["boxes", "lanes", "player", "hazards", "status"]:
            with self.subTest(category=category):
                switches = dict.fromkeys(["boxes", "lanes", "player", "hazards", "status"], False)
                switches[category] = True
                self.assertTrue(
                    np.any(render(self.image, self.state, config, switches) != self.image)
                )
        switches = {}
        self.assertTrue(toggle_key(switches, ord("b")))
        self.assertFalse(switches["boxes"])
        self.assertFalse(toggle_key(switches, ord("x")))

    def test_missing_unknown_and_small_image(self):
        self.state.tracks[0].missed = True
        result = render(self.image, self.state)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(render(np.zeros((8, 13, 3), np.uint8), self.state).shape, (8, 13, 3))

    def test_clicks_normalized_and_source_hash_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frame_001.png"
            source.write_bytes(b"original bytes")
            points = [(40, 20), (50, 20), (60, 20), (20, 90), (50, 90), (80, 90)]
            config = calibration_config(source, (100, 100, 3), points)
            self.assertEqual(config["lane_rows"], [[0.2, 0.4, 0.5, 0.6], [0.9, 0.2, 0.5, 0.8]])
            self.assertEqual(config["valid_frames"], ["frame_001.png"])
            self.assertEqual(len(config["provenance"]["image_sha256"]), 64)
            ranged = calibration_config(source, (100, 100, 3), points, "session", 1, 20)
            self.assertIn("PROVISIONAL", ranged["provenance"]["scope"])
            self.assertNotIn("valid_frames", ranged)
            from agent.geometry import Geometry

            geometry = Geometry(ranged)
            self.assertTrue(geometry.valid)
            self.assertIsNone(geometry._scope_reason((100, 100), "session/frame_010.png"))
            self.assertEqual(source.read_bytes(), b"original bytes")
            with self.assertRaises(ValueError):
                calibration_config(source, (100, 100, 3), points, first_frame=1)
            with self.assertRaises(ValueError):
                calibration_config(source, (100, 100, 3), points[:3])

    @patch("scripts.calibrate_lanes.cv2.destroyWindow")
    @patch("scripts.calibrate_lanes.cv2.waitKey", return_value=27)
    @patch("scripts.calibrate_lanes.cv2.imshow")
    @patch("scripts.calibrate_lanes.cv2.setMouseCallback")
    @patch("scripts.calibrate_lanes.cv2.namedWindow")
    def test_gui_cancel_creates_no_coordinates(self, *mocks):
        self.assertIsNone(collect_points(self.image))
