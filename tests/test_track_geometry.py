"""Scene-specific rail hypotheses: geometric gates, not accuracy measurements."""

import unittest
import cv2
import numpy as np
from agent.track_geometry import TrackGeometry


def clear_rails():
    image = np.zeros((500, 800, 3), np.uint8)
    # Six rails converge toward the horizon; ties/scenery must not supply lanes.
    for far, near in [(325, 60), (350, 190), (375, 325), (400, 460), (425, 590), (450, 725)]:
        cv2.line(image, (far, 160), (near, 475), (160, 160, 195), 7)
    return image


class TrackGeometryTests(unittest.TestCase):
    def test_no_rails_or_one_track_never_supplies_three_lanes(self):
        estimator = TrackGeometry()
        self.assertFalse(estimator.estimate(np.zeros((500, 800, 3), np.uint8), "f").valid)
        image = np.zeros((500, 800, 3), np.uint8)
        cv2.line(image, (350, 160), (190, 475), (160, 160, 195), 7)
        cv2.line(image, (375, 160), (325, 475), (160, 160, 195), 7)
        self.assertFalse(estimator.estimate(image, "f").valid)

    def test_clear_rail_pattern_is_frame_local_and_explicitly_inferred(self):
        g = TrackGeometry().estimate(clear_rails(), "session/frame_001.jpg")
        self.assertTrue(g.valid, g.reason)
        self.assertIn("automatic", g.reason.lower())
        self.assertNotIn("measured calibration", g.reason.lower())
        centers = g.lane_centers(0.8)
        self.assertIsNotNone(centers)
        self.assertLess(centers[0], centers[1])
        self.assertLess(centers[1], centers[2])
        self.assertFalse(g.validate_frame((500, 800), "session/frame_002.jpg")[0])

    def test_occlusion_does_not_reuse_last_success(self):
        estimator = TrackGeometry()
        self.assertTrue(estimator.estimate(clear_rails(), "a").valid)
        self.assertFalse(estimator.estimate(np.zeros((500, 800, 3), np.uint8), "b").valid)

    def test_parallel_nonperspective_lines_rejected(self):
        image = np.zeros((500, 800, 3), np.uint8)
        for x in (80, 180, 280, 380, 480, 580):
            cv2.line(image, (x, 200), (x, 475), (160, 160, 195), 7)
        self.assertFalse(TrackGeometry().estimate(image, "f").valid)

    def test_camera_horizontal_shift_reestimated_instead_of_holding_centers(self):
        image = clear_rails()
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 40], [0, 1, 0]]), (800, 500))
        estimator = TrackGeometry()
        a = estimator.estimate(image, "a")
        b = estimator.estimate(shifted, "b")
        self.assertTrue(a.valid, a.reason)
        self.assertTrue(b.valid, b.reason)
        delta = b.lane_centers(0.8)[1] - a.lane_centers(0.8)[1]
        self.assertAlmostEqual(delta, 40 / 800, delta=0.015)

    def test_original_clear_and_occluded_frames_qualitative_regression(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        folder = root / "data/raw/session_1795524944131662000"
        if not (folder / "frame_000500.jpg").is_file():
            self.skipTest("Local original gameplay recording unavailable")
        estimator = TrackGeometry()
        path = folder / "frame_000500.jpg"
        image = cv2.imread(str(path))
        geometry = estimator.estimate(image, str(path))
        self.assertTrue(geometry.valid, geometry.reason)
        # AI qualitative review of this source only: visible feet on center track.
        lane = geometry.map_player((624, 550, 706, 678), image.shape[:2], str(path), "RUNNING")
        self.assertEqual(lane.candidates, (1,))
        self.assertEqual(lane.status, "KNOWN")
        for n in (300, 420, 900):
            path = folder / f"frame_{n:06d}.jpg"
            self.assertFalse(estimator.estimate(cv2.imread(str(path)), str(path)).valid)

    def test_six_clean_rails_need_not_generate_duplicate_hough_segments(self):
        image = clear_rails()
        image[:275] = 0
        g = TrackGeometry().estimate(image, "short")
        self.assertTrue(g.valid, g.reason)

    def test_original_camera_follow_right_lane_inside_middle_image_third(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        path = root / "data/training/images/frame_000600.jpg"
        if not path.is_file():
            self.skipTest("Local original gameplay recording unavailable")
        image = cv2.imread(str(path))
        g = TrackGeometry().estimate(image, str(path))
        self.assertTrue(g.valid, g.reason)
        # Approximate AI-reviewed full-body box on this original, not a
        # detector result or an independent quantitative evaluation label.
        box = (736, 545, 814, 713)
        self.assertTrue(1 / 3 < (box[0] + box[2]) / (2 * image.shape[1]) < 2 / 3)
        lane = g.map_player(box, image.shape[:2], str(path), "RUNNING")
        self.assertEqual(lane.status, "KNOWN")
        self.assertEqual(lane.candidates, (2,))
        self.assertEqual(
            g.map_player(
                box, image.shape[:2], str(path), "RUNNING", footpoint_trustworthy=False
            ).status,
            "UNKNOWN",
        )

    def test_original_upperbody_projection_is_candidate_not_ground_state(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        path = root / "data/raw/session_1795524944131662000/frame_000500.jpg"
        if not path.is_file():
            self.skipTest("Local original gameplay recording unavailable")
        image = cv2.imread(str(path))
        g = TrackGeometry().estimate(image, str(path))
        estimate = g.player_candidates((630, 548, 694, 630), image.shape[:2], str(path))
        self.assertEqual(estimate.candidates, (1,))
        self.assertEqual(estimate.status, "UNKNOWN")
