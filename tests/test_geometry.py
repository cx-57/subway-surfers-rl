import unittest
from agent.geometry import Geometry
from agent.state import Detection


def calibration(**extra):
    return dict(
        lane_rows=[[0.2, 0.4, 0.5, 0.6], [0.9, 0.15, 0.5, 0.85]],
        provenance="synthetic test fixture",
        valid_frames=["frame_001.jpg"],
        **extra
    )


class GeometryTests(unittest.TestCase):
    def test_exact_source_path_cannot_match_another_recording(self):
        config = calibration()
        config["source_frame"] = "/recording_a/frame_001.jpg"
        g = Geometry(config)
        self.assertFalse(g.validate_frame((100, 100), "/recording_b/frame_001.jpg")[0])

    def test_missing_and_invalid_calibration(self):
        self.assertFalse(Geometry().valid)
        self.assertFalse(Geometry({"lane_rows": [[0.2, 0.6, 0.5, 0.4]]}).valid)

    def test_depth_interpolation_and_scope(self):
        g = Geometry(calibration())
        self.assertTrue(g.valid)
        self.assertAlmostEqual(g.lane_centers(0.55)[0], 0.275)
        d = Detection(1, 0.9, (40, 20, 60, 80), 0)
        self.assertEqual(
            g.assign(d, (100, 100), "other.jpg", footprint=(0.5, 0.8)).status, "UNKNOWN"
        )
        self.assertEqual(
            g.assign(d, (100, 100), "frame_001.jpg", footprint=(0.5, 0.8)).candidates, (1,)
        )

    def test_train_rectangle_does_not_define_lane(self):
        g = Geometry(calibration())
        self.assertEqual(
            g.assign(Detection(0, 0.9, (-20, 0, 95, 90), 0), (100, 100), "frame_001.jpg").status,
            "UNKNOWN",
        )

    def test_barrier_proxy_is_explicitly_uncertain(self):
        g = Geometry(calibration())
        estimate = g.assign(Detection(1, 0.9, (40, 30, 60, 80), 0), (100, 100), "frame_001.jpg")
        self.assertEqual(estimate.status, "UNKNOWN")
        self.assertEqual(estimate.candidates, (1,))
        self.assertIn("provisional", estimate.reason)

    def test_boundary_ambiguous_and_clipped_unknown(self):
        g = Geometry(calibration())
        centers = g.lane_centers(0.8)
        d = Detection(1, 0.9, (0, 0, 30, 80), 0)
        self.assertEqual(
            g.assign(
                d, (100, 100), "frame_001.jpg", footprint=((centers[0] + centers[1]) / 2, 0.8)
            ).status,
            "AMBIGUOUS",
        )
        self.assertEqual(g.assign(d, (100, 100), "frame_001.jpg").status, "UNKNOWN")
        self.assertEqual(
            g.assign(d, (100, 100), "frame_001.jpg", footprint=(0.5, 0.99)).status, "UNKNOWN"
        )

    def test_aspect_ratio_change_rejected(self):
        g = Geometry(calibration(aspect_ratio=1))
        self.assertEqual(
            g.assign(
                Detection(1, 0.9, (0, 0, 20, 80), 0),
                (100, 200),
                "frame_001.jpg",
                footprint=(0.5, 0.8),
            ).status,
            "UNKNOWN",
        )

    def test_invalid_scope_and_margin_rejected_at_load(self):
        for extra in (
            {"valid_frames": "frame_001.jpg"},
            {"frame_range": [5, 1]},
            {"boundary_margin": float("nan")},
            {"boundary_margin": -0.1},
        ):
            config = calibration()
            config.update(extra)
            self.assertFalse(Geometry(config).valid)

    def test_reviewed_train_footprint_mapping(self):
        config = calibration()
        config["footprints"] = {
            "frame_001.jpg": [
                {"bbox": [-20, 0, 95, 90], "point": [0.5, 0.8], "provenance": "synthetic test only"}
            ]
        }
        result = Geometry(config).assign(
            Detection(0, 0.9, (-20, 0, 95, 90), 0), (100, 100), "frame_001.jpg"
        )
        self.assertEqual(result.status, "KNOWN")
        self.assertEqual(result.candidates, (1,))

    def test_validate_current_frame_and_provisional_calibration(self):
        g = Geometry(calibration())
        self.assertTrue(g.validate_frame((100, 100), "frame_001.jpg")[0])
        self.assertFalse(g.validate_frame((100, 100), "frame_002.jpg")[0])
        g = Geometry(calibration(provisional=True))
        result = g.assign(
            Detection(1, 0.9, (40, 20, 60, 80), 0),
            (100, 100),
            "frame_001.jpg",
            footprint=(0.5, 0.8),
        )
        self.assertEqual(result.status, "UNKNOWN")
        self.assertIn("provisional", result.reason.lower())
        self.assertNotIn("measured", g.reason.lower())

    def test_map_player_respects_animation_scope_and_clipping(self):
        g = Geometry(calibration())
        box = (45, 55, 55, 80)
        self.assertEqual(g.map_player(box, (100, 100), "frame_001.jpg", "RUNNING").candidates, (1,))
        self.assertEqual(g.map_player(box, (100, 100), "frame_001.jpg", "RUNNING").status, "KNOWN")
        for movement in ("UNKNOWN", "JUMPING", "ON_TRAIN", "ROLLING"):
            self.assertEqual(
                g.map_player(box, (100, 100), "frame_001.jpg", movement).status, "UNKNOWN"
            )
        self.assertEqual(g.map_player(box, (100, 100), "other.jpg", "RUNNING").status, "UNKNOWN")
        self.assertEqual(
            g.map_player((45, 55, 55, 100), (100, 100), "frame_001.jpg", "RUNNING").status,
            "UNKNOWN",
        )

    def test_player_mapping_uses_observed_lane_curves_not_image_thirds(self):
        config = calibration()
        config["lane_rows"] = [[0.2, 0.25, 0.4, 0.55], [0.9, 0.3, 0.5, 0.7]]
        g = Geometry(config)
        # Player centered in the image still belongs to lane 2 at shallow depth.
        self.assertEqual(
            g.map_player((50, 10, 60, 20), (100, 100), "frame_001.jpg", "RUNNING").candidates, (2,)
        )

    def test_upperbody_candidates_never_claim_known_ground_lane(self):
        g = Geometry(calibration())
        estimate = g.player_candidates((45, 50, 55, 60), (100, 100), "frame_001.jpg")
        self.assertEqual(estimate.candidates, (1,))
        self.assertEqual(estimate.status, "UNKNOWN")
        self.assertIn("upper-body", estimate.reason)
        self.assertIn("foot contact unknown", estimate.reason)
        self.assertEqual(
            g.player_candidates((45, 70, 55, 95), (100, 100), "frame_001.jpg").candidates, ()
        )
        self.assertEqual(
            g.player_candidates((45, 50, 55, 60), (100, 100), "other.jpg").candidates, ()
        )

    def test_upperbody_candidate_union_respects_whole_span_and_supported_depth(self):
        g = Geometry(calibration())
        estimate = g.player_candidates((20, 50, 55, 60), (100, 100), "frame_001.jpg")
        self.assertEqual(estimate.candidates, (0, 1))
        self.assertEqual(estimate.status, "AMBIGUOUS")
        self.assertEqual(
            g.player_candidates((-5, 50, 55, 60), (100, 100), "frame_001.jpg").candidates, ()
        )
        self.assertEqual(
            Geometry().player_candidates((45, 50, 55, 60), (100, 100), "frame_001.jpg").candidates,
            (),
        )

    def test_malformed_box_and_nonfinite_shape_preserve_unknown(self):
        g = Geometry(calibration())
        for box in ((1, 2, 3), (1, None, 3, 4)):
            result = g.assign(Detection(1, 0.9, box, 0), (100, 100), "frame_001.jpg")
            self.assertEqual(result.status, "UNKNOWN")
        self.assertFalse(g.validate_frame((float("nan"), 100), "frame_001.jpg")[0])
        self.assertIsNone(g.lane_centers(None))
