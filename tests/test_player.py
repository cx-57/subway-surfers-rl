import unittest
from agent.player import PlayerEstimator


class PlayerTests(unittest.TestCase):
    def test_visual_location_and_candidate_lane_do_not_fabricate_ground_state(self):
        class Localizer:
            last_diagnostics = {}

            def locate(self, image, timestamp):
                return dict(
                    bbox=(40, 40, 60, 60),
                    score=0.9,
                    source="test pixel observation",
                    bbox_kind="upperbody",
                    movement="UNKNOWN",
                )

        class Geometry:
            def player_candidates(self, *args):
                from agent.state import LaneEstimate

                return LaneEstimate((1,), "UNKNOWN", "Upperbody projection only")

        estimator = PlayerEstimator(localizer=Localizer())
        state = estimator.update(object(), 1.0, geometry=Geometry(), image_shape=(100, 100))
        self.assertEqual(state.bbox, (40, 40, 60, 60))
        self.assertEqual(state.lane_candidates, (1,))
        self.assertIsNone(state.lane)
        self.assertEqual(state.movement, "UNKNOWN")
        self.assertIsNone(state.confidence)

    def test_unknown_without_observation(self):
        self.assertIsNone(PlayerEstimator().update(None, 1).lane)

    def test_reviewed_observations_never_propagate(self):
        estimator = PlayerEstimator(
            {
                "player_observations": {
                    "f1": {"lane": 0, "movement": "RUNNING", "provenance": "test review"},
                    "f2": {
                        "lane": 1,
                        "movement": "JUMPING",
                        "transitioning": True,
                        "provenance": "test review",
                    },
                }
            }
        )
        self.assertEqual(estimator.update(None, 1, "f1").lane, 0)
        self.assertTrue(estimator.update(None, 2, "f2").transitioning)
        self.assertEqual(estimator.update(None, 3, "f3").movement, "UNKNOWN")
        self.assertIsNone(estimator.update(None, 3, "f3").lane)

    def test_unreviewed_or_invalid_observation_unknown(self):
        estimator = PlayerEstimator(
            {"player_observations": {"f": {"lane": 2}, "bad": {"lane": 7, "provenance": "test"}}}
        )
        self.assertIsNone(estimator.update(None, 1, "f").lane)
        self.assertIsNone(estimator.update(None, 1, "bad").lane)

    def test_missing_frame_does_not_reuse_localizer_diagnostics(self):
        class Localizer:
            last_diagnostics = {"total_ms": 7.0, "reason": "prior frame match"}

            def locate(self, image, timestamp):
                return None

        estimator = PlayerEstimator(localizer=Localizer())
        estimator.update(object(), 1)
        estimator.update(None, 2)
        self.assertEqual(estimator.last_diagnostics, {})

    def test_localization_retains_template_provenance(self):
        class Localizer:
            last_diagnostics = {}

            def locate(self, image, timestamp):
                return {
                    "bbox": (40, 40, 60, 60),
                    "provenance": "reviewed crop",
                    "template_source": "session/frame.jpg",
                }

        state = PlayerEstimator(localizer=Localizer()).update(object(), 1, image_shape=(100, 100))
        self.assertEqual(state.localization["provenance"], "reviewed crop")
        self.assertEqual(state.localization["template_source"], "session/frame.jpg")

    def test_reviewed_observation_honors_exact_calibration_source(self):
        config = {
            "source_frame": "session_a/frame.jpg",
            "player_observations": {
                "frame.jpg": {"lane": 1, "movement": "RUNNING", "provenance": "review of session a"}
            },
        }
        estimator = PlayerEstimator(config, visual=False)
        self.assertEqual(estimator.update(None, 1, "session_a/frame.jpg").lane, 1)
        self.assertIsNone(estimator.update(None, 2, "session_b/frame.jpg").lane)
