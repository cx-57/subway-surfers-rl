import unittest
from agent.player import PlayerEstimator, PlayerLaneTracker
from agent.state import GameState


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

    def test_codex_player_bbox_produces_provisional_lane_without_new_detector(self):
        reference = {
            "provenance": "reviewed source screenshots",
            "centers": [0.4453, 0.5030, 0.5517],
            "aspect_ratio": 1354 / 823,
            "stable_frames": 2,
        }

        class Localizer:
            config = {"lane_tracking": reference}
            last_diagnostics = {}
            bbox = (650, 530, 710, 615)

            def locate(self, image, timestamp):
                if self.bbox is None:
                    return None
                return {
                    "bbox": self.bbox,
                    "bbox_kind": "upperbody",
                    "source": "automatic color-template NCC",
                }

        localizer = Localizer()
        player = PlayerEstimator(localizer=localizer)
        first = player.update(object(), 1.0, image_shape=(823, 1354))
        stable = player.update(object(), 1.1, image_shape=(823, 1354))
        self.assertIsNone(first.estimated_lane)
        self.assertEqual(stable.estimated_lane, 1)
        self.assertIsNone(stable.lane)
        self.assertEqual(stable.movement, "UNKNOWN")
        self.assertEqual(stable.source, "automatic color-template NCC")
        self.assertEqual(stable.lane_candidates, (1,))
        self.assertTrue(stable.localization["screen_lane"]["provisional"])
        vector = GameState(1.1, (823, 1354), player=stable).to_vector()
        self.assertEqual(list(vector[:4]), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(list(vector[-4:]), [0.0, 1.0, 0.0, 1.0])

        localizer.bbox = (685, 530, 745, 615)
        transitional = player.update(object(), 1.2, image_shape=(823, 1354))
        self.assertIsNone(transitional.estimated_lane)
        self.assertTrue(transitional.localization["screen_lane"]["transitioning"])
        localizer.bbox = (720, 530, 780, 615)
        self.assertIsNone(player.update(object(), 1.3, image_shape=(823, 1354)).estimated_lane)
        self.assertEqual(player.update(object(), 1.4, image_shape=(823, 1354)).estimated_lane, 2)
        localizer.bbox = None
        self.assertIsNone(player.update(object(), 1.5, image_shape=(823, 1354)).estimated_lane)
        localizer.bbox = (720, 530, 780, 615)
        self.assertIsNone(player.update(object(), 1.6, image_shape=(823, 1354)).estimated_lane)
        self.assertEqual(player.update(object(), 1.7, image_shape=(823, 1354)).estimated_lane, 2)
        self.assertIsNone(player.update(object(), 2.4, image_shape=(823, 1354)).estimated_lane)
        localizer.bbox = (574, 530, 634, 615)
        self.assertIsNone(player.update(object(), 2.5, image_shape=(823, 1354)).estimated_lane)
        self.assertEqual(player.update(object(), 2.6, image_shape=(823, 1354)).estimated_lane, 0)

    def test_screen_lane_independent_of_rail_geometry(self):
        reference = dict(
            provenance="reviewed gameplay",
            centers=[0.4453, 0.5030, 0.5517],
            boundaries=[0.468, 0.523],
            aspect_ratio=1000 / 600,
            stable_frames=2,
        )

        class Localizer:
            config = {"lane_tracking": reference}
            last_diagnostics = {}
            x = 0.443

            def locate(self, image, timestamp):
                center = self.x * 1000
                return dict(bbox=(center - 30, 300, center + 30, 400),
                            bbox_kind="upperbody")

        loc = Localizer()
        estimator = PlayerEstimator(localizer=loc)
        for n, (x, lane) in enumerate(((0.443, 0), (0.493, 1), (0.553, 2))):
            loc.x = x
            a = estimator.update(object(), 1.0 + 0.3 * n, image_shape=(600, 1000))
            b = estimator.update(object(), 1.1 + 0.3 * n, image_shape=(600, 1000))
            self.assertIsNone(a.estimated_lane)
            self.assertEqual(b.estimated_lane, lane)
            self.assertIsNone(b.lane)
        loc.x = 0.523
        transition = estimator.update(object(), 2.0, image_shape=(600, 1000))
        self.assertIsNone(transition.estimated_lane)
        self.assertTrue(transition.localization["screen_lane"]["transitioning"])

    def test_invalid_lane_reference_and_viewport_fail_closed(self):
        reference = {
            "provenance": "reviewed example", "centers": [0.4453, 0.5030, 0.5517],
            "aspect_ratio": 1354 / 823,
        }
        tracker = PlayerLaneTracker(reference)
        self.assertIsNone(tracker.estimate((650, 530, 710, 615), 1, (800, 800))["lane"])
        self.assertEqual(
            tracker.estimate((-5, 530, 710, 615), 1.1, (823, 1354))["candidates"], ()
        )
        for bad in (
            dict(reference, centers=[0.5, 0.5, 0.5]),
            dict(reference, boundaries=[0.6, 0.7]),
            dict(reference, max_gap_seconds=0),
            dict(reference, provenance=""),
        ):
            with self.assertRaises(ValueError):
                PlayerLaneTracker(bad)

    def test_rail_disagreement_stays_provisional_not_ground_truth(self):
        tracker = PlayerLaneTracker({
            "provenance": "reviewed example", "centers": [0.4453, 0.5030, 0.5517],
            "aspect_ratio": 1354 / 823,
        })
        tracker.estimate((650, 530, 710, 615), 1, (823, 1354), (2,))
        result = tracker.estimate((650, 530, 710, 615), 1.1, (823, 1354), (2,))
        self.assertEqual(result["lane"], 1)
        self.assertIs(result["geometry_agrees"], False)
        self.assertTrue(result["provisional"])

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
