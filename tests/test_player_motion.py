import unittest

from agent.player import VisualPlayerCalibration
from agent.player import PlayerEstimator


def calibration(validated=False, with_poses=True):
    data = {
        "provenance": "synthetic human-reviewed example",
        "validated": validated,
        "aspect_ratio": 1.0,
        "lane_x": [0.2, 0.5, 0.8],
        "stable_frames": 3,
        "max_x_error": 0.06,
        "lane_margin": 0.025,
    }
    if with_poses:
        data.update(
            running_y=[0.7, 0.7, 0.7],
            pose_offsets={"RUNNING": 0.0, "JUMPING": -0.09, "ROLLING": 0.08},
        )
    return data


def bbox(x, y):
    return (x * 100 - 5, y * 100 - 5, x * 100 + 5, y * 100 + 5)


class PlayerMotionTests(unittest.TestCase):
    def test_stable_lane_and_pose_are_only_hypotheses_before_validation(self):
        model = VisualPlayerCalibration(calibration())
        for frame in (0.0, 0.1):
            result = model.predict(bbox(0.5, 0.7), frame, (100, 100))
            self.assertFalse(result["stable"])
        result = model.predict(bbox(0.5, 0.7), 0.2, (100, 100))
        self.assertEqual(result["lane"], 1)
        self.assertEqual(result["pose"], "RUNNING")
        self.assertFalse(result["validated"])
        self.assertFalse(result["transitioning"])
        result = model.predict(bbox(0.5, 0.61), 0.3, (100, 100))
        self.assertEqual(result["pose"], "JUMPING")
        result = model.predict(bbox(0.5, 0.78), 0.4, (100, 100))
        self.assertEqual(result["pose"], "ROLLING")

    def test_transition_gap_and_bad_viewport_reset_instead_of_reusing_lane(self):
        model = VisualPlayerCalibration(calibration(validated=True))
        for time in (0.0, 0.1, 0.2):
            model.predict(bbox(0.2, 0.7), time, (100, 100))
        between = model.predict(bbox(0.35, 0.7), 0.3, (100, 100))
        self.assertIsNone(between["lane"])
        self.assertTrue(between["transitioning"])
        self.assertFalse(model.predict(bbox(0.8, 0.7), 0.4, (100, 100))["stable"])
        self.assertFalse(model.predict(bbox(0.8, 0.7), 1.0, (100, 100))["stable"])
        changed = model.predict(bbox(0.8, 0.7), 1.1, (100, 120))
        self.assertIsNone(changed["lane"])

    def test_realistic_narrow_spaced_lanes_can_be_calibrated(self):
        real = calibration()
        real["aspect_ratio"] = 1354 / 823
        real["lane_x"] = [0.44535, 0.50295, 0.55170]
        real["max_x_error"] = 0.033
        real["lane_margin"] = 0.012
        real["running_y"] = [0.726, 0.697, 0.713]
        real["pose_offsets"] = {
            "RUNNING": 0.0, "JUMPING": -0.2466, "ROLLING": 0.0802
        }
        model = VisualPlayerCalibration(real)
        center_x = 0.55170 * 1354
        center_y = 0.713 * 823
        for t in (0.0, 0.1, 0.2):
            result = model.predict(
                (center_x - 25, center_y - 30, center_x + 25, center_y + 30),
                t, (823, 1354),
            )
        self.assertEqual(result["lane"], 2)
        self.assertEqual(result["pose"], "RUNNING")
        self.assertFalse(result["validated"])

    def test_invalid_reference_data_rejected(self):
        for modifications in (
            {"lane_x": [0.5, 0.5, 0.8]},
            {"validated": "yes"},
            {"aspect_ratio": 0},
            {"max_x_error": float("nan")},
            {"running_y": [0.7, 0.7, 0.7], "pose_offsets": {"RUNNING": 0}},
        ):
            with self.subTest(modifications=modifications):
                with self.assertRaises(ValueError):
                    VisualPlayerCalibration(dict(calibration(), **modifications))

    def test_pose_remains_unknown_without_calibrated_distinct_actions(self):
        model = VisualPlayerCalibration(calibration(with_poses=False))
        for t in (0.0, 0.1, 0.2):
            result = model.predict(bbox(0.5, 0.7), t, (100, 100))
        self.assertEqual(result["pose"], "UNKNOWN")

    def test_player_estimator_does_not_promote_unvalidated_predictions(self):
        class Localizer:
            last_diagnostics = {}
            def locate(self, image, timestamp):
                return {"bbox": bbox(0.5, 0.7), "source": "test"}

        estimator = PlayerEstimator(
            {"player_visual_calibration": calibration(validated=False)},
            localizer=Localizer()
        )
        for t in (0.0, 0.1, 0.2):
            state = estimator.update(object(), t, image_shape=(100, 100))
        self.assertIsNone(state.lane)
        self.assertEqual(state.movement, "UNKNOWN")
        self.assertEqual(state.lane_candidates, (1,))
        self.assertTrue(state.localization["visual_calibration"]["stable"])

    def test_validated_model_promotes_after_repeated_visual_matches(self):
        class Localizer:
            last_diagnostics = {}
            def locate(self, image, timestamp):
                return {"bbox": bbox(0.5, 0.7), "source": "test"}

        estimator = PlayerEstimator(
            {"player_visual_calibration": calibration(validated=True)},
            localizer=Localizer()
        )
        for t in (0.0, 0.1, 0.2):
            state = estimator.update(object(), t, image_shape=(100, 100))
        self.assertEqual(state.lane, 1)
        self.assertEqual(state.movement, "RUNNING")
        self.assertIs(state.transitioning, False)
        estimator.update(None, 0.3)
        self.assertIsNone(estimator.update(object(), 0.4, image_shape=(100, 100)).lane)


if __name__ == "__main__":
    unittest.main()
