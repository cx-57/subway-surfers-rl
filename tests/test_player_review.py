"""Contract tests for reviewed session calibration and held-out evaluation."""

import json
from pathlib import Path
import statistics
import unittest

from agent.player import VisualPlayerCalibration
from scripts.evaluate_player_session import evaluate_rows

ROOT = Path(__file__).resolve().parents[1]
SESSION = "session_1795724241854738000"


class SessionCalibrationTests(unittest.TestCase):
    def test_reviewed_labels_match_fitted_calibration(self):
        path = ROOT / "data/reviews" / f"{SESSION}_player.json"
        calibration = ROOT / "config" / f"player_calibration_{SESSION}.json"
        labels = json.loads(path.read_text())
        config = json.loads(calibration.read_text())["player_visual_calibration"]
        self.assertFalse(config["validated"])
        self.assertEqual(labels["session"], SESSION)
        samples = labels["labels"]
        self.assertEqual(len(samples), len({x["frame"] for x in samples}))
        self.assertTrue(all(x["frame"] <= 350 for x in samples if x["split"] == "fit"))
        self.assertTrue(all(x["frame"] > 350 for x in samples if x["split"] == "holdout"))
        for lane in range(3):
            points = [
                x["center_px"][0] / labels["width"]
                for x in samples
                if x["split"] == "fit"
                and x["scene"] == "ground"
                and x["lane"] == lane
            ]
            self.assertGreaterEqual(len(points), 3)
            self.assertAlmostEqual(
                statistics.median(points), config["lane_x"][lane], places=6
            )
            neutral_y = [
                x["center_px"][1] / labels["height"]
                for x in samples
                if x["split"] == "fit"
                and x["scene"] == "ground"
                and x["lane"] == lane
                and x["movement"] == "RUNNING"
            ]
            self.assertGreaterEqual(len(neutral_y), 2)
            self.assertAlmostEqual(
                statistics.median(neutral_y), config["running_y"][lane], places=6
            )
        VisualPlayerCalibration(config)

    def test_unknown_and_false_positive_not_counted_as_successes(self):
        holdout = [
            dict(
                split="holdout", scene="ground", frame=361,
                lane=0, movement="RUNNING",
                bbox=None, center_error_px=None, predicted_lane=None,
                predicted_pose="UNKNOWN",
            ),
            dict(
                split="holdout", scene="ground", frame=362,
                lane=2, movement="ROLLING",
                bbox=[10, 10, 30, 30], center_error_px=100,
                predicted_lane=2, predicted_pose="ROLLING",
            ),
            dict(
                split="holdout", scene="ground", frame=363,
                lane=1, movement="JUMPING",
                bbox=[10, 10, 30, 30], center_error_px=12,
                predicted_lane=0, predicted_pose="RUNNING",
            ),
            dict(
                split="holdout", scene="non_gameplay", frame=560,
                lane=None, movement="NO_PLAYER",
                bbox=[10, 10, 30, 30], center_error_px=None,
                predicted_lane=0, predicted_pose="RUNNING",
            ),
        ]
        result = evaluate_rows(holdout)
        self.assertEqual(result["eligible_ground"], 3)
        self.assertEqual(result["plausible_location_ground"], 1)
        self.assertEqual(result["stable_lane_ground"], 1)
        self.assertEqual(result["correct_lane_ground"], 0)
        self.assertEqual(result["correct_pose_ground"], 0)
        self.assertEqual(result["false_localizations_on_negative_frames"], 1)
        self.assertIn(363, result["wrong_lane_examples"])
        self.assertIn(363, result["wrong_pose_examples"])

    def test_no_predictions_yield_null_accuracy_not_100_percent(self):
        score = evaluate_rows([])
        self.assertIsNone(score["ground_localization_recall"])
        self.assertIsNone(score["lane_accuracy_when_predicted"])
        self.assertIsNone(score["pose_accuracy_when_predicted"])


if __name__ == "__main__":
    unittest.main()
