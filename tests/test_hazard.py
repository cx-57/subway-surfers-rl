import unittest
from agent.hazard import HazardEstimator
from agent.state import Track, LaneEstimate, PlayerState


def track(lane=1, **kwargs):
    return Track(
        1,
        1,
        (40, 20, 60, 80),
        0.9,
        0,
        1,
        history=[(0, (40, 20, 60, 70)), (1, (40, 20, 60, 80))],
        lane=LaneEstimate((lane,), "KNOWN", depth=0.8),
        velocity_y=10,
        **kwargs
    )


class HazardTests(unittest.TestCase):
    def test_approaching_unknown_seconds(self):
        h = HazardEstimator().estimate(
            [track()], PlayerState(1, "RUNNING", timestamp=1), (100, 100), 0, True
        )[0]
        self.assertEqual(h.status, "THREAT")
        self.assertIsNone(h.ttc)
        self.assertEqual(h.feasible_actions, [])
        self.assertEqual(h.urgency, "HIGH")

    def test_reviewed_multilane_occupancy_threatens_each_covered_lane(self):
        obstacle = track()
        obstacle.lane = LaneEstimate((0, 1), "KNOWN", depth=0.8)
        evaluator = HazardEstimator()
        for lane in (0, 1):
            self.assertEqual(
                evaluator.estimate(
                    [obstacle], PlayerState(lane, "RUNNING"), (100, 100), 0, True
                )[0].status,
                "THREAT",
            )
        self.assertEqual(
            evaluator.estimate(
                [obstacle], PlayerState(2, "RUNNING"), (100, 100), 0, True
            )[0].status,
            "CLEAR",
        )

    def test_neighbor_clear(self):
        self.assertEqual(
            HazardEstimator()
            .estimate([track(0)], PlayerState(1, "RUNNING"), (100, 100), 0, True)[0]
            .status,
            "CLEAR",
        )

    def test_unknown_stale_or_missed(self):
        e = HazardEstimator()
        for player, age, valid, t in [
            (PlayerState(), 0, True, track()),
            (PlayerState(1, "RUNNING"), 2, True, track()),
            (PlayerState(1, "RUNNING"), 0, False, track()),
            (PlayerState(1, "RUNNING"), 0, True, track(missed=True)),
        ]:
            self.assertEqual(e.estimate([t], player, (100, 100), age, valid)[0].status, "UNKNOWN")

    def test_noisy_motion_does_not_create_ttc(self):
        t = track()
        t.history = [(0, (0, 0, 20, 70)), (0.5, (0, 0, 20, 85)), (1, (0, 0, 20, 80))]
        h = HazardEstimator().estimate([t], PlayerState(1, "RUNNING"), (100, 100), 0, True)[0]
        self.assertIsNone(h.ttc)
        self.assertIn("unstable", h.reason)

    def test_vertical_player_states_and_static_depth_unknown_urgency(self):
        for movement in ("JUMPING", "ROLLING", "ON_TRAIN"):
            h = HazardEstimator().estimate(
                [track()], PlayerState(1, movement), (100, 100), 0, True
            )[0]
            self.assertEqual(h.status, "UNKNOWN")
        t = track()
        t.history = []
        t.velocity_y = None
        h = HazardEstimator().estimate([t], PlayerState(1, "RUNNING"), (100, 100), 0, True)[0]
        self.assertEqual(h.urgency, "UNKNOWN")

    def test_invalid_freshness_parameter_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                HazardEstimator({"max_observation_age": value})

    def test_inconsistent_velocity_and_receding_history_unknown_urgency(self):
        t = track()
        t.history = [(0, (0, 0, 20, 85)), (1, (0, 0, 20, 80))]
        h = HazardEstimator().estimate([t], PlayerState(1, "RUNNING"), (100, 100), 0, True)[0]
        self.assertEqual(h.urgency, "UNKNOWN")
