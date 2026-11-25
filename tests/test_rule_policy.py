import unittest
from agent.rule_policy import ShadowPolicy
from agent.state import GameState, PlayerState, Hazard, Track, LaneEstimate


def state():
    t = Track(1, 1, (40, 20, 60, 80), 0.9, 0, 1, lane=LaneEstimate((1,), "KNOWN"))
    return GameState(
        1,
        (100, 100),
        player=PlayerState(1, "RUNNING", timestamp=1, transitioning=False),
        tracks=[t],
        hazards=[
            Hazard(1, "THREAT", "HIGH", 0.6, "test calibrated deadline", ["JUMP", "LEFT", "RIGHT"])
        ],
        available_actions=["JUMP", "LEFT", "RIGHT"],
        calibration_valid=True,
        valid=True,
    )


class PolicyTests(unittest.TestCase):
    def test_unknown_player_and_unknown_deadline(self):
        s = state()
        s.player = PlayerState()
        self.assertEqual(ShadowPolicy().recommend(s).action, "UNKNOWN")
        s = state()
        s.hazards[0].ttc = None
        self.assertEqual(ShadowPolicy().recommend(s).action, "UNKNOWN")

    def test_action_and_dedup(self):
        p = ShadowPolicy(
            {"provisional_action_durations": {"JUMP": 0.3}, "provisional_action_window": 0.8}
        )
        first = p.recommend(state())
        second = p.recommend(state())
        self.assertEqual(first.action, "JUMP")
        self.assertTrue(first.actionable)
        self.assertTrue(second.repeated)
        self.assertFalse(second.actionable)

    def test_no_early_jump_or_insufficient_reaction(self):
        p = ShadowPolicy({"provisional_action_durations": {"JUMP": 0.3}})
        for deadline in (0.1, 5):
            s = state()
            s.hazards[0].ttc = deadline
            self.assertEqual(p.recommend(s).action, "UNKNOWN")

    def test_unknown_neighbor_and_conflicting_hazards(self):
        s = state()
        s.tracks.append(Track(2, 0, (0, 0, 100, 100), 0.8, 0, 1))
        s.hazards.append(Hazard(2))
        p = ShadowPolicy({"provisional_action_durations": {"LEFT": 0.2, "RIGHT": 0.2}})
        self.assertEqual(p.recommend(s).action, "UNKNOWN")
        s = state()
        s.hazards.append(Hazard(2, "THREAT", "HIGH", 0.6, "test", ["ROLL"]))
        self.assertEqual(
            ShadowPolicy({"provisional_action_durations": {"JUMP": 0.3}}).recommend(s).action,
            "UNKNOWN",
        )

    def test_inconsistent_or_stale_player_cannot_drive_suggestion(self):
        p = ShadowPolicy({"provisional_action_durations": {"JUMP": 0.3}})
        s = state()
        s.tracks[0].lane.status = "UNKNOWN"
        self.assertEqual(p.recommend(s).action, "UNKNOWN")
        s = state()
        s.player.timestamp = 0
        self.assertEqual(p.recommend(s).action, "UNKNOWN")

    def test_adjacent_lane_needs_explicit_visibility(self):
        s = state()
        s.available_actions = ["LEFT"]
        s.hazards[0].feasible_actions = ["LEFT"]
        p = ShadowPolicy({"provisional_action_durations": {"LEFT": 0.2}})
        self.assertEqual(p.recommend(s).action, "UNKNOWN")
        s.lane_visibility = [True, None, None]
        self.assertEqual(p.recommend(s).action, "LEFT")

    def test_nonfinite_parameters_rejected(self):
        for config in (
            {"provisional_action_durations": {"JUMP": float("nan")}},
            {"max_observation_age": float("inf")},
            {"provisional_action_window": -1},
        ):
            with self.assertRaises(ValueError):
                ShadowPolicy(config)
