import json
from pathlib import Path
import tempfile
import unittest

from agent.control_validation import ControlValidation
from agent.state import GameState


class ControlValidationTests(unittest.TestCase):
    def test_commands_are_bounded_and_observations_are_not_ground_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            sent = []
            validation = ControlValidation(Path(folder), sent.append)
            state = GameState(100.0, (100, 100))
            validation.update(state, "first")
            state.timestamp = 105.0
            validation.update(state, "left")
            validation.update(state, "same_frame")
            self.assertEqual(sent, ["LEFT"])
            state.timestamp = 105.7
            state.player.lane_candidates = (0,)
            validation.update(state, "settled")
            events = []
            for x in (Path(folder) / "controls.jsonl").read_text().splitlines():
                events.append(json.loads(x))
            observation = next(x for x in events if x["event"] == "observation")
            self.assertEqual(observation["command_expected_lane"], 0)
            self.assertEqual(observation["lane_candidates"], [0])
            self.assertIsNone(state.player.lane)
            state.timestamp = 140.0
            validation.update(state, "late")
            self.assertEqual(sent, ["LEFT"])  # Never burst overdue commands.

    def test_focus_failure_aborts_remaining_commands(self):
        with tempfile.TemporaryDirectory() as folder:

            def failed(action):
                raise RuntimeError("Chrome is not focused")

            validation = ControlValidation(Path(folder), failed)
            state = GameState(0.0, (100, 100))
            validation.update(state, "start")
            state.timestamp = 5.0
            validation.update(state, "attempt")
            self.assertTrue(validation.aborted)
            state.timestamp = 8.0
            validation.update(state, "later")
            events = []
            for x in (Path(folder) / "controls.jsonl").read_text().splitlines():
                events.append(json.loads(x))
            self.assertEqual(sum(x["event"] == "command" for x in events), 1)
            self.assertFalse(events[-1]["sent"])
