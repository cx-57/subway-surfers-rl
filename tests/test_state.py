"""Unknowns, missing tracks and stable ordering in the future RL observation."""

import unittest
import numpy as np
from agent.state import GameState, Track, LaneEstimate, Hazard, OBSERVATION_FIELDS


class StateTests(unittest.TestCase):
    def test_unknown_visibility_differs_from_known_occlusion(self):
        state = GameState(0.0, (100, 100), lane_visibility=[None, False, True])
        values = dict(zip(OBSERVATION_FIELDS, state.to_vector()))
        self.assertEqual(values["lane0.visibility_known"], 0)
        self.assertEqual(values["lane1.visibility_known"], 1)
        self.assertEqual(values["lane1.visible"], 0)
        self.assertEqual(values["lane2.visible"], 1)

    def test_invalid_depth_and_negative_age_are_not_marked_known_or_fresh(self):
        t = Track(1, 1, (0, 0, 10, 10), 0.8, 0.0, 1.0, lane=LaneEstimate((0,), "KNOWN", depth=2.0))
        state = GameState(1.0, (100, 100), tracks=[t], observation_age=-1)
        values = dict(zip(OBSERVATION_FIELDS, state.to_vector()))
        self.assertEqual(values["fresh"], 0)
        self.assertEqual(values["lane0.slot0.depth_known"], 0)

    def test_all_six_classes_have_distinct_observation_flags(self):
        self.assertEqual(len(OBSERVATION_FIELDS), 137)
        names = ("train", "ground", "overhead", "bush", "small_obstacle", "pillar")
        for class_id, expected in enumerate(names):
            with self.subTest(class_id=class_id):
                track = Track(1, class_id, (0, 0, 10, 10), 0.8, 0, 1,
                              lane=LaneEstimate((1,), "KNOWN", depth=0.8))
                values = dict(zip(
                    OBSERVATION_FIELDS,
                    GameState(1, (100, 100), tracks=[track]).to_vector()
                ))
                for name in names:
                    self.assertEqual(values["lane1.slot0." + name], float(name == expected))

    def test_unlocated_new_class_remains_visible_without_inventing_a_lane(self):
        track = Track(7, 3, (10, 20, 30, 40), 0.9, 0, 1)
        state = GameState(1, (100, 100), tracks=[track])
        values = dict(zip(OBSERVATION_FIELDS, state.to_vector()))
        self.assertGreater(values["unlocated.bush"], 0.0)
        self.assertEqual(values["lane0.slot0.present"], 0.0)
        self.assertEqual(values["lane1.slot0.present"], 0.0)

    def test_empty_vector_is_finite_fixed_size_and_unknown_is_masked(self):
        state = GameState(0.0, (793, 1284))
        vector = state.to_vector()
        self.assertEqual(vector.shape, (len(OBSERVATION_FIELDS),))
        self.assertTrue(np.isfinite(vector).all())
        fields = dict(zip(OBSERVATION_FIELDS, vector))
        self.assertEqual(fields["player.lane_known"], 0)
        self.assertEqual(fields["lane0.slot0.present"], 0)
        self.assertEqual(fields["lane0.slot0.ttc_known"], 0)

    def test_unknown_ttc_differs_from_immediate_collision_and_missing_track(self):
        t = Track(1, 1, (0, 0, 10, 10), 0.8, 0.0, 1.0, lane=LaneEstimate((0,), "KNOWN", depth=0.8))
        state = GameState(1.0, (100, 100), tracks=[t], hazards=[Hazard(1)])
        a = dict(zip(OBSERVATION_FIELDS, state.to_vector()))
        self.assertEqual(a["lane0.slot0.present"], 1)
        self.assertEqual(a["lane0.slot0.observed"], 1)
        self.assertEqual(a["lane0.slot0.ttc_known"], 0)
        state.hazards[0].ttc = 0.0
        t.missed = True
        b = dict(zip(OBSERVATION_FIELDS, state.to_vector()))
        self.assertEqual(b["lane0.slot0.ttc_known"], 1)
        self.assertEqual(b["lane0.slot0.ttc"], 0)
        self.assertEqual(b["lane0.slot0.observed"], 0)

    def test_vector_order_is_independent_of_input_list_order(self):
        a = Track(2, 0, (0, 0, 10, 10), 0.9, 0.0, 1.0, lane=LaneEstimate((1,), "KNOWN", depth=0.4))
        b = Track(1, 2, (0, 0, 10, 10), 0.7, 0.0, 1.0, lane=LaneEstimate((1,), "KNOWN", depth=0.8))
        state = GameState(1.0, (100, 100), tracks=[a, b])
        first = state.to_vector()
        state.tracks.reverse()
        np.testing.assert_array_equal(first, state.to_vector())
        values = dict(zip(OBSERVATION_FIELDS, first))
        self.assertEqual(values["lane1.slot0.overhead"], 1)

    def test_unknown_lane_is_not_encoded_as_empty_scene(self):
        state = GameState(1.0, (100, 100), tracks=[Track(1, 0, (0, 0, 10, 10), 0.8, 0.0, 1.0)])
        values = dict(zip(OBSERVATION_FIELDS, state.to_vector()))
        self.assertGreater(values["unknown_lane_tracks"], 0)
        self.assertEqual(values["lane0.slot0.present"], 0)
