import unittest
from agent.state import Detection
from agent.tracking import ObstacleTracker


class TrackingTests(unittest.TestCase):
    def test_invalid_limits_fail_before_tracking(self):
        for settings in [
            {"max_age": float("nan")},
            {"max_age": float("inf")},
            {"history_size": 2.5},
            {"max_tracks": True},
            {"iou_threshold": float("nan")},
            {"iou_threshold": 1.1},
            {"ambiguity_margin": -0.1},
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ObstacleTracker(**settings)

    def d(self, x, t=0, c=0):
        return Detection(c, 0.9, (x, 10, x + 20, 40), t)

    def test_motion_missing_expiration(self):
        tr = ObstacleTracker(max_age=0.5, history_size=3)
        ident = tr.update([self.d(10)], 0, (100, 100))[0].id
        result = tr.update([self.d(12, 0.1)], 0.1, (100, 100))[0]
        self.assertEqual(ident, result.id)
        self.assertEqual(tr.update([], 0.3, (100, 100))[0].missed, True)
        self.assertEqual(tr.update([], 0.7, (100, 100)), [])
        self.assertEqual(tr.metrics["expired"], 1)

    def test_ambiguity_and_class(self):
        tr = ObstacleTracker()
        tr.update([self.d(10), self.d(10)], 0, (100, 100))
        tracks = tr.update([self.d(11, 0.1)], 0.1, (100, 100))
        self.assertTrue(all(t.association_ambiguous for t in tracks))
        self.assertGreater(tr.metrics["ambiguous_associations"], 0)

    def test_irregular_time_bounded_history(self):
        tr = ObstacleTracker(history_size=2)
        for t, y in [(0, 10), (0.1, 12), (0.4, 18)]:
            result = tr.update([Detection(0, 0.9, (10, y, 30, y + 20), t)], t, (100, 100))
        self.assertAlmostEqual(result[0].velocity_y, 20)
        self.assertEqual(len(result[0].history), 2)
        with self.assertRaises(ValueError):
            tr.update([], 0.2, (100, 100))

    def test_class_separation_and_capacity(self):
        tracker = ObstacleTracker(max_tracks=2)
        first = tracker.update([self.d(10)], 0, (100, 100))[0].id
        result = tracker.update([self.d(10, 0.1, c=1)], 0.1, (100, 100))
        self.assertNotEqual(first, next(t.id for t in result if not t.missed))
        result = tracker.update([self.d(70, 0.2, c=2)], 0.2, (100, 100))
        self.assertEqual(len(result), 2)
        self.assertEqual(tracker.metrics["capacity_evictions"], 1)
