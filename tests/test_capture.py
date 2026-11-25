import time
import unittest
import threading
import numpy as np
from agent.capture import Frame, LatestFrame


class CaptureTests(unittest.TestCase):
    def test_bounded_frame_iterator_keeps_final_frame_and_stops_at_eof(self):
        values = iter([Frame(np.zeros((2, 2, 3), dtype=np.uint8), 2, 3), None])
        with LatestFrame(lambda: next(values)) as capture:
            frames = list(capture.frames(duration=1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].timestamp, 2)

    def test_zero_duration_does_not_read_a_frame(self):
        capture = LatestFrame(lambda: None)
        self.assertEqual(list(capture.frames(duration=0)), [])

    def test_source_resources_close_on_the_capture_thread(self):
        class Source:
            def __call__(self):
                self.thread = threading.get_ident()
                return None

            def close(self):
                self.closed_on = threading.get_ident()

        source = Source()
        with LatestFrame(source) as capture:
            with self.assertRaises(EOFError):
                capture.get(timeout=1)
        self.assertEqual(source.closed_on, source.thread)
        self.assertNotEqual(source.closed_on, threading.get_ident())

    def test_latest_slot_and_shutdown(self):
        def source():
            time.sleep(0.001)
            return np.zeros((3, 4, 3), dtype=np.uint8)

        with LatestFrame(source) as capture:
            time.sleep(0.03)
            frame = capture.get(timeout=1)
            self.assertEqual(frame.image.shape, (3, 4, 3))
            self.assertGreater(capture.dropped, 0)
        self.assertFalse(capture.alive)

    def test_failure_propagates(self):
        def source():
            raise ValueError("permission")

        with LatestFrame(source) as capture:
            with self.assertRaisesRegex(RuntimeError, "capture"):
                capture.get(timeout=1)

    def test_eof_consumes_final_frame(self):
        values = iter([Frame(np.zeros((2, 2, 3), dtype=np.uint8), 2, 3), None])
        with LatestFrame(lambda: next(values)) as capture:
            self.assertEqual(capture.get(timeout=1).timestamp, 2)
            with self.assertRaises(EOFError):
                capture.get(timeout=1)

    def test_cleanup_does_not_mask_primary_failure(self):
        from unittest.mock import patch

        capture = LatestFrame(lambda: None)
        with patch.object(capture, "close", side_effect=RuntimeError("cleanup")):
            with self.assertRaisesRegex(ValueError, "primary"):
                with capture:
                    raise ValueError("primary")
