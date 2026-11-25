"""Recorded previews preserve source timing and never modify input frames."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from agent.video import VideoRecorder


class VideoTests(unittest.TestCase):
    def test_time_resampling_and_readable_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observation.mp4"
            image = np.full((80, 120, 3), 120, np.uint8)
            with VideoRecorder(path, fps=10) as recorder:
                recorder.write(image, 4.0)
                recorder.write(image, 4.05)
                recorder.write(image, 4.3)
            np.testing.assert_array_equal(image, np.full((80, 120, 3), 120, np.uint8))
            capture = cv2.VideoCapture(str(path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            self.assertTrue(capture.read()[0])
            capture.release()
            info = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(info["frames_written"], 4)
            self.assertAlmostEqual(info["source_duration_seconds"], 0.3)

    def test_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observation.mp4"
            path.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                with VideoRecorder(path):
                    pass
            self.assertEqual(path.read_bytes(), b"keep")

    def test_close_is_terminal_and_empty_recording_creates_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.mp4"
            recorder = VideoRecorder(path)
            recorder.close()
            recorder.close()
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(RuntimeError, "closed"):
                recorder.write(np.zeros((4, 4, 3), np.uint8), 0.0)

    def test_existing_sidecar_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observation.mp4"
            path.with_suffix(".json").write_text("keep")
            with self.assertRaises(FileExistsError):
                VideoRecorder(path)
            self.assertEqual(path.with_suffix(".json").read_text(), "keep")

    def test_context_preserves_primary_failure_when_cleanup_also_fails(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            recorder = VideoRecorder(Path(folder) / "observation.mp4")
            with patch.object(recorder, "close", side_effect=OSError("cleanup")):
                with self.assertRaisesRegex(ValueError, "primary"):
                    with recorder:
                        raise ValueError("primary")
