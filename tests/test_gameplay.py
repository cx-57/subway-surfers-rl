import contextlib
import importlib.util
import io
import json
import os
import signal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scripts import gameplay as g


class Screen:
    monitors = [{}, dict(left=-200, top=0, width=200, height=100)]

    def __init__(self, error=None):
        self.error = error
        self.regions = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def grab(self, region):
        self.regions.append(dict(region))
        if self.error and len(self.regions) > 1:
            raise self.error
        # Simulate a Retina screenshot with twice as many image pixels.
        return np.zeros((region["height"] * 2, region["width"] * 2, 4), np.uint8)


class GameplayTests(unittest.TestCase):
    def run_collector(self, output, screen=None, selection=None, fps=2):
        screen = screen or Screen()
        # A fake clock makes the capture loop run without waiting in real time.
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        def select(preview):
            if selection is not None:
                return selection
            height, width = preview.shape[:2]
            return width // 4, height // 4, width // 2, height // 2

        with (
            patch.multiple(
                g, FPS=fps, DURATION=1, START_DELAY=0, SELECTION_DELAY=0, OUTPUT=Path(output)
            ),
            patch.object(g.mss, "MSS", return_value=screen),
            patch.object(g.time, "perf_counter", side_effect=lambda: now[0]),
            patch.object(g.time, "sleep", side_effect=sleep),
            patch.object(g, "select_area", side_effect=select),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            g.main()
        return screen

    def test_captures_readable_images_and_maps_retina_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = self.run_collector(tmp)
            frames = list(Path(tmp).glob("session_*/frame_*.jpg"))
            self.assertEqual(len(frames), 2)
            self.assertTrue(all(cv2.imread(str(frame)) is not None for frame in frames))
            names = []
            for frame in sorted(frames):
                names.append(frame.name)
            self.assertEqual(names, ["frame_000001.jpg", "frame_000002.jpg"])
            region = screen.regions[1]
            for key, expected in dict(left=-150, top=25, width=100, height=50).items():
                self.assertAlmostEqual(region[key], expected, delta=1)
            self.assertTrue(screen.closed)
            metadata = json.loads(next(Path(tmp).glob("session_*/metadata.json")).read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["frames_saved"], len(frames))
            self.assertEqual(metadata["requested_fps"], 2)
            self.assertEqual(metadata["requested_duration"], 1)
            self.assertEqual(metadata["region"], region)
            self.assertIsNotNone(metadata["start_timestamp"])
            self.assertIsNotNone(metadata["end_timestamp"])

    def test_twenty_fps_saves_twenty_frames_in_one_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_collector(tmp, fps=20)
            metadata = json.loads(next(Path(tmp).glob("session_*/metadata.json")).read_text())
            self.assertEqual(metadata["frames_saved"], 20)
            self.assertEqual(len(list(Path(tmp).glob("session_*/frame_*.jpg"))), 20)
            self.assertAlmostEqual(metadata["actual_fps"], 20)

    def select_with_clicks(self, steps):
        callbacks = []

        def next_key(delay):
            clicks, key = steps.pop(0)
            for x, y in clicks:
                callbacks[0](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            return key

        with (
            patch.object(cv2, "namedWindow"),
            patch.object(
                cv2,
                "setMouseCallback",
                side_effect=lambda title, callback: callbacks.append(callback),
            ),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "waitKey", side_effect=next_key),
            patch.object(cv2, "getWindowProperty", return_value=1),
            patch.object(cv2, "destroyAllWindows"),
        ):
            return g.select_area(np.zeros((100, 100, 3), np.uint8))

    def test_two_click_selection_works_in_either_direction(self):
        self.assertEqual(self.select_with_clicks([([(80, 90), (10, 20)], 13)]), (10, 20, 70, 70))

    def test_two_click_selection_can_be_redone(self):
        self.assertEqual(
            self.select_with_clicks([([(1, 1), (4, 4)], ord("r")), ([(10, 20), (80, 90)], 13)]),
            (10, 20, 70, 70),
        )

    def test_two_click_selection_can_be_cancelled(self):
        self.assertIsNone(self.select_with_clicks([([], 27)]))

    def test_cancel_creates_no_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = self.run_collector(tmp, selection=(0, 0, 0, 0))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertEqual(len(screen.regions), 1)
            self.assertTrue(screen.closed)

    def test_recordings_never_replace_previous_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_collector(tmp)
            original = {}
            for path in Path(tmp).glob("session_*/*.jpg"):
                original[path] = path.read_bytes()
            self.run_collector(tmp)
            self.assertEqual(len(list(Path(tmp).glob("session_*"))), 2)
            self.assertTrue(all(path.read_bytes() == data for path, data in original.items()))

    def test_failed_write_removes_partial_image_and_closes_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = Screen()

            def fail_write(filename, *args):
                Path(filename).write_bytes(b"partial")
                return False

            with patch.object(cv2, "imwrite", side_effect=fail_write), self.assertRaises(OSError):
                self.run_collector(tmp, screen)
            self.assertEqual(list(Path(tmp).glob("session_*/*.jpg")), [])
            self.assertTrue(screen.closed)
            metadata = json.loads(next(Path(tmp).glob("session_*/metadata.json")).read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["frames_saved"], 0)
            self.assertIn("Could not save", metadata["reason"])

    def test_keyboard_interrupt_closes_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = self.run_collector(tmp, Screen(error=KeyboardInterrupt()))
            self.assertTrue(screen.closed)
            metadata = json.loads(next(Path(tmp).glob("session_*/metadata.json")).read_text())
            self.assertEqual(metadata["status"], "interrupted")
            self.assertEqual(metadata["frames_saved"], 0)

    def test_image_is_published_only_after_a_successful_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_write = cv2.imwrite
            real_link = os.link
            published = []

            def write(filename, *args):
                self.assertFalse((folder / "frame_000001.jpg").exists())
                return real_write(filename, *args)

            def publish(source, destination):
                self.assertIsNotNone(cv2.imread(str(source)))
                real_link(source, destination)
                published.append(destination)

            folder = Path(tmp) / "writer"
            folder.mkdir()
            with patch.object(cv2, "imwrite", side_effect=write), patch.object(
                os, "link", side_effect=publish
            ):
                g.save_image(folder / "frame_000001.jpg", np.zeros((8, 8, 3), np.uint8))
            self.assertEqual(len(published), 1)
            self.assertEqual(len(list(folder.iterdir())), 1)
            before = published[0].read_bytes()
            with self.assertRaises(FileExistsError):
                g.save_image(published[0], np.ones((8, 8, 3), np.uint8))
            self.assertEqual(published[0].read_bytes(), before)

    @unittest.skipUnless(hasattr(signal, "pthread_sigmask"), "requires macOS/POSIX signals")
    def test_ctrl_c_after_publication_keeps_metadata_count_correct(self):
        real_link = os.link

        def interrupt(source, destination):
            real_link(source, destination)
            signal.raise_signal(signal.SIGINT)

        with tempfile.TemporaryDirectory() as tmp, patch.object(os, "link", side_effect=interrupt):
            self.run_collector(tmp)
            metadata = json.loads(next(Path(tmp).glob("session_*/metadata.json")).read_text())
            frames = list(Path(tmp).glob("session_*/frame_*.jpg"))
            self.assertEqual(metadata["status"], "interrupted")
            self.assertEqual(metadata["frames_saved"], len(frames))
            self.assertEqual(len(frames), 1)

    def test_metadata_write_failure_preserves_previous_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            original = {"status": "recording", "frames_saved": 0}
            g.save_metadata(folder, original)
            with patch.object(Path, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    g.save_metadata(folder, {"status": "completed", "frames_saved": 2})
            self.assertEqual(json.loads((folder / "metadata.json").read_text()), original)
            names = []
            for path in folder.iterdir():
                names.append(path.name)
            self.assertEqual(names, ["metadata.json"])

    def test_import_does_not_start_capture(self):
        spec = importlib.util.spec_from_file_location("collector_import", g.__file__)
        with patch.object(
            g.mss, "MSS", side_effect=AssertionError("Capture started during import")
        ):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))


if __name__ == "__main__":
    unittest.main()
