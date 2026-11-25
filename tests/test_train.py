"""Check that training never includes excluded or neighboring holdout frames."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts import train


class TrainingDataTests(unittest.TestCase):
    def test_cleaned_dataset_preserves_selected_frames_without_resampling(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "kept"
            (batch / "images").mkdir(parents=True)
            (batch / "labels").mkdir()
            frames = []
            for number in (105, 108, 111, 850):
                name = f"frame_{number:06d}.jpg"
                (batch / "images" / name).write_bytes(b"original")
                (batch / "labels" / Path(name).with_suffix(".txt")).write_text("")
                frames.append(
                    dict(
                        image=name,
                        source_frame=number,
                        training_eligible=True,
                        verified_background=True,
                    )
                )
            splits = dict(
                train=["frame_000105.jpg", "frame_000108.jpg", "frame_000111.jpg"],
                val=["frame_000850.jpg"],
            )
            (batch / "manifest.json").write_text(json.dumps(dict(frames=frames, splits=splits)))
            (batch / "classes.txt").write_text("train\nground_barrier\noverhead_barrier\n")
            train.prepare_data(batch, batch)
            self.assertEqual(len((batch / "train.txt").read_text().splitlines()), 3)
            splits["val"] = ["frame_000105.jpg"]
            (batch / "manifest.json").write_text(json.dumps(dict(frames=frames, splits=splits)))
            with self.assertRaisesRegex(ValueError, "overlap"):
                train.prepare_data(batch, batch)

    def test_excludes_bad_frames_and_separates_time_ranges(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "batch"
            (batch / "images").mkdir(parents=True)
            (batch / "labels").mkdir()
            frames = []
            # The excluded frame and the gap must never appear in either list.
            for number, eligible in [
                (105, True),
                (108, True),
                (111, True),
                (114, True),
                (120, False),
                (780, True),
                (783, True),
                (849, True),
                (850, True),
                (853, True),
                (900, False),
            ]:
                name = f"frame_{number:06}.jpg"
                (batch / "images" / name).write_bytes(b"image")
                (batch / "labels" / name.replace(".jpg", ".txt")).write_text("0 0.5 0.5 0.1 0.2\n")
                frames.append(dict(image=name, source_frame=number, training_eligible=eligible))
            (batch / "manifest.json").write_text(json.dumps({"frames": frames}))
            (batch / "classes.txt").write_text("train\nground_barrier\noverhead_barrier\n")
            # YOLO's scan cache can miss coordinate edits with the same file size.
            stale_cache = batch / "labels.cache"
            stale_cache.write_bytes(b"old annotation scan")

            output = Path(folder) / "training"
            config = train.prepare_data(batch, output)

            training = []
            for p in (output / "train.txt").read_text().splitlines():
                training.append(Path(p).name)
            validation = []
            for p in (output / "val.txt").read_text().splitlines():
                validation.append(Path(p).name)
            self.assertEqual(training, ["frame_000105.jpg", "frame_000114.jpg"])
            self.assertEqual(validation, ["frame_000850.jpg", "frame_000853.jpg"])
            self.assertTrue(config.is_file())
            self.assertFalse(stale_cache.exists())
            # Preparing training data must leave the annotations untouched.
            self.assertEqual((batch / "labels/frame_000105.txt").read_text(), "0 0.5 0.5 0.1 0.2\n")

    def test_empty_usable_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "batch"
            batch.mkdir()
            (batch / "manifest.json").write_text(json.dumps({"frames": []}))
            with self.assertRaises(ValueError):
                train.prepare_data(batch, Path(folder) / "training")


if __name__ == "__main__":
    unittest.main()
