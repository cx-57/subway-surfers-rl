"""Protect annotation geometry, original pixels and chronological data splits."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import train


class DatasetTests(unittest.TestCase):
    def test_shared_curated_image_can_be_verified_without_optional_raw_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            (batch / 'images').mkdir()
            image = batch / 'images/frame.jpg'
            image.write_bytes(b'original pixels')
            frame = dict(image=image.name, source=str(batch/'unshared_raw/frame.jpg'),
                         sha256=hashlib.sha256(image.read_bytes()).hexdigest())
            (batch/'manifest.json').write_text(json.dumps(dict(frames=[frame])))
            self.assertEqual(train.verify_images(batch), 1)

    def test_new_obstacle_classes_are_valid_but_unknown_classes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            label = Path(folder) / 'objects.txt'
            label.write_text('3 .5 .5 .2 .2\n4 .5 .5 .2 .2\n5 .5 .5 .2 .2\n')
            self.assertEqual(len(train.read_labels(label)), 3)
            label.write_text('6 .5 .5 .2 .2\n')
            with self.assertRaises(ValueError):
                train.read_labels(label)

    def test_valid_edge_boxes_and_empty_background_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            label = Path(folder) / "frame.txt"
            label.write_text("0 0.5 0.5 1 1\n2 0.75 0.05 0.5 0.1\n")
            self.assertEqual(len(train.read_labels(label)), 2)
            label.write_text("")
            self.assertEqual(train.read_labels(label), [])

    def test_invalid_class_nan_size_and_out_of_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            label = Path(folder) / "frame.txt"
            for row in [
                "6 .5 .5 .2 .2",
                "1.5 .5 .5 .2 .2",
                "2 nan .5 .2 .2",
                "2 .5 .5 0 .1",
                "0 .05 .5 .2 .2",
                "1 .9 .5 .3 .2",
                "2 .5 .5 .1",
            ]:
                with self.subTest(row=row):
                    label.write_text(row)
                    with self.assertRaises(ValueError):
                        train.read_labels(label)

    def test_source_hash_detects_changes_without_rewriting_image(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            (batch / "images").mkdir()
            image = batch / "images/frame.jpg"
            image.write_bytes(b"original image bytes")
            manifest = {
                "frames": [
                    {"image": image.name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
                ]
            }
            (batch / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(train.verify_images(batch), 1)
            image.write_bytes(b"different image bytes")
            with self.assertRaises(ValueError):
                train.verify_images(batch)
            self.assertEqual(image.read_bytes(), b"different image bytes")

    def test_uncertain_frames_are_excluded_without_shifting_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "batch"
            (batch / "images").mkdir(parents=True)
            (batch / "labels").mkdir()
            frames = []
            for n in [105, 108, 111, 114, 117, 120, 123, 850, 853]:
                name = f"frame_{n:06}.jpg"
                (batch / "images" / name).write_bytes(b"original")
                (batch / "labels" / name.replace(".jpg", ".txt")).write_text("")
                frames.append(
                    dict(
                        image=name,
                        source_frame=n,
                        training_eligible=True,
                        annotation_uncertain=n in [114, 850],
                        verified_background=True,
                    )
                )
            (batch / "classes.txt").write_text("train\nground_barrier\noverhead_barrier\n")
            (batch / "manifest.json").write_text(json.dumps({"frames": frames}))
            output = Path(folder) / "prepared"
            train.prepare_data(batch, output)
            training, validation = [], []
            for filename in (output / "train.txt").read_text().splitlines():
                training.append(Path(filename).name)
            for filename in (output / "val.txt").read_text().splitlines():
                validation.append(Path(filename).name)
            self.assertEqual(training, ["frame_000105.jpg", "frame_000123.jpg"])
            self.assertEqual(validation, ["frame_000853.jpg"])
            before = (output / "dataset.yaml").read_bytes()
            train.prepare_data(batch, output)
            self.assertEqual(before, (output / "dataset.yaml").read_bytes())
            # Empty labels without a visual background review are not negatives.
            frames[0]["verified_background"] = False
            (batch / "manifest.json").write_text(json.dumps({"frames": frames}))
            with self.assertRaises(ValueError):
                train.prepare_data(batch, output)
            frames[0]["verified_background"] = True
            frames[0]["sha256"] = frames[-1]["sha256"] = "same-image-hash"
            (batch / "manifest.json").write_text(json.dumps({"frames": frames}))
            with self.assertRaisesRegex(ValueError, "identical screenshot"):
                train.prepare_data(batch, output)


if __name__ == "__main__":
    unittest.main()
