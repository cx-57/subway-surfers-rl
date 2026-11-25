import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.distance_map import score_records


class DistanceMapTests(unittest.TestCase):
    def test_ap_sweeps_below_live_cutoff_and_excludes_distant_objects(self):
        record = dict(
            image_shape=[100, 100],
            truth=[
                dict(class_id=1, xyxy=[40, 60, 60, 90]),
                dict(class_id=2, xyxy=[45, 10, 55, 20]),
            ],
            predictions=[dict(class_id=1, confidence=0.2, xyxy=[40, 60, 60, 90])],
        )
        result = score_records([record])
        self.assertGreater(result["per_class"]["1"]["ap50"], 0.99)
        self.assertEqual(result["per_class"]["1"]["objects"], 1)
        self.assertNotIn("2", result["per_class"])

    def test_unmatched_nearby_prediction_counts_as_false_positive(self):
        record = dict(
            image_shape=[100, 100],
            truth=[dict(class_id=1, xyxy=[40, 60, 60, 90])],
            predictions=[
                dict(class_id=1, confidence=0.9, xyxy=[10, 60, 30, 90]),
                dict(class_id=1, confidence=0.8, xyxy=[40, 60, 60, 90]),
            ],
        )
        result = score_records([record])
        self.assertLess(result["per_class"]["1"]["ap50"], 0.6)
        self.assertEqual(result["predictions"], 2)

    def test_snapshot_integrity_failure_is_a_runtime_error(self):
        from scripts import distance_map

        for changed in ("checkpoint", "annotation"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "models/barriers_640/weights/best.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"unchanged test checkpoint")
                frozen = root / "models/evaluation/frozen_validation"
                frozen.mkdir(parents=True)
                batch = root / "data/training"
                batch.mkdir(parents=True)
                checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                (frozen / "summary.json").write_text(
                    json.dumps(
                        dict(
                            checkpoint_sha256="wrong" if changed == "checkpoint" else checksum,
                            annotation_revision="wrong" if changed == "annotation" else 1,
                        )
                    )
                )
                (frozen / "predictions.json").write_text("[]")
                (batch / "manifest.json").write_text(
                    json.dumps(dict(annotation_revision=1, frames=[]))
                )
                with patch.object(distance_map, "ROOT", root), patch("ultralytics.YOLO"), patch(
                    "ultralytics.settings.update"
                ):
                    with self.assertRaisesRegex(ValueError, "Checkpoint|Annotation"):
                        distance_map.main()
