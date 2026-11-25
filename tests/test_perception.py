import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from agent.perception import Detector


class Array:
    def __init__(self, x):
        self.x = np.array(x)

    def cpu(self):
        return self

    def numpy(self):
        return self.x


class Model:
    names = {0: "train", 1: "ground_barrier", 2: "overhead_barrier", 3: "bush",
             4: "small_obstacle", 5: "pillar"}

    def predict(self, image, **kwargs):
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    cls=Array([1]), conf=Array([0.8]), xyxy=Array([[1, 2, 20, 30]])
                ),
                speed={"preprocess": 1, "inference": 2, "postprocess": 3},
            )
        ]


class PerceptionTests(unittest.TestCase):
    def test_postprocessing_timeout_retries_without_disabling_mps(self):
        from ultralytics.utils import LOGGER

        class ColdStart(Model):
            calls = 0

            def predict(self, image, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    LOGGER.warning("NMS time limit 2.050s exceeded")
                return super().predict(image, **kwargs)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f, patch(
            "agent.perception._load_model", return_value=ColdStart()
        ):
            detector = Detector(f.name, device="mps")
            result = detector.detect(np.zeros((30, 40, 3), np.uint8), 0.0)
            self.assertEqual(result[0].class_id, 1)
            self.assertEqual(detector.device, "mps")
            self.assertIsNone(detector.fallback_reason)
            self.assertEqual(detector.model.calls, 2)

    def test_repeated_postprocessing_timeout_fails_without_cpu_fallback(self):
        from ultralytics.utils import LOGGER

        class TimedOut(Model):
            calls = 0

            def predict(self, image, **kwargs):
                self.calls += 1
                LOGGER.warning("NMS time limit 2.050s exceeded")
                return super().predict(image, **kwargs)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f, patch(
            "agent.perception._load_model", return_value=TimedOut()
        ):
            detector = Detector(f.name, device="mps")
            with self.assertRaisesRegex(RuntimeError, "Incomplete NMS"):
                detector.detect(np.zeros((30, 40, 3), np.uint8), 0.0)
            self.assertEqual(detector.device, "mps")
            self.assertEqual(detector.model.calls, 2)

    def test_six_class_metadata_and_legacy_checkpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as file, patch(
            "agent.perception._load_model", return_value=Model()
        ):
            self.assertEqual(len(Detector(file.name, device="cpu").class_names), 6)

        class OldModel(Model):
            names = {0: "train", 1: "ground_barrier", 2: "overhead_barrier"}

        with tempfile.NamedTemporaryFile(suffix=".pt") as file, patch(
            "agent.perception._load_model", return_value=OldModel()
        ):
            self.assertEqual(len(Detector(file.name, device="cpu").class_names), 3)

    def test_mismatched_checkpoint_is_rejected(self):
        class BadModel(Model):
            names = {0: "pillar", 1: "ground_barrier", 2: "train"}

        with tempfile.NamedTemporaryFile(suffix=".pt") as file, patch(
            "agent.perception._load_model", return_value=BadModel()
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                Detector(file.name, device="cpu")

    def test_missing_path(self):
        with self.assertRaises(ValueError):
            Detector("/missing/checkpoint.pt")

    def test_schema_dimensions_timestamp(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as f, patch(
            "agent.perception._load_model", return_value=Model()
        ):
            detector = Detector(f.name, device="cpu")
            result = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8), 12.5)
            self.assertEqual(result[0].timestamp, 12.5)
            self.assertEqual(result[0].bbox, (1.0, 2.0, 20.0, 30.0))
            self.assertEqual(detector.image_shape, (60, 80))
            self.assertEqual(detector.last_timings["inference_ms"], 2)
            with self.assertRaises(ValueError):
                detector.detect(None, 13)

    def test_mps_failure_falls_back_once(self):
        class FailingMPS(Model):
            def to(self, device):
                self.device = device

            def predict(self, image, **kwargs):
                if kwargs["device"] == "mps":
                    raise RuntimeError("MPS unavailable")
                return super().predict(image, **kwargs)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f, patch(
            "agent.perception._load_model", return_value=FailingMPS()
        ):
            detector = Detector(f.name, device="mps")
            detector.detect(np.zeros((30, 40, 3), dtype=np.uint8), 0)
            self.assertEqual(detector.device, "cpu")
            self.assertIn("MPS", detector.fallback_reason)
