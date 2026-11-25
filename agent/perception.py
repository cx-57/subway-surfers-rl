"""One retained detector instance for replay and live BGR frames."""

import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from .state import Detection
from .detection import CONFIDENCE, IncompleteNMS, complete_nms


def _load_model(path):
    # Ultralytics can otherwise install optional packages during import/use.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    return YOLO(str(path))


class Detector:
    def __init__(
        self,
        checkpoint="models/barriers_640/weights/best.pt",
        device="auto",
        imgsz=640,
        confidence=CONFIDENCE,
    ):
        path = Path(checkpoint)
        if not path.is_file():
            raise ValueError(f"Detector checkpoint does not exist: {path}")
        if imgsz <= 0 or not 0 <= confidence <= 1:
            raise ValueError("Invalid detector resolution/confidence")
        if device == "auto":
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device, self.imgsz, self.confidence = device, imgsz, confidence
        self.last_timings = {}
        self.image_shape = None
        self.fallback_reason = None
        try:
            self.model = _load_model(path)
        except Exception as exc:
            raise RuntimeError(f"Cannot load detector {path}: {exc}") from exc

    def detect(self, image, timestamp):
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or not image.size
            or image.dtype != np.uint8
        ):
            raise ValueError("Expected nonempty uint8 BGR frame")
        if not math.isfinite(timestamp):
            raise ValueError("Timestamp must be finite")
        self.image_shape = image.shape[:2]
        start = perf_counter()

        def predict():
            # A cold postprocessing timeout is not evidence of an MPS backend error.
            for attempt in range(2):
                try:
                    with complete_nms():
                        return self.model.predict(
                            image,
                            imgsz=self.imgsz,
                            conf=self.confidence,
                            device=self.device,
                            verbose=False,
                        )[0]
                except IncompleteNMS:
                    if attempt:
                        raise

        try:
            result = predict()
        except IncompleteNMS:
            raise
        except Exception as exc:
            if self.device != "mps":
                raise RuntimeError(f"Detector inference failed: {exc}") from exc
            self.fallback_reason = str(exc)
            self.device = "cpu"
            self.model.to("cpu")
            try:
                result = predict()
            except Exception as cpu_exc:
                raise RuntimeError(f"Detector CPU fallback failed: {cpu_exc}") from cpu_exc
        boxes = result.boxes
        detections = []
        # Copy model tensors into ordinary Python values for the rest of the engine.
        for class_id, confidence, box in zip(
            boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()
        ):
            detection = Detection(
                int(class_id), float(confidence), tuple(map(float, box)), float(timestamp)
            )
            detections.append(detection)
        self.last_timings = {}
        for stage in ("preprocess", "inference", "postprocess"):
            self.last_timings[stage + "_ms"] = float(result.speed.get(stage, 0))
        self.last_timings.update(total_ms=(perf_counter() - start) * 1000, device=self.device)
        return detections
