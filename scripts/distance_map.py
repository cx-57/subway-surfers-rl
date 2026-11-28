"""Fresh distance-ROI AP50 on reviewed holdouts whose source images are retained."""

import json
from pathlib import Path
import sys
import tempfile
from time import time_ns

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.detection import CLASS_NAMES, nearby_indexes, complete_nms
from scripts.evaluate import match_objects, write_json, check_coverage
from scripts.train import ROOT, file_sha256


def score_records(records, distance=True):
    from ultralytics.utils.metrics import ap_per_class

    true_positive, scores, classes, targets = [], [], [], []
    for record in records:
        truth, predictions = record["truth"], record["predictions"]
        if distance:
            truth = select_nearby(truth, record["image_shape"])
            predictions = select_nearby(predictions, record["image_shape"])
        matched, _ = match_objects(truth, predictions)
        hits = set(matched)
        hits.discard(None)
        for i, p in enumerate(predictions):
            true_positive.append([i in hits])
            scores.append(p["confidence"])
            classes.append(p["class_id"])
        for obj in truth:
            targets.append(obj["class_id"])
    values = ap_per_class(
        np.asarray(true_positive, dtype=bool).reshape(-1, 1),
        np.asarray(scores),
        np.asarray(classes),
        np.asarray(targets),
        names=dict(enumerate(CLASS_NAMES)),
    )
    ap, class_ids = values[5], values[6]
    per_class = {}
    for index, class_id in enumerate(class_ids):
        per_class[str(int(class_id))] = dict(
            objects=targets.count(int(class_id)), ap50=float(ap[index, 0])
        )
    return dict(
        map50=float(ap.mean()) if ap.size else None,
        per_class=per_class,
        predictions=len(scores),
        frames=len(records),
        objects=len(targets),
    )


def select_nearby(objects, image_shape):
    """AP sweeps confidence; use geometry alone when selecting its distance subset."""
    candidates = []
    for obj in objects:
        candidates.append(dict(obj, confidence=1.0))
    selected = []
    for index in nearby_indexes(candidates, image_shape):
        selected.append(objects[index])
    return selected


def main():
    import os

    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    checkpoint = ROOT / "models/barriers_640/weights/best.pt"
    sha = file_sha256(checkpoint)
    output = ROOT / f"models/evaluation/distance_map50_{time_ns()}"
    output.mkdir()
    model = YOLO(checkpoint)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cohorts = {}
    all_records = []
    batches = [
        ("frozen_validation", ROOT / "data/training"),
        ("frozen_regression", ROOT / "data/labeling/test_001"),
    ]
    for name, batch in batches:
        if not (batch / "manifest.json").exists():
            print("Skipping removed dataset:", name, flush=True)
            continue
        previous = ROOT / "models/evaluation" / name
        summary = json.loads((previous / "summary.json").read_text())
        cached = json.loads((previous / "predictions.json").read_text())
        manifest = json.loads((batch / "manifest.json").read_text())
        if summary["checkpoint_sha256"] != sha:
            raise ValueError(f"Checkpoint does not match frozen snapshot: {name}")
        if summary["annotation_revision"] != manifest["annotation_revision"]:
            raise ValueError(f"Annotation revision does not match frozen snapshot: {name}")
        frames, truth, paths = {}, {}, []
        for frame in manifest["frames"]:
            frames[frame["image"]] = frame
        for record in cached:
            truth[record["image"]] = record["truth"]
            paths.append(ROOT / frames[record["image"]]["source"])
        records = []
        seen = []
        print("Fresh inference:", name, len(paths), "frames", flush=True)
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / "images.txt"
            listing.write_text("\n".join(map(str, paths)) + "\n")
            with complete_nms():
                for r in model.predict(
                    str(listing),
                    imgsz=640,
                    batch=2,
                    device=device,
                    conf=0.001,
                    iou=0.7,
                    max_det=1000,
                    augment=False,
                    verbose=False,
                    stream=True,
                ):
                    image = Path(r.path)
                    seen.append(str(image.absolute()))
                    if (
                        file_sha256(image)
                        != frames[image.name]["sha256"]
                    ):
                        raise ValueError(f"Original image changed: {image}")
                    if len(r.boxes) >= 1000:
                        raise RuntimeError("Detection cap reached")
                    predictions = []
                    for class_id, confidence, box in zip(
                        r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()
                    ):
                        predictions.append(
                            dict(class_id=int(class_id), confidence=float(confidence), xyxy=box)
                        )
                    records.append(
                        dict(
                            cohort=name,
                            image=image.name,
                            image_shape=list(r.orig_shape),
                            truth=truth[image.name],
                            predictions=predictions,
                        )
                    )
                    if len(records) % 20 == 0:
                        print("Checked", len(records), "/", len(paths), flush=True)
        check_coverage(list(map(str, paths)), seen)
        cohorts[name] = dict(
            distance=score_records(records), all_distances=score_records(records, False)
        )
        write_json(output / f"{name}_predictions.json", records)
        all_records += records
        print(name, json.dumps(cohorts[name]), flush=True)
    if file_sha256(checkpoint) != sha:
        raise ValueError("Checkpoint changed during evaluation")
    result = dict(
        checkpoint_sha256=sha,
        confidence_floor=0.001,
        matching_iou=0.5,
        nms_iou=0.7,
        cohorts=cohorts,
        combined=score_records(all_records),
        protocol="Strict ROI: apply identical class-specific bottom/width geometry to labels and predictions, independent of confidence. Unmatched retained predictions are false positives, including boundary mismatches. AP uses confidence-ranked one-to-one IoU>=0.5 matching and Ultralytics 101-point integration. Known regression/validation data; not an independent unseen test. No training. Labels reused from verified frozen snapshots.",
    )
    write_json(output / "summary.json", result)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
