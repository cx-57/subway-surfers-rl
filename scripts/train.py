"""Train a small obstacle detector. Run with .venv/bin/python scripts/train.py."""

import json
import hashlib
import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.detection import CLASS_NAMES, complete_nms

# Keep settings here instead of adding command-line options.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/training"
IMAGE_SIZE = 640
BATCH_SIZE = 2
FRAME_STEP = 3
TRAIN_END = 780
VAL_START = 850
# Fine-tune the existing model; a separate run keeps the first model intact.
MODEL = ROOT / "models/barriers_640/weights/best.pt"
LEARNING_RATE = 0.0003


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_labels(path):
    """Read YOLO labels; reject invalid classes, sizes and visible bounds."""
    rows = []
    for line in path.read_text().splitlines():
        values = list(map(float, line.split()))
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO row in {path}: {line}")
        for value in values:
            if not math.isfinite(value):
                raise ValueError(f"Invalid YOLO row in {path}: {line}")
        c, x, y, w, h = values
        if (
            c != int(c) or not 0 <= c < len(CLASS_NAMES)
            or w <= 0
            or h <= 0
            or min(x - w / 2, y - h / 2) < -1e-6
            or max(x + w / 2, y + h / 2) > 1 + 1e-6
        ):
            raise ValueError(f"Invalid YOLO bounds/class in {path}: {line}")
        rows.append(values)
    return rows


def verify_images(batch=DATA):
    """Compare every original screenshot with its recorded SHA256 fingerprint."""
    manifest = json.loads((batch / "manifest.json").read_text())
    frames = manifest.get("all_source_frames", manifest["frames"])
    for frame in frames:
        paths = [batch / "images" / frame["image"]]
        if "source" in frame:
            source = ROOT / frame['source']
            # GitHub shares the curated original; the full raw clip is optional.
            if source.is_file() and source != paths[0]:
                paths.append(source)
        for path in paths:
            if file_sha256(path) != frame["sha256"]:
                raise ValueError(f"Original image changed: {path}")
    return len(frames)


def training_settings():
    """One conservative training configuration; existing checkpoints stay intact."""
    return dict(
        epochs=12,
        patience=5,
        time=None,
        imgsz=IMAGE_SIZE,
        batch=DATA_SIZE,
        seed=42,
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        warmup_epochs=1,
        mosaic=0,
        close_mosaic=3,
        scale=0.2,
        translate=0.05,
        mixup=0,
        cutmix=0,
        degrees=0,
        perspective=0,
        flipud=0,
    )


def prepare_data(batch=DATA, output=DATA):
    """Write image lists without copying images or changing their labels."""
    manifest = json.loads((batch / "manifest.json").read_text())
    if "splits" in manifest:
        # These images were already sampled before cleanup. Never sample them twice.
        selected = manifest["splits"]
        if set(selected["train"]) & set(selected["val"]):
            raise ValueError("Training and validation splits overlap")
        by_name = {}
        for frame in manifest["frames"]:
            by_name[frame["image"]] = frame
        groups = {}
        for split in ("train", "val"):
            groups[split] = []
            names = selected[split]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate image in split")
            for name in names:
                frame = by_name[name]
                if frame.get("annotation_uncertain") or not frame.get("training_eligible", True):
                    raise ValueError("Excluded frame in saved split: " + name)
                groups[split].append(frame)
        training, validation = groups["train"], groups["val"]
    else:
        frames = []
        for frame in manifest["frames"]:
            if frame["training_eligible"] is True:
                frames.append(frame)
        frames.sort(key=source_frame_number)
        # Neighboring frames are almost identical. Keep one in three for training.
        training, validation = [], []
        for frame in frames:
            if frame["source_frame"] <= TRAIN_END:
                training.append(frame)
            if frame["source_frame"] >= VAL_START and not frame.get("annotation_uncertain", False):
                validation.append(frame)
        # A later time block and a gap reduce overlap; this is still one session.
        # Filter after sampling so an uncertain frame does not shift later selections.
        sampled = training[::FRAME_STEP]
        training = []
        for frame in sampled:
            if not frame.get("annotation_uncertain", False):
                training.append(frame)
    if not training or not validation:
        raise ValueError("Need usable frames in both training and validation time ranges.")
    train_hashes = set()
    for frame in training:
        if "sha256" in frame:
            train_hashes.add(frame["sha256"])
    for frame in validation:
        if "sha256" in frame and frame["sha256"] in train_hashes:
            raise ValueError("An identical screenshot occurs in training and validation.")

    # Force a fresh label scan even when coordinate edits keep the same file size.
    (batch / "labels.cache").unlink(missing_ok=True)

    output.mkdir(parents=True, exist_ok=True)
    for name, group in [("train", training), ("val", validation)]:
        images = []
        for frame in group:
            image = batch / "images" / frame["image"]
            label = batch / "labels" / Path(frame["image"]).with_suffix(".txt")
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f'Missing image or label: {frame["image"]}')
            if not read_labels(label) and not frame.get("verified_background", False):
                raise ValueError(f"Empty label needs a visual background review: {label}")
            # Keep /images/ in the path so YOLO finds the matching /labels/ file.
            # YOLO resolves ./ paths relative to this split file, on any computer.
            images.append('./' + os.path.relpath(image.absolute(), output.absolute()))
        (output / f"{name}.txt").write_text("\n".join(images) + "\n")

    # JSON is valid YAML and safely handles spaces in paths and class names.
    config = output / "dataset.yaml"
    config.write_text(
        json.dumps(
            dict(
                train='train.txt',
                val='val.txt',
                names=(batch / "classes.txt").read_text().splitlines(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Training: {len(training)} frames; validation: {len(validation)} frames.")
    print("Validation includes an older chronological block and a continuation clip; collect a truly independent test recording.")
    return config


def source_frame_number(frame):
    return frame["source_frame"]


def main():
    # Import the training libraries only when actually starting a run.
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    config = training_settings()
    print(f"Verified {verify_images()} unchanged original screenshots.")
    dataset = prepare_data()
    files = [DATA / "train.txt", DATA / "val.txt", DATA / "manifest.json"]
    if (DATA / "partial_audit.json").exists():
        files.append(DATA / "partial_audit.json")
    files += sorted((DATA / "labels").glob("*.txt"))
    fingerprints = {}
    for path in files:
        fingerprints[str(path.relative_to(ROOT))] = file_sha256(path)
    record = dict(
        settings=config,
        initial_checkpoint=str(MODEL),
        status="running",
        training_images=(DATA / "train.txt").read_text().splitlines(),
        validation_images=(DATA / "val.txt").read_text().splitlines(),
        checkpoint_sha256=file_sha256(MODEL),
        inputs=fingerprints,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    model = YOLO(MODEL)

    def save_run(trainer):
        (trainer.save_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")

    def release_gpu_memory(trainer):
        torch.mps.empty_cache()

    model.add_callback("on_train_start", save_run)
    # Release unused GPU allocations between epochs on this 8 GB Mac.
    if device == "mps":
        model.add_callback("on_train_epoch_end", release_gpu_memory)
    with complete_nms():
        model.train(
            data=str(dataset),
            **config,
            device=device,
            # Avoid extra worker processes and image caches on this 8 GB Mac.
            workers=0,
            cache=False,
            project=str(models),
            name="training",
            plots=True,
        )
    for path in files:
        if (
            file_sha256(path)
            != record["inputs"][str(path.relative_to(ROOT))]
        ):
            raise ValueError("Dataset inputs changed during training; do not compare this run.")
    record.update(status="complete", epochs_completed=model.trainer.epoch + 1)
    save_run(model.trainer)
    print(f"Best checkpoint: {model.trainer.best}")


if __name__ == "__main__":
    main()
