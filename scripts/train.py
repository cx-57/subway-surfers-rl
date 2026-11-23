"""Train a small obstacle detector. Run with .venv/bin/python scripts/train.py."""

import json
from pathlib import Path

# Keep settings here instead of adding command-line options.
ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'data/labeling/batch_001'
DATA = ROOT / 'data/training'
MINUTES = 12
IMAGE_SIZE = 640
BATCH_SIZE = 4
FRAME_STEP = 3
TRAIN_END = 780
VAL_START = 850
# Fine-tune the existing model; a separate run keeps the first model intact.
MODEL = ROOT / 'models/baseline/weights/best.pt'
RUN_NAME = 'barriers_640'
LEARNING_RATE = 0.0003


def prepare_data(batch=BATCH, output=DATA):
    """Write image lists without copying images or changing their labels."""
    manifest = json.loads((batch / 'manifest.json').read_text())
    frames = sorted((f for f in manifest['frames'] if f['training_eligible'] is True),
                    key=lambda f: f['source_frame'])
    # Neighboring frames are almost identical. Keep one in three for training.
    training = [f for f in frames if f['source_frame'] <= TRAIN_END][::FRAME_STEP]
    # A later time block and a gap reduce overlap; this is still one session.
    validation = [f for f in frames if f['source_frame'] >= VAL_START]
    if not training or not validation:
        raise ValueError('Need usable frames in both training and validation time ranges.')

    # Force a fresh label scan even when coordinate edits keep the same file size.
    (batch / 'labels.cache').unlink(missing_ok=True)

    output.mkdir(parents=True, exist_ok=True)
    for name, group in [('train', training), ('val', validation)]:
        images = []
        for frame in group:
            image = batch / 'images' / frame['image']
            label = batch / 'labels' / Path(frame['image']).with_suffix('.txt')
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f'Missing image or label: {frame["image"]}')
            # Keep /images/ in the path so YOLO finds the matching /labels/ file.
            images.append(str(image.absolute()))
        (output / f'{name}.txt').write_text('\n'.join(images) + '\n')

    # JSON is valid YAML and safely handles spaces in paths and class names.
    config = output / 'dataset.yaml'
    config.write_text(json.dumps(dict(path=str(batch.absolute()),
                                     train=str((output / 'train.txt').absolute()),
                                     val=str((output / 'val.txt').absolute()),
                                     names=(batch / 'classes.txt').read_text().splitlines()), indent=2) + '\n')
    print(f'Training: {len(training)} frames; validation: {len(validation)} frames.')
    print('Diagnostic split from one session; a new recording is needed for a real test.')
    return config


def main():
    # Import the training libraries only when actually starting a run.
    import torch
    from ultralytics import YOLO, settings

    settings.update({'sync': False})
    dataset = prepare_data()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    models = ROOT / 'models'
    models.mkdir(exist_ok=True)
    model = YOLO(MODEL)
    model.train(data=str(dataset), time=MINUTES / 60, epochs=100,
                imgsz=IMAGE_SIZE, batch=BATCH_SIZE, device=device,
                # Avoid extra worker processes and image caches on this 8 GB Mac.
                workers=0, cache=False, project=str(models), name=RUN_NAME,
                # Small updates help preserve what the first model learned.
                optimizer='AdamW', lr0=LEARNING_RATE, warmup_epochs=1,
                mosaic=0, scale=0.2, translate=0.05, seed=42, plots=True)
    print(f'Best checkpoint: {model.trainer.best}')


if __name__ == '__main__':
    main()
