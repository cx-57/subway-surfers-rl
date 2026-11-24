"""Train a small obstacle detector. Run with .venv/bin/python scripts/train.py."""

import json
import hashlib
import math
from contextlib import contextmanager
from pathlib import Path

# Keep settings here instead of adding command-line options.
ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'data/labeling/batch_001'
DATA = ROOT / 'data/training'
IMAGE_SIZE = 640
BATCH_SIZE = 2
FRAME_STEP = 3
TRAIN_END = 780
VAL_START = 850
# Fine-tune the existing model; a separate run keeps the first model intact.
MODEL = ROOT / 'models/barriers_640/weights/best.pt'
LEARNING_RATE = 0.0003


@contextmanager
def complete_nms():
    """Do not accept scores if Ultralytics abandons part of NMS processing."""
    from ultralytics.utils import LOGGER
    errors = []
    def watch(record):
        if 'NMS time limit' in record.getMessage():
            errors.append(record.getMessage())
        return True
    LOGGER.addFilter(watch)
    try:
        yield
    finally:
        LOGGER.removeFilter(watch)
    if errors:
        raise RuntimeError('Incomplete NMS output: ' + '; '.join(errors))


def read_labels(path):
    """Read YOLO labels; reject invalid classes, sizes and visible bounds."""
    rows = []
    for line in path.read_text().splitlines():
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid YOLO row in {path}: {line}')
        c, x, y, w, h = values
        if (c not in (0, 1, 2) or w <= 0 or h <= 0 or
                min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
            raise ValueError(f'Invalid YOLO bounds/class in {path}: {line}')
        rows.append(values)
    return rows


def verify_images(batch=BATCH):
    """Compare every original screenshot with its recorded SHA256 fingerprint."""
    manifest = json.loads((batch / 'manifest.json').read_text())
    frames = manifest.get('all_source_frames', manifest['frames'])
    for frame in frames:
        paths = [batch / 'images' / frame['image']]
        if 'source' in frame:
            paths.append(ROOT / frame['source'])
        for path in paths:
            with path.open('rb') as source:
                actual = hashlib.file_digest(source, 'sha256').hexdigest()
            if actual != frame['sha256']:
                raise ValueError(f'Original image changed: {path}')
    return len(frames)


def training_settings():
    """One conservative training configuration; existing checkpoints stay intact."""
    return dict(epochs=12, patience=5, time=None, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                seed=42, optimizer='AdamW', lr0=LEARNING_RATE, warmup_epochs=1,
                mosaic=0, close_mosaic=3, scale=.2, translate=.05,
                mixup=0, cutmix=0, degrees=0, perspective=0, flipud=0)


def prepare_data(batch=BATCH, output=DATA):
    """Write image lists without copying images or changing their labels."""
    manifest = json.loads((batch / 'manifest.json').read_text())
    frames = sorted((f for f in manifest['frames'] if f['training_eligible'] is True),
                    key=lambda f: f['source_frame'])
    # Neighboring frames are almost identical. Keep one in three for training.
    training = [f for f in frames if f['source_frame'] <= TRAIN_END][::FRAME_STEP]
    # A later time block and a gap reduce overlap; this is still one session.
    validation = [f for f in frames if f['source_frame'] >= VAL_START]
    # Filter after sampling so an uncertain frame does not shift later selections.
    training = [f for f in training if not f.get('annotation_uncertain', False)]
    validation = [f for f in validation if not f.get('annotation_uncertain', False)]
    if not training or not validation:
        raise ValueError('Need usable frames in both training and validation time ranges.')
    train_hashes = {f['sha256'] for f in training if 'sha256' in f}
    if train_hashes & {f['sha256'] for f in validation if 'sha256' in f}:
        raise ValueError('An identical screenshot occurs in training and validation.')

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
            if not read_labels(label) and not frame.get('verified_background', False):
                raise ValueError(f'Empty label needs a visual background review: {label}')
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
    config = training_settings()
    print(f'Verified {verify_images()} unchanged original screenshots.')
    dataset = prepare_data()
    files = [DATA/'train.txt', DATA/'val.txt', BATCH/'manifest.json']
    if (BATCH/'partial_audit.json').exists():
        files.append(BATCH/'partial_audit.json')
    files += sorted((BATCH/'labels').glob('*.txt'))
    record = dict(settings=config, initial_checkpoint=str(MODEL), status='running',
                  training_images=(DATA/'train.txt').read_text().splitlines(),
                  validation_images=(DATA/'val.txt').read_text().splitlines(),
                  checkpoint_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in files})
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    models = ROOT / 'models'
    models.mkdir(exist_ok=True)
    model = YOLO(MODEL)
    model.add_callback('on_train_start', lambda trainer:
                       (trainer.save_dir/'run.json').write_text(json.dumps(record, indent=2)+'\n'))
    # Release unused GPU allocations between epochs on this 8 GB Mac.
    if device == 'mps':
        model.add_callback('on_train_epoch_end', lambda _: torch.mps.empty_cache())
    with complete_nms():
        model.train(data=str(dataset), **config, device=device,
                    # Avoid extra worker processes and image caches on this 8 GB Mac.
                    workers=0, cache=False, project=str(models),
                    name='training', plots=True)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != record['inputs'][str(p.relative_to(ROOT))]
           for p in files):
        raise ValueError('Dataset inputs changed during training; do not compare this run.')
    record.update(status='complete', epochs_completed=model.trainer.epoch+1)
    (model.trainer.save_dir/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'Best checkpoint: {model.trainer.best}')


if __name__ == '__main__':
    main()
