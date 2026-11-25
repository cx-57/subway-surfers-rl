"""Merge reviewed obstacle labels into the existing train/validation dataset."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from agent.detection import CLASS_NAMES
from scripts.train import prepare_data, read_labels

ROOT = Path(__file__).resolve().parents[1]


def label_rows(boxes, width, height):
    """Convert reviewed native-pixel boxes to YOLO coordinates without resizing images."""
    rows = []
    for box in boxes:
        class_id = box['class_id']
        if not isinstance(class_id, int) or isinstance(class_id, bool) or not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError('Unsupported class ID')
        left, top, right, bottom = box['xyxy']
        for value in (left, top, right, bottom):
            if not math.isfinite(value):
                raise ValueError('Box coordinates must be finite')
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError('Box outside original image')
        x = (left + right) / (2 * width)
        y = (top + bottom) / (2 * height)
        w = (right - left) / width
        h = (bottom - top) / height
        rows.append(f'{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}')
    return rows


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def integrate(root=ROOT):
    dataset = root / 'data/training'
    annotations = root / 'models/evaluation/obstacle_labels'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    frames = {}
    for frame in manifest['frames']:
        frames[frame['image']] = frame
    splits = manifest['splits']
    legacy = json.loads((annotations / 'legacy_new_objects.json').read_text())
    legacy_reviewed = set(legacy['reviewed'])
    legacy_uncertain = {}
    for item in legacy.get('uncertain', []):
        legacy_uncertain[item['image']] = item['reason']
    updates = []
    assets = []

    # Old images need the new-class audit too; otherwise omitted bushes become false negatives.
    for name, frame in frames.items():
        if frame.get('annotation_file'):
            continue
        if name not in legacy_reviewed:
            raise ValueError('Legacy image has not been reviewed for new classes: ' + name)
        image_path = dataset / 'images' / name
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('Legacy original image changed: ' + name)
        source = frame.get('source')
        if source and hashlib.sha256((root / source).read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('Legacy source image changed: ' + name)
        if name in legacy_uncertain:
            frame['annotation_uncertain'] = True
            for split in ('train', 'val'):
                if name in splits[split]:
                    splits[split].remove(name)
            continue
        additions = legacy.get('additions', {}).get(name, [])
        if additions:
            image = cv2.imread(str(dataset / 'images' / name))
            height, width = image.shape[:2]
            label = dataset / 'labels' / Path(name).with_suffix('.txt')
            rows = label.read_text().splitlines()
            for row in label_rows(additions, width, height):
                if row not in rows:
                    rows.append(row)
            updates.append((label, '\n'.join(rows) + '\n'))

    imported = {}
    active_imports = set()
    for annotation_file in sorted(annotations.glob('session_*/annotations.json')):
        session = annotation_file.parent.name
        source_folder = root / 'data/raw' / session
        review = json.loads((source_folder / 'review.json').read_text())
        # A whole recording stays in one split, including every adjacent rare-object view.
        split = 'val' if review.get('reserved_for_validation') else 'train'
        source_hashes = {}
        for record in review['files']:
            source_hashes[record['image']] = record['sha256']
        annotation_bytes = annotation_file.read_bytes()
        document = json.loads(annotation_bytes)
        annotation_checksum = hashlib.sha256(annotation_bytes).hexdigest()
        if document['class_names'] != list(CLASS_NAMES):
            raise ValueError('Annotation class order differs from the detector')
        imported[session] = 0
        seen = set()
        for item in document['frames']:
            name = item['image']
            if Path(name).name != name or not item.get('reviewed'):
                raise ValueError('Only reviewed frames with plain filenames can be imported')
            if name in seen:
                raise ValueError('Duplicate reviewed image: ' + name)
            seen.add(name)
            number = int(Path(name).stem.split('_')[-1])
            for excluded in review.get('excluded_ranges', []):
                if excluded['first'] <= number <= excluded['last']:
                    raise ValueError('Non-gameplay frame in reviewed labels: ' + name)
            source = source_folder / name
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            if checksum != source_hashes[name]:
                raise ValueError('Original image changed: ' + str(source))
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError('Unreadable original: ' + str(source))
            height, width = image.shape[:2]
            rows = label_rows(item['boxes'], width, height)
            if not rows and not item.get('verified_background'):
                raise ValueError('Empty label requires a visual background review: ' + name)
            unique_name = session + '_' + name
            active_imports.add(unique_name)
            other = 'train' if split == 'val' else 'val'
            if unique_name in splits[other]:
                raise ValueError('Recording would cross train/validation splits')
            destination = dataset / 'images' / unique_name
            label = dataset / 'labels' / Path(unique_name).with_suffix('.txt')
            updates.append((label, '\n'.join(rows) + ('\n' if rows else '')))
            appearances = []
            for box in item['boxes']:
                if box.get('appearance'):
                    appearances.append(box['appearance'])
            frames[unique_name] = dict(image=unique_name, source=str(source.relative_to(root)),
                source_session=session, source_frame=number, sha256=checksum, width=width, height=height,
                training_eligible=True, annotation_uncertain=False, split=split,
                verified_background=item.get('verified_background', False),
                annotation_status='AI-assisted visually reviewed draft; not verified human ground truth',
                annotation_file=str(annotation_file.relative_to(root)), review_notes=item.get('notes', ''),
                object_appearances=appearances,
                annotation_sha256=annotation_checksum,
                label_sha256=hashlib.sha256(('\n'.join(rows) + ('\n' if rows else '')).encode()).hexdigest())
            if unique_name not in splits[split]:
                splits[split].append(unique_name)
            imported[session] += 1
            # Hard links retain original bytes and avoid duplicating large screenshots.
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
                    raise ValueError('Existing curated image differs: ' + unique_name)
            else:
                assets.append((source, destination))

    retired = []
    for name, frame in list(frames.items()):
        if frame.get('annotation_file') and name not in active_imports:
            retired.append(frame)
            del frames[name]
            for split in ('train', 'val'):
                if name in splits[split]:
                    splits[split].remove(name)

    # Reject leaks before changing labels, classes, manifests or active image files.
    train_hashes = set()
    if set(splits['train']) & set(splits['val']):
        raise ValueError('Training and validation splits overlap')
    for name in splits['train']:
        train_hashes.add(frames[name]['sha256'])
    for name in splits['val']:
        if frames[name]['sha256'] in train_hashes:
            raise ValueError('An identical screenshot occurs in training and validation')
    for frame in retired:
        source = root / frame['source']
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('Cannot retire a curated copy without its unchanged original')

    # Publish labels and split metadata only after every input has passed validation.
    for source, destination in assets:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    for path, text in updates:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text)
        temporary.replace(path)
        read_labels(path)
    manifest['frames'] = list(frames.values())
    manifest['classes'] = list(CLASS_NAMES)
    manifest['annotation_revision'] = 'six_obstacles_v1'
    manifest['new_recording_imports'] = imported
    write_json(dataset / 'manifest.json', manifest)
    (dataset / 'classes.txt').write_text('\n'.join(CLASS_NAMES) + '\n')
    prepare_data(dataset, dataset)
    for frame in retired:
        name = frame['image']
        (dataset / 'images' / name).unlink(missing_ok=True)
        (dataset / 'labels' / Path(name).with_suffix('.txt')).unlink(missing_ok=True)
    report = dict(classes=list(CLASS_NAMES), splits={}, source_sessions=imported,
                  counting='Labeled boxes and labeled frames; repeated views are not distinct physical obstacles.',
                  label_quality='AI-assisted visual drafts; not independently human-verified.',
                  validation_note='Third new clip is held out. Recordings appear to continue the same game, so this is not an independent unseen test.')
    for split in ('train', 'val'):
        counts, image_counts = {}, {}
        appearance_counts = {}
        for name in CLASS_NAMES:
            counts[name] = 0
            image_counts[name] = 0
        for name in splits[split]:
            for appearance in frames[name].get('object_appearances', []):
                appearance_counts[appearance] = appearance_counts.get(appearance, 0) + 1
            present = set()
            for row in read_labels(dataset / 'labels' / Path(name).with_suffix('.txt')):
                class_name = CLASS_NAMES[int(row[0])]
                counts[class_name] += 1
                present.add(class_name)
            for class_name in present:
                image_counts[class_name] += 1
        report['splits'][split] = dict(images=len(splits[split]), boxes=counts, frames_by_class=image_counts,
                                      appearances=appearance_counts)
    write_json(dataset / 'dataset_report.json', report)
    print('Reviewed frames imported:', imported)
    return manifest


if __name__ == '__main__':
    integrate()
