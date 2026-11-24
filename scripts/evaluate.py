"""Evaluate a checkpoint on fixed labels, with visible-object error counts."""

import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

try:
    from .train import read_labels, verify_images, complete_nms
except ImportError:  # Also support `python scripts/evaluate.py ...`.
    from train import read_labels, verify_images, complete_nms


# Minimum confidence for individual detections; mAP still sweeps score thresholds.
CONFIDENCE = .5
# (Minimum bottom position / image height, minimum box width / image width).
# Starting proximity estimates, not measured distances or collision deadlines.
NEARBY_LIMITS = {0: (.65, .12), 1: (.55, .06), 2: (.55, .06)}


def nearby_indexes(predictions, image_shape):
    """Select confident, nearby candidates. Lane occupancy must be checked separately."""
    height, width = image_shape
    nearby = []
    for index, prediction in enumerate(predictions):
        if prediction['confidence'] < CONFIDENCE:
            continue
        limits = NEARBY_LIMITS.get(prediction['class_id'])
        if limits is None:
            continue
        left, top, right, bottom = prediction['xyxy']
        min_bottom, min_width = limits
        if bottom >= min_bottom*height and right-left >= min_width*width:
            nearby.append(index)
    return nearby


def barrier_strokes(predictions, image_shape):
    """Suggest keys per nearby barrier; check player path and timing before sending."""
    strokes = {1: ('up', 'jump'), 2: ('down', 'roll')}
    recommendations = []
    for index in nearby_indexes(predictions, image_shape):
        stroke = strokes.get(predictions[index]['class_id'])
        if stroke is not None:
            key, action = stroke
            recommendations.append(dict(prediction_index=index, key=key, action=action))
    return recommendations


def write_json(path, value):
    # Ultralytics includes NumPy integer scalars in some metric summaries.
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item())+'\n')


def predict_frames(model, image_list, device):
    with complete_nms():
        # A Python list bypasses file-loader batching. Pass the split file itself.
        for result in model.predict(str(image_list), imgsz=640, batch=2,
                                    device=device, conf=CONFIDENCE, iou=.7, max_det=300,
                                    half=False, augment=False, stream=True, verbose=False):
            yield result


def check_coverage(expected, seen):
    """Reject incomplete, duplicated or unexpected inference results."""
    if len(expected) != len(set(expected)) or len(seen) != len(set(seen)) or set(expected) != set(seen):
        raise ValueError('Prediction coverage does not match the requested split.')


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def match_objects(truth, predictions, iou_threshold=.5):
    """Match each detection once, in confidence order; duplicates remain errors."""
    matched = [None] * len(truth)
    false_positives = []
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i]['confidence']):
        prediction = predictions[index]
        choices = [(overlap(obj['xyxy'], prediction['xyxy']), j)
                   for j, obj in enumerate(truth) if matched[j] is None and
                   obj['class_id'] == prediction['class_id']]
        score, target = max(choices, default=(0, -1))
        if score >= iou_threshold:
            matched[target] = index
        else:
            false_positives.append(index)
    return matched, false_positives


def evaluate(checkpoint, dataset, split='val', output=None):
    import torch
    from ultralytics import YOLO, settings, __version__
    from ultralytics.utils import YAML

    settings.update({'sync': False})
    checkpoint, dataset = Path(checkpoint), Path(dataset)
    output = Path(output) if output else checkpoint.parent.parent / f'evaluation_{split}'
    requested, number = output, 2
    while (output/'summary.json').exists():
        output = requested.with_name(f'{requested.name}_{number}')
        number += 1
    output.mkdir(parents=True, exist_ok=True)
    data = YAML.load(dataset)
    batch = Path(data['path'])
    image_list = Path(data[split])
    images = image_list.read_text().splitlines()
    manifest = json.loads((batch / 'manifest.json').read_text())
    frames = {f['image']: f for f in manifest['frames']}
    for image in images:
        frame = frames[Path(image).name]
        if frame.get('annotation_uncertain') or frame.get('evaluation_eligible') is False:
            raise ValueError(f'Uncertain/excluded frame in quantitative evaluation: {image}')
    files = [dataset, image_list, batch/'manifest.json']
    files += [batch/'labels'/Path(p).with_suffix('.txt').name for p in images]
    files += [Path(p) for p in images]
    if (batch/'partial_audit.json').exists():
        files.append(batch/'partial_audit.json')
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    verify_images(batch)
    # Equal-size coordinate edits otherwise leave Ultralytics' scan cache stale.
    (batch/'labels.cache').unlink(missing_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(checkpoint)
    with complete_nms():
        result = model.val(data=str(dataset), split=split, imgsz=640, batch=2, device=device,
                           workers=0, augment=False, plots=False, verbose=False,
                           conf=.001, iou=.7, max_det=300, half=False,
                           project=str(output), name='metrics', exist_ok=True)
    summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   dataset=str(dataset), split=split, image_count=len(images),
                   metrics=result.results_dict, per_class=result.summary(),
                   operating_point=dict(confidence=CONFIDENCE, matching_iou=.5, nms_iou=.7, batch=2),
                   proximity_gate=dict(limits=NEARBY_LIMITS,
                       meaning='Minimum normalized box bottom and width; rough proximity only, not lane occupancy or verified reaction timing.'),
                   input_hashes=fingerprints, versions=dict(ultralytics=__version__, torch=str(torch.__version__)),
                   annotation_revision=manifest.get('annotation_revision'), status='complete',
                   caveat='Same-session validation or previously inspected regression data; not proof of unseen generalization.')
    audit = batch / 'partial_audit.json'
    cases = json.loads(audit.read_text()).get('evaluation_cases', []) if audit.exists() else []
    tags = {(c['image'], c['label']): c['tags'] for c in cases}
    counts = {str(c): dict(tp=0, fp=0, fn=0) for c in range(3)}
    groups = {g: dict(objects=0, detected=0, missed=0) for g in
              ['image_edge', 'small', 'occluded', 'hood_only', 'fully_visible', 'near_player_barrier']}
    records, negatives, seen = [], [], []
    for prediction in predict_frames(model, image_list, device):
        image = Path(prediction.path)
        seen.append(str(image.absolute()))
        height, width = prediction.orig_shape
        label = batch / 'labels' / image.with_suffix('.txt').name
        rows = label.read_text().splitlines()
        truth = []
        for row, (c, x, y, w, h) in zip(rows, read_labels(label)):
            box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
            categories = list(tags.get((image.name, row), []))
            if min(box[:2]) <= 1 or box[2] >= width-1 or box[3] >= height-1:
                categories.append('image_edge')
            if max(w*width, h*height)*640/max(width, height) <= 32:
                categories.append('small')
            if c in (1, 2) and box[3] >= height*.7:
                categories.append('near_player_barrier')
            truth.append(dict(class_id=int(c), xyxy=box, tags=categories))
        boxes = [dict(class_id=int(c), confidence=float(s), xyxy=b.tolist())
                 for c, s, b in zip(prediction.boxes.cls.cpu().numpy(), prediction.boxes.conf.cpu().numpy(),
                                    prediction.boxes.xyxy.cpu().numpy())]
        matched, extra = match_objects(truth, boxes)
        for obj, index in zip(truth, matched):
            counts[str(obj['class_id'])]['tp' if index is not None else 'fn'] += 1
            for tag in obj['tags']:
                groups[tag]['objects'] += 1
                groups[tag]['detected' if index is not None else 'missed'] += 1
        for index in extra:
            counts[str(boxes[index]['class_id'])]['fp'] += 1
        if not truth:
            if not frames[image.name].get('verified_background', False):
                raise ValueError(f'Empty label lacks a verified background review: {image}')
            negatives.append(dict(image=image.name, false_positives=len(boxes)))
        records.append(dict(image=image.name, truth=truth, predictions=boxes, matched=matched,
                            image_shape=[height, width], nearby_prediction_indexes=nearby_indexes(boxes, (height, width)),
                            barrier_strokes=barrier_strokes(boxes, (height, width)),
                            false_positive_indexes=extra))
    check_coverage([str(Path(p).absolute()) for p in images], seen)
    # A synchronized single-image measurement is closer to future live inference.
    timings = []
    for i in range(25):
        if device == 'mps':
            torch.mps.synchronize()
        start = time.perf_counter()
        model.predict(images[0], imgsz=640, conf=CONFIDENCE, device=device, verbose=False)
        if device == 'mps':
            torch.mps.synchronize()
        if i >= 5:
            timings.append((time.perf_counter()-start)*1000)
    summary.update(class_counts=counts, subgroups=groups, negative_frames=negatives,
                   median_single_frame_ms=statistics.median(timings),
                   subgroup_definitions=dict(small='Longest side <=32 pixels at 640px inference size; size proxy, not measured distance.',
                       image_edge='Visible box touches an image boundary within one native pixel.',
                       occluded='Visually tagged subset only; not an exhaustive population estimate.',
                       hood_only='Visually reviewed blue-cap views with the chevron mostly hidden; preliminary examples.',
                       near_player_barrier='Barrier bottom in lower 30% of image; position proxy, not collision timing.'),
                   subgroup_false_positives='FPs lack ground-truth visibility categories; reported per class and verified negative frame.')
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != fingerprints[str(p)] for p in files):
        raise ValueError('Evaluation inputs changed during execution.')
    verify_images(batch)
    write_json(output / 'summary.json', summary)
    write_json(output / 'predictions.json', records)
    print(f'Report saved to: {output}')
    print((output / 'summary.json').read_text())
    return summary


if __name__ == '__main__':
    evaluate(*sys.argv[1:])
