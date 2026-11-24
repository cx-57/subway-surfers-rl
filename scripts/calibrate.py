"""Baseline confidence check: frozen YOLO weights, ten original screenshots."""

import hashlib
import json
from pathlib import Path
import tempfile

try:
    from .evaluate import match_objects, check_coverage, write_json
    from .train import ROOT, complete_nms
except ImportError:
    from evaluate import match_objects, check_coverage, write_json
    from train import ROOT, complete_nms


FRAMES = [830, 840, 850, 860, 870, 880, 890, 900, 910, 920]
THRESHOLDS = [.25, .1, .05, .025, .01, .005, .0025, .001, .0001, .00001]


def score_threshold(records, threshold):
    counts = dict(tp=0, fp=0, fn=0, hidden_detected=0, hidden_total=0, negative_frame_fp=0)
    for record in records:
        predictions = [p for p in record['predictions'] if p['confidence'] >= threshold]
        matched, extra = match_objects(record['truth'], predictions)
        tp = sum(index is not None for index in matched)
        counts['tp'] += tp
        counts['fp'] += len(extra)
        counts['fn'] += len(matched)-tp
        if record['frame'] <= 870:
            counts['hidden_detected'] += tp
            counts['hidden_total'] += len(matched)
        if not record['truth']:
            counts['negative_frame_fp'] += len(extra)
    counts['precision'] = counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else 0
    counts['recall'] = counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else 0
    return counts


def main():
    import torch
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    checkpoint = ROOT/'models/barriers_640/weights/best.pt'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    batch = ROOT/'data/labeling/test_001'
    manifest = json.loads((batch/'manifest.json').read_text())
    source = {f['source_frame']: f for f in manifest['all_source_frames']}
    selected = [source[n] for n in FRAMES]
    images = [ROOT/f['source'] for f in selected]
    # Reuse the verified label snapshot; individual cloud-backed label reads can stall.
    previous = ROOT/'models/partial_experiments/frozen_regression'
    summary = json.loads((previous/'summary.json').read_text())
    assert summary['checkpoint_sha256'] == checkpoint_hash
    assert summary['annotation_revision'] == manifest['annotation_revision']
    label_frames = {f['image']: f for f in manifest['frames']}
    cached_truth = {r['image']: [obj for obj in r['truth'] if obj['class_id']==2]
                    for r in json.loads((previous/'predictions.json').read_text())}
    for frame in selected:
        label = batch/'labels'/Path(frame['image']).with_suffix('.txt').name
        assert summary['input_hashes'][str(label)] == label_frames[frame['image']]['label_sha256']
    # Check only these ten originals, rather than re-reading the whole recording.
    for frame, image in zip(selected, images):
        assert hashlib.sha256(image.read_bytes()).hexdigest() == frame['sha256'], image
    print('Checking ten original frames with the frozen model; no training.', flush=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(checkpoint)
    records, seen = [], []
    with tempfile.TemporaryDirectory() as folder:
        listing = Path(folder)/'frames.txt'
        listing.write_text('\n'.join(map(str, images))+'\n')
        with complete_nms():
            for r in model.predict(str(listing), imgsz=640, batch=2, device=device,
                                   conf=1e-5, classes=[2], iou=.7, max_det=1000,
                                   augment=False, verbose=False, stream=True):
                name = Path(r.path).name
                seen.append(str(Path(r.path).absolute()))
                truth = cached_truth[name]
                predictions = [dict(class_id=2, confidence=float(s), xyxy=b.tolist())
                               for s, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
                if len(predictions) >= 1000:
                    raise RuntimeError('Proposal cap reached; do not use censored counts.')
                records.append(dict(image=name, frame=int(Path(name).stem.split('_')[-1]),
                                    truth=truth, predictions=predictions))
                print('Checked', name, flush=True)
    check_coverage(list(map(str, images)), seen)
    results = {str(t): score_threshold(records, t) for t in THRESHOLDS}
    output = ROOT/'models/partial_experiments/confidence_baseline'
    number = 2
    while output.exists():
        output = ROOT/f'models/partial_experiments/confidence_baseline_{number}'
        number += 1
    output.mkdir()
    write_json(output/'results.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
               note='Small, previously inspected regression diagnostic. Overhead-only FPs; no threshold selected or deployed. Model scores and weights are unchanged.',
               label_snapshot=str(previous/'predictions.json'), annotation_revision=manifest['annotation_revision'],
               frame_count=len(records), confidence_floor=1e-5, nms_iou=.7, matching_iou=.5, results=results))
    write_json(output/'predictions.json', records)
    for frame, image in zip(selected, images):
        assert hashlib.sha256(image.read_bytes()).hexdigest() == frame['sha256'], image
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    print(json.dumps(results, indent=2))
    print('Saved baseline check:', output)


if __name__=='__main__':
    main()
