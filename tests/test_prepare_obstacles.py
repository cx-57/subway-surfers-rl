"""Keep holdout recordings separate and reject unsafe annotation imports."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from agent.detection import CLASS_NAMES
from scripts.prepare_obstacles import integrate, label_rows


class PrepareObstaclesTests(unittest.TestCase):
    def fixture(self, root):
        dataset = root / 'data/training'
        (dataset / 'images').mkdir(parents=True)
        (dataset / 'labels').mkdir()
        frames = []
        for name, color in [('train.jpg', 0), ('val.jpg', 255)]:
            path = dataset / 'images' / name
            cv2.imwrite(str(path), np.full((40, 80, 3), color, np.uint8))
            (dataset / 'labels' / Path(name).with_suffix('.txt')).write_text('0 .5 .5 .2 .2\n')
            frames.append(dict(image=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest = dict(frames=frames, splits=dict(train=['train.jpg'], val=['val.jpg']))
        (dataset / 'manifest.json').write_text(json.dumps(manifest))
        annotations = root / 'models/evaluation/obstacle_labels'
        annotations.mkdir(parents=True)
        (annotations / 'legacy_new_objects.json').write_text(json.dumps(dict(
            reviewed=['train.jpg', 'val.jpg'], additions={}, uncertain=[])))
        session = 'session_123'
        raw = root / 'data/raw' / session
        raw.mkdir(parents=True)
        source = raw / 'frame_000001.jpg'
        cv2.imwrite(str(source), np.full((40, 80, 3), 127, np.uint8))
        (raw / 'review.json').write_text(json.dumps(dict(reserved_for_validation=True,
            files=[dict(image=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest())])))
        folder = annotations / session
        folder.mkdir()
        document = dict(class_names=list(CLASS_NAMES), frames=[dict(image=source.name,
            reviewed=True, boxes=[dict(class_id=3, xyxy=[20, 10, 40, 30])])])
        path = folder / 'annotations.json'
        path.write_text(json.dumps(document))
        return source, path

    def test_holdout_stays_separate_and_repeat_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _ = self.fixture(root)
            before = source.read_bytes()
            first = integrate(root)
            second = integrate(root)
            self.assertEqual(first['splits'], second['splits'])
            self.assertEqual(len(second['splits']['train']), 1)
            self.assertEqual(len(second['splits']['val']), 2)
            self.assertEqual(source.read_bytes(), before)
            label = root / 'data/training/labels/session_123_frame_000001.txt'
            self.assertEqual(label.read_text(), '3 0.37500000 0.50000000 0.25000000 0.50000000\n')

    def test_unreviewed_empty_frames_cannot_become_negatives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, path = self.fixture(root)
            document = json.loads(path.read_text())
            document['frames'][0]['boxes'] = []
            path.write_text(json.dumps(document))
            before = (root / 'data/training/manifest.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'background review'):
                integrate(root)
            self.assertEqual((root / 'data/training/manifest.json').read_bytes(), before)

    def test_boxes_cannot_be_clipped_or_invalid_classes_silently_accepted(self):
        for box in [dict(class_id=6, xyxy=[0, 0, 20, 20]),
                    dict(class_id=3, xyxy=[-1, 0, 20, 20])]:
            with self.assertRaises(ValueError):
                label_rows([box], 80, 40)

    def test_removed_annotation_is_removed_from_the_active_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, path = self.fixture(root)
            integrate(root)
            document = json.loads(path.read_text())
            document['frames'] = []
            document['excluded'] = [dict(image=source.name, reason='ambiguous')]
            path.write_text(json.dumps(document))
            result = integrate(root)
            self.assertEqual(result['splits']['val'], ['val.jpg'])
            self.assertTrue(source.is_file())

    def test_legacy_pixel_change_fails_before_publishing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            manifest = root / 'data/training/manifest.json'
            before = manifest.read_bytes()
            (root / 'data/training/images/train.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                integrate(root)
            self.assertEqual(manifest.read_bytes(), before)

    def test_duplicate_pixels_across_splits_fail_before_publishing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _ = self.fixture(root)
            source.write_bytes((root / 'data/training/images/train.jpg').read_bytes())
            review = source.parent / 'review.json'
            record = json.loads(review.read_text())
            record['files'][0]['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            review.write_text(json.dumps(record))
            manifest = root / 'data/training/manifest.json'
            before = manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, 'identical'):
                integrate(root)
            self.assertEqual(manifest.read_bytes(), before)
