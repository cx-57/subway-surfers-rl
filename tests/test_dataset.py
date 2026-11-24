"""Protect annotation geometry, original pixels and chronological data splits."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import train


class DatasetTests(unittest.TestCase):
    def test_partial_view_selection_keeps_only_confident_early_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)/'batch'
            (batch/'images').mkdir(parents=True)
            (batch/'labels').mkdir()
            frames, changes = [], []
            for n in [105, 108, 111, 114, 850, 853]:
                name = f'frame_{n:06}.jpg'
                (batch/'images'/name).write_bytes(b'original')
                (batch/'labels'/name.replace('.jpg','.txt')).write_text('2 .5 .5 .1 .2\n')
                frames.append(dict(image=name, source_frame=n, training_eligible=True,
                                   annotation_uncertain=n==111))
                if n in [108,111,853]:
                    changes.append(dict(image=name, class_id=2, old_box=None))
            (batch/'manifest.json').write_text(json.dumps({'frames':frames}))
            (batch/'partial_audit.json').write_text(json.dumps({'changes':changes}))
            (batch/'classes.txt').write_text('train\nground_barrier\noverhead_barrier\n')
            output = Path(folder)/'prepared'
            train.prepare_data(batch, output, keep_partial_views=True)
            selected = [Path(p).name for p in (output/'train.txt').read_text().splitlines()]
            self.assertEqual(selected,['frame_000105.jpg','frame_000108.jpg','frame_000114.jpg'])
            self.assertEqual(len(selected),len(set(selected)))
            self.assertEqual(train.experiment_settings('partial_views'),train.experiment_settings('control'))

    def test_valid_edge_boxes_and_empty_background_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            label = Path(folder) / 'frame.txt'
            label.write_text('0 0.5 0.5 1 1\n2 0.75 0.05 0.5 0.1\n')
            self.assertEqual(len(train.read_labels(label)), 2)
            label.write_text('')
            self.assertEqual(train.read_labels(label), [])

    def test_invalid_class_nan_size_and_out_of_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            label = Path(folder) / 'frame.txt'
            for row in ['3 .5 .5 .2 .2', '1.5 .5 .5 .2 .2',
                        '2 nan .5 .2 .2', '2 .5 .5 0 .1',
                        '0 .05 .5 .2 .2', '1 .9 .5 .3 .2', '2 .5 .5 .1']:
                with self.subTest(row=row):
                    label.write_text(row)
                    with self.assertRaises(ValueError):
                        train.read_labels(label)

    def test_source_hash_detects_changes_without_rewriting_image(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            (batch / 'images').mkdir()
            image = batch / 'images/frame.jpg'
            image.write_bytes(b'original image bytes')
            manifest = {'frames': [{'image': image.name,
                         'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}]}
            (batch / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(train.verify_images(batch), 1)
            image.write_bytes(b'different image bytes')
            with self.assertRaises(ValueError):
                train.verify_images(batch)
            self.assertEqual(image.read_bytes(), b'different image bytes')

    def test_uncertain_frames_are_excluded_without_shifting_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / 'batch'
            (batch / 'images').mkdir(parents=True)
            (batch / 'labels').mkdir()
            frames = []
            for n in [105, 108, 111, 114, 117, 120, 123, 850, 853]:
                name = f'frame_{n:06}.jpg'
                (batch / 'images' / name).write_bytes(b'original')
                (batch / 'labels' / name.replace('.jpg', '.txt')).write_text('')
                frames.append(dict(image=name, source_frame=n, training_eligible=True,
                                   annotation_uncertain=n in [114, 850], verified_background=True))
            (batch / 'classes.txt').write_text('train\nground_barrier\noverhead_barrier\n')
            (batch / 'manifest.json').write_text(json.dumps({'frames': frames}))
            output = Path(folder) / 'prepared'
            train.prepare_data(batch, output)
            self.assertEqual([Path(p).name for p in (output/'train.txt').read_text().splitlines()],
                             ['frame_000105.jpg', 'frame_000123.jpg'])
            self.assertEqual([Path(p).name for p in (output/'val.txt').read_text().splitlines()],
                             ['frame_000853.jpg'])
            before = (output / 'dataset.yaml').read_bytes()
            train.prepare_data(batch, output)
            self.assertEqual(before, (output / 'dataset.yaml').read_bytes())
            # Empty labels without a visual background review are not negatives.
            frames[0]['verified_background'] = False
            (batch / 'manifest.json').write_text(json.dumps({'frames': frames}))
            with self.assertRaises(ValueError):
                train.prepare_data(batch, output)
            frames[0]['verified_background'] = True
            frames[0]['sha256'] = frames[-1]['sha256'] = 'same-image-hash'
            (batch / 'manifest.json').write_text(json.dumps({'frames': frames}))
            with self.assertRaisesRegex(ValueError, 'identical screenshot'):
                train.prepare_data(batch, output)

    def test_experiments_change_only_the_intended_training_transform(self):
        control = train.experiment_settings('control')
        scale = train.experiment_settings('translate_scale')
        mosaic = train.experiment_settings('mosaic')
        self.assertEqual(scale, {**control, 'translate': .10, 'scale': .25})
        self.assertEqual(mosaic, {**control, 'mosaic': .25})
        self.assertEqual(control['seed'], 42)
        self.assertEqual(control['epochs'], 12)
        self.assertEqual(control['mixup'], 0)
        self.assertEqual(control['cutmix'], 0)
        self.assertEqual(control['degrees'], 0)
        self.assertEqual(control['perspective'], 0)
        self.assertEqual(control['flipud'], 0)
        self.assertIsNone(control['time'])
        self.assertNotIn('erasing', control)
        with self.assertRaises(ValueError):
            train.experiment_settings('unknown')


if __name__ == '__main__':
    unittest.main()
