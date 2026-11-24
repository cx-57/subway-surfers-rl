"""Ensure operating-point counts use one-to-one, class-correct matches."""

import unittest
import tempfile
from pathlib import Path
import json
import numpy as np
from scripts.evaluate import match_objects, write_json, predict_frames, check_coverage, complete_nms


class MatchingTests(unittest.TestCase):
    def test_proximity_waits_for_exposed_barrier_but_rejects_low_confidence(self):
        from scripts.evaluate import nearby_indexes
        # Real predictions: frame 880 is farther away; frame 885 is closer.
        boxes = [dict(class_id=2, confidence=.532, xyxy=[585, 317, 681, 405]),
                 dict(class_id=2, confidence=.836, xyxy=[584, 347, 693, 446]),
                 dict(class_id=1, confidence=.49, xyxy=[330, 690, 510, 792])]
        self.assertEqual(nearby_indexes(boxes, (793, 1284)), [1])

    def test_tiny_low_screen_boxes_do_not_count_as_nearby(self):
        from scripts.evaluate import nearby_indexes
        boxes = [dict(class_id=1, confidence=.9, xyxy=[400, 700, 410, 730]),
                 dict(class_id=1, confidence=.9, xyxy=[330, 690, 510, 792])]
        self.assertEqual(nearby_indexes(boxes, (793, 1284)), [1])

    def test_train_proximity_uses_size_and_position_without_changing_detections(self):
        from scripts.evaluate import nearby_indexes
        boxes = [dict(class_id=0, confidence=.9, xyxy=[600, 300, 680, 790]),
                 dict(class_id=0, confidence=.9, xyxy=[490, 427, 810, 793]),
                 dict(class_id=0, confidence=.9, xyxy=[100, 100, 500, 300])]
        before = json.dumps(boxes)
        self.assertEqual(nearby_indexes(boxes, (793, 1284)), [1])
        scaled = [dict(p, xyxy=[v/2 for v in p['xyxy']]) for p in boxes]
        self.assertEqual(nearby_indexes(scaled, (396.5, 642)), [1])
        self.assertEqual(json.dumps(boxes), before)

    def test_validation_nms_warning_cannot_produce_a_successful_score(self):
        from ultralytics.utils import LOGGER
        with self.assertRaisesRegex(RuntimeError, 'Incomplete NMS'):
            with complete_nms():
                LOGGER.warning('NMS time limit 2.100s exceeded')

    def test_nms_timeout_is_an_execution_error(self):
        from ultralytics.utils import LOGGER
        class TimedOutModel:
            def predict(self, *args, **kwargs):
                LOGGER.warning('NMS time limit 2.100s exceeded')
                yield None
        with self.assertRaisesRegex(RuntimeError, 'Incomplete NMS'):
            list(predict_frames(TimedOutModel(), Path('unused.txt'), 'cpu'))

    def test_missing_duplicate_and_unexpected_predictions_are_rejected(self):
        check_coverage(['a.jpg', 'b.jpg'], ['b.jpg', 'a.jpg'])
        for seen in [['a.jpg'], ['a.jpg', 'a.jpg'], ['a.jpg', 'c.jpg']]:
            with self.subTest(seen=seen), self.assertRaises(ValueError):
                check_coverage(['a.jpg', 'b.jpg'], seen)

    def test_file_source_honors_two_image_batch_limit(self):
        import cv2
        from ultralytics.data.build import load_inference_source
        sizes = []
        class LoaderModel:
            def predict(self, source, **kwargs):
                loader = load_inference_source(source, batch=kwargs['batch'])
                sizes.append(loader.bs)
                return loader
        with tempfile.TemporaryDirectory() as folder:
            images = []
            for n in range(3):
                path = Path(folder) / f'{n}.jpg'
                cv2.imwrite(str(path), np.zeros((16, 24, 3), np.uint8))
                images.append(str(path))
            listing = Path(folder) / 'images.txt'
            listing.write_text('\n'.join(images)+'\n')
            list(predict_frames(LoaderModel(), listing, 'cpu'))
            self.assertEqual(sizes, [2])

    def test_numpy_metric_scalars_are_saved_as_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'summary.json'
            write_json(path, {'instances': np.int64(8), 'recall': np.float64(.375)})
            self.assertEqual(json.loads(path.read_text()), {'instances': 8, 'recall': .375})

    def test_duplicate_predictions_and_wrong_classes_are_false_positives(self):
        truth = [{'class_id': 2, 'xyxy': [10, 10, 30, 30]},
                 {'class_id': 1, 'xyxy': [40, 40, 50, 50]}]
        predictions = [dict(class_id=2, confidence=.9, xyxy=[10, 10, 30, 30]),
                       dict(class_id=2, confidence=.8, xyxy=[10, 10, 30, 30]),
                       dict(class_id=0, confidence=.7, xyxy=[40, 40, 50, 50])]
        matched, false_positives = match_objects(truth, predictions)
        self.assertEqual(matched, [0, None])
        self.assertEqual(false_positives, [1, 2])

    def test_poor_localization_and_empty_background_are_handled(self):
        truth = [dict(class_id=2, xyxy=[10, 10, 30, 30])]
        predictions = [dict(class_id=2, confidence=.9, xyxy=[20, 20, 40, 40])]
        self.assertEqual(match_objects(truth, predictions), ([None], [0]))
        self.assertEqual(match_objects([], predictions), ([], [0]))
        self.assertEqual(match_objects(truth, []), ([None], []))


if __name__ == '__main__':
    unittest.main()
