import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from agent.player_vision import PlayerLocalizer


class PlayerVisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        rng = np.random.default_rng(4)
        self.template = rng.integers(0, 255, (30, 20, 3), dtype=np.uint8)
        p = Path(self.tmp.name) / "actor.png"
        cv2.imwrite(str(p), self.template)
        self.config = {
            "working_width": 200,
            "roi": [0, 0, 1, 1],
            "aspect_range": [0.5, 3.0],
            "score_threshold": 0.8,
            "margin_threshold": 0.1,
            "scales": [1.0],
            "templates": [
                {
                    "path": str(p),
                    "reference_height": 100,
                    "bbox_kind": "fullbody",
                    "provenance": "deterministic test patch",
                }
            ],
        }

    def test_positive_and_original_dimensions(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:70, 80:100] = self.template
        result = PlayerLocalizer(self.config).locate(image, 1.0)
        self.assertEqual(result["bbox"], (80.0, 40.0, 100.0, 70.0))
        self.assertEqual(result["movement"], "UNKNOWN")
        self.assertEqual(result["bbox_kind"], "fullbody")

    def test_default_search_region_includes_jump_height(self):
        import json

        path = Path(__file__).resolve().parents[1] / "config/player_templates.json"
        roi = json.loads(path.read_text())["roi"]
        localizer = PlayerLocalizer(dict(self.config, roi=roi))
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:50, 90:110] = self.template
        result = localizer.locate(image, 1.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["bbox"], (90.0, 20.0, 110.0, 50.0))

    def test_screen_lane_corridor_rejects_false_background_identity(self):
        config = dict(self.config)
        config["lane_tracking"] = {
            "centers": [0.4453, 0.5030, 0.5517],
            "max_error": 0.035,
        }
        image = np.zeros((100, 200, 3), np.uint8)
        # Both patches are perfect NCC matches. Only the center-positioned
        # patch can be the runner; the scenery patch must not compete.
        image[40:70, 20:40] = self.template
        image[40:70, 80:100] = self.template
        localizer = PlayerLocalizer(config)
        result = localizer.locate(image, 1.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["bbox"], (80.0, 40.0, 100.0, 70.0))
        # Without the actual runner, a convincing background template must
        # not seed a false character track or optical flow.
        image[40:70, 80:100] = 0
        self.assertIsNone(PlayerLocalizer(config).locate(image, 2.0))

    def test_no_match_does_not_propagate(self):
        localizer = PlayerLocalizer(self.config)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:70, 80:100] = self.template
        self.assertIsNotNone(localizer.locate(image, 1))
        self.assertIsNone(localizer.locate(np.zeros_like(image), 10))

    def test_duplicate_and_occluded_rejected(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:70, 40:60] = self.template
        image[40:70, 120:140] = self.template
        self.assertIsNone(PlayerLocalizer(self.config).locate(image, 1))
        image[:] = 0
        image[40:55, 40:60] = self.template[:15]
        self.assertIsNone(PlayerLocalizer(self.config).locate(image, 2))

    def test_invalid_input_and_time(self):
        localizer = PlayerLocalizer(self.config)
        self.assertIsNone(localizer.locate(None, 1))
        self.assertIsNone(localizer.locate(np.zeros((100, 200, 3), dtype=np.uint8), float("nan")))

    def test_corrupt_scoped_template_rejected(self):
        self.config["templates"][0]["template_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            PlayerLocalizer(self.config)

    def test_real_heldout_frame_if_available(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "data/raw/session_1795524944131662000/frame_000500.jpg"
        )
        if not path.is_file():
            self.skipTest("Local original footage unavailable")
        image = cv2.imread(str(path))
        if image is None:
            self.skipTest("Local original footage not readable")
        result = PlayerLocalizer().locate(image, 0.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["bbox_kind"], "upperbody")
        self.assertFalse(result["footpoint_valid"])
        self.assertGreater(result["score"], 0.8)
        x1, y1, x2, y2 = result["bbox"]
        self.assertTrue(610 < x1 < 660 and 530 < y1 < 580 and 680 < x2 < 725 and 610 < y2 < 660)

    def flow_config(self):
        return dict(
            self.config,
            enable_flow=True,
            ncc_refresh_interval=0.15,
            flow_max_gap=0.2,
            flow_identity_ttl=0.4,
            flow_min_features=5,
            working_width=200,
        )

    def test_flow_translation_and_blank_occlusion(self):
        localizer = PlayerLocalizer(self.flow_config())
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:70, 80:100] = self.template
        self.assertIsNotNone(localizer.locate(image, 0))
        shifted = np.zeros_like(image)
        shifted[42:72, 84:104] = self.template
        result = localizer.locate(shifted, 0.04)
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "opticalflow")
        self.assertAlmostEqual(result["bbox"][0], 84, delta=1)
        __import__("json").dumps(result)
        self.assertIsNone(localizer.locate(np.zeros_like(image), 0.08))

    def test_flow_ttl_and_dimension_reset(self):
        localizer = PlayerLocalizer(self.flow_config())
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:70, 80:100] = self.template
        localizer.locate(image, 0)
        result = localizer.locate(image, 0.5)
        self.assertNotEqual(result["source"], "opticalflow")
        result = localizer.locate(cv2.resize(image, (100, 50)), 0.54)
        self.assertTrue(result is None or result["source"] != "opticalflow")

    def test_flow_identity_ttl_without_refresh(self):
        config = self.flow_config()
        config.update(ncc_refresh_interval=99, flow_identity_ttl=0.1)
        localizer = PlayerLocalizer(config)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:70, 80:100] = self.template
        self.assertIsNotNone(localizer.locate(image, 0))
        self.assertEqual(localizer.locate(image, 0.04)["source"], "opticalflow")
        self.assertEqual(localizer.locate(image, 0.08)["source"], "opticalflow")
        self.assertEqual(localizer.locate(image, 0.12)["source"], "automatic color-template NCC")
        # A failed flow update can reacquire even before the next scheduled refresh.
        localizer._flow = lambda image, timestamp: None
        self.assertEqual(localizer.locate(image, 0.16)["source"], "automatic color-template NCC")

    def test_flow_refresh_rejects_ambiguous_identity(self):
        localizer = PlayerLocalizer(self.flow_config())
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:70, 40:60] = self.template
        localizer.locate(image, 0)
        image[40:70, 120:140] = self.template
        self.assertIsNone(localizer.locate(image, 0.16))
        self.assertIsNone(localizer.locate(image, 0.18))

    def test_invalid_configuration_fails_before_opencv(self):
        for extra in (
            {"roi": [0.8, 0, 0.2, 1]},
            {"working_width": float("nan")},
            {"flow_identity_ttl": float("inf")},
            {"scales": [0]},
            {
                "templates": [
                    {"path": str(Path(self.tmp.name) / "actor.png"), "reference_height": 0}
                ]
            },
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                PlayerLocalizer(dict(self.config, **extra))

    def test_malformed_timestamp_and_tiny_input_return_unknown(self):
        for flow in (False, True):
            localizer = PlayerLocalizer(dict(self.config, enable_flow=flow))
            self.assertIsNone(localizer.locate(np.zeros((100, 200, 3), np.uint8), "bad"))
            self.assertIsNone(localizer.locate(np.zeros((1, 2, 3), np.uint8), 0))

    def test_flow_refresh_uses_configured_ambiguity_margin(self):
        localizer = PlayerLocalizer(self.flow_config())
        image = np.zeros((100, 200, 3), np.uint8)
        image[40:70, 40:60] = self.template
        self.assertIsNotNone(localizer.locate(image, 0))
        noise = np.random.default_rng(9).normal(0, 24, self.template.shape)
        image[40:70, 120:140] = np.clip(self.template.astype(float) + noise, 0, 255).astype(
            np.uint8
        )
        self.assertIsNone(localizer.locate(image, 0.16))
        self.assertIsNone(localizer.locate(image, 0.18))
