"""Color template localization with expiring optical flow; scores are not probabilities.

Templates are small separate derivatives with provenance. No position survives a
missing observation. This recognizes one character/scene, not arbitrary avatars.
"""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np


class PlayerLocalizer:
    def __init__(self, config=None):
        root = Path(__file__).resolve().parents[1]
        if config is None:
            config = root / "config/player_templates.json"
        if isinstance(config, (str, Path)):
            path = Path(config)
            self.config = json.loads(path.read_text())
            base = path.parent
        else:
            self.config = dict(config)
            base = root
        self._validate_config()
        self.templates = []
        for entry in self.config.get("templates", []):
            path = Path(entry["path"])
            path = path if path.is_absolute() else base / path
            if (
                entry.get("template_sha256")
                and hashlib.sha256(path.read_bytes()).hexdigest() != entry["template_sha256"]
            ):
                raise ValueError(f"Player template hash mismatch: {path}")
            image = cv2.imread(str(path))
            if image is None or image.size == 0 or image.std() < 2:
                raise ValueError(f"Invalid player template: {path}")
            self.templates.append((entry, image))
        if not self.templates:
            raise ValueError("Player template bank is empty")
        self.last_diagnostics = {}
        self._timestamp = None
        self._flow_state = None
        self._last_ncc = None
        self._last_identity = None

    def _validate_config(self):
        """Reject invalid search/timing settings before they reach OpenCV."""
        if not isinstance(self.config, dict):
            raise ValueError("Player localizer configuration must be an object")
        for key, default in (
            ("working_width", 480),
            ("max_time_gap", 0.5),
            ("flow_max_gap", 0.2),
            ("flow_identity_ttl", 0.4),
            ("ncc_refresh_interval", 0.15),
        ):
            value = self.config.get(key, default)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(key + " must be finite and positive")
        roi = self.config.get("roi", [0.15, 0.38, 0.85, 0.93])
        roi_error = "roi must contain ordered normalized x1,y1,x2,y2"
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError(roi_error)
        for value in roi:
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(roi_error)
        if not (roi[0] < roi[2] and roi[1] < roi[3]):
            raise ValueError(roi_error)
        aspect = self.config.get("aspect_range", [0, float("inf")])
        aspect_error = "aspect_range must contain ordered nonnegative bounds"
        if not isinstance(aspect, (list, tuple)) or len(aspect) != 2:
            raise ValueError(aspect_error)
        for value in aspect:
            if not isinstance(value, (int, float)) or math.isnan(value):
                raise ValueError(aspect_error)
        if not 0 <= aspect[0] < aspect[1]:
            raise ValueError(aspect_error)
        scales = self.config.get("scales", [0.88, 1, 1.12])
        scale_error = "scales must contain finite positive values"
        if not isinstance(scales, (list, tuple)) or not scales:
            raise ValueError(scale_error)
        for value in scales:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(scale_error)
        for key, default, low, high in (
            ("score_threshold", 0.73, -1, 1),
            ("margin_threshold", 0.07, 0, 2),
        ):
            value = self.config.get(key, default)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(key + " outside supported correlation range")
        features = self.config.get("flow_min_features", 5)
        if not isinstance(features, int) or isinstance(features, bool) or features < 1:
            raise ValueError("flow_min_features must be a positive integer")
        entries = self.config.get("templates", [])
        if not isinstance(entries, list):
            raise ValueError("templates must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), (str, Path)):
                raise ValueError("Each player template needs a file path")
            dimension = entry.get("reference_width", entry.get("reference_height"))
            if (
                not isinstance(dimension, (int, float))
                or not math.isfinite(dimension)
                or dimension <= 0
            ):
                raise ValueError("Each player template needs a finite positive reference dimension")

    @staticmethod
    def _valid_input(image, timestamp):
        return (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[2] == 3
            and image.dtype == np.uint8
            and min(image.shape[:2]) >= 5
            and isinstance(timestamp, (int, float))
            and math.isfinite(timestamp)
        )

    def _clear_flow(self):
        self._flow_state = None
        self._last_identity = None
        self._last_ncc = None

    def _gray(self, image):
        ratio = min(1.0, self.config.get("working_width", 320) / image.shape[1])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray,
            (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
            interpolation=cv2.INTER_LINEAR,
        )
        return small, ratio

    def _remember(self, image, timestamp, result):
        gray, ratio = self._gray(image)
        x1, y1, x2, y2 = result["bbox"]
        x1, y1, x2, y2 = round(x1 * ratio), round(y1 * ratio), round(x2 * ratio), round(y2 * ratio)
        mask = np.zeros_like(gray)
        mask[max(0, y1) : min(gray.shape[0], y2), max(0, x1) : min(gray.shape[1], x2)] = 255
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=40, qualityLevel=0.01, minDistance=2, mask=mask, blockSize=3
        )
        self._flow_state = (gray, points, ratio, image.shape, timestamp, dict(result))

    def _flow(self, image, timestamp):
        """Follow image features briefly between full template checks."""
        if self._flow_state is None:
            return None
        previous, points, ratio, shape, previous_time, result = self._flow_state
        if points is None or len(points) < self.config.get("flow_min_features", 5):
            return None
        gray, new_ratio = self._gray(image)
        if shape != image.shape or new_ratio != ratio:
            return None
        if not 0 < timestamp - previous_time <= self.config.get("flow_max_gap", 0.2):
            return None
        forward, status, error = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(15, 15), maxLevel=2
        )
        if forward is None or status is None or error is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous, forward, None, winSize=(15, 15), maxLevel=2
        )
        if backward is None or back_status is None:
            return None
        fb = np.linalg.norm(points - backward, axis=2).reshape(-1)
        valid = (
            (status.reshape(-1) > 0)
            & (back_status.reshape(-1) > 0)
            & (fb < 1.0)
            & (error.reshape(-1) < 25.0)
        )
        if int(valid.sum()) < self.config.get("flow_min_features", 5):
            return None
        motion = (forward - points).reshape(-1, 2)[valid]
        displacement = np.median(motion, axis=0)
        coherent = np.linalg.norm(motion - displacement, axis=1) < 2.0
        if int(coherent.sum()) < self.config.get("flow_min_features", 5) or coherent.mean() < 0.65:
            return None
        displacement = np.median(motion[coherent], axis=0)
        max_motion = gray.shape[1] * 2 * (timestamp - previous_time) + 2
        if np.linalg.norm(displacement) > max_motion:
            return None
        dx, dy = displacement / ratio
        x1, y1, x2, y2 = result["bbox"]
        box = (float(x1 + dx), float(y1 + dy), float(x2 + dx), float(y2 + dy))
        h, w = image.shape[:2]
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
            return None
        # No NCC score is manufactured for a flow-only observation.
        result.update(
            bbox=box,
            score=None,
            score_margin=None,
            source="opticalflow",
            timestamp=float(timestamp),
            flow_features=int(coherent.sum()),
            flow_fb_error=float(np.median(fb[valid])),
            identity_age=timestamp - self._last_identity,
        )
        return result

    def locate(self, image, timestamp):
        started = perf_counter()
        if not self._valid_input(image, timestamp):
            self._clear_flow()
            self.last_diagnostics = {"reason": "invalid input"}
            return None
        if not self.config.get("enable_flow", False):
            return self._match(image, timestamp)
        if self._flow_state is not None:
            _, _, _, shape, previous_time, _ = self._flow_state
            if (
                shape != image.shape
                or timestamp < previous_time
                or timestamp - previous_time > self.config.get("flow_max_gap", 0.2)
            ):
                self._clear_flow()
        refresh = self._last_ncc is None or timestamp - self._last_ncc >= self.config.get(
            "ncc_refresh_interval", 0.15
        )
        if refresh:
            match = self._match(image, timestamp)
            self._last_ncc = timestamp
            if match is not None:
                self._last_identity = timestamp
                self._remember(image, timestamp, match)
                self.last_diagnostics["source"] = "NCC"
                return match
            # Two strong spatial candidates cannot establish the tracked identity.
            if self.last_diagnostics.get("second_score", 0) >= self.config.get(
                "score_threshold", 0.73
            ) and self.last_diagnostics.get("score_margin", 1) < self.config.get(
                "margin_threshold", 0.07
            ):
                self._clear_flow()
                return None
        if self._last_identity is None or timestamp - self._last_identity > self.config.get(
            "flow_identity_ttl", 0.4
        ):
            self._clear_flow()
            self.last_diagnostics = {
                "reason": "No recent strong identity match; flow identity expired"
            }
            return None
        result = self._flow(image, timestamp)
        if result is None:
            self._clear_flow()
            self.last_diagnostics = {"reason": "optical flow failed; identity discarded"}
            return None
        self._remember(image, timestamp, result)
        self.last_diagnostics = dict(
            reason="verified forward/backward optical flow",
            source="opticalflow",
            features=result["flow_features"],
            fb_error=result["flow_fb_error"],
            identity_age=result["identity_age"],
            total_ms=(perf_counter() - started) * 1000,
        )
        return result

    def _match(self, image, timestamp):
        """Find the best character template, rejecting weak or competing matches."""
        start = perf_counter()
        self.last_diagnostics = {"reason": "invalid input", "movement": "UNKNOWN"}
        h, w = image.shape[:2]
        aspect = self.config.get("aspect_range", [0, float("inf")])
        if not aspect[0] <= w / h <= aspect[1]:
            self.last_diagnostics["reason"] = "image aspect outside template scope"
            return None
        # Independent matching on every call means reset/gaps cannot retain stale positions.
        reset = self._timestamp is not None and (
            timestamp < self._timestamp
            or timestamp - self._timestamp > self.config.get("max_time_gap", 0.5)
        )
        self._timestamp = timestamp
        ratio = min(1.0, self.config.get("working_width", 480) / w)
        small = cv2.resize(
            image,
            (max(1, round(w * ratio)), max(1, round(h * ratio))),
            interpolation=cv2.INTER_AREA,
        )
        sh, sw = small.shape[:2]
        rx1, ry1, rx2, ry2 = self.config.get("roi", [0.15, 0.38, 0.85, 0.93])
        # The old wide search sometimes locked onto underwater scenery far
        # left of the runner, then propagated that false match by optical flow.
        # Limit actor identity to the visually reviewed screen-lane corridor.
        # This gate is intentionally independent of track/obstacle geometry.
        allowed_x = None
        lane_reference = self.config.get("lane_tracking", {})
        if isinstance(lane_reference, dict):
            centers = lane_reference.get("centers")
            if isinstance(centers, list) and len(centers) == 3:
                margin = float(lane_reference.get("max_error", 0.035)) + 0.015
                allowed_x = (
                    max(0.0, min(centers) - margin),
                    min(1.0, max(centers) + margin),
                )
                # Leave room for the actor's box width around its center.
                rx1 = max(rx1, allowed_x[0] - 0.06)
                rx2 = min(rx2, allowed_x[1] + 0.06)
        ox, oy = int(rx1 * sw), int(ry1 * sh)
        roi = small[oy : int(ry2 * sh), ox : int(rx2 * sw)]
        if min(roi.shape[:2]) < 5:
            self.last_diagnostics["reason"] = "Search region too small for player templates"
            return None
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        candidates = []
        for entry, original in self.templates:
            for scale in self.config.get("scales", [0.88, 1, 1.12]):
                factor = (
                    (
                        w / entry["reference_width"]
                        if "reference_width" in entry
                        else h / entry["reference_height"]
                    )
                    * ratio
                    * scale
                )
                tw, th = round(original.shape[1] * factor), round(original.shape[0] * factor)
                if min(tw, th) < 5 or tw > roi.shape[1] or th > roi.shape[0]:
                    continue
                template = cv2.resize(original, (tw, th), interpolation=cv2.INTER_AREA)
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                scores = cv2.matchTemplate(gray_roi, gray_template, cv2.TM_CCOEFF_NORMED)
                if allowed_x is not None:
                    # A match's center, not its top-left corner, must lie in
                    # the plausible player corridor. Exclude out-of-scope
                    # peaks BEFORE selecting the strongest candidates.
                    low = max(0, int(np.ceil(allowed_x[0] * sw - ox - tw / 2)))
                    high = min(
                        scores.shape[1],
                        int(np.floor(allowed_x[1] * sw - ox - tw / 2)) + 1,
                    )
                    if low >= high:
                        continue
                    scores[:, :low] = -1
                    scores[:, high:] = -1
                # Keep three spatially distinct peaks per template/scale.
                for _ in range(3):
                    _, score, _, location = cv2.minMaxLoc(scores)
                    x, y = location
                    color_score = float(
                        cv2.matchTemplate(
                            roi[y : y + th, x : x + tw], template, cv2.TM_CCOEFF_NORMED
                        )[0, 0]
                    )
                    candidates.append(
                        (color_score, (x + ox, y + oy, x + ox + tw, y + oy + th), entry)
                    )
                    scores[
                        max(0, y - th // 2) : y + th // 2 + 1, max(0, x - tw // 2) : x + tw // 2 + 1
                    ] = -1

        def match_score(candidate):
            return candidate[0]

        candidates.sort(key=match_score, reverse=True)
        if not candidates:
            self.last_diagnostics["reason"] = "No player template fits reviewed screen-lane corridor"
            return None
        score, box, entry = candidates[0]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        # Scales/templates agreeing on the same actor are not competitors.
        competitors = []
        for other_score, other_box, _ in candidates[1:]:
            other_x = (other_box[0] + other_box[2]) / 2
            other_y = (other_box[1] + other_box[3]) / 2
            separate_x = abs(other_x - cx) > (box[2] - box[0]) * 0.55
            separate_y = abs(other_y - cy) > (box[3] - box[1]) * 0.55
            if separate_x or separate_y:
                competitors.append(other_score)
        second = max(competitors, default=-1.0)
        margin = score - second
        reason = (
            "accepted"
            if score >= self.config.get("score_threshold", 0.73)
            and margin >= self.config.get("margin_threshold", 0.07)
            else "weak or spatially ambiguous correlation"
        )
        original_box = []
        for coordinate in box:
            original_box.append(float(coordinate / ratio))
        original_box = tuple(original_box)
        self.last_diagnostics = dict(
            reason=reason,
            score=score,
            score_margin=margin,
            second_score=second,
            best_bbox=original_box,
            reset=reset,
            total_ms=(perf_counter() - start) * 1000,
        )
        if reason != "accepted":
            return None
        return dict(
            bbox=original_box,
            score=score,
            score_margin=margin,
            movement="UNKNOWN",
            bbox_kind=entry.get("bbox_kind", "unknown"),
            source="automatic color-template NCC",
            provenance=entry.get("provenance", ""),
            template_source=entry.get("source"),
            timestamp=float(timestamp),
            footpoint_valid=False,
        )
