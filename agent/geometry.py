"""Scoped, piecewise-linear lane calibration. Boxes are not occupancy maps.

Rows and measured footprint points are normalized to the original image. An unreviewed object box supplies possible occupied lanes, never a known lane.
"""

import math
from pathlib import Path
import re

from .state import Detection, LaneEstimate


def _finite(values):
    for value in values:
        if not math.isfinite(value):
            return False
    return True


class Geometry:
    def __init__(self, config=None):
        self.config = config or {}
        self.rows = []
        self.valid = False
        self.reason = "No measured lane calibration"
        try:
            rows = self.config.get("lane_rows", [])
            if not self.config.get("provenance") or len(rows) < 2:
                return
            if not (
                self.config.get("valid_frames")
                or self.config.get("frame_range")
                or self.config.get("source_session")
            ):
                self.reason = "Calibration requires explicit frame/session scope"
                return
            frames = self.config.get("valid_frames")
            if frames is not None:
                if not isinstance(frames, list):
                    raise ValueError("invalid frame scope")
                for frame in frames:
                    if not isinstance(frame, str) or not frame:
                        raise ValueError("invalid frame scope")
            frame_range = self.config.get("frame_range")
            if frame_range is not None:
                if not isinstance(frame_range, list) or len(frame_range) != 2:
                    raise ValueError("invalid frame range")
                for number in frame_range:
                    if not isinstance(number, int) or isinstance(number, bool) or number < 0:
                        raise ValueError("invalid frame range")
                if frame_range[0] > frame_range[1]:
                    raise ValueError("invalid frame range")
            session = self.config.get("source_session")
            if session is not None and (not isinstance(session, str) or not session):
                raise ValueError("invalid session scope")
            margin = float(self.config.get("boundary_margin", 0.015))
            if not math.isfinite(margin) or not 0 <= margin <= 0.1:
                raise ValueError("invalid lane boundary margin")
            previous = -1.0
            for row in rows:
                y, left, center, right = map(float, row)
                for value in (y, left, center, right):
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("unordered or out-of-range rows")
                if not (previous < y and left < center < right):
                    raise ValueError("unordered or out-of-range rows")
                previous = y
                self.rows.append([y, left, center, right])
            ratio = self.config.get("aspect_ratio")
            if ratio is not None and (not math.isfinite(float(ratio)) or float(ratio) <= 0):
                raise ValueError("invalid aspect ratio")
            self.valid = True
            source = (
                "Automatic visual rail-pattern estimate; scene-specific heuristic"
                if self.config.get("automatic")
                else "Measured calibration"
            )
            self.reason = (
                "Provisional calibration candidates" if self.config.get("provisional") else source
            ) + "; usable only inside recorded scope and depth"
        except (ValueError, TypeError):
            self.rows = []
            self.reason = "Invalid lane calibration"

    def lane_centers(self, y_norm):
        if (
            not self.valid
            or not isinstance(y_norm, (int, float))
            or not math.isfinite(y_norm)
            or not self.rows[0][0] <= y_norm <= self.rows[-1][0]
        ):
            return None
        for a, b in zip(self.rows, self.rows[1:]):
            if a[0] <= y_norm <= b[0]:
                weight = (y_norm - a[0]) / (b[0] - a[0])
                centers = []
                for index in (1, 2, 3):
                    centers.append(a[index] + weight * (b[index] - a[index]))
                return centers
        return None

    @staticmethod
    def _lane_boundaries(centers):
        left, center, right = centers
        return [
            left - (center - left) / 2,
            (left + center) / 2,
            (center + right) / 2,
            right + (right - center) / 2,
        ]

    def _scope_reason(self, image_shape, frame_id):
        if not self.valid:
            return self.reason
        try:
            h, w = image_shape[:2]
            if not _finite((h, w)) or min(h, w) <= 0:
                return "Invalid image shape"
            ratio = self.config.get("aspect_ratio")
            if ratio and abs(w / h - float(ratio)) > 0.01 * float(ratio):
                return "Image aspect ratio differs from calibration"
            name = Path(frame_id).name
            exact_source = self.config.get("source_frame")
            if exact_source and Path(frame_id).resolve() != Path(exact_source).resolve():
                return "Frame differs from calibrated source image"
            frames = self.config.get("valid_frames")
            if frames and name not in frames:
                return "Frame outside calibration scope"
            session = self.config.get("source_session")
            if session and session not in Path(frame_id).parts:
                return "Session outside calibration scope; pass session/frame filename"
            frame_range = self.config.get("frame_range")
            if frame_range:
                match = re.search(r"(\d+)(?=\.[^.]+$)", name)
                if not match or not int(frame_range[0]) <= int(match.group()) <= int(
                    frame_range[1]
                ):
                    return "Frame outside calibration range"
        except (TypeError, ValueError, IndexError):
            return "Invalid image shape or calibration scope"
        return None

    def validate_frame(self, image_shape, frame_id=""):
        """Check current crop and frame scope, rather than config validity alone."""
        reason = self._scope_reason(image_shape, frame_id)
        return reason is None, reason or self.reason

    def map_player(
        self, bbox, image_shape, frame_id="", movement="UNKNOWN", footpoint_trustworthy=True
    ):
        """Map observed ground feet, never project elevated/unknown feet to ground.

        RUNNING must come from independent visual state evidence. A head/template
        box is not a full-body footpoint: callers must pass trustworthy=False.
        No position or lane is propagated across frames by this method.
        """
        valid, reason = self.validate_frame(image_shape, frame_id)
        if not valid:
            return LaneEstimate(reason=reason)
        if movement != "RUNNING" or not footpoint_trustworthy:
            return LaneEstimate(
                reason="Ground footpoint unavailable: elevated, unsupported, or unknown player pose"
            )
        try:
            x1, y1, x2, y2 = map(float, bbox)
            h, w = image_shape[:2]
            if not _finite((x1, y1, x2, y2)) or not (0 < x1 < x2 < w and 0 <= y1 < y2 < h):
                return LaneEstimate(reason="Invalid or clipped full-body player box")
        except (ValueError, TypeError):
            return LaneEstimate(reason="Missing or invalid full-body player box")
        result = self.assign(
            Detection(-1, 0.0, (x1, y1, x2, y2), 0.0),
            image_shape,
            frame_id,
            footprint=((x1 + x2) / (2 * w), y2 / h),
        )
        if result.status == "KNOWN":
            result.reason = (
                "Automatic observed rails and running ground footpoint; heuristic estimate"
                if self.config.get("automatic")
                else "Calibrated lanes and observed running ground footpoint"
            )
        return result

    def player_candidates(self, bbox, image_shape, frame_id=""):
        """Project an upper-body x span across observed lower track depths.

        This is an image-column hypothesis, not evidence of ground contact or
        elevation. Even one candidate is UNKNOWN, never a known player lane.
        The box bottom is only a lower limit for projection, not invented feet.
        """
        valid, reason = self.validate_frame(image_shape, frame_id)
        if not valid:
            return LaneEstimate(reason=reason)
        try:
            x1, y1, x2, y2 = map(float, bbox)
            h, w = image_shape[:2]
            if not _finite((x1, y1, x2, y2)) or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 < h):
                return LaneEstimate(
                    reason="Invalid or clipped upper-body box; foot contact unknown"
                )
        except (ValueError, TypeError):
            return LaneEstimate(reason="Missing upper-body box; foot contact unknown")
        lower = max(self.rows[0][0], y2 / h)
        upper = self.rows[-1][0]
        if lower > upper:
            return LaneEstimate(
                reason="No supported ground depth below upper-body box; foot contact unknown"
            )
        depths = [lower]
        for row in self.rows:
            if lower < row[0] <= upper:
                depths.append(row[0])
        boundaries = []
        for y in depths:
            boundaries.append(self._lane_boundaries(self.lane_centers(y)))
        margin = float(self.config.get("boundary_margin", 0.015))
        candidates = []
        # Compare the player's image column with each supported lane region.
        columns = list(zip(*boundaries))
        for lane in range(3):
            left_edge = min(columns[lane]) - margin
            right_edge = max(columns[lane + 1]) + margin
            if x2 / w >= left_edge and x1 / w <= right_edge:
                candidates.append(lane)
        candidates = tuple(candidates)
        reason = "upper-body image-column projection only; foot contact unknown"
        if not candidates:
            reason += "; no intersection with supported ground lane regions"
        return LaneEstimate(candidates, "AMBIGUOUS" if len(candidates) > 1 else "UNKNOWN", reason)

    def _provisional_occupancy(self, bbox, image_shape):
        """Project the entire visible box width onto calibrated lane intervals.

        This estimates *possible* occupied lanes, not physical ground contact.
        A clipped box or depth outside observed rails cannot safely define a
        collision footprint. Multiple candidates are not proof of multiple blocks.
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        if x1 <= 0 or x2 >= w or y2 >= h:
            return LaneEstimate(reason="Clipped detection; obstacle occupancy unverified")
        depth = y2 / h
        centers = self.lane_centers(depth)
        if centers is None:
            return LaneEstimate(reason="Obstacle bottom outside calibrated depth")
        boundaries = self._lane_boundaries(centers)
        left, right = x1 / w, x2 / w
        candidates = []
        for lane in range(3):
            lane_width = boundaries[lane + 1] - boundaries[lane]
            overlap = min(right, boundaries[lane + 1]) - max(left, boundaries[lane])
            if overlap > 0 and overlap >= 0.12 * lane_width:
                candidates.append(lane)
        if not candidates:
            return LaneEstimate(
                reason="No supported lane overlap; obstacle occupancy unverified",
                depth=depth,
                footprint=((left + right) / 2, depth),
            )
        return LaneEstimate(
            tuple(candidates),
            "UNKNOWN",
            "provisional box-width lane overlap; physical blocking footprint unverified",
            depth,
            ((left + right) / 2, depth),
        )

    def assign(self, detection, image_shape, frame_id="", footprint=None):
        reason = self._scope_reason(image_shape, frame_id)
        if reason:
            return LaneEstimate(reason=reason)
        h, w = image_shape[:2]
        try:
            x1, y1, x2, y2 = map(float, detection.bbox)
        except (ValueError, TypeError):
            return LaneEstimate(reason="Invalid detection box")
        if not _finite((x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return LaneEstimate(reason="Invalid detection box")
        if detection.class_id not in (0, 1, 2, 3, 4, 5, -1):
            return LaneEstimate(reason="Unsupported object class")
        # Explicitly reviewed *multi-lane* occupancy is supported. A detector box
        # alone never receives KNOWN status, irrespective of confidence or history.
        if detection.class_id >= 0:
            for item in self.config.get("occupied_lanes", {}).get(Path(frame_id).name, []):
                if not isinstance(item, dict):
                    continue
                if tuple(item.get("bbox", [])) != tuple(detection.bbox):
                    continue
                lanes = item.get("lanes")
                if not item.get("provenance") or not isinstance(lanes, list):
                    continue
                if (
                    not lanes
                    or any(type(lane) is not int or lane not in (0, 1, 2) for lane in lanes)
                    or len(set(lanes)) != len(lanes)
                ):
                    continue
                status = "UNKNOWN" if self.config.get("provisional") else "KNOWN"
                return LaneEstimate(
                    tuple(sorted(lanes)), status, "Reviewed occupied lanes", y2 / h
                )
        measured = footprint is not None
        if footprint is None:
            for item in self.config.get("footprints", {}).get(Path(frame_id).name, []):
                if item.get("provenance") and tuple(item.get("bbox", [])) == tuple(detection.bbox):
                    footprint = item.get("point")
                    measured = footprint is not None
                    break
        if footprint is None:
            if detection.class_id == -1:
                return LaneEstimate(reason="Player ground footpoint missing")
            return self._provisional_occupancy((x1, y1, x2, y2), image_shape)
        try:
            x, y = map(float, footprint)
        except (TypeError, ValueError):
            return LaneEstimate(reason="Invalid measured footprint")
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            return LaneEstimate(reason="Footprint outside image")
        centers = self.lane_centers(y)
        if centers is None:
            return LaneEstimate(reason="Footprint outside calibrated depth", footprint=(x, y))
        # Reject points beyond extrapolated outer track extents.
        boundaries = self._lane_boundaries(centers)
        if not boundaries[0] <= x <= boundaries[3]:
            return LaneEstimate(
                reason="Footprint outside calibrated lanes", depth=y, footprint=(x, y)
            )
        tolerance = float(self.config.get("boundary_margin", 0.015))
        for index, boundary in enumerate(boundaries[1:3]):
            if abs(x - boundary) <= tolerance:
                return LaneEstimate(
                    (index, index + 1), "AMBIGUOUS", "Footprint near lane boundary", y, (x, y)
                )
        lane = 0 if x < boundaries[1] else 1 if x < boundaries[2] else 2
        provisional = bool(self.config.get("provisional"))
        reason = (
            "provisional calibration; lane candidate only"
            if provisional
            else (
                "Reviewed visible footprint"
                if measured
                else "provisional obstacle bottom-center candidate; visible footprint unverified"
            )
        )
        return LaneEstimate(
            (lane,), "KNOWN" if measured and not provisional else "UNKNOWN", reason, y, (x, y)
        )
