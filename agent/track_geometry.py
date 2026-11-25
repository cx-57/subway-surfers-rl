"""Per-frame rail-pattern geometry for the currently recorded pink-rail scene.

This is a small visual heuristic, not a general rail detector or an accuracy
calibration. All six rails must be visible with perspective-consistent gauges.
Every frame is re-estimated; occlusion/curves/camera changes can yield UNKNOWN.
No lane numbering is inferred from a solitary visible track or screen thirds.
"""

import cv2
import numpy as np

from .geometry import Geometry


class TrackGeometry:
    def __init__(self, config=None):
        self.config = config or {}
        self.last_diagnostics = {}

    def _reject(self, reason, groups=None):
        self.last_diagnostics = {"accepted": False, "reason": reason, "rail_groups": groups or []}
        geometry = Geometry()
        geometry.reason = "Automatic rail geometry unavailable: " + reason
        return geometry

    def estimate(self, image, frame_id=""):
        """Return a frame-scoped Geometry or an explicitly invalid Geometry.

        Rail material thresholds describe the inspected scene and may reject
        other themes. Successful pattern agreement supplies no numeric confidence.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or not image.size
        ):
            return self._reject("Expected nonempty uint8 BGR original image")
        if not frame_id:
            return self._reject("Unique frame identity required")
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return self._reject("Image too small to establish six visible rails")
        reference_y = 0.7 * h
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 40, 100)
        edges[: int(h * 0.42)] = 0
        edges[int(h * 0.95) :] = 0
        # One-degree bins retain the inspected rail fans with much less CPU work.
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 45, minLineLength=round(h * 0.17), maxLineGap=round(h * 0.025)
        )
        candidates = []
        for ((x1, y1, x2, y2),) in ([] if lines is None else lines):
            span = abs(int(y2) - int(y1))
            if span < h * 0.17 or max(y1, y2) < h * 0.82 or min(y1, y2) > h * 0.75:
                continue
            slope = (float(x2) - x1) / (float(y2) - y1)
            if abs(slope) > 1.5:
                continue
            points = np.linspace([x1, y1], [x2, y2], 40).astype(int)
            blue, green, red = image[points[:, 1], points[:, 0]].astype(float).T
            # Pink rail highlights, evaluated along the actual line pixels;
            # ground ties and gray paths do not establish longitudinal rails.
            material = (
                (red - green > 16) & (red - blue > 16) & (np.abs(blue - green) < 20) & (red > 115)
            )
            if np.mean(material) < 0.5:
                continue
            projected = float(x1) + (reference_y - float(y1)) * slope
            if not 0 <= projected < w:
                continue
            candidates.append((projected, slope, float(min(y1, y2)), float(max(y1, y2))))
        groups = []
        for candidate in sorted(candidates):
            if groups:
                previous_group = np.asarray(groups[-1])
                previous_x = float(np.mean(previous_group[:, 0]))
            if groups and candidate[0] - previous_x < w * 0.025:
                groups[-1].append(candidate)
            else:
                groups.append([candidate])
        summary = []
        positions, slopes, tops, bottoms = [], [], [], []
        for group in groups:
            # Columns contain each segment's x position, slope and visible depth.
            values = np.asarray(group)
            x = float(np.median(values[:, 0]))
            slope = float(np.median(values[:, 1]))
            top = float(values[:, 2].min())
            bottom = float(values[:, 3].max())
            summary.append(dict(x=x, slope=slope, segments=len(group), top=top, bottom=bottom))
            positions.append(x)
            slopes.append(slope)
            tops.append(top)
            bottoms.append(bottom)
        # Duplicate Hough segments are correlated edges, not independent
        # confidence votes. Shared observed depth and perspective gates below
        # determine whether the six distinct rail groups support geometry.
        if len(groups) != 6:
            return self._reject("Six independently supported visible rails not found", summary)
        positions = np.array(positions)
        slopes = np.array(slopes)
        slope_gaps = np.diff(slopes)
        if slopes[0] >= -0.05 or slopes[-1] <= 0.05 or np.any(slope_gaps <= 0.04):
            return self._reject("Rail directions do not form an ordered perspective fan", summary)
        intersections = reference_y - np.diff(positions) / slope_gaps
        if (
            np.any(intersections < 0)
            or np.any(intersections > h * 0.65)
            or np.ptp(intersections) > h * 0.3
        ):
            return self._reject("Inconsistent rail convergence or curved geometry", summary)
        low = max(tops) / h
        high = min(bottoms) / h
        if high - low < 0.15:
            return self._reject("Insufficient shared observed rail depth", summary)
        rows = []
        for y in (low, (low + high) / 2, high):
            rails = positions + slopes * (y * h - reference_y)
            gaps = np.diff(rails)
            gauges = rails[1::2] - rails[::2]
            centers = (rails[1::2] + rails[::2]) / 2
            spacing = np.diff(centers)
            if (
                np.any(rails < 0)
                or np.any(rails >= w)
                or np.min(gaps) <= w * 0.025
                or max(gauges) / min(gauges) > 1.5
                or max(spacing) / min(spacing) > 1.4
                or max(gaps) / min(gaps) > 1.8
            ):
                return self._reject("Rail pair gauges or lane spacing inconsistent", summary)
            rows.append([float(y)] + (centers / w).tolist())
        geometry = Geometry(
            {
                "lane_rows": rows,
                "aspect_ratio": w / h,
                "automatic": True,
                "valid_frames": [str(frame_id).split("/")[-1]],
                "source_frame": str(frame_id),
                "provenance": {
                    "method": "per-frame six visible pink rail pattern with perspective gates",
                    "scope": "current image only; scene-specific heuristic, no measured accuracy",
                },
            }
        )
        self.last_diagnostics = {
            "accepted": geometry.valid,
            "reason": geometry.reason,
            "rail_groups": summary,
            "lane_rows": rows,
        }
        return geometry
