"""Frame-local provisional lane geometry from visible track-rail perspective.

Fit a regular six-rail fan from four or more independently visible metal/colored
rail segments. Missing rails may be interpolated *within* observed ground depths.
Never hold geometry through an occlusion, treat a box as verified occupancy, or
infer lanes from screen thirds. This is a scene-dependent visual heuristic.
"""

import cv2
import numpy as np

from .geometry import Geometry


class TrackGeometry:
    def __init__(self, config=None):
        self.config = config or {}
        self.last_diagnostics = {}

    def _reject(self, reason, groups=0):
        self.last_diagnostics = {
            "accepted": False,
            "reason": reason,
            "material_rail_groups": groups,
        }
        geometry = Geometry()
        geometry.reason = "Automatic rail geometry unavailable: " + reason
        return geometry

    def estimate(self, image, frame_id=""):
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
            return self._reject("Image too small for rail geometry")
        reference_y = 0.70 * h
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 40, 100)
        edges[: int(0.42 * h)] = 0
        edges[int(0.95 * h) :] = 0
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 45,
            minLineLength=round(0.17 * h), maxLineGap=round(0.025 * h),
        )
        candidates = []
        for x1, y1, x2, y2 in ([] if lines is None else lines.reshape(-1, 4)):
            span = abs(int(y2) - int(y1))
            if span < 0.17 * h or max(y1, y2) < 0.82 * h or min(y1, y2) > 0.75 * h:
                continue
            slope = (float(x2) - x1) / (float(y2) - y1)
            if abs(slope) > 1.35:
                continue
            projected = float(x1) + (reference_y - float(y1)) * slope
            if not 0.16 * w < projected < 0.84 * w:
                continue
            # Longitudinal metal rails are relatively neutral gray/blue; older
            # scene-specific rails are pink. Reject purple platform edges.
            top = max(min(y1, y2), 0.57 * h)
            bottom = min(max(y1, y2), 0.93 * h)
            ys = np.rint(np.linspace(top, bottom, 30)).astype(int)
            xs = np.rint(projected + (ys - reference_y) * slope).astype(int)
            inside = (xs >= 3) & (xs < w - 3)
            xs, ys = xs[inside], ys[inside]
            if len(xs) < 15:
                continue
            pixels = image[ys[:, None], xs[:, None] + np.arange(-2, 3)[None, :]]
            blue, green, red = np.moveaxis(pixels.astype(np.int16), 2, 0)
            neutral = (blue - green < 38) & (red - green < 28) & (green > 42)
            pink = (
                (red - green > 16) & (red - blue > 16)
                & (np.abs(blue - green) < 20) & (red > 115)
            )
            if max(float(neutral.mean()), float(pink.mean())) < 0.40:
                continue
            candidates.append(
                (projected, slope, float(min(y1, y2)), float(max(y1, y2)), span)
            )
        # Parallel Hough fragments / highlight edges are correlated and count as
        # one rail. Their quantity is not interpreted as detection confidence.
        groups = []
        for candidate in sorted(candidates):
            if groups:
                previous_x = float(np.median([segment[0] for segment in groups[-1]]))
                previous_slope = float(np.median([segment[1] for segment in groups[-1]]))
                if (
                    candidate[0] - previous_x <= 0.014 * w
                    and abs(candidate[1] - previous_slope) < 0.11
                ):
                    groups[-1].append(candidate)
                    continue
            groups.append([candidate])
        observed = []
        for group in groups:
            values = np.asarray(group)
            observed.append((
                float(np.median(values[:, 0])),
                float(np.median(values[:, 1])),
                float(np.min(values[:, 2])),
                float(np.max(values[:, 3])),
                float(np.max(values[:, 4])),
            ))
        if len(observed) < 4:
            return self._reject(
                f"Only {len(observed)} independently visible material-matched rail groups; need 4",
                len(observed),
            )
        best = None
        hypotheses = []
        # Search 6-rail lattice phases without assuming where the player stands.
        # Positions and line slopes must agree with the SAME perspective fan.
        for a_index, a in enumerate(observed):
            for b in observed[a_index + 1 :]:
                gap = b[0] - a[0]
                if gap <= 0:
                    continue
                convergence = (b[1] - a[1]) / gap
                if not 1 / (0.65 * h) < convergence < 1 / (0.25 * h):
                    continue
                for separation in range(1, 6):
                    step = gap / separation
                    if not 0.035 * w <= step <= 0.135 * w:
                        continue
                    slope_step = (b[1] - a[1]) / separation
                    for first_index in range(6 - separation):
                        x0 = a[0] - first_index * step
                        slope0 = a[1] - first_index * slope_step
                        if x0 < 0.10 * w or x0 + 5 * step > 0.90 * w:
                            continue
                        used = set()
                        matches = []
                        error = 0.0
                        for rail in range(6):
                            target_x = x0 + rail * step
                            target_slope = slope0 + rail * slope_step
                            nearest = None
                            for group_index, group in enumerate(observed):
                                if group_index in used:
                                    continue
                                dx = abs(group[0] - target_x) / max(8.0, 0.18 * step)
                                ds = abs(group[1] - target_slope) / 0.085
                                residual = dx * dx + ds * ds
                                if dx <= 1 and ds <= 1 and (
                                    nearest is None or residual < nearest[0]
                                ):
                                    nearest = (residual, group_index, group)
                            if nearest is not None:
                                residual, group_index, group = nearest
                                used.add(group_index)
                                matches.append((rail, group_index, group))
                                error += residual
                        if len(matches) < 4 or matches[-1][0] - matches[0][0] < 4:
                            continue
                        center = x0 + 2.5 * step
                        centeredness = abs(center - 0.50 * w) / (0.11 * w)
                        exterior = 0
                        for group_index, group in enumerate(observed):
                            if group_index in used:
                                continue
                            if (
                                abs(group[0] - (x0 - step)) < 0.25 * step
                                or abs(group[0] - (x0 + 6 * step)) < 0.25 * step
                            ):
                                exterior += 1
                        score = (
                            12 * len(matches) - 2 * error
                            + 0.7 * min(exterior, 2)
                            - 1.7 * max(0.0, centeredness - 0.25)
                            + 0.02 * sum(m[2][4] for m in matches) / h
                        )
                        hypothesis = (score, center, step, matches)
                        hypotheses.append(hypothesis)
                        if best is None or score > best[0]:
                            best = hypothesis
        if best is None:
            return self._reject("Visible rail groups fail six-rail perspective consistency", len(observed))
        score, center, step, matches = best
        # Multiple equally credible lateral phases must remain UNKNOWN.
        alternatives = []
        for hypothesis in hypotheses:
            if abs(hypothesis[1] - center) > 0.60 * step:
                alternatives.append(hypothesis)
        if alternatives and score - max(alt[0] for alt in alternatives) < 0.65:
            return self._reject("Ambiguous six-rail lattice phase", len(observed))
        shared_top = max(match[2][2] for match in matches) / h
        shared_bottom = min(match[2][3] for match in matches) / h
        if shared_bottom - shared_top < 0.12:
            return self._reject("Insufficient shared visible rail depth", len(observed))
        sample = np.asarray([(rail, group[0], group[1]) for rail, _, group in matches])
        pos_step, pos_origin = np.polyfit(sample[:, 0], sample[:, 1], 1)
        slope_step, slope_origin = np.polyfit(sample[:, 0], sample[:, 2], 1)
        rows = []
        for y in (shared_top, (shared_top + shared_bottom) / 2, shared_bottom):
            projected = []
            for rail in range(6):
                x = pos_origin + pos_step * rail
                x += (slope_origin + slope_step * rail) * (y * h - reference_y)
                projected.append(x / w)
            if (
                min(projected) <= 0 or max(projected) >= 1
                or min(np.diff(projected)) <= 0.025
            ):
                return self._reject("Inferred rail fan leaves visible track bounds", len(observed))
            centers = []
            for lane in range(3):
                centers.append((projected[2 * lane] + projected[2 * lane + 1]) / 2)
            rows.append([float(y)] + centers)
        geometry = Geometry({
            "lane_rows": rows,
            "aspect_ratio": w / h,
            "automatic": True,
            "valid_frames": [str(frame_id).split("/")[-1]],
            "source_frame": str(frame_id),
            "provenance": {
                "method": "partial metal/colored rail lattice; 4+ visible rails and perspective gates",
                "scope": "current image only; interpolated rails are provisional, not measured",
            },
        })
        self.last_diagnostics = {
            "accepted": geometry.valid,
            "reason": geometry.reason,
            "material_rail_groups": len(observed),
            "matched_rails": len(matches),
            "lane_rows": rows,
        }
        return geometry
