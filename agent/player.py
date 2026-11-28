"""Visual character tracking plus scoped reviewed observations; uncertainty is explicit."""

import math
from pathlib import Path
from .state import PlayerState


class PlayerLaneTracker:
    """Provisional screen-lane tracking from Codex's existing upper-body boxes.

    The character's horizontal location is compared with reviewed scene-specific
    columns. Consecutive matching observations stabilize a provisional lane.
    No pose, feet, ground contact, or safe action is inferred.
    """

    def __init__(self, config):
        if not isinstance(config, dict) or not config.get("provenance"):
            raise ValueError("Player lane reference requires provenance")
        centers = config.get("centers")
        if (
            not isinstance(centers, list)
            or len(centers) != 3
            or any(
                not isinstance(x, (float, int)) or isinstance(x, bool)
                or not math.isfinite(x) or not 0 <= x <= 1
                for x in centers
            )
            or not 0.035 <= centers[1] - centers[0]
            or not 0.035 <= centers[2] - centers[1]
        ):
            raise ValueError("Player lane centers must be three ordered screen columns")
        self.centers = tuple(centers)
        boundaries = config.get("boundaries")
        if boundaries is None:
            boundaries = [
                (centers[0] + centers[1]) / 2,
                (centers[1] + centers[2]) / 2,
            ]
        if (
            not isinstance(boundaries, (list, tuple))
            or len(boundaries) != 2
            or any(
                not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(value)
                for value in boundaries
            )
            or not centers[0] < boundaries[0] < centers[1]
            or not centers[1] < boundaries[1] < centers[2]
        ):
            raise ValueError("Lane boundaries must separate reviewed screen positions")
        self.boundaries = tuple(boundaries)
        self.aspect_ratio = float(config.get("aspect_ratio", 0))
        self.aspect_tolerance = float(config.get("aspect_tolerance", 0.08))
        self.max_error = float(config.get("max_error", 0.035))
        self.margin = float(config.get("boundary_margin", 0.012))
        self.max_gap = float(config.get("max_gap_seconds", 0.35))
        self.stable_frames = config.get("stable_frames", 2)
        if (
            not all(
                math.isfinite(value) for value in
                (self.aspect_ratio, self.aspect_tolerance, self.max_error,
                 self.margin, self.max_gap)
            )
            or self.aspect_ratio <= 0
            or not 0 < self.aspect_tolerance <= 0.15
            or not 0 < self.max_error <= 0.08
            or not 0 < self.margin <= 0.05
            or not 0 < self.max_gap <= 1
            or type(self.stable_frames) is not int
            or not 2 <= self.stable_frames <= 8
        ):
            raise ValueError("Invalid provisional player lane tracking settings")
        self.reset()

    def reset(self):
        self.last_timestamp = None
        self.last_lane = None
        self.streak = 0

    def estimate(self, bbox, timestamp, shape, geometry_candidates=()):
        """Return a lane hypothesis; never turn it into a confirmed ground lane."""
        result = dict(
            lane=None, candidates=(), stable=False, transitioning=None,
            provisional=True, reason="Screen lane unavailable",
        )
        try:
            h, w = shape[:2]
            x1, y1, x2, y2 = map(float, bbox)
            valid = (
                isinstance(timestamp, (int, float))
                and math.isfinite(timestamp)
                and h > 0 and w > 0
                and all(math.isfinite(v) for v in (x1, y1, x2, y2))
                and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
                and abs(w / h - self.aspect_ratio) / self.aspect_ratio
                <= self.aspect_tolerance
            )
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            valid = False
        if not valid:
            self.reset()
            result["reason"] = "Player box or viewport outside lane reference"
            return result
        if self.last_timestamp is not None:
            if timestamp <= self.last_timestamp or timestamp - self.last_timestamp > self.max_gap:
                self.reset()
        self.last_timestamp = timestamp
        x = (x1 + x2) / (2 * w)
        distances = []
        for lane, center in enumerate(self.centers):
            distances.append((abs(x - center), lane))
        distances.sort()
        error = distances[0][0]
        left_boundary, right_boundary = self.boundaries
        result["normalized_x"] = x
        result["boundaries"] = self.boundaries
        result["anchor_error"] = error
        if error > self.max_error:
            self.last_lane, self.streak = None, 0
            result["reason"] = "Player outside reviewed screen positions"
            result["transitioning"] = True
            return result
        # Deadbands prevent a mid-swipe box being called a stable lane.
        if abs(x - left_boundary) <= self.margin:
            self.last_lane, self.streak = None, 0
            result["candidates"] = (0, 1)
            result["reason"] = "Player crossing left/center screen boundary"
            result["transitioning"] = True
            return result
        if abs(x - right_boundary) <= self.margin:
            self.last_lane, self.streak = None, 0
            result["candidates"] = (1, 2)
            result["reason"] = "Player crossing center/right screen boundary"
            result["transitioning"] = True
            return result
        lane = 0 if x < left_boundary else 1 if x < right_boundary else 2
        result["candidates"] = (lane,)
        # Report the current screen-position candidate immediately. It remains
        # provisional until the required consecutive observations stabilize.
        # Missing/transitioning frames still return lane=None.
        result["lane"] = lane
        if geometry_candidates:
            result["geometry_agrees"] = lane in geometry_candidates
        if self.last_lane == lane:
            self.streak += 1
        else:
            self.last_lane = lane
            self.streak = 1
        if self.streak < self.stable_frames:
            result["reason"] = "Provisional screen lane; waiting for stability"
            result["transitioning"] = True
            return result
        result["lane"] = lane
        result["stable"] = True
        result["transitioning"] = False
        result["reason"] = "Provisional screen-column lane, NOT verified ground contact"
        return result


class PlayerEstimator:
    def __init__(self, config=None, localizer=None, visual=True):
        self.config = config or {}
        self.localizer = localizer
        self.visual = visual
        self.last_diagnostics = {}
        self.localizer_error = None
        self.lane_tracker = None
        self.lane_settings = self.config.get("lane_tracking")
        self.last_rail_lane = None
        self.last_rail_time = None

    def update(self, image, timestamp, frame_id="", image_shape=None, geometry=None):
        previous_lane, previous_time = self.last_rail_lane, self.last_rail_time
        self.last_rail_lane = self.last_rail_time = None
        self.last_diagnostics = {}
        reviewed = self._reviewed(timestamp, frame_id)
        if reviewed is not None:
            if self.lane_tracker is not None:
                self.lane_tracker.reset()
            return reviewed
        unknown = PlayerState(timestamp=timestamp)
        if image is None or not self.visual:
            if self.lane_tracker is not None:
                self.lane_tracker.reset()
            return unknown
        if self.localizer is None:
            if self.localizer_error is not None:
                unknown.lane_reason = self.localizer_error
                if self.lane_tracker is not None:
                    self.lane_tracker.reset()
                return unknown
            try:
                from .player_vision import PlayerLocalizer

                self.localizer = PlayerLocalizer(self.config.get("player_templates"))
            except (ValueError, OSError) as error:
                self.localizer_error = f"Player templates unavailable: {error}"
                unknown.lane_reason = self.localizer_error
                return unknown
        if self.lane_tracker is None:
            settings = self.lane_settings
            if settings is None:
                settings = getattr(self.localizer, "config", {}).get("lane_tracking")
            if settings is not None:
                self.lane_tracker = PlayerLaneTracker(settings)
        observation = self.localizer.locate(image, timestamp)
        self.last_diagnostics = dict(self.localizer.last_diagnostics)
        if observation is None:
            unknown.lane_reason = self.last_diagnostics.get("reason", "Character match unavailable")
            if self.lane_tracker is not None:
                self.lane_tracker.reset()
            return unknown
        unknown.bbox = tuple(observation["bbox"])
        unknown.bbox_kind = observation.get("bbox_kind", "upperbody")
        unknown.match_score = observation.get("score")
        unknown.source = observation.get("source", "visual character observation")
        for key in (
            "score_margin",
            "flow_features",
            "flow_fb_error",
            "identity_age",
            "footpoint_valid",
            "provenance",
            "template_source",
        ):
            unknown.localization[key] = observation.get(key)
        if geometry is not None:
            estimate = geometry.player_candidates(
                unknown.bbox, image_shape or image.shape[:2], frame_id
            )
            unknown.lane_candidates = estimate.candidates
            unknown.lane_reason = estimate.reason
        else:
            unknown.lane_reason = (
                "Visual upperbody located; no verified ground contact or lane geometry"
            )
        if unknown.lane_candidates:
            # A fixed screen column cannot resolve visibly ambiguous rails.
            if self.lane_tracker is not None:
                self.lane_tracker.reset()
            if len(unknown.lane_candidates) == 1 and math.isfinite(timestamp):
                lane = unknown.lane_candidates[0]
                self.last_rail_lane, self.last_rail_time = lane, timestamp
                if previous_lane == lane and 0 < timestamp - previous_time <= 0.35:
                    unknown.estimated_lane = lane
                    unknown.lane_reason = "Estimated from consecutive rail observations"
            return unknown
        if self.lane_tracker is not None:
            estimated = self.lane_tracker.estimate(
                unknown.bbox, timestamp, image_shape or image.shape[:2],
                unknown.lane_candidates,
            )
            unknown.localization["screen_lane"] = estimated
            # Screen coordinates are primary for player lane; rail candidates
            # remain only a diagnostic cross-check, not a requirement.
            unknown.lane_candidates = estimated["candidates"]
            unknown.lane_reason = estimated["reason"]
            # Expose the provisional observation even on the first valid
            # frame. The separate stable flag is retained in localization.
            if estimated["lane"] in (0, 1, 2):
                unknown.estimated_lane = estimated["lane"]
        # Preserve Codex's actual localizer, verified-lane semantics and UNKNOWN pose.
        return unknown

    def _reviewed(self, timestamp, frame_id):
        source = self.config.get("source_frame")
        if source and Path(frame_id).resolve() != Path(source).resolve():
            return None
        observation = self.config.get("player_observations", {}).get(frame_id)
        # Basenames are accepted only when no session scope is requested.
        if observation is None and not self.config.get("source_session"):
            observation = self.config.get("player_observations", {}).get(Path(frame_id).name)
        session = self.config.get("source_session")
        if session and session not in Path(frame_id).parts:
            return None
        if not isinstance(observation, dict) or not observation.get("provenance"):
            return None
        lane = observation.get("lane")
        movement = observation.get("movement", "UNKNOWN")
        if (
            lane not in (0, 1, 2)
            or isinstance(lane, bool)
            or movement not in ("UNKNOWN", "RUNNING", "JUMPING", "ROLLING", "ON_TRAIN")
        ):
            return None
        confidence = observation.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            return None
        transition = observation.get("transitioning")
        if transition is not None and not isinstance(transition, bool):
            return None
        return PlayerState(
            lane,
            movement,
            confidence,
            "provisional reviewed frame observation",
            timestamp,
            transition,
        )
