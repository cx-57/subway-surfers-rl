"""Visual character tracking plus scoped reviewed observations; uncertainty is explicit."""

import math
from pathlib import Path
from .state import PlayerState


class VisualPlayerCalibration:
    def __init__(self, config):
        self.config = dict(config)
        self.validated = self.config.get("validated", False)
        anchors = self.config.get("lane_x")
        if (
            not self.config.get("provenance")
            or not isinstance(self.validated, bool)
            or not isinstance(anchors, list)
            or len(anchors) != 3
            or any(not self._unit(value) for value in anchors)
            or not 0.08 <= anchors[1] - anchors[0]
            or not 0.08 <= anchors[2] - anchors[1]
        ):
            raise ValueError("Player lane calibration needs reviewed provenance and three separated normalized lane_x anchors")
        self.anchors = anchors
        self.aspect_ratio = self.config.get("aspect_ratio")
        if not self._positive(self.aspect_ratio):
            raise ValueError("Player calibration needs a positive image aspect_ratio")
        self.max_x_error = self.config.get("max_x_error", 0.045)
        self.margin = self.config.get("lane_margin", 0.025)
        self.stable_frames = self.config.get("stable_frames", 3)
        self.max_gap = self.config.get("max_gap_seconds", 0.35)
        if (
            not self._positive(self.max_x_error)
            or self.max_x_error > 0.1
            or not self._positive(self.margin)
            or self.margin > 0.1
            or not isinstance(self.stable_frames, int)
            or isinstance(self.stable_frames, bool)
            or not 2 <= self.stable_frames <= 12
            or not self._positive(self.max_gap)
            or self.max_gap > 1.0
        ):
            raise ValueError("Invalid player calibration thresholds")
        self.running_y = self.config.get("running_y")
        self.pose_offsets = self.config.get("pose_offsets")
        self.pose_enabled = self.running_y is not None and self.pose_offsets is not None
        if (self.running_y is None) != (self.pose_offsets is None):
            raise ValueError("Pose calibration must include both running_y and pose_offsets")
        if self.pose_enabled:
            if (
                not isinstance(self.running_y, list)
                or len(self.running_y) != 3
                or any(not self._unit(value) for value in self.running_y)
                or not isinstance(self.pose_offsets, dict)
                or set(self.pose_offsets) != {"RUNNING", "JUMPING", "ROLLING"}
                or any(not self._finite(value) or abs(value) > 0.3 for value in self.pose_offsets.values())
                or abs(self.pose_offsets["RUNNING"]) > 0.01
                or self.pose_offsets["JUMPING"] >= -0.025
                or self.pose_offsets["ROLLING"] <= 0.025
            ):
                raise ValueError("Pose calibration needs distinct running, jumping and rolling reference offsets")
        self.max_pose_error = self.config.get("max_pose_error", 0.04)
        self.pose_margin = self.config.get("pose_margin", 0.02)
        if not self._positive(self.max_pose_error) or not self._positive(self.pose_margin):
            raise ValueError("Invalid pose-matching thresholds")
        self.reset()

    @staticmethod
    def _finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    @classmethod
    def _unit(cls, value):
        return cls._finite(value) and 0 <= value <= 1

    @classmethod
    def _positive(cls, value):
        return cls._finite(value) and value > 0

    def reset(self):
        self.last_timestamp = None
        self.last_lane = None
        self.streak = 0

    def predict(self, bbox, timestamp, image_shape):
        """Return candidates for review; a validated calibration may promote a stable lane.

        A frame gap, unrecognized player, changed crop, or ambiguous horizontal
        position drops temporal certainty instead of carrying forward a lane.
        """
        result = dict(lane=None, candidates=(), pose="UNKNOWN", transitioning=None,
                      stable=False, validated=self.validated, reason="Uncalibrated player frame")
        try:
            h, w = image_shape[:2]
            x1, y1, x2, y2 = map(float, bbox)
        except (ValueError, TypeError, AttributeError):
            self.reset()
            return result
        if (
            not self._positive(h) or not self._positive(w)
            or not self._finite(timestamp)
            or not all(self._finite(v) for v in (x1, y1, x2, y2))
            or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h)
            or abs(w / h - self.aspect_ratio) > 0.01 * self.aspect_ratio
        ):
            self.reset()
            result["reason"] = "Player box or viewport outside visual calibration"
            return result
        if self.last_timestamp is not None and (
            timestamp <= self.last_timestamp or timestamp - self.last_timestamp > self.max_gap
        ):
            self.reset()
        self.last_timestamp = timestamp
        x = ((x1 + x2) / 2) / w
        y = ((y1 + y2) / 2) / h
        distances = sorted((abs(x - anchor), lane) for lane, anchor in enumerate(self.anchors))
        closest, lane = distances[0]
        separation = distances[1][0] - closest
        result["x_error"] = closest
        result["x_separation"] = separation
        result["center"] = (x, y)
        if closest > self.max_x_error:
            self.last_lane = None
            self.streak = 0
            result["reason"] = "Player outside reviewed lane anchors; possible transition"
            result["transitioning"] = True
            return result
        if separation < self.margin:
            self.last_lane = None
            self.streak = 0
            result["candidates"] = tuple(sorted((distances[0][1], distances[1][1])))
            result["transitioning"] = True
            result["reason"] = "Player between lane anchors"
            return result
        if self.last_lane == lane:
            self.streak += 1
        else:
            self.last_lane = lane
            self.streak = 1
        result["candidates"] = (lane,)
        result["stable"] = self.streak >= self.stable_frames
        result["transitioning"] = not result["stable"]
        if not result["stable"]:
            result["reason"] = "Awaiting repeated player-lane observations"
            return result
        result["lane"] = lane
        result["reason"] = "Stable horizontal anchor match; ground contact unverified"
        if self.pose_enabled:
            displacement = y - self.running_y[lane]
            scores = sorted(
                (abs(displacement - offset), name)
                for name, offset in self.pose_offsets.items()
            )
            if scores[0][0] <= self.max_pose_error and scores[1][0] - scores[0][0] >= self.pose_margin:
                result["pose"] = scores[0][1]
                result["pose_error"] = scores[0][0]
            else:
                result["reason"] += "; pose ambiguous"
        return result


class PlayerEstimator:
    def __init__(self, config=None, localizer=None, visual=True):
        self.config = config or {}
        self.localizer = localizer
        self.visual = visual
        self.visual_model = (
            VisualPlayerCalibration(self.config["player_visual_calibration"])
            if self.config.get("player_visual_calibration") else None
        )
        self.last_diagnostics = {}
        self.localizer_error = None

    def update(self, image, timestamp, frame_id="", image_shape=None, geometry=None):
        self.last_diagnostics = {}
        reviewed = self._reviewed(timestamp, frame_id)
        if reviewed is not None:
            if self.visual_model is not None:
                self.visual_model.reset()
            return reviewed
        unknown = PlayerState(timestamp=timestamp)
        if image is None or not self.visual:
            if self.visual_model is not None:
                self.visual_model.reset()
            return unknown
        if self.localizer is None:
            if self.localizer_error is not None:
                unknown.lane_reason = self.localizer_error
                if self.visual_model is not None:
                    self.visual_model.reset()
                return unknown
            try:
                from .player_vision import PlayerLocalizer

                self.localizer = PlayerLocalizer(self.config.get("player_templates"))
            except (ValueError, OSError) as error:
                self.localizer_error = f"Player templates unavailable: {error}"
                unknown.lane_reason = self.localizer_error
                if self.visual_model is not None:
                    self.visual_model.reset()
                return unknown
        observation = self.localizer.locate(image, timestamp)
        self.last_diagnostics = dict(self.localizer.last_diagnostics)
        if observation is None:
            unknown.lane_reason = self.last_diagnostics.get("reason", "Character match unavailable")
            if self.visual_model is not None:
                self.visual_model.reset()
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
        if self.visual_model is not None:
            prediction = self.visual_model.predict(
                unknown.bbox, timestamp, image_shape or image.shape[:2]
            )
            unknown.localization["visual_calibration"] = prediction
            if prediction["candidates"]:
                unknown.lane_candidates = prediction["candidates"]
                unknown.lane_reason = prediction["reason"]
            if prediction["validated"] and prediction["stable"]:
                # Only a human-reviewed *and held-out validated* calibration
                # is allowed to promote hypotheses into the game state.
                unknown.lane = prediction["lane"]
                unknown.movement = prediction["pose"]
                unknown.transitioning = prediction["transitioning"]
                unknown.source = "validated scene-specific visual lane/pose estimates"
            elif self.visual_model is not None:
                unknown.lane_reason += "; not held-out validated"
        # NCC scores are correlations; visual pose estimates are not ground truth.
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
