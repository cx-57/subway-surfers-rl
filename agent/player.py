"""Visual character tracking plus scoped reviewed observations; uncertainty is explicit."""

import math
from pathlib import Path
from .state import PlayerState


class PlayerEstimator:
    def __init__(self, config=None, localizer=None, visual=True):
        self.config = config or {}
        self.localizer = localizer
        self.visual = visual
        self.last_diagnostics = {}
        self.localizer_error = None

    def update(self, image, timestamp, frame_id="", image_shape=None, geometry=None):
        self.last_diagnostics = {}
        reviewed = self._reviewed(timestamp, frame_id)
        if reviewed is not None:
            return reviewed
        unknown = PlayerState(timestamp=timestamp)
        if image is None or not self.visual:
            return unknown
        if self.localizer is None:
            if self.localizer_error is not None:
                unknown.lane_reason = self.localizer_error
                return unknown
            try:
                from .player_vision import PlayerLocalizer

                self.localizer = PlayerLocalizer(self.config.get("player_templates"))
            except (ValueError, OSError) as error:
                self.localizer_error = f"Player templates unavailable: {error}"
                unknown.lane_reason = self.localizer_error
                return unknown
        observation = self.localizer.locate(image, timestamp)
        self.last_diagnostics = dict(self.localizer.last_diagnostics)
        if observation is None:
            unknown.lane_reason = self.last_diagnostics.get("reason", "Character match unavailable")
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
        # Template/flow scores are not probabilities or movement evidence.
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
