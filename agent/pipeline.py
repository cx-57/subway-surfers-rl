"""Shared sequential engine for replay and live frames; logging is streaming."""

from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
from time import monotonic, perf_counter

from .geometry import Geometry
from .track_geometry import TrackGeometry
from .hazard import HazardEstimator
from .perception import Detector
from .player import PlayerEstimator
from .rule_policy import ShadowPolicy
from .state import Detection, GameState, OBSERVATION_FIELDS
from .tracking import ObstacleTracker
from .detection import nearby_indexes


class Engine:
    def __init__(self, detector=None, calibration=None, settings=None):
        self.detector = detector if detector is not None else Detector()
        self.calibration = calibration or {}
        settings = settings or {}
        for name in ("distance_filter", "automatic_geometry", "visual_player"):
            if not isinstance(settings.get(name, True), bool):
                raise ValueError(f"{name} must be true or false")
        self.distance_filter = settings.get("distance_filter", True)
        self.geometry = Geometry(calibration)
        self.configured_geometry = self.geometry
        self.auto_geometry = TrackGeometry() if settings.get("automatic_geometry", True) else None
        self.player = PlayerEstimator(calibration, visual=settings.get("visual_player", True))
        self.tracker = ObstacleTracker(**settings.get("tracking", {}))
        self.max_age = float(settings.get("max_observation_age", 0.5))
        if not math.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError("Maximum observation age must be positive and finite")
        for component in ("hazards", "policy"):
            limit = settings.get(component, {}).get("max_observation_age", self.max_age)
            if limit != self.max_age:
                raise ValueError("Configure shared max_observation_age at the top level")
        self.hazards = HazardEstimator(
            dict(settings.get("hazards", {}), max_observation_age=self.max_age)
        )
        self.policy = ShadowPolicy(
            dict(settings.get("policy", {}), max_observation_age=self.max_age)
        )
        self.recent = deque(maxlen=16)

    def process(self, frame):
        """Detect obstacles, estimate lanes/player, assess hazards, then save a snapshot."""
        start = perf_counter()
        if (
            not math.isfinite(frame.captured_at)
            or not math.isfinite(frame.capture_ms)
            or frame.capture_ms < 0
        ):
            raise ValueError(
                "Capture clock and duration must be finite; duration cannot be negative"
            )
        detections = self.detector.detect(frame.image, frame.timestamp)
        shape = frame.image.shape[:2]
        frame_id = str(frame.frame_id)
        geometry_start = perf_counter()
        self.geometry = self.configured_geometry
        valid, reason = self.geometry.validate_frame(shape, frame_id)
        if not valid and self.auto_geometry is not None:
            self.geometry = self.auto_geometry.estimate(frame.image, frame_id)
            valid, reason = self.geometry.validate_frame(shape, frame_id)
        geometry_ms = (perf_counter() - geometry_start) * 1000
        tracks = self.tracker.update(detections, frame.timestamp, shape)
        for track in tracks:
            track.lane = self.geometry.assign(
                Detection(track.class_id, track.confidence, track.bbox, track.last_seen),
                shape,
                frame_id,
            )
        player_start = perf_counter()
        player = self.player.update(frame.image, frame.timestamp, frame_id, shape, self.geometry)
        player_ms = (perf_counter() - player_start) * 1000
        # captured_at is read/capture completion; include capture time in frame age.
        age = max(0.0, monotonic() - frame.captured_at + frame.capture_ms / 1000)
        # Keep distant tracks visible for lookahead, but limit current hazards.
        active = tracks
        nearby_ids = None
        if self.distance_filter:
            predictions = []
            for track in tracks:
                predictions.append(
                    dict(class_id=track.class_id, confidence=track.confidence, xyxy=track.bbox)
                )
            active = []
            nearby_ids = []
            for index in nearby_indexes(predictions, shape):
                active.append(tracks[index])
                nearby_ids.append(tracks[index].id)
        hazards = self.hazards.estimate(active, player, shape, age, valid)
        visibility = self.calibration.get("lane_visibility", {}).get(
            Path(frame_id).name, [None, None, None]
        )
        if not valid or not isinstance(visibility, list) or len(visibility) != 3:
            visibility = [None, None, None]
        visibility = list(visibility)
        for index, value in enumerate(visibility):
            if not isinstance(value, bool):
                visibility[index] = None
        state = GameState(
            frame.timestamp,
            shape,
            player=player,
            tracks=tracks,
            hazards=hazards,
            calibration_valid=valid,
            calibration_reason=reason,
            observation_age=age,
            timestamp_quality=frame.timestamp_quality,
            valid=valid and age <= self.max_age and player.lane in (0, 1, 2),
            lane_visibility=visibility,
            max_observation_age=self.max_age,
            nearby_track_ids=nearby_ids,
            recent_recommendations=list(self.recent),
        )
        # An action must be feasible for every simultaneous threat.
        actions = None
        for hazard in hazards:
            if hazard.status == "THREAT":
                if actions is None:
                    actions = set(hazard.feasible_actions)
                else:
                    actions.intersection_update(hazard.feasible_actions)
        state.available_actions = sorted(actions or [])
        state.recommendation = self.policy.recommend(state)
        if state.recommendation.action == "NO_OP":
            state.available_actions.append("NO_OP")
        self.recent.append(state.recommendation.action)
        state.timings = dict(self.detector.last_timings)
        state.timings.update(
            capture_ms=frame.capture_ms,
            state_ms=(perf_counter() - start) * 1000,
            geometry_ms=geometry_ms,
            player_ms=player_ms,
            observation_age_ms=age * 1000,
        )
        # Return an owned snapshot: tracker mutates its internal records next frame.
        return deepcopy(state)


class StateLog:
    """JSONL rows are written incrementally; existing results are never overwritten."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x")
        return self

    def write(self, state, frame_id):
        self.stream.write(
            json.dumps(
                dict(
                    schema_version=1,
                    frame_id=str(frame_id),
                    state=state.to_dict(),
                    observation=state.to_vector().tolist(),
                ),
                allow_nan=False,
            )
            + "\n"
        )

    def __exit__(self, *args):
        self.stream.close()


def write_schema(path):
    Path(path).write_text(
        json.dumps(
            dict(version=1, fields=OBSERVATION_FIELDS, shape=[len(OBSERVATION_FIELDS)]), indent=2
        )
        + "\n"
    )
