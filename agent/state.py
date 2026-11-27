"""Shared, lightweight contracts. Unknown measurements remain explicit."""

from dataclasses import asdict, dataclass, field
import math

from .detection import CLASS_NAMES

# v2 includes all six classes; do not interpret 113-value v1 logs as v2.
OBSERVATION_SCHEMA_VERSION = 3

ACTIONS = ("NO_OP", "LEFT", "RIGHT", "JUMP", "ROLL")
MOVEMENTS = ("RUNNING", "JUMPING", "ROLLING", "ON_TRAIN")
GLOBAL_FIELDS = (
    "player.left",
    "player.center",
    "player.right",
    "player.lane_known",
    "player.running",
    "player.jumping",
    "player.rolling",
    "player.on_train",
    "player.movement_known",
    "player.transitioning",
    "player.transition_known",
    "calibration_valid",
    "fresh",
    "age",
    "timestamp_exact",
    "valid",
    "unknown_lane_tracks",
    "missing_tracks",
)
# Preserve unlocated detections without inventing lane occupancy.
for name in CLASS_NAMES:
    GLOBAL_FIELDS += (f"unlocated.{name}",)
for lane in range(3):
    GLOBAL_FIELDS += (f"lane{lane}.visible", f"lane{lane}.visibility_known")
SLOT_FIELDS = (
    "present",
    "observed",
    "lane_known",
    "train",
    "ground",
    "overhead",
    "bush",
    "small_obstacle",
    "pillar",
    "confidence",
    "depth",
    "depth_known",
    "ttc",
    "ttc_known",
    "urgency",
    "urgency_known",
    "association_known",
)
# Keep this ordering fixed: saved logs and future RL inputs use the same schema.
observation_fields = list(GLOBAL_FIELDS)
for action in ACTIONS:
    observation_fields.append("available." + action)
for lane in range(3):
    for slot in range(2):
        for name in SLOT_FIELDS:
            observation_fields.append(f"lane{lane}.slot{slot}.{name}")
# Keep the 137 legacy fields in their original order.
observation_fields += [
    "player.screen_left",
    "player.screen_center",
    "player.screen_right",
    "player.screen_lane_available",
]
OBSERVATION_FIELDS = tuple(observation_fields)


def _unit(value, scale=1.0):
    return min(1.0, max(0.0, value / scale)) if value is not None and math.isfinite(value) else 0.0


def _known_lane(track):
    return track.lane.status == "KNOWN" and len(track.lane.candidates) == 1


def _nearest_first(track):
    depth = track.lane.depth
    if depth is None:
        depth = -1
    return -depth, track.id


def _unknown_visibility():
    return [None, None, None]


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    timestamp: float


@dataclass
class LaneEstimate:
    candidates: tuple[int, ...] = ()
    status: str = "UNKNOWN"
    reason: str = "No calibrated footprint"
    depth: float | None = None
    footprint: tuple[float, float] | None = None


@dataclass
class Track:
    id: int
    class_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    first_seen: float
    last_seen: float
    history: list = field(default_factory=list)
    missed: bool = False
    lane: LaneEstimate = field(default_factory=LaneEstimate)
    velocity_y: float | None = None
    association_ambiguous: bool = False


@dataclass
class PlayerState:
    lane: int | None = None
    movement: str = "UNKNOWN"
    confidence: float | None = None
    source: str = "unavailable"
    timestamp: float | None = None
    transitioning: bool | None = None
    bbox: tuple[float, float, float, float] | None = None
    bbox_kind: str | None = None
    match_score: float | None = None
    lane_candidates: tuple[int, ...] = ()
    lane_reason: str = "Player location unavailable"
    localization: dict = field(default_factory=dict)
    estimated_lane: int | None = None  # Provisional x-coordinate estimate; NOT a verified ground lane.


@dataclass
class Hazard:
    track_id: int
    status: str = "UNKNOWN"
    urgency: str = "UNKNOWN"
    ttc: float | None = None
    reason: str = "Insufficient evidence"
    feasible_actions: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    action: str = "UNKNOWN"
    reason: str = "Insufficient evidence"
    track_id: int | None = None
    timing: float | None = None
    confidence: float | None = None
    actionable: bool = False
    repeated: bool = False


@dataclass
class GameState:
    """One frame's results. None/UNKNOWN means we could not measure that value."""

    timestamp: float
    image_shape: tuple[int, int]
    player: PlayerState = field(default_factory=PlayerState)
    tracks: list[Track] = field(default_factory=list)
    hazards: list[Hazard] = field(default_factory=list)
    recommendation: Recommendation = field(default_factory=Recommendation)
    available_actions: list[str] = field(default_factory=list)
    calibration_valid: bool = False
    calibration_reason: str = "No calibration"
    observation_age: float = 0.0
    timestamp_quality: str = "approximate"
    timings: dict = field(default_factory=dict)
    confidence: float | None = None
    valid: bool = False
    recent_recommendations: list[str] = field(default_factory=list)
    lane_visibility: list[bool | None] = field(default_factory=_unknown_visibility)
    max_observation_age: float = 0.5
    nearby_track_ids: list[int] | None = None

    def to_dict(self):
        return asdict(self)

    def to_vector(self):
        """Schema v3: 141 float32 values with separate provisional screen-lane mask."""
        import numpy as np

        player = self.player
        values = []
        for lane in range(3):
            values.append(float(player.lane == lane))
        values += [float(player.lane in (0, 1, 2))]
        for movement in MOVEMENTS:
            values.append(float(player.movement == movement))
        unknown_tracks = 0
        missing_tracks = 0
        unlocated_counts = [0] * len(CLASS_NAMES)
        hazards = {}
        for track in self.tracks:
            if not _known_lane(track):
                unknown_tracks += 1
                if not track.missed and 0 <= track.class_id < len(CLASS_NAMES):
                    unlocated_counts[track.class_id] += 1
            if track.missed:
                missing_tracks += 1
        for hazard in self.hazards:
            hazards[hazard.track_id] = hazard
        values += [
            float(player.movement in MOVEMENTS),
            float(player.transitioning is True),
            float(player.transitioning is not None),
            float(self.calibration_valid),
            float(
                math.isfinite(self.observation_age)
                and 0 <= self.observation_age <= self.max_observation_age
            ),
            _unit(self.observation_age, 1.0),
            float(self.timestamp_quality in ("exact", "monotonic")),
            float(self.valid),
            _unit(unknown_tracks, 10.0),
            _unit(missing_tracks, 10.0),
        ]
        for count in unlocated_counts:
            values.append(_unit(count, 10.0))
        for lane in range(3):
            visibility = self.lane_visibility[lane] if lane < len(self.lane_visibility) else None
            values += [float(visibility is True), float(isinstance(visibility, bool))]
        for action in ACTIONS:
            values.append(float(action in self.available_actions))
        for lane in range(3):
            tracks = []
            for track in self.tracks:
                if _known_lane(track) and track.lane.candidates[0] == lane:
                    tracks.append(track)
            tracks.sort(key=_nearest_first)
            for slot in range(2):
                if slot >= len(tracks):
                    values += [0.0] * len(SLOT_FIELDS)
                    continue
                track = tracks[slot]
                hazard = hazards.get(track.id)
                ttc = hazard.ttc if hazard else None
                ttc_known = ttc is not None and math.isfinite(ttc) and ttc >= 0
                urgency = hazard.urgency if hazard else "UNKNOWN"
                values += [1.0, float(not track.missed), 1.0]
                for class_id in range(len(CLASS_NAMES)):
                    values.append(float(track.class_id == class_id))
                values += [
                    _unit(track.confidence),
                    _unit(track.lane.depth),
                    float(
                        track.lane.depth is not None
                        and math.isfinite(track.lane.depth)
                        and 0 <= track.lane.depth <= 1
                    ),
                    _unit(ttc, 10.0) if ttc_known else 0.0,
                    float(ttc_known),
                    {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 1.0}.get(urgency, 0.0),
                    float(urgency in ("LOW", "MEDIUM", "HIGH")),
                    float(not track.association_ambiguous),
                ]
        # This estimate does not establish running pose or verified ground contact.
        screen_lane = player.estimated_lane
        screen_valid = type(screen_lane) is int and screen_lane in (0, 1, 2)
        for lane in range(3):
            values.append(float(screen_valid and screen_lane == lane))
        values.append(float(screen_valid))
        return np.asarray(values, dtype=np.float32)
