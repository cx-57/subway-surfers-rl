"""Versioned float32 features; coordinates and velocities are image-relative."""

import math
import numpy as np
from agent.detection import CLASS_NAMES, RAMP_TRAIN_ID

VERSION = 1
MAX_OBSTACLES = 8
GLOBAL_FIELDS = (
    "player_present", "player_x1", "player_y1", "player_x2", "player_y2",
    "player_vx", "player_vy", "player_velocity_known", "observation_age", "dt",
    "previous_noop", "previous_left", "previous_right", "previous_jump", "previous_roll",
)
SLOT_FIELDS = (
    "present", "observed", "x1", "y1", "x2", "y2", "confidence",
    "approach_velocity", "velocity_known", "time_since_seen", "association_ambiguous",
) + tuple(CLASS_NAMES)
FIELDS = list(GLOBAL_FIELDS)
for slot in range(MAX_OBSTACLES):
    for name in SLOT_FIELDS:
        FIELDS.append(f"obstacle{slot}.{name}")
FIELDS = tuple(FIELDS)


def finite(value, low=-1.0, high=1.0):
    if value is None or not math.isfinite(value):
        return 0.0
    return float(np.clip(value, low, high))


def box_values(box, shape):
    height, width = shape
    x1, y1, x2, y2 = box
    return [finite(x1 / width, 0), finite(y1 / height, 0),
            finite(x2 / width, 0), finite(y2 / height, 0)]


def obstacle_order(track):
    # Current observations first, then closest screen-bottom extent; IDs break ties.
    return track.missed, -track.bbox[3], track.id


class ObservationEncoder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None

    def encode(self, state, previous_action=0):
        height, width = state.image_shape
        if height <= 0 or width <= 0 or not math.isfinite(state.timestamp):
            raise ValueError("Invalid observation shape/timestamp")
        dt = 0.0
        vx, vy, velocity_known = 0.0, 0.0, 0.0
        box = state.player.bbox
        if self.previous is not None:
            old_time, old_box, old_shape = self.previous
            dt = state.timestamp - old_time
            if dt <= 0:
                raise ValueError("RL observation timestamps must strictly increase")
            if box is not None and old_box is not None and old_shape == state.image_shape:
                vx = ((box[0] + box[2]) - (old_box[0] + old_box[2])) / (2 * width * dt)
                vy = ((box[1] + box[3]) - (old_box[1] + old_box[3])) / (2 * height * dt)
                velocity_known = 1.0
        values = [float(box is not None)]
        values.extend(box_values(box, state.image_shape) if box is not None else [0.0] * 4)
        values.extend([finite(vx), finite(vy), velocity_known,
                       finite(state.observation_age, 0), finite(dt, 0)])
        for action in range(5):
            values.append(float(previous_action == action))
        tracks = sorted(state.tracks, key=obstacle_order)[:MAX_OBSTACLES]
        for track in tracks:
            values.extend([1.0, float(not track.missed)])
            values.extend(box_values(track.bbox, state.image_shape))
            values.extend([
                finite(track.confidence, 0), finite(
                    track.velocity_y / height if track.velocity_y is not None else None),
                float(track.velocity_y is not None),
                finite(state.timestamp - track.last_seen, 0),
                float(track.association_ambiguous),
            ])
            track_class = 0 if track.class_id == RAMP_TRAIN_ID else track.class_id
            for class_id in range(len(CLASS_NAMES)):
                values.append(float(track_class == class_id))
        values.extend([0.0] * ((MAX_OBSTACLES - len(tracks)) * len(SLOT_FIELDS)))
        self.previous = (state.timestamp, box, state.image_shape)
        return np.asarray(values, dtype=np.float32)


def schema():
    return dict(version=VERSION, fields=FIELDS, shape=[len(FIELDS)], dtype="float32",
                bounds=[-1, 1], max_obstacles=MAX_OBSTACLES,
                velocity_units="image widths/heights per second, clipped to [-1, 1]",
                ordering="observed first, descending box bottom, then track ID")
