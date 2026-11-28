"""Shared detector thresholds, box geometry and incomplete-inference guard.

This module has no CLI or training side effects. Scripts re-export existing public
names for compatibility; live perception and tracking import the rules directly.
"""

from contextlib import contextmanager

CONFIDENCE = 0.5
CLASS_NAMES = ('train', 'ground_barrier', 'overhead_barrier', 'bush', 'small_obstacle', 'pillar')
LEGACY_CLASS_NAMES = CLASS_NAMES[:3]
RAMP_TRAIN_ID = len(CLASS_NAMES)
SPLIT_TRAIN_CLASS_NAMES = ("blocking_train",) + CLASS_NAMES[1:] + ("ramp_train",)
TRAIN_CLASS_IDS = (0, RAMP_TRAIN_ID)


def validate_checkpoint_classes(names):
    """Reject silent class-ID remapping between checkpoint and game-state code."""
    if isinstance(names, dict):
        count = len(names)
        if all(index in names for index in range(count)):
            ordered = tuple(names[index] for index in range(count))
        elif all(str(index) in names for index in range(count)):
            ordered = tuple(names[str(index)] for index in range(count))
        else:
            raise ValueError("Checkpoint class IDs must be consecutive from zero")
    elif isinstance(names, (tuple, list)):
        ordered = tuple(names)
    else:
        raise ValueError("Checkpoint is missing usable class names")
    if ordered not in (LEGACY_CLASS_NAMES, CLASS_NAMES, SPLIT_TRAIN_CLASS_NAMES):
        raise ValueError(
            f"Checkpoint class IDs/names {ordered!r} are incompatible with "
            f"expected a known class order, including {SPLIT_TRAIN_CLASS_NAMES!r}"
        )
    return ordered


# Minimum normalized box bottom and width; proximity estimates, not meters.
NEARBY_LIMITS = {0: (0.65, 0.12), 1: (0.55, 0.06), 2: (0.55, 0.06),
                 3: (0.55, 0.06), 4: (0.55, 0.06), 5: (0.55, 0.06)}
NEARBY_LIMITS[RAMP_TRAIN_ID] = NEARBY_LIMITS[0]


class IncompleteNMS(RuntimeError):
    """Postprocessing stopped before all predictions were processed."""


def nearby_indexes(predictions, image_shape):
    """Select confident, nearby candidates. Lane occupancy must be checked separately."""
    height, width = image_shape
    nearby = []
    for index, prediction in enumerate(predictions):
        if prediction["confidence"] < CONFIDENCE:
            continue
        limits = NEARBY_LIMITS.get(prediction["class_id"])
        if limits is None:
            continue
        left, top, right, bottom = prediction["xyxy"]
        min_bottom, min_width = limits
        if bottom >= min_bottom * height and right - left >= min_width * width:
            nearby.append(index)
    return nearby


def overlap(a, b):
    """Intersection-over-union: shared box area divided by total covered area."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


@contextmanager
def complete_nms():
    """Do not accept scores if Ultralytics abandons part of NMS processing."""
    from ultralytics.utils import LOGGER

    errors = []

    def watch(record):
        if "NMS time limit" in record.getMessage():
            errors.append(record.getMessage())
        return True

    LOGGER.addFilter(watch)
    try:
        yield
    finally:
        LOGGER.removeFilter(watch)
    if errors:
        raise IncompleteNMS("Incomplete NMS output: " + "; ".join(errors))
