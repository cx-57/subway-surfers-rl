"""Conservative path threats with relative image urgency, never guessed seconds."""

import math
from .state import Hazard


class HazardEstimator:
    def __init__(self, config=None):
        self.config = config or {}
        self.max_age = float(self.config.get("max_observation_age", 0.5))
        if not math.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError("max_observation_age must be finite and positive")

    def estimate(self, tracks, player, image_shape, observation_age, geometry_valid):
        result = []
        for track in tracks:
            hazard = Hazard(track.id)
            result.append(hazard)
            if (
                not geometry_valid
                or not math.isfinite(observation_age)
                or observation_age < 0
                or observation_age > self.max_age
            ):
                hazard.reason = "Invalid calibration or stale observation"
                continue
            if (
                player.lane not in (0, 1, 2)
                or player.movement != "RUNNING"
                or player.transitioning is True
            ):
                hazard.reason = "Unknown, transitioning, or unsupported vertical player state"
                continue
            if (
                track.missed
                or track.association_ambiguous
                or track.lane.status != "KNOWN"
                or len(track.lane.candidates) != 1
            ):
                hazard.reason = "Missing track or uncertain footprint/association"
                continue
            if player.lane not in track.lane.candidates:
                hazard.status = "CLEAR"
                hazard.reason = "Known footprint in another lane"
                continue
            hazard.status = "THREAT"
            depth = track.lane.depth
            deltas = []
            for (ta, ba), (tb, bb) in zip(track.history, track.history[1:]):
                if tb > ta:
                    deltas.append((bb[3] - ba[3]) / (tb - ta))
            positive = False
            nonpositive = False
            all_approaching = bool(deltas)
            for speed in deltas:
                if speed > 0:
                    positive = True
                if speed <= 0:
                    nonpositive = True
                if not math.isfinite(speed) or speed <= 0:
                    all_approaching = False
            unstable = positive and nonpositive
            if unstable:
                hazard.reason = "Path conflict; unstable apparent motion; relative urgency only"
            elif (
                track.velocity_y is not None
                and math.isfinite(track.velocity_y)
                and track.velocity_y > 0
                and all_approaching
            ):
                if depth is not None and math.isfinite(depth):
                    if depth >= 0.75:
                        hazard.urgency = "HIGH"
                    elif depth >= 0.5:
                        hazard.urgency = "MEDIUM"
                    else:
                        hazard.urgency = "LOW"
                hazard.reason = "Approaching path conflict; relative image urgency only; collision boundary uncalibrated"
            else:
                hazard.reason = (
                    "Path conflict; insufficient approach history; collision timing unknown"
                )
            # No action can be certified against a deadline when TTC is unknown.
        return result
