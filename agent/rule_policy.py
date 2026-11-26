"""Read-only deterministic suggestions. This module has no input-control API."""

from collections import deque
import math
from .state import Recommendation


class ShadowPolicy:
    def __init__(self, config=None):
        self.config = config or {}
        self.max_age = float(self.config.get("max_observation_age", 0.5))
        for key, default in (
            ("max_observation_age", 0.5),
            ("provisional_action_window", 1.0),
            ("provisional_movement_cooldown", 0.0),
        ):
            value = float(self.config.get(key, default))
            if (
                not math.isfinite(value)
                or value < 0
                or (value == 0 and key != "provisional_movement_cooldown")
            ):
                raise ValueError(key + " must be finite and positive (cooldown may be zero)")
        self.action_window = float(self.config.get("provisional_action_window", 1.0))
        self.cooldown = float(self.config.get("provisional_movement_cooldown", 0.0))
        self.durations = dict(self.config.get("provisional_action_durations", {}))
        for action, duration in self.durations.items():
            if (
                action not in ("JUMP", "ROLL", "LEFT", "RIGHT")
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(
                    "Action durations must use supported actions and finite positive seconds"
                )
        self.seen = deque(maxlen=max(1, int(self.config.get("history_size", 128))))

    def recommend(self, state):
        if not state.calibration_valid:
            return Recommendation(reason="Lane geometry unavailable")
        if state.player.lane is None:
            return Recommendation(reason="Player position or confirmed ground lane unavailable")
        if (
            not state.valid
            or not math.isfinite(state.observation_age)
            or not 0 <= state.observation_age <= self.max_age
        ):
            return Recommendation(reason="Invalid or stale game state")
        player = state.player
        if (
            player.lane not in (0, 1, 2)
            or player.movement != "RUNNING"
            or player.transitioning is not False
        ):
            return Recommendation(reason="Player lane, running state, or transition status unknown")
        if (
            player.timestamp is None
            or not math.isfinite(player.timestamp)
            or not 0 <= state.timestamp - player.timestamp <= self.max_age
        ):
            return Recommendation(reason="Player observation unavailable or stale")
        for track in state.tracks:
            if (
                track.missed
                or track.association_ambiguous
                or track.lane.status != "KNOWN"
                or len(track.lane.candidates) != 1
            ):
                return Recommendation(reason="Uncertain obstacle occupancy")
        threats = []
        for hazard in state.hazards:
            if hazard.status == "UNKNOWN":
                return Recommendation(reason="Unresolved obstacle hazard")
            if hazard.status == "THREAT":
                threats.append(hazard)
        if not threats:
            return Recommendation("NO_OP", "No observed path threat", actionable=False)
        threat = threats[0]
        for hazard in threats:
            if hazard.ttc is None or not math.isfinite(hazard.ttc) or hazard.ttc <= 0:
                return Recommendation(
                    reason="Collision deadline unavailable; relative urgency cannot schedule an action"
                )
            if hazard.ttc < threat.ttc:
                threat = hazard
        for action in ("JUMP", "ROLL", "LEFT", "RIGHT"):
            duration = self.durations.get(action)
            if duration is None:
                continue
            if action not in state.available_actions:
                continue
            fits_all = True
            for hazard in threats:
                deadline_ok = (
                    duration + self.cooldown + state.observation_age
                    < hazard.ttc
                    <= self.action_window
                )
                if action not in hazard.feasible_actions or not deadline_ok:
                    fits_all = False
                    break
            if not fits_all:
                continue
            if action in ("LEFT", "RIGHT"):
                target = player.lane + (-1 if action == "LEFT" else 1)
                if target not in (0, 1, 2):
                    continue
                visibility = getattr(state, "lane_visibility", [None, None, None])
                if len(visibility) != 3 or visibility[target] is not True:
                    continue
                occupied = False
                for track in state.tracks:
                    if target in track.lane.candidates:
                        occupied = True
                        break
                if occupied:
                    continue
            # Simultaneous class conflicts are reflected by the intersection of
            # feasible actions above; never choose based only on the first box.
            track_ids = []
            repeated = False
            for hazard in threats:
                track_ids.append(hazard.track_id)
                if hazard.track_id in self.seen:
                    repeated = True
            key = tuple(sorted(track_ids))
            if not repeated:
                self.seen.extend(key)
            return Recommendation(
                action,
                "Provisional action duration fits supplied deadlines; shadow suggestion only",
                threat.track_id,
                threat.ttc,
                None,
                not repeated,
                repeated,
            )
        return Recommendation(
            reason="No action meets timing, simultaneous threats, and known target-lane constraints"
        )
