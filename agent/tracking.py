"""Conservative timestamp-based association, bounded histories and finite occlusion TTL.

No identity ground truth is available. Ambiguous candidates are deliberately not
matched, and are surfaced rather than declaring a confident identity switch.
"""

import math
from .state import Track
from .detection import overlap


class ObstacleTracker:
    def __init__(
        self,
        max_age=0.6,
        history_size=12,
        iou_threshold=0.15,
        ambiguity_margin=0.08,
        max_tracks=128,
    ):
        if (
            not isinstance(max_age, (int, float))
            or isinstance(max_age, bool)
            or not math.isfinite(max_age)
            or max_age <= 0
        ):
            raise ValueError("Track lifetime must be finite and positive")
        for value in (history_size, max_tracks):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    "History size and track capacity must be positive integers (history >=2)"
                )
        if history_size < 2 or max_tracks < 1:
            raise ValueError(
                "History size and track capacity must be positive integers (history >=2)"
            )
        if (
            not math.isfinite(iou_threshold)
            or not 0 < iou_threshold <= 1
            or not math.isfinite(ambiguity_margin)
            or not 0 <= ambiguity_margin <= 1
        ):
            raise ValueError("Invalid tracker matching thresholds")
        self.max_age, self.history_size = max_age, history_size
        self.iou_threshold, self.ambiguity_margin = iou_threshold, ambiguity_margin
        self.max_tracks = max_tracks
        self._tracks = {}
        self._next_id = 1
        self._timestamp = None
        self.metrics = dict(
            created=0,
            expired=0,
            lost_events=0,
            ambiguous_associations=0,
            capacity_evictions=0,
            id_switches=None,
            id_switch_proxy="ambiguous_associations; no identity ground truth",
        )

    def update(self, detections, timestamp, image_shape):
        if not math.isfinite(timestamp) or (
            self._timestamp is not None and timestamp < self._timestamp
        ):
            raise ValueError("Tracking timestamps must be finite and nondecreasing")
        if len(image_shape) < 2 or min(image_shape[:2]) <= 0:
            raise ValueError("Invalid image dimensions")
        self._timestamp = timestamp
        for ident, track in list(self._tracks.items()):
            if timestamp - track.last_seen > self.max_age:
                del self._tracks[ident]
                self.metrics["expired"] += 1
        candidates = []
        for ident, track in self._tracks.items():
            track.association_ambiguous = False
            predicted = track.bbox
            if len(track.history) >= 2:
                t0, b0 = track.history[-2]
                t1, b1 = track.history[-1]
                dt = t1 - t0
                if dt > 0:
                    horizon = min(timestamp - t1, self.max_age)
                    predicted = []
                    for coordinate in range(4):
                        velocity = (b1[coordinate] - b0[coordinate]) / dt
                        predicted.append(b1[coordinate] + velocity * horizon)
            for j, detection in enumerate(detections):
                if detection.class_id != track.class_id:
                    continue
                score = max(
                    overlap(predicted, detection.bbox), overlap(track.bbox, detection.bbox) * 0.8
                )
                if score >= self.iou_threshold:
                    candidates.append((score, ident, j))
        ambiguous_tracks = set()
        ambiguous_detections = set()
        for score, ident, j in candidates:
            alternatives = []
            for other_score, other_track, other_detection in candidates:
                same_track = other_track == ident and other_detection != j
                same_detection = other_detection == j and other_track != ident
                if same_track or same_detection:
                    alternatives.append(other_score)
            if alternatives and abs(score - max(alternatives)) <= self.ambiguity_margin:
                ambiguous_tracks.add(ident)
                ambiguous_detections.add(j)
        self.metrics["ambiguous_associations"] += len(ambiguous_detections)
        used_tracks = set()
        used_detections = set()
        for score, ident, j in sorted(candidates, reverse=True):
            if (
                ident in used_tracks
                or j in used_detections
                or ident in ambiguous_tracks
                or j in ambiguous_detections
            ):
                continue
            track = self._tracks[ident]
            detection = detections[j]
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.last_seen = timestamp
            track.missed = False
            track.history.append((timestamp, detection.bbox))
            track.history = track.history[-self.history_size :]
            if len(track.history) >= 2:
                t0, b0 = track.history[0]
                t1, b1 = track.history[-1]
                track.velocity_y = (b1[3] - b0[3]) / (t1 - t0) if t1 > t0 else None
            used_tracks.add(ident)
            used_detections.add(j)
        for ident, track in self._tracks.items():
            if ident not in used_tracks:
                if not track.missed:
                    self.metrics["lost_events"] += 1
                track.missed = True
                track.velocity_y = None
                track.association_ambiguous = ident in ambiguous_tracks
        for j, detection in enumerate(detections):
            if j in used_detections:
                continue
            ident = self._next_id
            self._next_id += 1
            self._tracks[ident] = Track(
                ident,
                detection.class_id,
                detection.bbox,
                detection.confidence,
                timestamp,
                timestamp,
                history=[(timestamp, detection.bbox)],
                association_ambiguous=j in ambiguous_detections,
            )
            self.metrics["created"] += 1
        while len(self._tracks) > self.max_tracks:
            # Evict missing tracks first, then the oldest observation.
            def eviction_order(ident):
                track = self._tracks[ident]
                return not track.missed, track.last_seen, ident

            ident = min(self._tracks, key=eviction_order)
            del self._tracks[ident]
            self.metrics["capacity_evictions"] += 1
        # Track IDs are the dictionary keys, so no custom sorting function is needed.
        tracks = []
        for ident in sorted(self._tracks):
            tracks.append(self._tracks[ident])
        return tracks
