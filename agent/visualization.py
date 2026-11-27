"""Read-only BGR overlays shared by offline replay and live observation.

Cyan boxes are observations; muted boxes marked UPCOMING are distant tracks.
Amber annotations are inferences. Missing tracks
use gray boxes. Display owners may use TOGGLE_KEYS with ``toggle_key``.
"""

import textwrap

import cv2
import numpy as np
from .detection import CLASS_NAMES as DETECTOR_CLASSES

CLASS_NAMES = dict(enumerate(DETECTOR_CLASSES))
TOGGLE_KEYS = {"b": "boxes", "l": "lanes", "p": "player", "h": "hazards", "s": "status"}
OBSERVED = (255, 220, 30)
UPCOMING = (170, 140, 60)
INFERRED = (0, 190, 255)
MISSING = (150, 150, 150)


def toggle_key(toggles, key):
    """Mutate overlay switches; return whether a recognized key was received."""
    name = TOGGLE_KEYS.get(chr(key & 0xFF))
    if name:
        toggles[name] = not toggles.get(name, True)
    return name is not None


def _label(image, text, x, y, color, scale=0.4):
    h, w = image.shape[:2]
    x, y = max(0, min(int(x), w - 1)), max(10, min(int(y), h - 2))
    cv2.putText(
        image, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(image, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _lane_curves(geometry):
    if geometry is None:
        return []
    config = getattr(geometry, "config", geometry if isinstance(geometry, dict) else {})
    rows = config.get("lane_rows", []) if isinstance(config, dict) else []
    if rows:
        depths = []
        for row in rows:
            depths.append(row[0])
        lo, hi = min(depths), max(depths)
    else:
        lo, hi = 0.15, 1.0
    samples = []
    for y in np.linspace(lo, hi, 45):
        if hasattr(geometry, "lane_centers"):
            centers = geometry.lane_centers(float(y))
        elif rows:
            ordered = sorted(rows)
            columns = np.asarray(ordered).T
            centers = []
            for lane in range(3):
                centers.append(np.interp(y, columns[0], columns[lane + 1]))
        else:
            return []
        if centers is None or len(centers) != 3:
            continue
        a, b, c = map(float, centers)
        samples.append((y, (a - (b - a) / 2, (a + b) / 2, (b + c) / 2, c + (c - b) / 2)))
    return samples


def render(image, state, geometry=None, toggles=None):
    """Return an annotated copy with exactly the input dimensions and aspect ratio."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    output = image.copy()
    switches = dict.fromkeys(TOGGLE_KEYS.values(), True)
    switches.update(toggles or {})
    h, w = image.shape[:2]
    if switches["lanes"] and state.calibration_valid:
        samples = _lane_curves(geometry)
        for boundary in range(4):
            points = []
            for y, xs in samples:
                points.append((round(xs[boundary] * (w - 1)), round(y * (h - 1))))
            points = np.array(points, dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(output, [points], False, INFERRED, 1, cv2.LINE_AA)
    hazards = {}
    for hazard in state.hazards:
        hazards[hazard.track_id] = hazard
    for track in state.tracks:
        # Show early detections; proximity gates hazards, not box visibility.
        distant = state.nearby_track_ids is not None and track.id not in state.nearby_track_ids
        proximity = (
            (" UPCOMING" if distant else " NEARBY") if state.nearby_track_ids is not None else ""
        )
        x1, y1, x2, y2 = map(round, track.bbox)
        color = MISSING if track.missed else (UPCOMING if distant else OBSERVED)
        if switches["boxes"]:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            _label(
                output,
                f"#{track.id} {CLASS_NAMES.get(track.class_id, track.class_id)} "
                f"{track.confidence:.2f}" + proximity + (" MISSING" if track.missed else ""),
                x1,
                y1 - 5,
                color,
            )
        if distant:
            continue
        if switches["lanes"]:
            candidates = ",".join(map(str, track.lane.candidates)) or "?"
            _label(output, f"lane {candidates} {track.lane.status}", x1, y2 + 14, INFERRED)
        if switches["hazards"] and track.id in hazards:
            hazard = hazards[track.id]
            ttc = "?" if hazard.ttc is None else f"{hazard.ttc:.2f}s"
            _label(output, f"{hazard.status} {hazard.urgency} TTC {ttc}", x1, y2 + 28, INFERRED)
    if switches["player"]:
        player = state.player
        if player.bbox is not None:
            x1, y1, x2, y2 = map(round, player.bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), (80, 255, 80), 2)
            match = "" if player.match_score is None else f" NCC={player.match_score:.2f}"
            _label(output, f"Player {player.bbox_kind}{match}", x1, y1 - 6, (80, 255, 80))
        candidates = ",".join(map(str, player.lane_candidates)) or "?"
        if player.lane is not None:
            shown_lane = ("LEFT", "CENTER", "RIGHT")[player.lane] + " REVIEWED"
        elif player.estimated_lane is not None:
            stable = player.localization.get("screen_lane", {}).get("stable", False)
            status = "STABLE SCREEN ESTIMATE" if stable else "SCREEN CANDIDATE"
            shown_lane = ("LEFT", "CENTER", "RIGHT")[player.estimated_lane] + " " + status
        else:
            shown_lane = "? (transition or uncertain)"
        screen = player.localization.get("screen_lane", {})
        x = screen.get("normalized_x")
        x_label = f" | x={x * 100:.1f}%" if isinstance(x, (int, float)) else ""
        _label(
            output,
            f"Player lane {shown_lane}{x_label} | {player.movement}",
            6,
            h - 10,
            INFERRED,
        )
    if switches["status"]:
        recommendation = state.recommendation
        timings = state.timings
        fps = timings.get("processed_fps")
        latency = timings.get("inference_ms", timings.get("inference"))
        lines = [
            f"t={state.timestamp:.3f} {state.timestamp_quality} "
            f"age={state.observation_age*1000:.0f}ms"
            + (f" FPS={fps:.1f}" if fps is not None else ""),
            f"Action {recommendation.action} | "
            + ("actionable" if recommendation.actionable else "uncertain"),
            recommendation.reason,
            f'Calibration: {"valid" if state.calibration_valid else "UNKNOWN"} {state.calibration_reason}',
        ]
        if state.nearby_track_ids is not None:
            lines[1] += f" | nearby {len(state.nearby_track_ids)}/{len(state.tracks)}"
        if latency is not None:
            lines[0] += f" inference={latency:.1f}ms"
        if timings.get("capture_fps") is not None:
            lines[0] += f' capture={timings["capture_fps"]:.1f}FPS'
        # Keep the status to a narrow, bounded corner, even with verbose reasons.
        chars = max(12, min(68, int(w * 0.72 / 7)))
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, chars)[:2])
        lines = wrapped[:7]
        panel_h, panel_w = min(h, 8 + 15 * len(lines)), min(w, chars * 7 + 12)
        patch = output[:panel_h, :panel_w]
        patch[:] = (patch.astype(np.float32) * 0.35).astype(np.uint8)
        for index, line in enumerate(lines):
            _label(output, line, 6, 15 + index * 15, (235, 235, 235))
    return output
