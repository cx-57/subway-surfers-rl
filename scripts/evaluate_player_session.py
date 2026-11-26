"""Evaluate the scene-specific *draft* player calibration on held-out frames.

The review file has manual labels for sampled original JPEGs only. All other
images are unlabeled and MUST NOT be interpreted as correct model predictions.

Usage:
  python scripts/evaluate_player_session.py \
    data/raw/session_1795724241854738000 \
    --calibration config/player_calibration_session_1795724241854738000.json \
    --labels data/reviews/session_1795724241854738000_player.json

Writes outputs into models/evaluation/ without changing model weights or images.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.player import PlayerEstimator


def accuracy(correct, attempts):
    """Do not report a percentage if there were no eligible predictions."""
    if attempts == 0:
        return None
    return round(correct / attempts, 4)


def evaluate_rows(entries):
    """Aggregate annotated held-out rows; do not count unknown as correct.

    Each entry is a dictionary containing reference labels and output of the
    real pixel localizer. A location far from a reviewed player is a false
    match, not a correct prediction merely because the x-lane looks plausible.
    """
    stats = {
        "reviewed_holdout": 0,
        "eligible_ground": 0,
        "localized_ground": 0,
        "plausible_location_ground": 0,
        "stable_lane_ground": 0,
        "correct_lane_ground": 0,
        "classified_pose_ground": 0,
        "correct_pose_ground": 0,
        "false_localizations_on_negative_frames": 0,
        "negative_frames": 0,
        "wrong_lane_examples": [],
        "wrong_pose_examples": [],
        "unrecognized_examples": [],
        "unscored_context_examples": [],
    }
    for row in entries:
        if row["split"] != "holdout":
            continue
        stats["reviewed_holdout"] += 1
        scene = row["scene"]
        frame = row["frame"]
        if scene in ("non_gameplay", "editor"):
            stats["negative_frames"] += 1
            if row["bbox"] is not None:
                stats["false_localizations_on_negative_frames"] += 1
            continue
        if scene != "ground" or row["lane"] is None or row["movement"] not in (
            "RUNNING", "JUMPING", "ROLLING"
        ):
            stats["unscored_context_examples"].append(frame)
            continue
        stats["eligible_ground"] += 1
        if row["bbox"] is None:
            stats["unrecognized_examples"].append(frame)
            continue
        stats["localized_ground"] += 1
        center_error = row.get("center_error_px")
        if center_error is None or center_error > 75:
            stats["unrecognized_examples"].append(frame)
            continue
        stats["plausible_location_ground"] += 1
        predicted_lane = row.get("predicted_lane")
        if predicted_lane is not None:
            stats["stable_lane_ground"] += 1
            if predicted_lane == row["lane"]:
                stats["correct_lane_ground"] += 1
            else:
                stats["wrong_lane_examples"].append(frame)
        pose = row.get("predicted_pose", "UNKNOWN")
        if pose != "UNKNOWN":
            stats["classified_pose_ground"] += 1
            if pose == row["movement"]:
                stats["correct_pose_ground"] += 1
            else:
                stats["wrong_pose_examples"].append(frame)
    stats["ground_localization_recall"] = accuracy(
        stats["plausible_location_ground"], stats["eligible_ground"]
    )
    stats["lane_coverage_ground"] = accuracy(
        stats["stable_lane_ground"], stats["eligible_ground"]
    )
    stats["lane_accuracy_when_predicted"] = accuracy(
        stats["correct_lane_ground"], stats["stable_lane_ground"]
    )
    stats["pose_coverage_ground"] = accuracy(
        stats["classified_pose_ground"], stats["eligible_ground"]
    )
    stats["pose_accuracy_when_predicted"] = accuracy(
        stats["correct_pose_ground"], stats["classified_pose_ground"]
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.labels.read_text())
    config = json.loads(args.calibration.read_text())
    if not config.get("player_visual_calibration"):
        parser.error("Missing player_visual_calibration in calibration JSON")
    if config["player_visual_calibration"].get("validated") is not False:
        parser.error("This evaluation requires validated=false until independent verification")
    if args.session.name != data["session"]:
        parser.error("Review labels belong to a different original recording")

    meta = json.loads((args.session / "metadata.json").read_text())
    fps = float(meta.get("actual_fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        parser.error("Recording metadata has no positive actual FPS")
    if (meta.get("frame_width"), meta.get("frame_height")) != (
        data["width"], data["height"]
    ):
        parser.error("Original image geometry differs from reviewed labels")
    frames = sorted(args.session.glob("frame_*.jpg"))
    if not frames:
        parser.error("Session contains no original gameplay frames")

    reviewed = {}
    for row in data["labels"]:
        number = row.get("frame")
        if type(number) is not int or number in reviewed:
            parser.error("Invalid or duplicate reviewed frame identifier")
        reviewed[number] = row

    output = args.output or (
        ROOT / "models/evaluation" / args.session.name.replace("session_", "player_session_", 1)
    )
    output.mkdir(parents=True, exist_ok=True)
    player = PlayerEstimator(config)
    results = []
    diagnostics = {
        "total_frames": len(frames),
        "localized_all_frames": 0,
        "unreviewed_frames_not_ground_truth": 0,
        "calibration_validated": False,
        "source": args.session.name,
        "method": "real PlayerEstimator / PlayerLocalizer on ORIGINAL JPEGs",
    }
    for path in frames:
        number = int(path.stem.rsplit("_", 1)[1])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Unreadable original frame: {path}")
        if (image.shape[1], image.shape[0]) != (data["width"], data["height"]):
            raise ValueError(f"Image crop changed: {path}")
        timestamp = (number - 1) / fps
        state = player.update(image, timestamp, str(path), image.shape[:2])
        prediction = state.localization.get("visual_calibration", {})
        bbox = state.bbox
        if bbox is not None:
            diagnostics["localized_all_frames"] += 1
        label = reviewed.get(number)
        if label is None:
            diagnostics["unreviewed_frames_not_ground_truth"] += 1
        center_error_px = None
        if bbox is not None and label and label.get("center_px") is not None:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            tx, ty = label["center_px"]
            center_error_px = round(math.hypot(cx - tx, cy - ty), 2)
        row = {
            "frame": number,
            "split": label["split"] if label else "unreviewed",
            "scene": label["scene"] if label else "unreviewed",
            "lane": label["lane"] if label else None,
            "movement": label["movement"] if label else None,
            "bbox": list(bbox) if bbox is not None else None,
            "center_error_px": center_error_px,
            "predicted_lane": prediction.get("lane"),
            "predicted_pose": prediction.get("pose", "UNKNOWN"),
            "stable": prediction.get("stable", False),
            "reason": prediction.get("reason", state.lane_reason),
            "match_score": state.match_score,
        }
        results.append(row)
    summary = evaluate_rows(results)
    summary.update(diagnostics)
    summary["caveats"] = [
        "Holdout is later time block in the SAME run, not an independent session.",
        "Manual reference centers are approximate, not validated pixel-perfect boxes.",
        "Only reviewed frames contribute to classification metrics.",
        "Unreviewed frames are not ground truth or evidence of accuracy.",
        "Pose from upper-body y-offset cannot certify contact with ground or train roof.",
        "Draft model remains validated=false and sends no keyboard input.",
    ]
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "frame_predictions.jsonl").open("w") as log:
        for row in results:
            log.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print("Saved evaluation to", output)


if __name__ == "__main__":
    main()
