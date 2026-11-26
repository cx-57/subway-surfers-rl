"""Interactively label original gameplay frames for a scene-specific player model.

Usage:
    .venv/bin/python scripts/calibrate_player.py data/raw/session_<ID> --every 12

Press 1/2/3 for observed lane, then r/j/o for running/jumping/rolling.
Space skips; q ends and saves. Samples are reviewed labels, not held-out accuracy.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.player_vision import PlayerLocalizer


def median(values):
    return float(statistics.median(values))


def build_model(samples, width, height, session):
    anchors = []
    running_y = []
    for lane in range(3):
        positions = []
        neutral = []
        for sample in samples:
            if sample["lane"] != lane:
                continue
            positions.append(sample["x"])
            if sample["movement"] == "RUNNING":
                neutral.append(sample["y"])
        if len(positions) < 3 or len(neutral) < 2:
            raise ValueError(
                f"Lane {lane} needs at least 3 labeled positions and 2 RUNNING examples"
            )
        anchors.append(median(positions))
        running_y.append(median(neutral))
    gap = min(anchors[1] - anchors[0], anchors[2] - anchors[1])
    if gap < 0.035:
        raise ValueError("Reviewed lane centers are not separated enough; check labels and crop")
    model = {
        "provenance": {
            "recording": str(session),
            "method": "operator-labelled original frames and observed upper-body boxes",
            "samples": len(samples),
            "validation": "NOT YET TESTED on a separate gameplay recording",
        },
        "validated": False,
        "aspect_ratio": width / height,
        "lane_x": anchors,
        "max_x_error": min(0.055, gap / 2),
        "lane_margin": min(0.035, gap / 5),
        "stable_frames": 3,
        "max_gap_seconds": 0.35,
    }
    offsets = {"RUNNING": 0.0}
    counts = {}
    for movement in ("JUMPING", "ROLLING"):
        differences = []
        for sample in samples:
            if sample["movement"] == movement:
                differences.append(sample["y"] - running_y[sample["lane"]])
        counts[movement] = len(differences)
        if len(differences) >= 3:
            offsets[movement] = median(differences)
    if (
        len(offsets) == 3
        and offsets["JUMPING"] < -0.025
        and offsets["ROLLING"] > 0.025
        and min(abs(offsets["JUMPING"]), abs(offsets["ROLLING"])) >= 0.045
    ):
        model["running_y"] = running_y
        model["pose_offsets"] = offsets
        model["max_pose_error"] = 0.04
        model["pose_margin"] = 0.02
        print("Pose estimates enabled as drafts: verify on unseen footage.")
    else:
        print("Pose calibration omitted: not enough visually separable jump/roll examples.")
        print("Pose label counts:", counts)
    return model


def select_label(image, bbox, index, total):
    preview = image.copy()
    if bbox is not None:
        x1, y1, x2, y2 = map(round, bbox)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
    h, w = preview.shape[:2]
    font_scale = max(0.55, w / 1900)
    lines = [
        f"Frame {index}/{total} | 1=LEFT  2=CENTER  3=RIGHT",
        "Then r=RUNNING j=JUMPING o=ROLLING",
        "SPACE=skip   q=finish   Only label clearly observed player",
    ]
    for line_number, line in enumerate(lines):
        cv2.putText(preview, line, (16, 34 + 34 * line_number),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 5)
        cv2.putText(preview, line, (16, 34 + 34 * line_number),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    ratio = min(1.0, 1100 / w, 800 / h)
    if ratio < 1:
        preview = cv2.resize(preview, (round(w * ratio), round(h * ratio)))
    cv2.imshow("Label player frames", preview)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == 32:
            return None
        if key not in (ord("1"), ord("2"), ord("3")):
            continue
        lane = key - ord("1")
        while True:
            pose = cv2.waitKey(0) & 0xFF
            if pose in (ord("q"), 27):
                return "quit"
            if pose == 32:
                return None
            movements = {ord("r"): "RUNNING", ord("j"): "JUMPING", ord("o"): "ROLLING"}
            if pose in movements:
                return lane, movements[pose]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Original gameplay frame directory")
    parser.add_argument("--every", type=int, default=12, help="Review every Nth frame")
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.every < 1 or args.max_frames < 1:
        parser.error("Sampling intervals must be positive")
    paths = sorted(args.session.glob("frame_*.jpg"))[::args.every][:args.max_frames]
    if not paths:
        parser.error("No original frame_*.jpg files found in recording")
    output = args.output or Path("config") / f"player_calibration_{args.session.name}.json"
    if output.exists():
        parser.error("Output already exists; select a new --output path to preserve it")
    localizer = PlayerLocalizer()
    samples = []
    dimensions = None
    try:
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                continue
            height, width = image.shape[:2]
            if dimensions is None:
                dimensions = (height, width)
            if dimensions != (height, width):
                raise ValueError("Recording changes image dimensions; use one consistent crop")
            observation = localizer.locate(image, float(index))
            bbox = None if observation is None else observation["bbox"]
            if bbox is None:
                # The operator cannot calibrate an invisible/uncertain character.
                continue
            chosen = select_label(image, bbox, index, len(paths))
            if chosen == "quit":
                break
            if chosen is None:
                continue
            lane, movement = chosen
            x1, y1, x2, y2 = bbox
            samples.append({
                "frame": path.name, "lane": lane, "movement": movement,
                "x": (x1 + x2) / (2 * width),
                "y": (y1 + y2) / (2 * height),
            })
            print(f"{len(samples)} labels: {path.name} lane={lane} pose={movement}", flush=True)
    finally:
        cv2.destroyAllWindows()
    if dimensions is None:
        raise SystemExit("No readable images")
    try:
        model = build_model(samples, dimensions[1], dimensions[0], args.session)
    except ValueError as error:
        raise SystemExit(f"Insufficient or inconsistent calibration labels: {error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump({"player_visual_calibration": model}, stream, indent=2)
        stream.write("\n")
    print("Draft calibration saved to", output)
    print("It is NOT validated; do not enable automatic controls.")


if __name__ == "__main__":
    main()
