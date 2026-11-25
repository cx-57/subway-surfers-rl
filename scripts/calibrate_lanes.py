"""Click measured lane centers; save geometry separately from the source image."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def calibration_config(
    image_path, image_shape, points, session=None, first_frame=None, last_frame=None
):
    """Convert actual original-pixel clicks into a scoped calibration document."""
    h, w = image_shape[:2]
    if len(points) < 6 or len(points) % 3:
        raise ValueError("At least two complete rows of three centers are required")
    rows = []
    for start in range(0, len(points), 3):
        group = points[start : start + 3]
        xs, ys = [], []
        for x, y in group:
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError("Clicks must be inside the original image")
            xs.append(x / w)
            ys.append(y)
        if not group[0][0] < group[1][0] < group[2][0]:
            raise ValueError("Click each row left, center, right")
        if max(ys) - min(ys) > 0.04 * h:
            raise ValueError("Centers in each row must share approximately the same depth")
        rows.append([sum(ys) / (3 * h)] + xs)
    rows.sort()
    for first, second in zip(rows, rows[1:]):
        if second[0] - first[0] < 0.05:
            raise ValueError("Use distinct far and near rows separated by at least 5% image height")
    if (first_frame is None) != (last_frame is None):
        raise ValueError("Supply both --first-frame and --last-frame")
    if first_frame is not None and (first_frame < 0 or last_frame < first_frame or not session):
        raise ValueError("A frame range requires --session and ordered nonnegative frame numbers")
    path = Path(image_path)
    config = {
        "lane_rows": rows,
        "aspect_ratio": w / h,
        "valid_frames": [path.name],
        "source_frame": str(path.resolve()),
        "provenance": {
            "source_image": str(path.resolve()),
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "method": "operator clicks on original image pixels",
            "scope": "measured source image only",
        },
    }
    if session:
        config["source_session"] = session
    if first_frame is not None:
        config.pop("valid_frames")
        config.pop("source_frame")
        config["frame_range"] = [first_frame, last_frame]
        config["provisional"] = True
        config["provenance"][
            "scope"
        ] = "PROVISIONAL operator assumption: geometry applies throughout requested session frame range"
    return config


def collect_points(image):
    """Use original resolution so mouse coordinates never depend on display scaling."""
    title = "Lane calibration: left center right; far then near | Enter save | R redo | Esc cancel"
    points = []
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            points.append((x, y))

    cv2.setMouseCallback(title, click)
    try:
        while True:
            display = image.copy()
            for index, point in enumerate(points):
                cv2.circle(display, point, 5, (0, 190, 255), -1)
                cv2.putText(
                    display,
                    str(index + 1),
                    point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            cv2.imshow(title, display)
            key = cv2.waitKey(30) & 255
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) >= 6 and len(points) % 3 == 0:
                return points
    finally:
        cv2.destroyWindow(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session")
    parser.add_argument("--first-frame", type=int)
    parser.add_argument("--last-frame", type=int)
    args = parser.parse_args()
    if args.image.resolve() == args.output.resolve():
        parser.error("Output must differ from the source image")
    if args.output.exists():
        parser.error("Output already exists; choose a new configuration filename")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    try:
        while True:
            points = collect_points(image)
            if points is None:
                print("Calibration cancelled; nothing saved.")
                return
            try:
                config = calibration_config(
                    args.image, image.shape, points, args.session, args.first_frame, args.last_frame
                )
                break
            except ValueError as error:
                print(f"{error}. Please redo the clicks.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        print(f'Saved {args.output}; {config["provenance"]["scope"]}')
    except cv2.error as error:
        parser.exit(1, f"Calibration display unavailable: {error}\n")


if __name__ == "__main__":
    main()
