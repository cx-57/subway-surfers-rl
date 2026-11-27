"""Configure Chrome lifecycle, train DQN, or compare it with uniform random actions."""

import argparse
import json
from pathlib import Path
import platform
import sys
from time import time_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from agent.sources import ScreenSource, select_region
from scripts.gameplay import countdown, select_area


def prompt_capture(message, delay):
    input(message + "\nPress Enter here, then switch to Chrome. ")
    countdown(delay)


def select_point(image):
    title = "Click the START button position | Enter: confirm | Esc: cancel"
    point = []

    def mouse(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            point[:] = [x, y]

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            preview = image.copy()
            if point:
                cv2.circle(preview, tuple(point), 8, (0, 255, 0), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Setup cancelled")
            if key in (10, 13) and point:
                return point
    finally:
        cv2.destroyAllWindows()


def configure_ui(config, path, source, delay, start_image=None):
    from agent.rl.lifecycle import GameOverDetector

    if start_image is None:
        prompt_capture("Reload manually and leave Chrome on the initial START menu after any ad ends.", delay)
        start_image = source().image
        start_x, start_y = select_point(start_image)
        for command in config["reset_commands"]:
            if command["type"] == "click":
                command["x"] = start_x / start_image.shape[1]
                command["y"] = start_y / start_image.shape[0]
                break
    print("Select a static START-menu-only label (such as PRESS TO PLAY), then Enter.", flush=True)
    start_patch = select_area(start_image)
    if start_patch is None:
        raise RuntimeError("UI calibration cancelled")
    prompt_capture("Start a run and keep playing during the countdown. Capture must show ACTIVE gameplay.", delay)
    game_image = source().image
    print("Select a small static gameplay-only HUD icon, such as PAUSE. Exclude changing numbers.", flush=True)
    gameplay_patch = select_area(game_image)
    if gameplay_patch is None:
        raise RuntimeError("UI calibration cancelled")
    specs = []
    for name, image, rectangle, other in (
        ("start_ui", start_image, start_patch, game_image),
        ("gameplay_ui", game_image, gameplay_patch, start_image),
    ):
        x, y, width, height = rectangle
        h, w = image.shape[:2]
        roi = [x / w, y / h, (x + width) / w, (y + height) / h]
        template = image[y:y + height, x:x + width]
        detector = GameOverDetector(template, roi)
        _, score = detector.update(other, 0.0)
        if score >= detector.threshold:
            raise ValueError(f"{name} marker matches both menu and gameplay; select a more distinctive icon")
        template_path = path.with_suffix(f".{name}.png")
        specs.append((name, template_path, template, roi))
    for name, template_path, template, roi in specs:
        if not cv2.imwrite(str(template_path), template):
            raise OSError("Could not save UI template")
        config[name] = dict(template=template_path.name, roi=roi, threshold=0.94)
    config["reset_timeout"] = max(90.0, config.get("reset_timeout", 0))


def calibrate_ui(args):
    path = args.config.resolve()
    config = json.loads(path.read_text())
    source = ScreenSource(config["region"])
    try:
        configure_ui(config, path, source, args.countdown)
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Updated UI calibration in {path}. Run check again.")
    finally:
        source.close()


def setup(args):
    from agent.rl.lifecycle import GameOverDetector

    path = args.config.resolve()
    template_path = path.with_suffix(".game_over.png")
    if path.exists() or template_path.exists():
        raise ValueError("Setup output already exists; use a new --config filename")
    prompt_capture("Show the complete gameplay area in Chrome.", args.countdown)
    region = select_region(args.monitor)
    if region is None:
        return
    source = ScreenSource(region)
    try:
        prompt_capture("Play until the final game-over/restart screen is visible. "
                       "Dismiss revive offers first; keep browser position/zoom fixed.", args.countdown)
        image = source().image
        print("Select a small, distinctive STATIC game-over label: two clicks, then Enter.", flush=True)
        patch = select_area(image)
        if patch is None:
            return
        x, y, width, height = patch
        height_image, width_image = image.shape[:2]
        roi = [x / width_image, y / height_image,
               (x + width) / width_image, (y + height) / height_image]
        template = image[y:y + height, x:x + width]
        terminal_detector = GameOverDetector(template, roi)
        prompt_capture("Reload Chrome manually and leave the game on its initial START menu. "
                       "Dismiss any overlays, but do not start playing.", args.countdown)
        start_image = source().image
        _, menu_score = terminal_detector.update(start_image, 0.0)
        if menu_score >= terminal_detector.threshold:
            raise ValueError("Game-over patch also matches the start menu; choose a distinctive death-only label")
        start_x, start_y = select_point(start_image)
        config = dict(
            region=region,
            game_over=dict(template=template_path.name, roi=roi, threshold=0.94),
            reset_commands=[
                dict(type="hotkey", keys=["command" if platform.system() == "Darwin" else "ctrl", "r"]),
                dict(type="wait", seconds=5.0),
                dict(type="click", x=start_x / width_image, y=start_y / height_image),
                dict(type="wait", seconds=0.5),
            ],
            reset_timeout=90.0, freeze_seconds=5.0, terminal_grace_seconds=8.0,
            motion_threshold=0.005,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        configure_ui(config, path, source, args.countdown, start_image=start_image)
        if not cv2.imwrite(str(template_path), template):
            raise OSError("Could not save game-over template")
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Saved {path}. Run the check command before training.")
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "calibrate-ui", "check", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("config/rl.local.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/training/weights/best.pt"))
    parser.add_argument("--model", type=Path, help="DQN model for evaluation or training resume")
    parser.add_argument("--replay", type=Path, help="Replay buffer to restore with --model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--action-seconds", type=float, default=0.12)
    parser.add_argument("--episode-seconds", type=float, default=120.0)
    parser.add_argument("--countdown", type=int, default=10)
    parser.add_argument("--monitor", type=int, default=1)
    args = parser.parse_args()
    if args.countdown < 0 or args.episodes < 1 or args.steps < 1:
        parser.error("Countdown must be nonnegative; episodes/steps must be positive")
    if args.mode == "setup":
        setup(args)
        return
    if args.mode == "calibrate-ui":
        calibrate_ui(args)
        return
    if args.mode == "evaluate" and args.model is None:
        parser.error("evaluate requires --model")
    if args.replay is not None and args.model is None:
        parser.error("--replay requires --model")
    from stable_baselines3.common.monitor import Monitor
    from agent.rl.chrome import ChromeBackend, load_config
    from agent.rl.env import SubwayEnv
    from agent.rl.learning import evaluate, load_model, train

    config = load_config(args.config)
    output = args.output or Path("models/rl") / f"{args.mode}_{time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    backend = ChromeBackend(config, args.checkpoint)
    env = Monitor(SubwayEnv(backend, action_seconds=args.action_seconds,
                            max_episode_seconds=args.episode_seconds), str(output / "episodes"))
    try:
        # Model load happens before binding controls to the foreground Chrome window.
        model = load_model(args.model) if args.mode == "evaluate" else None
        print(f"AUTOMATED CONTROL: focus the selected Chrome window. Output: {output}", flush=True)
        print("Switching away stops controls. PyAutoGUI's top-left mouse failsafe is enabled.", flush=True)
        countdown(args.countdown)
        if args.mode == "check":
            # Observe a complete no-op episode to test death recognition and resetting twice.
            for episode in range(2):
                _, info = env.reset(seed=args.seed)
                while True:
                    _, _, terminated, truncated, info = env.step(0)
                    if terminated or truncated:
                        print(json.dumps(info, indent=2), flush=True)
                        if truncated:
                            raise RuntimeError("No game-over detected before time limit; review template/reset setup")
                        break
            print("Two game-over detections and resets completed.")
        elif args.mode == "train":
            train(env, output, steps=args.steps, seed=args.seed,
                  checkpoint_every=args.checkpoint_every, learning_starts=args.learning_starts,
                  buffer_size=args.buffer_size, resume=args.model, replay=args.replay)
        else:
            print(json.dumps(evaluate(env, model, output, args.episodes, args.seed), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("RL run stopped.")
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"RL failed: {exc}")
