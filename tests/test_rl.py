"""Deterministic contract tests and actual SB3 replay/gradient/save/load smoke test.

No test sends desktop input or claims learned gameplay performance.
"""

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
try:
    from stable_baselines3.common.env_checker import check_env
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Install requirements-rl.txt to run RL tests") from exc
from agent.state import GameState, PlayerState, Track
from agent.rl.env import Sample, SubwayEnv
from agent.rl.lifecycle import GameOverDetector
from agent.rl.observation import FIELDS, GLOBAL_FIELDS, SLOT_FIELDS, ObservationEncoder
from agent.rl.learning import evaluate, load_model, train


def state(timestamp, box=(40, 55, 60, 85)):
    return GameState(timestamp, (100, 100), player=PlayerState(bbox=box),
                     observation_age=0.01, timestamp_quality="monotonic")


class FakeBackend:
    """A clocked test fixture, not a browser/game simulator or benchmark."""

    def __init__(self):
        self.closed = False
        self.clock = 10.0
        self.steps = 0
        self.reset_count = 0
        self.actions = []

    def reset(self):
        self.clock = 10.0
        self.steps = 0
        self.reset_count += 1
        return Sample(state(self.clock))

    def act(self, action):
        self.actions.append(action)
        return self.clock + 0.01, self.clock + 0.02

    def observe(self, after):
        self.clock = after + 0.01
        self.steps += 1
        sample = state(self.clock, (40 + self.steps, 55, 60 + self.steps, 85))
        return Sample(sample, terminal=self.steps >= 4)

    def close(self):
        self.closed = True


class ObservationTests(unittest.TestCase):
    def test_boxes_and_velocity_survive_unknown_geometry_and_missing_player(self):
        encoder = ObservationEncoder()
        sample = state(1, box=None)
        sample.tracks = [Track(1, 2, (10, 20, 30, 60), .8, 0, 1, velocity_y=20)]
        vector = encoder.encode(sample)
        self.assertEqual(vector.shape, (len(FIELDS),))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.isfinite(vector).all())
        slot = len(GLOBAL_FIELDS)
        self.assertEqual(vector[0], 0)
        self.assertAlmostEqual(vector[slot + SLOT_FIELDS.index("x1")], .1)
        self.assertAlmostEqual(vector[slot + SLOT_FIELDS.index("approach_velocity")], .2)
        self.assertEqual(vector[slot + SLOT_FIELDS.index("velocity_known")], 1)
        self.assertFalse(sample.calibration_valid)

    def test_bounded_slots_sorting_player_velocity_and_episode_reset(self):
        encoder = ObservationEncoder()
        first = state(1)
        first.tracks = []
        for ident in range(20):
            first.tracks.append(Track(ident, ident % 6, (0, 0, 100, ident + 50), .9, 0, 1))
        first.tracks[-1].missed = True
        vector = encoder.encode(first)
        self.assertAlmostEqual(vector[len(GLOBAL_FIELDS) + SLOT_FIELDS.index("y2")], .68)
        second = state(1.2, box=(42, 55, 62, 85))
        vector = encoder.encode(second, previous_action=2)
        self.assertAlmostEqual(vector[FIELDS.index("player_vx")], .1)
        self.assertEqual(vector[FIELDS.index("previous_right")], 1)
        self.assertEqual(vector[FIELDS.index("player_velocity_known")], 1)
        self.assertTrue((np.abs(vector) <= 1).all())
        with self.assertRaises(ValueError):
            encoder.encode(second)
        encoder.reset()
        self.assertEqual(encoder.encode(state(.1))[FIELDS.index("player_velocity_known")], 0)


class LifecycleTests(unittest.TestCase):
    def test_template_requires_persistent_match_and_resets_on_mismatch(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        detector = GameOverDetector(image[20:40, 10:50], (.1, .2, .5, .4))
        self.assertFalse(detector.update(image, 1)[0])
        self.assertTrue(detector.update(image, 1.2)[0])
        self.assertFalse(detector.update(np.zeros_like(image), 1.4)[0])
        self.assertFalse(detector.update(image, 1.6)[0])
        detector.reset()
        self.assertFalse(detector.update(image, 2)[0])

    def test_blank_terminal_patch_is_rejected(self):
        with self.assertRaises(ValueError):
            GameOverDetector(np.zeros((20, 20, 3), np.uint8), (0, 0, 1, 1))


class EnvironmentTests(unittest.TestCase):
    def test_delayed_game_over_excludes_offer_time_and_corrects_prior_survival_credit(self):
        backend = FakeBackend()
        env = SubwayEnv(backend)
        env.reset()
        _, initial_reward, _, _, _ = env.step(0)
        # Confirmed death appears several seconds later. Motion stopped before the
        # preceding sample, so its provisional survival credit must be retracted.
        delayed = Sample(state(17.0), True, 1.0, terminal_at=10.1,
                         observation_wait_seconds=6.0)
        with patch.object(backend, "observe", return_value=delayed):
            _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["survival_seconds"], .1)
        self.assertAlmostEqual(initial_reward + reward, -.9)
        self.assertEqual(info["terminal_wait_seconds"], 6.0)
        self.assertAlmostEqual(info["episode_wall_seconds"], 7.0)

    def test_unconfirmed_wait_cannot_bypass_timestamp_limits(self):
        backend = FakeBackend()
        env = SubwayEnv(backend)
        env.reset()
        sample = Sample(state(17), observation_wait_seconds=6)
        with patch.object(backend, "observe", return_value=sample):
            with self.assertRaisesRegex(RuntimeError, "terminal observation wait"):
                env.step(0)

    def test_gym_contract_and_survival_reward_terminal_and_reset(self):
        backend = FakeBackend()
        env = SubwayEnv(backend)
        check_env(env, warn=True)
        env.reset(seed=1)
        reward_sum = 0
        for action in range(4):
            _, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            self.assertGreaterEqual(info["observation_timestamp"], info["action_finished_at"] + .12)
            self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward_sum, info["survival_seconds"] - 1)
        with self.assertRaises(RuntimeError):
            env.step(0)
        observation, _ = env.reset()
        self.assertEqual(observation[FIELDS.index("player_velocity_known")], 0)
        env.close()
        self.assertTrue(backend.closed)

    def test_time_limit_is_truncation_and_invalid_actions_do_not_send_keys(self):
        backend = FakeBackend()
        env = SubwayEnv(backend, max_episode_seconds=.1)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(5)
        self.assertEqual(backend.actions, [])
        _, _, terminated, truncated, _ = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_stale_pre_action_and_post_action_frames_are_rejected(self):
        backend = FakeBackend()
        env = SubwayEnv(backend)
        env.reset()
        with patch.object(backend, "observe", return_value=Sample(state(10.01))):
            with self.assertRaisesRegex(RuntimeError, "Unsynchronized"):
                env.step(1)
        env.reset()
        stale = state(10.2)
        stale.observation_age = .9
        with patch.object(backend, "observe", return_value=Sample(stale)):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                env.step(1)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_capture_failure_propagates_without_a_fabricated_transition(self):
        backend = FakeBackend()
        env = SubwayEnv(backend)
        env.reset()
        with patch.object(backend, "observe", side_effect=TimeoutError("capture failed")):
            with self.assertRaises(TimeoutError):
                env.step(0)
        self.assertEqual(env.steps, 0)
        self.assertTrue(env.finished)


class ChromeContractTests(unittest.TestCase):
    def test_static_revive_offer_waits_for_confirmed_death_without_more_keys(self):
        from agent.rl.chrome import ChromeBackend

        backend = ChromeBackend.__new__(ChromeBackend)
        backend.config = {}
        backend.controller = Mock()
        backend.last_motion = 10.0
        backend._sample = Mock(side_effect=[
            (Sample(state(15.1)), False),
            (Sample(state(15.3), True, 1.0), False),
        ])
        with patch("agent.rl.chrome.monotonic", return_value=15.1), redirect_stdout(io.StringIO()):
            sample = backend.observe(15.0)
        self.assertTrue(sample.terminal)
        self.assertEqual(sample.terminal_at, 10.0)
        self.assertAlmostEqual(sample.observation_wait_seconds, .2)
        backend.controller.act.assert_not_called()
        self.assertEqual(backend.last_state_timestamp, 15.3)

    def test_real_freeze_still_aborts_after_bounded_grace(self):
        from agent.rl.chrome import ChromeBackend

        backend = ChromeBackend.__new__(ChromeBackend)
        backend.config = {}
        backend.controller = Mock()
        backend.last_motion = 10.0
        backend._sample = Mock(return_value=(Sample(state(15.1)), False))
        with patch("agent.rl.chrome.monotonic", side_effect=[15.1, 15.1, 24.0]), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "revive grace period"):
                backend.observe(15.0)
        backend.controller.act.assert_not_called()

    def test_reset_guards_start_click_then_waits_for_gameplay_before_creating_engine(self):
        from agent.capture import Frame
        from agent.rl.chrome import ChromeBackend

        backend = ChromeBackend.__new__(ChromeBackend)
        backend.config = dict(reset_commands=[dict(type="hotkey", keys=["ctrl", "r"]),
                                              dict(type="click", x=.5, y=.5)],
                              region=dict(left=0, top=0, width=100, height=100))
        backend.controller = Mock()
        backend.terminal = Mock(threshold=.94)
        backend.terminal.update.return_value = False, .1
        backend.start_ui = Mock()
        backend.gameplay_ui = Mock()
        backend.detector = Mock()
        backend._open = Mock()
        frame = Frame(np.zeros((100, 100, 3), np.uint8), 1.6, 1.6)
        backend._wait_for_ui = Mock(return_value=frame)
        with patch("agent.rl.chrome.Engine") as engine:
            engine.return_value.process.return_value = state(1.6)
            sample = backend.reset()
        self.assertEqual(sample.state.timestamp, 1.6)
        self.assertEqual(backend._wait_for_ui.call_count, 2)
        self.assertIs(backend._wait_for_ui.call_args_list[0].args[0], backend.start_ui)
        self.assertIs(backend._wait_for_ui.call_args_list[1].args[0], backend.gameplay_ui)
        self.assertTrue(backend._wait_for_ui.call_args.kwargs["require_motion"])
        engine.assert_called_once()
        self.assertEqual(backend.terminal.reset.call_count, 1)
        self.assertEqual(backend.controller.reset_game.call_count, 2)

    def test_animated_ads_do_not_satisfy_gameplay_ui_gate(self):
        from agent.capture import Frame
        from agent.rl.chrome import ChromeBackend

        backend = ChromeBackend.__new__(ChromeBackend)
        backend.config = {}
        backend.controller = Mock()
        template = np.random.default_rng(3).integers(0, 255, (20, 20, 3), dtype=np.uint8)
        detector = GameOverDetector(template, (.1, .1, .3, .3))
        frames = []
        for index, brightness in enumerate((0, 255, 80, 100)):
            image = np.full((100, 100, 3), brightness, np.uint8)
            if index >= 2:
                image[10:30, 10:30] = template
            frames.append(Frame(image, 1 + index * .2, 1 + index * .2))
        backend.source = Mock(side_effect=frames)
        with patch("agent.rl.chrome.monotonic", return_value=0), redirect_stdout(io.StringIO()):
            frame = backend._wait_for_ui(detector, "gameplay HUD", 90, require_motion=True)
        self.assertIs(frame, frames[-1])
        self.assertEqual(backend.source.call_count, 4)
        backend.controller.act.assert_not_called()
        backend.controller.reset_game.assert_not_called()

    def test_ad_ui_wait_is_bounded(self):
        from agent.rl.chrome import ChromeBackend

        backend = ChromeBackend.__new__(ChromeBackend)
        backend.config = {}
        backend.controller = Mock()
        detector = Mock()
        with patch("agent.rl.chrome.monotonic", return_value=91), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "ad may need dismissal"):
                backend._wait_for_ui(detector, "start menu", 90)
        backend.controller.reset_game.assert_not_called()

    def test_focus_loss_prevents_keyboard_event(self):
        from agent.rl.chrome import ChromeController

        controller = ChromeController.__new__(ChromeController)
        controller.keyboard = Mock()
        controller.target = 123
        controller.foreground = Mock(return_value=456)
        with self.assertRaisesRegex(RuntimeError, "lost focus"):
            controller.act(1)
        controller.keyboard.press.assert_not_called()

    def test_backend_waits_for_action_boundary_and_rejects_old_capture(self):
        from agent.rl.chrome import ChromeBackend

        backend = ChromeBackend.__new__(ChromeBackend)
        backend.controller = Mock()
        backend.config = {}
        backend.last_motion = 9
        backend._sample = Mock(return_value=(Sample(state(10.2)), True))
        with patch("agent.rl.chrome.monotonic", return_value=10):
            sample = backend.observe(10.1)
        backend.controller.wait.assert_called_once_with(unittest.mock.ANY)
        self.assertEqual(sample.state.timestamp, 10.2)
        backend._sample.return_value = Sample(state(10.05)), True
        with self.assertRaisesRegex(RuntimeError, "precedes"):
            backend.observe(10.1)


class LearningTests(unittest.TestCase):
    def test_interrupted_run_preserves_only_completed_replay_transitions(self):
        backend = FakeBackend()
        original_observe = backend.observe

        def fail_capture(after):
            if len(backend.actions) > 6:
                raise RuntimeError("capture interrupted")
            return original_observe(after)

        backend.observe = fail_capture
        env = SubwayEnv(backend)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            output = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "capture interrupted"):
                train(env, output, steps=16, learning_starts=4,
                      checkpoint_every=100, buffer_size=32)
            model = load_model(output / "dqn_final.zip", env,
                               output / "dqn_final.replay.pkl")
            self.assertEqual(model.replay_buffer.size(), 6)
            self.assertEqual(json.loads((output / "training.json").read_text())["status"], "interrupted")
            self.assertEqual(len((output / "transitions.jsonl").read_text().splitlines()), 6)
        env.close()

    def test_real_dqn_replay_gradients_checkpoint_resume_and_random_comparison(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            output = Path(folder)
            env = SubwayEnv(FakeBackend())
            model = train(env, output, steps=48, learning_starts=8,
                          checkpoint_every=24, buffer_size=64)
            self.assertGreater(model._n_updates, 0)
            self.assertEqual(model.replay_buffer.size(), 48)
            self.assertTrue((output / "dqn_24.zip").exists())
            restored = load_model(output / "dqn_final.zip", env,
                                  output / "dqn_final.replay.pkl")
            self.assertEqual(restored.replay_buffer.size(), 48)
            resumed = train(env, output / "resumed", steps=8,
                            resume=output / "dqn_final.zip",
                            replay=output / "dqn_final.replay.pkl")
            self.assertEqual(resumed.num_timesteps, 56)
            self.assertEqual(resumed.replay_buffer.size(), 56)
            observation, _ = env.reset()
            np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0],
                                          restored.predict(observation, deterministic=True)[0])
            summary = evaluate(env, restored, output, episodes=2)
            self.assertEqual(summary["episodes_per_policy"], 2)
            self.assertEqual(summary["random"]["truncations"], 0)
            self.assertAlmostEqual(summary["dqn_minus_random_survival_seconds"], 0)
            rows = (output / "transitions.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 48)
            manifest = output / "dqn_final.schema.json"
            altered = json.loads(manifest.read_text())
            altered["version"] = -1
            manifest.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_model(output / "dqn_final.zip")
            env.close()


if __name__ == "__main__":
    unittest.main()
