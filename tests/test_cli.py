"""The simple command entrypoint must preserve arguments and never run on import."""

import unittest
from unittest.mock import patch

import run


class RunTests(unittest.TestCase):
    def test_default_is_observe(self):
        command = run.build_command([])
        self.assertEqual(command[-1], "scripts/observe_live.py")
        self.assertEqual(command[0], run.sys.executable)

    def test_observe_flags_pass_through(self):
        command = run.build_command(["--duration", "60", "--headless"])
        self.assertEqual(
            command[-3:], ["--duration", "60", "--headless"]
        )
        self.assertIn("scripts/observe_live.py", command)

    def test_named_commands(self):
        for name, suffix in (
            ("record", "scripts/gameplay.py"),
            ("calibrate", "scripts/calibrate_player.py"),
            ("replay", "scripts/replay_gameplay.py"),
            ("test", "unittest"),
        ):
            with self.subTest(name=name):
                self.assertIn(suffix, run.build_command([name]))

    def test_control_check_only_on_explicit_mode(self):
        self.assertNotIn("--validate-controls", run.build_command([]))
        self.assertIn("--validate-controls", run.build_command(["controls"]))

    @patch("run.subprocess.run")
    def test_command_passes_args_and_returns_child_exit_code(self, executor):
        executor.return_value.returncode = 3
        result = run.main(["calibrate", "data/raw/session_x", "--every", "12"])
        self.assertEqual(result, 3)
        self.assertIn("data/raw/session_x", executor.call_args.args[0])
        self.assertEqual(executor.call_args.kwargs["cwd"], run.ROOT)

    @patch("run.subprocess.run")
    def test_help_and_unknown_command_do_not_launch(self, executor):
        self.assertEqual(run.main(["help"]), 0)
        self.assertEqual(run.main(["unknown"]), 2)
        executor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
