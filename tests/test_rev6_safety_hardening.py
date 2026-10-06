from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def block(start: str, end: str) -> str:
    a = TRACKER.index(start)
    b = TRACKER.index(end, a + len(start))
    return TRACKER[a:b]


class Rev6SafetyHardeningTests(unittest.TestCase):
    def test_failed_active_brake_enters_ptz_command_lockout(self):
        pause = block(
            "    async def _pause_hybrid_actuator",
            "    async def _stop_hybrid_chase",
        )
        stop = block(
            "    async def _stop_hybrid_chase",
            "    def _apply_axis_reversal_holdoff",
        )
        self.assertIn("_enter_ptz_command_lockout(", pause)
        self.assertIn("_enter_ptz_command_lockout(", stop)
        self.assertIn('"PTZ_ERROR"', TRACKER)
        self.assertIn('"ptz_command_lockout"', TRACKER)
        self.assertLess(
            pause.index("_enter_ptz_command_lockout("),
            pause.index("self._hybrid_pan_speed = 0"),
        )

    def test_home_and_calibration_respect_lockout(self):
        home = block("    async def home(", "    async def debug_jpeg")
        calibrate = block("    async def calibrate(", "    async def _wait_zoom_idle")
        goto_home = block(
            "    async def _goto_home_with_retry",
            "    async def home(",
        )
        self.assertIn('_ensure_ptz_command_unlocked("home")', home)
        self.assertIn('"home_command_blocked"', home)
        self.assertIn('_ensure_ptz_command_unlocked("calibrate")', calibrate)
        self.assertIn("if self._ptz_command_lockout:", goto_home)

    def test_camera_status_or_explicit_restart_can_clear_lockout(self):
        self.assertIn(
            '_clear_ptz_command_lockout_from_status(status, source="camera_status")',
            TRACKER,
        )
        self.assertIn('source="explicit_restart"', TRACKER)
        self.assertIn('"ptz_command_lockout": self._ptz_command_lockout', TRACKER)

    def test_continuous_handoff_precedes_move_directly_gate(self):
        drive = block(
            "    async def _drive_to_target(",
            "    async def _maybe_autozoom(",
        )
        self.assertLess(
            drive.index("if hybrid_entry:"),
            drive.index("if not self.cfg.move_directly_enabled:"),
        )
        self.assertIn(
            "Continuous tracking is independent of moveDirectly",
            drive,
        )

    def test_scene_timeout_releases_control_but_quarantines_velocity_learning(self):
        process = block(
            "    async def _process_observation(",
            "    async def _drive_to_target(",
        )
        self.assertIn("released_by_timeout = False", process)
        self.assertIn("if released_by_timeout:", process)
        self.assertIn("self._velocity_learning_ready = False", process)
        self.assertIn('"velocity_learning_resumed"', process)
        self.assertIn("and self._velocity_learning_ready", process)
        self.assertIn(
            '"velocity_learning_ready": self._velocity_learning_ready',
            TRACKER,
        )
        self.assertIn(
            '"velocity_stable_frames": self._velocity_stable_frames',
            TRACKER,
        )


if __name__ == "__main__":
    unittest.main()
