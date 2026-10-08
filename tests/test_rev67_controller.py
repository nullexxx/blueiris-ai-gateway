from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def block(start, end):
    a = TRACKER.index(start)
    b = TRACKER.index(end, a + len(start))
    return TRACKER[a:b]


class Rev67ControllerTests(unittest.TestCase):
    def test_native_rescue_latches_only_triggered_axes(self):
        chase = block("    async def _drive_hybrid_chase(", "    def _begin_ptz_operation")
        self.assertIn("self._native_edge_rescue_axes = set(requested_axes)", chase)
        self.assertIn('if "pan" in self._native_edge_rescue_axes else 0', chase)
        self.assertIn('if "tilt" in self._native_edge_rescue_axes else 0', chase)
        self.assertIn('"native_edge_rescue_axis_added"', chase)

    def test_native_rescue_has_progress_abort(self):
        chase = block("    async def _drive_hybrid_chase(", "    def _begin_ptz_operation")
        self.assertIn("_native_edge_rescue_progress_check_s", chase)
        self.assertIn("_native_edge_rescue_min_improvement", chase)
        self.assertIn('"native_edge_rescue_progress_abort"', chase)
        self.assertIn('reason="axes_complete_or_stalled"', chase)

    def test_self_induced_motion_uses_frozen_loss_clock(self):
        process = block("    async def _process_observation(", "    def _associate(")
        self.assertIn("self._pause_loss_clock(self.target.last_seen)", process)
        self.assertIn("self._pause_loss_clock(now)", process)
        self.assertIn("self._resume_loss_clock(now)", process)
        self.assertIn("missing_for = self._target_missing_seconds(now)", process)
        self.assertIn("missing_for_match = self._target_missing_seconds(now)", process)

    def test_applied_ptz_commands_have_authoritative_event(self):
        speed = block("    async def _set_hybrid_chase_speed(", "    async def _set_hybrid_chase_velocity(")
        velocity = block("    async def _set_hybrid_chase_velocity(", "    async def _drive_hybrid_chase(")
        self.assertIn('"ptz_command_applied"', speed)
        self.assertIn('actuator="native_discrete"', speed)
        self.assertIn('"ptz_command_applied"', velocity)
        self.assertIn('actuator="onvif_fractional"', velocity)

    def test_fractional_calibration_covers_runtime_high_end(self):
        self.assertIn("[0.04, 0.08, 0.12, 0.16, 0.24, 0.32]", TRACKER)

    def test_status_identifies_rev67(self):
        self.assertIn('"controller_patch": "6.7.4"', TRACKER)
        self.assertIn('"active_axes": sorted(self._native_edge_rescue_axes)', TRACKER)
        self.assertIn('"loss_clock_paused"', TRACKER)


if __name__ == "__main__":
    unittest.main()
