from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def block(start, end):
    a = TRACKER.index(start)
    b = TRACKER.index(end, a + len(start))
    return TRACKER[a:b]


class Rev672FastContinuityTests(unittest.TestCase):
    def test_status_identifies_rev672(self):
        self.assertIn('"controller_patch": "6.7.4"', TRACKER)
        self.assertIn(
            '"response_aware_native_handoff_decoupled_fast_continuity"',
            TRACKER,
        )
        for token in (
            '"acquire_gap_grace_s"',
            '"fast_target_missing_grace_s"',
            '"native_fast_handoff"',
            '"native_axis_response"',
        ):
            self.assertIn(token, TRACKER)

    def test_acquisition_survives_short_detector_gap(self):
        process = block(
            "    async def _process_observation(",
            "    def _associate(",
        )
        self.assertIn("acquire_gap_s <= self._acquire_gap_grace_s", process)
        self.assertIn('"acquire_gap_grace"', process)
        self.assertIn('"acquire_gap_recovered"', process)

    def test_servo_hold_detector_miss_does_not_zero_stop_chase(self):
        process = block(
            "    async def _process_observation(",
            "    def _associate(",
        )
        self.assertIn('policy=coast_policy', process)
        self.assertIn('"servo_hold_pause"', process)
        self.assertIn("if self._servo_hold_since is not None:", process)
        self.assertIn("self.state = \"ESCAPE_CHASE\"", process)

    def test_fast_target_missing_coast_holds_then_tapers(self):
        process = block(
            "    async def _process_observation(",
            "    def _associate(",
        )
        self.assertIn("self._fast_target_missing_grace_s", process)
        self.assertIn("self._fast_target_coast_hold_s", process)
        self.assertIn("self._fast_target_coast_end_scale", process)
        self.assertIn('"fast_fractional_hold_then_taper"', process)

    def test_fast_native_handoff_is_separate_from_general_edge_rescue(self):
        drive = block(
            "    async def _drive_to_target(",
            "    def _render_debug_frame(",
        )
        self.assertIn("self._native_fast_handoff_enabled", drive)
        self.assertIn("self._native_fast_handoff_speed_norm", drive)
        self.assertIn("self._native_fast_handoff_projected_travel", drive)
        self.assertIn("self._native_fast_handoff_future_error", drive)
        self.assertIn('"native_fast_handoff_enter"', drive)
        self.assertIn("fast_native_axes", drive)
        self.assertIn("TRACKER_NATIVE_EDGE_RESCUE_ENABLED", TRACKER)

    def test_native_handoff_is_bounded_and_can_exit_early(self):
        chase = block(
            "    async def _drive_hybrid_chase(",
            "    def _begin_ptz_operation",
        )
        self.assertIn('"native_fast_handoff_exit"', chase)
        self.assertIn('exit_reason = "inner_region"', chase)
        self.assertIn('"native_fast_handoff_inner_region"', chase)
        self.assertIn("now < self._native_fast_handoff_until", chase)

    def test_native_takeup_latency_is_instrumented(self):
        helper = block(
            "    def _reset_native_response_monitor(",
            "    def _apply_axis_reversal_holdoff(",
        )
        self.assertIn('"native_axis_command_direction"', helper)
        self.assertIn('"native_axis_response_armed"', helper)
        self.assertIn("takeup_ms=", helper)
        speed = block(
            "    async def _set_hybrid_chase_speed(",
            "    async def _set_hybrid_chase_velocity(",
        )
        self.assertIn('_note_native_applied_command("pan"', speed)
        self.assertIn('source=(', speed)
        self.assertIn('"native_fast_handoff"', speed)

    def test_existing_fail_closed_stop_safety_remains(self):
        stop = block(
            "    async def _stop_hybrid_chase(",
            "    @staticmethod\n    def _servo_command_sign",
        )
        self.assertIn("self._enter_ptz_command_lockout(", stop)
        self.assertIn('"hybrid_chase_stop_failed"', stop)


if __name__ == "__main__":
    unittest.main()
