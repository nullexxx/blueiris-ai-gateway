from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def block(start, end):
    a = TRACKER.index(start)
    b = TRACKER.index(end, a + len(start))
    return TRACKER[a:b]


class Rev674ResponseAwareHandoffTests(unittest.TestCase):
    def test_status_identifies_rev674(self):
        self.assertIn('"controller_patch": "6.7.4"', TRACKER)
        self.assertIn(
            '"response_aware_native_handoff_decoupled_fast_continuity"',
            TRACKER,
        )
        for token in (
            '"takeup_timeout_s"',
            '"response_window_s"',
            '"hard_max_s"',
            '"pending_response_tail"',
        ):
            self.assertIn(token, TRACKER)

    def test_fast_continuity_is_decoupled_from_native_thresholds(self):
        drive = block(
            "    async def _drive_to_target(",
            "    def _render_debug_frame(",
        )
        self.assertIn(
            "self._hybrid_fast_target = bool(fast_predictive_follow)",
            drive,
        )
        self.assertIn("self._native_fast_handoff_speed_norm", drive)
        self.assertIn("self._native_fast_handoff_projected_travel", drive)
        self.assertIn("fast_native_axes", drive)

    def test_native_handoff_waits_for_measured_response(self):
        chase = block(
            "    async def _drive_hybrid_chase(",
            "    def _begin_ptz_operation",
        )
        self.assertIn('"native_fast_handoff_response_confirmed"', chase)
        self.assertIn("self._native_fast_handoff_takeup_deadline", chase)
        self.assertIn("self._native_fast_handoff_response_started_at", chase)
        self.assertIn('"takeup_timeout"', chase)
        self.assertIn('"response_window_complete"', chase)
        self.assertIn('"hard_max"', chase)

    def test_delayed_native_motion_is_attributed_to_native(self):
        helper = block(
            "    def _reset_native_response_monitor(",
            "    def _apply_axis_reversal_holdoff(",
        )
        self.assertIn("def _archive_native_response_tail(", helper)
        self.assertIn('"native_axis_response_tail"', helper)
        self.assertIn('"native_axis_response_delayed"', helper)
        self.assertIn('"native_axis_response_tail_expired"', helper)
        self.assertIn("after_stop_ms=", helper)

    def test_onvif_response_cannot_claim_pending_native_tail(self):
        helper = block(
            "    def _servo_axis_response_ready(",
            "    def _reset_native_response_monitor(",
        )
        self.assertIn("_servo_response_attribution_block_until", helper)
        self.assertIn("_native_response_tail.get(axis)", helper)

    def test_native_stop_archives_unconfirmed_response_before_reset(self):
        pause = block(
            "    async def _pause_hybrid_actuator(",
            "    async def _stop_hybrid_chase(",
        )
        stop = block(
            "    async def _stop_hybrid_chase(",
            "    @staticmethod\n    def _servo_command_sign",
        )
        self.assertIn("_archive_native_response_tail(t1, reason)", pause)
        self.assertIn("_archive_native_response_tail(t1, reason)", stop)
        self.assertIn("self._enter_ptz_command_lockout(", stop)

    def test_response_aware_env_contracts_exist(self):
        for token in (
            "TRACKER_NATIVE_FAST_HANDOFF_TAKEUP_TIMEOUT",
            "TRACKER_NATIVE_FAST_HANDOFF_RESPONSE_SECONDS",
            "TRACKER_NATIVE_FAST_HANDOFF_HARD_MAX",
            "TRACKER_NATIVE_RESPONSE_TAIL_WINDOW",
            "TRACKER_NATIVE_RESPONSE_ATTRIBUTION_GUARD",
        ):
            self.assertIn(token, TRACKER)


if __name__ == "__main__":
    unittest.main()
