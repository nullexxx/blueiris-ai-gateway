from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def block(start, end):
    a = TRACKER.index(start)
    b = TRACKER.index(end, a + len(start))
    return TRACKER[a:b]


class Rev671DivergenceResponseTests(unittest.TestCase):
    def test_applied_fractional_command_updates_axis_response_state(self):
        velocity = block(
            "    async def _set_hybrid_chase_velocity(",
            "    async def _drive_hybrid_chase(",
        )
        self.assertIn(
            'self._note_servo_applied_command("pan", desired[0], t1)',
            velocity,
        )
        self.assertIn(
            'self._note_servo_applied_command("tilt", desired[1], t1)',
            velocity,
        )

    def test_direction_change_disarms_divergence_until_camera_takes_up_command(self):
        helper = block(
            "    @staticmethod\n    def _servo_command_sign",
            "    def _apply_axis_reversal_holdoff(",
        )
        self.assertIn('state["armed"] = False', helper)
        self.assertIn("if shift * correction_sign >= 0.0:", helper)
        self.assertIn('state["armed"] = True', helper)
        self.assertIn('"servo_axis_response_armed"', helper)

    def test_divergence_uses_applied_direction_not_fresh_servo_desire(self):
        chase = block(
            "    async def _drive_hybrid_chase(",
            "    def _begin_ptz_operation",
        )
        self.assertIn("pan_applied_sign", chase)
        self.assertIn("tilt_applied_sign", chase)
        self.assertIn("if pan_ready else 0", chase)
        self.assertIn("if tilt_ready else 0", chase)
        self.assertIn(
            'divergence_evidence="armed_wrong_direction_camera_motion"',
            chase,
        )

    def test_repeated_divergence_cools_down_without_session_disable(self):
        chase = block(
            "    async def _drive_hybrid_chase(",
            "    def _begin_ptz_operation",
        )
        self.assertNotIn("self._hybrid_disabled_for_session = trip", chase)
        self.assertIn("self._servo_divergence_escalated_cooldown_s", chase)
        self.assertIn("session_disabled=False", chase)
        self.assertIn('"repeated_divergence_cooldown"', chase)

    def test_status_identifies_rev671_and_response_monitor(self):
        self.assertIn('"controller_patch": "6.7.2"', TRACKER)
        self.assertIn(
            '"fractional_servo_response_armed_divergence_frozen_loss_clock"',
            TRACKER,
        )
        self.assertIn('"axis_response": {', TRACKER)
        self.assertIn('"divergence_session_disable": False', TRACKER)


if __name__ == "__main__":
    unittest.main()
