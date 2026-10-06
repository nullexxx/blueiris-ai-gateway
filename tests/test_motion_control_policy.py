import ast
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def load_motion_policy():
    tree = ast.parse(TRACKER)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_motion_control_decision"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, "tracker.py", "exec"), namespace)
    return namespace["_motion_control_decision"]


class MotionControlPolicyTests(unittest.TestCase):
    def call_policy(self, **overrides):
        values = dict(
            err_x=0.65,
            err_y=0.10,
            vx=80.0,
            vy=0.0,
            velocity_valid=True,
            velocity_sample_ms=300,
            frame_shape=(480, 640),
            move_eta_s=1.6,
            hybrid_enabled=True,
            hybrid_disabled=False,
            cooldown_ready=True,
            edge_clipped=False,
            exit_error=0.50,
            entry_error=0.82,
            motion_error=0.55,
            motion_speed_norm=0.03,
            min_sample_ms=250,
            deadline_travel_norm=0.18,
            stationary_speed_norm=0.012,
            moving_error=0.35,
        )
        values.update(overrides)
        return load_motion_policy()(**values)

    def test_mature_moving_target_uses_continuous_before_edge(self):
        decision = self.call_policy(err_x=0.40)
        self.assertTrue(decision["use_continuous"])
        self.assertTrue(decision["moving_target"])
        self.assertFalse(decision["precision_move_allowed"])
        self.assertIn(decision["reason"], {"deadline_motion", "motion_escape"})

    def test_moving_target_inside_entry_region_holds_instead_of_move_direct(self):
        decision = self.call_policy(err_x=0.25, vx=35.0)
        self.assertFalse(decision["use_continuous"])
        self.assertTrue(decision["moving_target"])
        self.assertFalse(decision["precision_move_allowed"])
        self.assertEqual(decision["precision_hold_reason"], "moving_target")

    def test_stationary_mature_target_allows_precision_move(self):
        decision = self.call_policy(err_x=0.30, vx=2.0)
        self.assertFalse(decision["use_continuous"])
        self.assertFalse(decision["moving_target"])
        self.assertTrue(decision["precision_move_allowed"])
        self.assertIsNone(decision["precision_hold_reason"])

    def test_immature_velocity_waits_before_precision_move(self):
        decision = self.call_policy(velocity_sample_ms=120)
        self.assertFalse(decision["velocity_mature"])
        self.assertFalse(decision["precision_move_allowed"])
        self.assertEqual(decision["precision_hold_reason"], "velocity_sample")

    def test_inward_motion_does_not_start_continuous_chase(self):
        decision = self.call_policy(vx=-100.0)
        self.assertTrue(decision["moving_inward_dominant"])
        self.assertFalse(decision["use_continuous"])
        self.assertFalse(decision["precision_move_allowed"])

    def test_hard_escape_can_chase_without_velocity(self):
        decision = self.call_policy(
            err_x=0.95,
            vx=0.0,
            velocity_valid=False,
            velocity_sample_ms=0,
        )
        self.assertTrue(decision["use_continuous"])
        self.assertEqual(decision["reason"], "hard_escape")

    def test_hybrid_disabled_preserves_move_direct_fallback(self):
        decision = self.call_policy(hybrid_enabled=False, velocity_sample_ms=0)
        self.assertFalse(decision["use_continuous"])
        self.assertTrue(decision["precision_move_allowed"])

    def test_live_target_quality_no_longer_mutates_spatial_calibration(self):
        start = TRACKER.index("    def _complete_move_quality")
        end = TRACKER.index("    def _hybrid_axis_speed", start)
        block = TRACKER[start:end]
        self.assertNotIn("set_spatial_scales", block)
        self.assertIn("Live-target telemetry is observational only", block)

    def test_rev4_precision_contracts_are_present(self):
        self.assertIn('"controller_revision": 4', TRACKER)
        self.assertIn('"persistent_velocity_continuous_servo_precision"', TRACKER)
        self.assertIn('"adaptive_hybrid_current_center"', TRACKER)
        self.assertIn("self._precision_hold_until", TRACKER)
        self.assertIn("hybrid_chase_confidence_grace", TRACKER)
        self.assertIn("scale = max(1.0", TRACKER)
        self.assertIn("precision_move_deferred", TRACKER)

    def test_continuous_exit_is_inside_old_escape_exit(self):
        self.assertIn("TRACKER_MOTION_CONTROL_CONTINUOUS_EXIT_ERROR", TRACKER)
        self.assertIn("self._motion_control_continuous_exit_error", TRACKER)
        start = TRACKER.index("    async def _drive_hybrid_chase")
        end = TRACKER.index("    def _begin_ptz_operation", start)
        block = TRACKER[start:end]
        self.assertIn("<= self._motion_control_continuous_exit_error", block)


if __name__ == "__main__":
    unittest.main()
