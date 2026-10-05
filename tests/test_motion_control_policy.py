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
        )
        values.update(overrides)
        return load_motion_policy()(**values)

    def test_mature_fast_target_uses_continuous_before_edge(self):
        decision = self.call_policy()
        self.assertTrue(decision["use_continuous"])
        self.assertTrue(decision["velocity_mature"])
        self.assertEqual(decision["reason"], "deadline_motion")
        self.assertGreater(decision["projected_travel_norm"], 0.18)

    def test_immature_velocity_does_not_drive_deadline_handoff(self):
        decision = self.call_policy(velocity_sample_ms=120)
        self.assertFalse(decision["velocity_mature"])
        self.assertFalse(decision["use_continuous"])

    def test_inward_motion_does_not_start_continuous_chase(self):
        decision = self.call_policy(vx=-100.0)
        self.assertTrue(decision["moving_inward_dominant"])
        self.assertFalse(decision["use_continuous"])

    def test_hard_escape_can_chase_without_velocity(self):
        decision = self.call_policy(
            err_x=0.95,
            vx=0.0,
            velocity_valid=False,
            velocity_sample_ms=0,
        )
        self.assertTrue(decision["use_continuous"])
        self.assertEqual(decision["reason"], "hard_escape")

    def test_global_disable_is_authoritative(self):
        decision = self.call_policy(hybrid_enabled=False)
        self.assertFalse(decision["use_continuous"])

    def test_live_target_quality_no_longer_mutates_spatial_calibration(self):
        start = TRACKER.index("    def _complete_move_quality")
        end = TRACKER.index("    def _hybrid_axis_speed", start)
        block = TRACKER[start:end]
        self.assertNotIn("set_spatial_scales", block)
        self.assertIn("Live-target telemetry is observational only", block)

    def test_velocity_maturity_and_lead_are_bounded(self):
        self.assertIn(
            "max(self.cfg.velocity_min_sample_ms, self._motion_control_min_sample_ms)",
            TRACKER,
        )
        self.assertIn(
            "lead_horizon_s = min(predicted_move_eta_s, self._move_direct_lead_horizon_max)",
            TRACKER,
        )
        self.assertIn("max_lead_x = w * self._move_direct_lead_max_fraction", TRACKER)
        self.assertIn("controller_revision", TRACKER)


if __name__ == "__main__":
    unittest.main()
