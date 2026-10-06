import ast
import math
from pathlib import Path
from typing import Dict
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()
ACTIVE_CAL = (ROOT / "ptz_active_calibration.py").read_text()


def load_servo_policy():
    tree = ast.parse(TRACKER)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_servo_axis_decision"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "Dict": Dict}
    exec(compile(module, "tracker.py", "exec"), namespace)
    return namespace["_servo_axis_decision"]


class Rev5ServoTests(unittest.TestCase):
    def call_servo(self, **overrides):
        values = dict(
            error=0.60,
            error_rate=0.0,
            feedforward_rate=0.0,
            exit_error=0.22,
            kp=0.70,
            kd=0.24,
            feedforward_gain=0.45,
            brake_horizon_s=0.24,
        )
        values.update(overrides)
        return load_servo_policy()(**values)

    def test_fast_closing_target_brakes_before_center(self):
        decision = self.call_servo(error=0.30, error_rate=-1.0)
        self.assertTrue(decision["brake"])
        self.assertEqual(decision["desired_rate"], 0.0)
        self.assertIn(decision["phase"], {"predicted_stop", "closing_fast"})

    def test_outward_motion_increases_servo_demand(self):
        still = self.call_servo(error=0.60, error_rate=0.0)
        outward = self.call_servo(error=0.60, error_rate=0.25)
        self.assertGreater(outward["desired_rate"], still["desired_rate"])

    def test_closing_motion_damps_servo_demand(self):
        still = self.call_servo(error=0.60, error_rate=0.0)
        closing = self.call_servo(error=0.60, error_rate=-0.25)
        self.assertGreater(still["desired_rate"], closing["desired_rate"])

    def test_controller_uses_calibrated_rate_selection(self):
        self.assertIn("choose_continuous_speed_for_rate", TRACKER)
        self.assertIn("def choose_continuous_speed_for_rate", ACTIVE_CAL)
        self.assertIn("TRACKER_SERVO_START_SPEED_MAX", TRACKER)
        self.assertIn("TRACKER_SERVO_ACCEL_STEP", TRACKER)
        self.assertIn('"hybrid_servo"', TRACKER)

    def test_servo_entry_is_seeded_from_pre_chase_velocity(self):
        self.assertIn("def _seed_servo_feedback", TRACKER)
        self.assertIn('self._servo_feedforward["pan"] = self.target.vx', TRACKER)
        self.assertIn('self._servo_feedforward["tilt"] = self.target.vy', TRACKER)
        self.assertIn("self._seed_servo_feedback(frame_shape, now, err_x, err_y)", TRACKER)

    def test_semantic_continuity_is_spatially_gated(self):
        start = TRACKER.index("    def _associate(")
        end = TRACKER.index("    async def _drive_to_target", start)
        block = TRACKER[start:end]
        self.assertNotIn("if det.class_id != self.target.class_id:\n                continue", block)
        self.assertIn("self._class_mismatch_compatible(", block)
        self.assertIn("class_penalty", block)
        self.assertIn("TRACKER_CLASS_CONTINUITY_LABELS", TRACKER)
        self.assertIn("target_class_switched", TRACKER)

    def test_class_flicker_does_not_bypass_explicit_update_guard(self):
        tree = ast.parse(TRACKER)
        target = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TargetTrack"
        )
        update = next(
            node for node in target.body
            if isinstance(node, ast.FunctionDef) and node.name == "update"
        )
        source = ast.get_source_segment(TRACKER, update) or ""
        self.assertIn("allow_class_mismatch", source)
        self.assertIn("and not allow_class_mismatch", source)

    def test_rev5_status_exposes_tuning_and_last_servo_decision(self):
        self.assertIn('"controller_revision": 5', TRACKER)
        self.assertIn('"damped_feedback_servo_semantic_continuity"', TRACKER)
        self.assertIn('"last_decision": self._servo_last_decision or None', TRACKER)
        self.assertIn('"semantic_continuity": {', TRACKER)


if __name__ == "__main__":
    unittest.main()
