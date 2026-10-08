import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def load_function(name: str):
    tree = ast.parse(TRACKER)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "tracker.py", "exec"), namespace)
    return namespace[name]


def block(start: str, end: str) -> str:
    a = TRACKER.index(start)
    b = TRACKER.index(end, a + len(start))
    return TRACKER[a:b]


class Rev63TrackingRefinementTests(unittest.TestCase):
    def test_dynamic_brake_horizon_scales_with_fractional_velocity(self):
        horizon = load_function("_servo_dynamic_brake_horizon")
        kwargs = dict(max_velocity=0.35, base_horizon_s=0.30, velocity_extension_s=0.08)
        self.assertAlmostEqual(horizon(0.0, **kwargs), 0.30, places=4)
        self.assertAlmostEqual(horizon(0.175, **kwargs), 0.34, places=4)
        self.assertAlmostEqual(horizon(0.35, **kwargs), 0.38, places=4)
        self.assertAlmostEqual(horizon(-0.70, **kwargs), 0.38, places=4)

    def test_servo_uses_effective_brake_horizon(self):
        servo = block("    def _servo_axis_command(", "    def _servo_camera_command(")
        self.assertIn("_servo_dynamic_brake_horizon(", servo)
        self.assertIn("brake_horizon_s=effective_brake_horizon", servo)
        self.assertIn('"brake_horizon_s": round(effective_brake_horizon, 3)', servo)

    def test_rev63_loss_windows_are_extended(self):
        self.assertIn('TRACKER_HYBRID_MISSING_GRACE", 0.50', TRACKER)
        self.assertIn('TRACKER_TARGET_RELEASE_TIMEOUT", 5.0', TRACKER)
        self.assertIn("if missing_for < self._target_release_timeout:", TRACKER)
        self.assertIn("self.cfg.home_timeout", TRACKER)

    def test_association_miss_diagnostics_capture_candidate_evidence(self):
        associate = block("    def _associate(", "    async def _drive_to_target(")
        for token in [
            '"nearest_same_class"',
            '"distance_norm"',
            '"inside_distance_gate"',
            '"same_class_count"',
            '"best_score"',
            '"distance_or_class_gate"',
        ]:
            self.assertIn(token, associate)
        process = block("    async def _process_observation(", "    async def _drive_to_target(")
        self.assertIn('"association_miss"', process)
        self.assertIn("association=self._last_association_diagnostic", process)
        self.assertIn('"reacquire_confidence_gate"', process)

    def test_rev63_policy_remains_present_in_newer_patch(self):
        self.assertIn('"controller_patch": "6.7.2"', TRACKER)
        self.assertIn('"target_release_timeout_s"', TRACKER)
        self.assertIn('"brake_velocity_extension_s"', TRACKER)
        self.assertIn('"last_association_diagnostic"', TRACKER)


if __name__ == "__main__":
    unittest.main()
