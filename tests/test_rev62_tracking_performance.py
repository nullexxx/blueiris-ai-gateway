import ast
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def load_large_error_gain():
    tree = ast.parse(TRACKER)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_servo_large_error_gain"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "tracker.py", "exec"), namespace)
    return namespace["_servo_large_error_gain"]


def block(start: str, end: str) -> str:
    a = TRACKER.index(start)
    b = TRACKER.index(end, a + len(start))
    return TRACKER[a:b]


class Rev62TrackingPerformanceTests(unittest.TestCase):
    def test_large_error_gain_is_progressive_and_bounded(self):
        gain = load_large_error_gain()
        kwargs = dict(start_error=0.45, full_error=0.75, max_gain=1.45)
        self.assertEqual(gain(0.20, **kwargs), 1.0)
        self.assertEqual(gain(0.45, **kwargs), 1.0)
        self.assertAlmostEqual(gain(0.60, **kwargs), 1.225, places=3)
        self.assertEqual(gain(0.75, **kwargs), 1.45)
        self.assertEqual(gain(-0.90, **kwargs), 1.45)

    def test_large_error_gain_only_boosts_proportional_term(self):
        servo = block("    def _servo_axis_command(", "    def _servo_camera_command(")
        self.assertIn("effective_kp = self._servo_kp * error_gain", servo)
        self.assertIn("kp=effective_kp", servo)
        self.assertIn("kd=self._servo_kd", servo)
        self.assertIn("feedforward_gain=self._servo_feedforward_gain", servo)
        self.assertIn('"error_gain": round(error_gain, 3)', servo)
        self.assertIn('"effective_kp": round(effective_kp, 3)', servo)

    def test_fractional_servo_is_not_stopped_by_native_duration_watchdog(self):
        chase = block("    async def _drive_hybrid_chase(", "    def _begin_ptz_operation(")
        self.assertIn('self._hybrid_actuator != "onvif_fractional"', chase)
        self.assertIn("self.cfg.hybrid_chase_max_seconds", chase)
        self.assertIn('_stop_hybrid_chase("max_duration"', chase)

    def test_active_chase_uses_hold_confidence_not_reacquire_confidence(self):
        chase = block("    async def _drive_hybrid_chase(", "    def _begin_ptz_operation(")
        self.assertIn("chase_confidence_threshold = self.cfg.hold_conf", chase)
        self.assertIn("self.target.confidence < chase_confidence_threshold", chase)
        self.assertNotIn("self.target.confidence < self.cfg.reacquire_conf", chase)
        self.assertIn("threshold=round(chase_confidence_threshold, 3)", chase)

    def test_rev62_policy_remains_present_in_newer_patch(self):
        self.assertIn('"controller_revision": 6', TRACKER)
        self.assertIn('"controller_patch": "6.3"', TRACKER)
        self.assertIn('"active_chase_confidence_threshold"', TRACKER)
        self.assertIn('"native_max_chase_seconds"', TRACKER)
        self.assertIn('"fractional_max_chase_seconds": None', TRACKER)
        self.assertIn('"large_error_start"', TRACKER)
        self.assertIn('"large_error_full"', TRACKER)
        self.assertIn('"large_error_gain"', TRACKER)


if __name__ == "__main__":
    unittest.main()
