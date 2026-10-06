import ast
import json
import math
from pathlib import Path
import tempfile
from typing import Dict
import unittest

from ptz_active_calibration import ActivePtzCalibrationStore


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


class Rev6ControllerTests(unittest.TestCase):
    def test_closing_rate_preserves_error_rate_sign_for_negative_error(self):
        servo = load_servo_policy()
        closing = servo(
            error=-0.50,
            error_rate=0.40,
            feedforward_rate=0.0,
            exit_error=0.22,
            kp=0.70,
            kd=0.24,
            feedforward_gain=0.45,
            brake_horizon_s=0.24,
        )
        outward = servo(
            error=-0.50,
            error_rate=-0.40,
            feedforward_rate=0.0,
            exit_error=0.22,
            kp=0.70,
            kd=0.24,
            feedforward_gain=0.45,
            brake_horizon_s=0.24,
        )
        self.assertGreater(closing["closing_rate"], 0.0)
        self.assertLess(outward["closing_rate"], 0.0)

    def test_fractional_onvif_velocity_interpolates_below_native_speed_floor(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.json"
            path.write_text(json.dumps({
                "version": 1,
                "cameras": {
                    "cam": {
                        "move_directly": {"samples": [{"ok": True}]},
                        "continuous": {"samples": [{"ok": True}]},
                        "onvif_benchmark": {
                            "fractional_continuous": {
                                "usable": True,
                                "signs": {"pan": 1, "tilt": -1},
                                "samples": [
                                    {"axis": "pan", "velocity": 0.04, "normalized_rate_per_s": 0.10},
                                    {"axis": "pan", "velocity": 0.08, "normalized_rate_per_s": 0.20},
                                    {"axis": "tilt", "velocity": 0.04, "normalized_rate_per_s": 0.16},
                                    {"axis": "tilt", "velocity": 0.08, "normalized_rate_per_s": 0.32},
                                ],
                            }
                        },
                    }
                },
            }))
            store = ActivePtzCalibrationStore(str(path), "cam")
            self.assertTrue(store.has_onvif_fractional_continuous())
            self.assertAlmostEqual(
                store.onvif_velocity_for_rate("pan", 0.05, max_velocity=0.35),
                0.02,
                places=3,
            )
            self.assertEqual(store.onvif_continuous_sign("tilt", 1), -1)

    def test_continuous_stop_requires_quiet_time_and_scene_stability(self):
        self.assertIn("TRACKER_SERVO_POST_STOP_SETTLE", TRACKER)
        self.assertIn("self._continuous_settle_until", TRACKER)
        self.assertIn("self._scene_stability.reset(t1)", TRACKER)
        self.assertIn("time.monotonic() >= self._continuous_settle_until", TRACKER)
        self.assertIn('"camera_settling"', TRACKER)

    def test_divergence_uses_strikes_before_session_disable(self):
        self.assertIn("TRACKER_SERVO_DIVERGENCE_TRIP_LIMIT", TRACKER)
        self.assertIn("self._hybrid_divergence_strikes.append(now)", TRACKER)
        self.assertIn("trip = strikes >= self._servo_divergence_trip_limit", TRACKER)
        self.assertIn("self._hybrid_disabled_for_session = trip", TRACKER)
        self.assertIn('"repeated_divergence" if trip else "diverging"', TRACKER)

    def test_fractional_onvif_runtime_and_calibration_contracts(self):
        self.assertIn("async def continuous_move(self, pan: float, tilt: float)", ACTIVE_CAL)
        self.assertIn("async def stop(self) -> bool", ACTIVE_CAL)
        self.assertIn("TRACKER_ONVIF_SERVO_CALIBRATION_VELOCITIES", TRACKER)
        self.assertIn('"fractional_continuous"', TRACKER)
        self.assertIn('"onvif_fractional"', TRACKER)
        self.assertIn("onvif_velocity_for_rate", TRACKER)

    def test_native_discrete_is_edge_rescue_fallback_without_fractional_profile(self):
        self.assertIn('"fractional_actuator_unavailable"', TRACKER)
        self.assertIn("not self._servo_onvif_available", TRACKER)
        self.assertIn("dominant_error < hard_escape_error", TRACKER)

    def test_rev6_status_contract(self):
        self.assertIn('"controller_revision": 6', TRACKER)
        self.assertIn(
            '"fractional_servo_settled_velocity_semantic_continuity"',
            TRACKER,
        )
        self.assertIn('"fractional_onvif_available"', TRACKER)
        self.assertIn('"post_stop_settle_remaining_ms"', TRACKER)


if __name__ == "__main__":
    unittest.main()
