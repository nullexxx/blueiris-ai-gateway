import ast
import math
from pathlib import Path
from typing import Tuple
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


def load_functions(*names):
    tree = ast.parse(TRACKER)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"math": math, "Tuple": Tuple}
    exec(compile(module, "tracker.py", "exec"), ns)
    return [ns[name] for name in names]


class Rev66PredictiveTrackingTests(unittest.TestCase):
    def test_fast_center_cross_starts_continuous_before_old_error_threshold(self):
        decision, = load_functions("_motion_control_decision")
        result = decision(
            err_x=-0.27, err_y=-0.08,
            vx=950.0, vy=0.0,
            velocity_valid=True, velocity_sample_ms=300,
            frame_shape=(480, 640, 3), move_eta_s=1.3,
            hybrid_enabled=True, hybrid_disabled=False, cooldown_ready=True,
            edge_clipped=False, exit_error=0.50, entry_error=0.82,
            motion_error=0.55, motion_speed_norm=0.03,
            min_sample_ms=250, deadline_travel_norm=0.18,
            stationary_speed_norm=0.012, moving_error=0.35,
            fast_follow_speed_norm=0.10, fast_follow_horizon_s=0.45,
            fast_follow_error=0.22,
        )
        self.assertTrue(result["fast_predictive_follow"])
        self.assertTrue(result["use_continuous"])
        self.assertEqual(result["reason"], "fast_predictive_follow")
        self.assertGreater(result["future_error_x"], 0.22)

    def test_slow_center_cross_does_not_trigger_fast_follow(self):
        decision, = load_functions("_motion_control_decision")
        result = decision(
            err_x=-0.20, err_y=0.0,
            vx=40.0, vy=0.0,
            velocity_valid=True, velocity_sample_ms=300,
            frame_shape=(480, 640, 3), move_eta_s=1.3,
            hybrid_enabled=True, hybrid_disabled=False, cooldown_ready=True,
            edge_clipped=False, exit_error=0.50, entry_error=0.82,
            motion_error=0.55, motion_speed_norm=0.03,
            min_sample_ms=250, deadline_travel_norm=0.18,
            stationary_speed_norm=0.012, moving_error=0.35,
            fast_follow_speed_norm=0.10, fast_follow_horizon_s=0.45,
            fast_follow_error=0.22,
        )
        self.assertFalse(result["fast_predictive_follow"])

    def test_false_positive_boxes_quantize_to_same_hotspot(self):
        key, = load_functions("_acquisition_hotspot_key")
        a = key((334.0, 40.0, 369.0, 156.0), (480, 640, 3))
        b = key((335.0, 39.0, 368.0, 155.0), (480, 640, 3))
        dog = key((65.0, 204.0, 186.0, 279.0), (480, 640, 3))
        self.assertEqual(a, b)
        self.assertNotEqual(a, dog)

    def test_animal_continuity_includes_bear_but_not_person(self):
        self.assertIn('continuity_default = "dog,cat,bird,bear"', TRACKER)
        self.assertNotIn('continuity_default = "person', TRACKER)

    def test_history_primary_time_is_local_and_utc_is_retained(self):
        self.assertIn('"time_utc": wall_utc.isoformat', TRACKER)
        self.assertIn('wall_utc.astimezone(self._event_tz)', TRACKER)
        self.assertIn('"event_timezone": self._event_timezone_name', TRACKER)
        self.assertIn('"session_started_utc"', TRACKER)

    def test_static_hotspot_guard_is_conservative_not_permanent(self):
        self.assertIn('"acquire_static_hotspot_suspected"', TRACKER)
        self.assertIn('"acquire_static_hotspot_rejected"', TRACKER)
        self.assertIn("self._static_hotspot_override_conf", TRACKER)
        self.assertIn("self._static_hotspot_motion_threshold", TRACKER)
        self.assertIn("self._static_hotspot_block_s", TRACKER)

    def test_hotspot_guard_preserves_home_context_until_confirmation(self):
        self.assertIn("self._active_acquire_home_context = home_context", TRACKER)
        self.assertIn("if self._active_acquire_home_context and self._ptz_operation is None:", TRACKER)
        self.assertIn("self._home_sent = True", TRACKER)

    def test_motion_aware_hold_keeps_chase_identity_when_projection_leaves_center(self):
        hold, = load_functions("_servo_hold_action")
        action = hold(
            0.375, -0.083,
            0.10, -0.04,
            exit_error=0.22,
            resume_growth=0.06,
            elapsed_s=0.33,
            hold_seconds=0.30,
            frames=5,
            min_frames=3,
            projected_error_x=-0.31,
            projected_error_y=-0.02,
            max_hold_seconds=0.75,
        )
        self.assertEqual(action, "wait_motion")

    def test_motion_aware_hold_still_settles_when_projection_remains_centered(self):
        hold, = load_functions("_servo_hold_action")
        action = hold(
            0.24, 0.04,
            0.10, 0.03,
            exit_error=0.22,
            resume_growth=0.06,
            elapsed_s=0.33,
            hold_seconds=0.30,
            frames=5,
            min_frames=3,
            projected_error_x=0.15,
            projected_error_y=0.04,
            max_hold_seconds=0.75,
        )
        self.assertEqual(action, "settled")

    def test_hold_resume_renews_divergence_grace(self):
        self.assertIn("self._hybrid_divergence_grace_started_at = now", TRACKER)
        self.assertIn("divergence_grace_anchor = max(", TRACKER)
        self.assertIn('"servo_hold_motion_guard"', TRACKER)
        self.assertIn("divergence_grace_reset_ms=", TRACKER)

    def test_status_identifies_rev66(self):
        self.assertIn('"controller_patch": "6.7.3"', TRACKER)
        self.assertIn('"fast_follow_speed_norm"', TRACKER)
        self.assertIn('"static_acquisition_guard"', TRACKER)
        self.assertIn('"divergence_min_camera_shift_px"', TRACKER)


if __name__ == "__main__":
    unittest.main()
