from pathlib import Path


def one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def scoped(text: str, start_marker: str, end_marker: str):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return start, end, text[start:end]


tracker_path = Path("tracker.py")
text = tracker_path.read_text()

# ---------------------------------------------------------------------------
# Rev 3 motion policy: continuous PTZ owns moving subjects; moveDirectly is
# reserved for slow/stationary precision corrections. This deliberately borrows
# Frigate's strongest control ideas: classify motion before commanding a slow
# positional move, keep a meaningful no-move region, and never let stale motion
# prediction become a new long-latency positional command.
# ---------------------------------------------------------------------------
policy = r'''
def _motion_control_decision(
    *,
    err_x,
    err_y,
    vx,
    vy,
    velocity_valid,
    velocity_sample_ms,
    frame_shape,
    move_eta_s,
    hybrid_enabled,
    hybrid_disabled,
    cooldown_ready,
    edge_clipped,
    exit_error,
    entry_error,
    motion_error,
    motion_speed_norm,
    min_sample_ms,
    deadline_travel_norm,
    stationary_speed_norm,
    moving_error,
):
    """Choose continuous chase vs slow/stationary precision positioning.

    Rev 3 treats moveDirectly as a precision controller, not a predictor. A
    moving target either receives continuous PTZ (when sufficiently displaced)
    or is intentionally held until its motion settles. This avoids committing
    the camera to a 1.3-1.8 second move based on a target that will be somewhere
    else when the move completes.
    """
    h, w = frame_shape[:2]
    half_w = max(1.0, float(w) / 2.0)
    half_h = max(1.0, float(h) / 2.0)
    frame_diag = max(1.0, math.hypot(w, h))
    dominant_error = max(abs(err_x), abs(err_y))
    target_speed_norm = math.hypot(vx, vy) / frame_diag if velocity_valid else 0.0
    velocity_mature = bool(velocity_valid and velocity_sample_ms >= min_sample_ms)
    moving_target = bool(
        velocity_mature and target_speed_norm > stationary_speed_norm
    )

    if abs(err_x) >= abs(err_y):
        dominant_axis_error = err_x
        dominant_axis_velocity = vx
    else:
        dominant_axis_error = err_y
        dominant_axis_velocity = vy

    moving_inward_dominant = velocity_mature and (
        abs(dominant_axis_error) >= moving_error
        and dominant_axis_error * dominant_axis_velocity < 0.0
        and abs(dominant_axis_velocity) / frame_diag >= stationary_speed_norm
    )
    moving_outward = velocity_mature and (
        (abs(err_x) >= moving_error and err_x * vx > 0.0)
        or (abs(err_y) >= moving_error and err_y * vy > 0.0)
    )

    projected_travel_norm = 0.0
    if velocity_mature:
        projected_travel_norm = max(
            abs(vx) * max(0.0, move_eta_s) / half_w,
            abs(vy) * max(0.0, move_eta_s) / half_h,
        )

    motion_escape = (
        moving_target
        and dominant_error >= moving_error
        and moving_outward
        and target_speed_norm >= min(motion_speed_norm, stationary_speed_norm * 1.25)
    )
    deadline_escape = (
        moving_target
        and dominant_error >= moving_error
        and projected_travel_norm >= deadline_travel_norm
        and not moving_inward_dominant
    )
    hard_escape_error = max(0.90, entry_error + 0.08)
    # A clipped target deserves continuous recovery sooner than the historical
    # emergency-only entry threshold.
    edge_escape = (
        edge_clipped
        and dominant_error >= max(moving_error, min(exit_error, entry_error))
        and not moving_inward_dominant
    )
    hard_escape = dominant_error >= hard_escape_error and not moving_inward_dominant

    use_continuous = bool(
        hybrid_enabled
        and not hybrid_disabled
        and cooldown_ready
        and (edge_escape or hard_escape or motion_escape or deadline_escape)
    )
    reason = None
    if use_continuous:
        if motion_escape:
            reason = "motion_escape"
        elif deadline_escape:
            reason = "deadline_motion"
        elif edge_escape:
            reason = "edge_clipped"
        else:
            reason = "hard_escape"

    adaptive_hybrid_available = bool(hybrid_enabled and not hybrid_disabled)
    precision_move_allowed = True
    precision_hold_reason = None
    if adaptive_hybrid_available:
        if not velocity_mature:
            precision_move_allowed = False
            precision_hold_reason = "velocity_sample"
        elif moving_target:
            precision_move_allowed = False
            precision_hold_reason = "moving_target"

    return {
        "use_continuous": use_continuous,
        "reason": reason,
        "velocity_mature": velocity_mature,
        "moving_target": moving_target,
        "target_speed_norm": target_speed_norm,
        "projected_travel_norm": projected_travel_norm,
        "moving_outward": moving_outward,
        "moving_inward_dominant": moving_inward_dominant,
        "motion_escape": motion_escape,
        "deadline_escape": deadline_escape,
        "precision_move_allowed": precision_move_allowed,
        "precision_hold_reason": precision_hold_reason,
    }
'''
start = text.index("def _motion_control_decision(")
end = text.index("\n\nclass DogTracker:", start)
text = text[:start] + policy.rstrip() + text[end:]

# Runtime knobs are intentionally internal defaults so existing Compose files do
# not need to change for Rev 3.
old_runtime = '''        self._move_direct_lead_max_fraction = max(\n            0.05, min(0.25, _env_float("TRACKER_MOVE_DIRECT_LEAD_MAX_FRACTION", 0.12))\n        )\n\n        self.active = False\n'''
new_runtime = '''        self._move_direct_lead_max_fraction = max(\n            0.05, min(0.25, _env_float("TRACKER_MOVE_DIRECT_LEAD_MAX_FRACTION", 0.12))\n        )\n        self._motion_control_stationary_speed_norm = max(\n            0.003, min(0.05, _env_float("TRACKER_MOTION_CONTROL_STATIONARY_SPEED_NORM", 0.012))\n        )\n        self._motion_control_moving_error = max(\n            0.18, min(0.60, _env_float("TRACKER_MOTION_CONTROL_MOVING_ERROR", 0.35))\n        )\n        self._motion_control_continuous_exit_error = max(\n            0.12, min(0.45, _env_float("TRACKER_MOTION_CONTROL_CONTINUOUS_EXIT_ERROR", 0.22))\n        )\n        self._post_chase_precision_holdoff = max(\n            0.0, min(1.5, _env_float("TRACKER_POST_CHASE_PRECISION_HOLDOFF", 0.35))\n        )\n        self._hybrid_confidence_grace = max(\n            0.0, min(1.0, _env_float("TRACKER_HYBRID_CHASE_CONFIDENCE_GRACE", 0.25))\n        )\n\n        self.active = False\n'''
text = one(text, old_runtime, new_runtime, "Rev3 runtime controls")

# Session state for confidence grace and the continuous->precision handoff.
s, e, block = scoped(text, "    def __init__(self, cfg: TrackerConfig", "    def _record_event")
block = one(
    block,
    '''        self._hybrid_last_error: Optional[float] = None\n        self._hybrid_divergence_count = 0\n\n        self.total_inferences = 0\n''',
    '''        self._hybrid_last_error: Optional[float] = None\n        self._hybrid_divergence_count = 0\n        self._hybrid_low_confidence_since: Optional[float] = None\n        self._precision_hold_until = 0.0\n        self._precision_defer_reason: Optional[str] = None\n        self._precision_defer_last_event_at = 0.0\n\n        self.total_inferences = 0\n''',
    "Rev3 session state",
)
text = text[:s] + block + text[e:]

s, e, block = scoped(text, "    def _reset_tracking_state", "    def _session_valid")
block = one(
    block,
    '''        self._hybrid_last_error = None\n        self._hybrid_divergence_count = 0\n        self._move_failure_count = 0\n''',
    '''        self._hybrid_last_error = None\n        self._hybrid_divergence_count = 0\n        self._hybrid_low_confidence_since = None\n        self._precision_hold_until = 0.0\n        self._precision_defer_reason = None\n        self._precision_defer_last_event_at = 0.0\n        self._move_failure_count = 0\n''',
    "Rev3 reset state",
)
text = text[:s] + block + text[e:]

# Frigate-like no-move behavior: never shrink below the configured precision
# deadzone, and make the no-move region larger while target motion is present.
s, e, block = scoped(text, "    def _effective_deadzone", "    def _update_frame_quality")
new_deadzone = '''    def _effective_deadzone(self, target_span: float, frame_shape: Tuple[int, ...]) -> Tuple[float, float]:\n        h, w = frame_shape[:2]\n        zoom_scale = math.sqrt(self._current_zoom_factor())\n        size_scale = max(0.90, min(1.45, math.sqrt(0.12 / max(0.03, target_span))))\n        speed_norm = 0.0\n        if self.target is not None and self.target.velocity_valid:\n            speed_norm = math.hypot(self.target.vx, self.target.vy) / max(1.0, math.hypot(w, h))\n        motion_scale = (\n            1.35 if speed_norm > self._motion_control_stationary_speed_norm else 1.0\n        )\n        # Rev 3 never makes the precision deadzone smaller than the configured\n        # base. A 1+ second positional move for ~0.11 normalized error was a\n        # measurable regression in Rev 2.\n        scale = max(1.0, min(1.75, zoom_scale * size_scale * motion_scale))\n        result = (\n            max(self.cfg.move_deadzone_x, min(0.45, self.cfg.move_deadzone_x * scale)),\n            max(self.cfg.move_deadzone_y, min(0.50, self.cfg.move_deadzone_y * scale)),\n        )\n        self._last_effective_deadzone = result\n        return result\n\n'''
text = text[:s] + new_deadzone + text[e:]

# Continuous speed mapping must be able to operate inside the old emergency
# exit threshold now that it is the primary moving-target controller.
s, e, block = scoped(text, "    def _hybrid_axis_speed", "    async def _stop_hybrid_chase")
new_axis = '''    def _hybrid_axis_speed(self, error: float, axis: str) -> int:\n        magnitude = abs(error)\n        axis_exit_error = min(\n            self.cfg.hybrid_chase_exit_error,\n            self._motion_control_continuous_exit_error,\n        )\n        if magnitude <= axis_exit_error:\n            return 0\n        zoom_factor = self._current_zoom_factor()\n        calibrated = self._active_calibration.choose_continuous_speed(\n            axis,\n            magnitude,\n            zoom_factor,\n            min_speed=self.cfg.hybrid_chase_min_speed,\n            max_speed=self.cfg.hybrid_chase_max_speed,\n            exit_error=axis_exit_error,\n            full_speed_error=self.cfg.hybrid_chase_full_speed_error,\n        )\n        if calibrated is not None:\n            return calibrated if error > 0 else -calibrated\n        span = max(0.01, self.cfg.hybrid_chase_full_speed_error - axis_exit_error)\n        ratio = max(0.0, min(1.0, (magnitude - axis_exit_error) / span))\n        zoom_max = max(\n            self.cfg.hybrid_chase_min_speed,\n            int(round(self.cfg.hybrid_chase_max_speed / math.sqrt(zoom_factor))),\n        )\n        speed = int(round(self.cfg.hybrid_chase_min_speed + ratio * (zoom_max - self.cfg.hybrid_chase_min_speed)))\n        speed = max(self.cfg.hybrid_chase_min_speed, min(zoom_max, speed))\n        return speed if error > 0 else -speed\n\n'''
text = text[:s] + new_axis + text[e:]

# A clean chase stop must settle, rebase velocity and then prove the subject is
# slow before moveDirectly gets control.
s, e, block = scoped(text, "    async def _stop_hybrid_chase", "    async def _set_hybrid_chase_speed")
block = one(
    block,
    '''        self._hybrid_last_stopped_at = t1\n        self._hybrid_last_error = None\n        self._hybrid_divergence_count = 0\n''',
    '''        self._hybrid_last_stopped_at = t1\n        self._precision_hold_until = max(\n            self._precision_hold_until, t1 + self._post_chase_precision_holdoff\n        )\n        self._hybrid_last_error = None\n        self._hybrid_divergence_count = 0\n        self._hybrid_low_confidence_since = None\n''',
    "post-chase precision holdoff",
)
text = text[:s] + block + text[e:]

# Low-confidence motion blur gets a short coast window. Do not stop the motor on
# a single weak detection and then immediately restart it on the next frame.
s, e, block = scoped(text, "    async def _drive_hybrid_chase", "    def _begin_ptz_operation")
block = one(
    block,
    '''        if max(abs(err_x), abs(err_y)) <= self.cfg.hybrid_chase_exit_error:\n            await self._stop_hybrid_chase("safe_inner_region", seq=seq)\n            return\n        if self.target is not None and self.target.confidence < self.cfg.reacquire_conf:\n            await self._stop_hybrid_chase("low_confidence", seq=seq)\n            return\n        pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)\n''',
    '''        if max(abs(err_x), abs(err_y)) <= self._motion_control_continuous_exit_error:\n            await self._stop_hybrid_chase("safe_inner_region", seq=seq)\n            return\n        if self.target is not None and self.target.confidence < self.cfg.reacquire_conf:\n            if self._hybrid_low_confidence_since is None:\n                self._hybrid_low_confidence_since = now\n                self._record_event(\n                    "hybrid_chase_confidence_grace",\n                    confidence=round(self.target.confidence, 3),\n                    grace_ms=int(self._hybrid_confidence_grace * 1000),\n                )\n            if (now - self._hybrid_low_confidence_since) <= self._hybrid_confidence_grace:\n                self.state = "ESCAPE_CHASE"\n                return\n            await self._stop_hybrid_chase("low_confidence", seq=seq)\n            return\n        self._hybrid_low_confidence_since = None\n        pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)\n''',
    "confidence grace",
)
text = text[:s] + block + text[e:]

# Feed Rev 3's moving/stationary thresholds into the policy.
s, e, block = scoped(text, "    async def _drive_to_target", "    def _render_debug_frame")
block = one(
    block,
    '''                deadline_travel_norm=self._motion_control_deadline_travel,\n            )\n''',
    '''                deadline_travel_norm=self._motion_control_deadline_travel,\n                stationary_speed_norm=self._motion_control_stationary_speed_norm,\n                moving_error=self._motion_control_moving_error,\n            )\n''',
    "Rev3 policy arguments",
)

# Demote moveDirectly to slow/stationary precision only. If motion is still being
# classified, hold; if motion is material, continuous control owns it.
needle = '''                await self._set_hybrid_chase_speed(\n                    pan_speed,\n                    tilt_speed,\n                    seq=seq,\n                    now=now,\n                    error_x=err_x,\n                    error_y=err_y,\n                    target_span=target_span,\n                )\n                return\n\n            # moveDirectly centers the point we give the camera, but a native 3D\n'''
replacement = '''                await self._set_hybrid_chase_speed(\n                    pan_speed,\n                    tilt_speed,\n                    seq=seq,\n                    now=now,\n                    error_x=err_x,\n                    error_y=err_y,\n                    target_span=target_span,\n                )\n                return\n\n            adaptive_hybrid_available = (\n                self.cfg.hybrid_chase_enabled and not self._hybrid_disabled_for_session\n            )\n            precision_hold_reason = None\n            if adaptive_hybrid_available and now < self._precision_hold_until:\n                precision_hold_reason = "post_chase_holdoff"\n            elif adaptive_hybrid_available and not bool(control_decision["precision_move_allowed"]):\n                precision_hold_reason = str(control_decision["precision_hold_reason"] or "moving_target")\n\n            if precision_hold_reason is not None:\n                next_state = (\n                    "VELOCITY_SAMPLE"\n                    if precision_hold_reason == "velocity_sample"\n                    else "PRECISION_HOLD"\n                    if precision_hold_reason == "post_chase_holdoff"\n                    else "MOTION_HOLD"\n                )\n                if (\n                    precision_hold_reason != self._precision_defer_reason\n                    or (now - self._precision_defer_last_event_at) >= 1.0\n                ):\n                    self._record_event(\n                        "precision_move_deferred",\n                        reason=precision_hold_reason,\n                        error_x=round(err_x, 3),\n                        error_y=round(err_y, 3),\n                        target_speed_norm=round(target_speed_norm, 4),\n                        velocity_mature=bool(control_decision["velocity_mature"]),\n                    )\n                    self._precision_defer_reason = precision_hold_reason\n                    self._precision_defer_last_event_at = now\n                self.state = next_state\n                return\n\n            self._precision_defer_reason = None\n\n            # moveDirectly centers the CURRENT measured point. In adaptive hybrid\n'''
block = one(block, needle, replacement, "precision motion gate")

# No future-position extrapolation for moveDirectly when adaptive hybrid is
# available. The continuous controller is the predictor/follower; moveDirectly
# is only the final stationary precision correction.
block = one(
    block,
    '''            lead_horizon_s = min(predicted_move_eta_s, self._move_direct_lead_horizon_max)\n\n            lead_suppressed_reason: Optional[str] = None\n            if not self.target.velocity_valid:\n                lead_suppressed_reason = "velocity_sample"\n            elif self.target.confidence < self.cfg.lead_min_conf:\n                lead_suppressed_reason = "low_confidence"\n            elif target_span < self.cfg.lead_min_span:\n                lead_suppressed_reason = "small_target"\n            elif edge_clipped:\n                lead_suppressed_reason = "edge_clipped"\n            else:\n                velocity_ok, velocity_reason = self._validate_velocity_for_lead(frame_shape)\n                if not velocity_ok:\n                    lead_suppressed_reason = velocity_reason\n''',
    '''            lead_horizon_s = (\n                0.0\n                if adaptive_hybrid_available\n                else min(predicted_move_eta_s, self._move_direct_lead_horizon_max)\n            )\n\n            lead_suppressed_reason: Optional[str] = (\n                "adaptive_hybrid_current_center" if adaptive_hybrid_available else None\n            )\n            if lead_suppressed_reason is None:\n                if not self.target.velocity_valid:\n                    lead_suppressed_reason = "velocity_sample"\n                elif self.target.confidence < self.cfg.lead_min_conf:\n                    lead_suppressed_reason = "low_confidence"\n                elif target_span < self.cfg.lead_min_span:\n                    lead_suppressed_reason = "small_target"\n                elif edge_clipped:\n                    lead_suppressed_reason = "edge_clipped"\n                else:\n                    velocity_ok, velocity_reason = self._validate_velocity_for_lead(frame_shape)\n                    if not velocity_ok:\n                        lead_suppressed_reason = velocity_reason\n''',
    "disable positional prediction in adaptive hybrid",
)
text = text[:s] + block + text[e:]

# Status must make it unambiguous which controller is deployed and why a
# precision move is being held.
text = one(
    text,
    '''                    "controller_revision": 2,\n                    "min_velocity_sample_ms": self._motion_control_min_sample_ms,\n                    "deadline_travel_norm": round(self._motion_control_deadline_travel, 3),\n                    "move_direct_lead_horizon_max_s": round(self._move_direct_lead_horizon_max, 3),\n                    "move_direct_lead_max_fraction": round(self._move_direct_lead_max_fraction, 3),\n                    "live_spatial_learning": False,\n''',
    '''                    "controller_revision": 3,\n                    "strategy": "continuous_moving_precision_stationary",\n                    "min_velocity_sample_ms": self._motion_control_min_sample_ms,\n                    "deadline_travel_norm": round(self._motion_control_deadline_travel, 3),\n                    "stationary_speed_norm": round(self._motion_control_stationary_speed_norm, 4),\n                    "moving_entry_error": round(self._motion_control_moving_error, 3),\n                    "continuous_exit_error": round(self._motion_control_continuous_exit_error, 3),\n                    "post_chase_precision_holdoff_s": round(self._post_chase_precision_holdoff, 3),\n                    "confidence_grace_s": round(self._hybrid_confidence_grace, 3),\n                    "precision_hold_until_ms": max(0, int((self._precision_hold_until - now) * 1000)),\n                    "move_direct_predictive_lead": False,\n                    "live_spatial_learning": False,\n''',
    "Rev3 status",
)

text = one(
    text,
    '        self.logger.info("PTZ tracker STARTED (hybrid moveDirectly + escape chase)")\n',
    '        self.logger.info("PTZ tracker STARTED (Rev3 continuous-moving + stationary-precision control)")\n',
    "Rev3 startup log",
)

tracker_path.write_text(text)

# ---------------------------------------------------------------------------
# Behavioral/contract tests for Rev 3. This file deliberately loads only the
# standalone policy function with AST so it does not need camera/network deps.
# ---------------------------------------------------------------------------
test_path = Path("tests/test_motion_control_policy.py")
test_path.write_text(r'''import ast
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

    def test_rev3_precision_contracts_are_present(self):
        self.assertIn('"controller_revision": 3', TRACKER)
        self.assertIn('"continuous_moving_precision_stationary"', TRACKER)
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
''')

print("Rev 3 controller patch applied")
