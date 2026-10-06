from pathlib import Path


def one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


tracker_path = Path("tracker.py")
phase2_path = Path("ptz_phase2.py")
policy_test_path = Path("tests/test_motion_control_policy.py")
rev4_test_path = Path("tests/test_rev4_controller.py")

tracker = tracker_path.read_text()
phase2 = phase2_path.read_text()
policy_test = policy_test_path.read_text()

# ---------------------------------------------------------------------------
# 1. Velocity estimates are persistent state, not a 250 ms on/off square wave.
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """    velocity_sample_ms: int = 0
    velocity_valid: bool = False
""",
    """    velocity_sample_ms: int = 0
    velocity_valid: bool = False
    velocity_estimate_time: Optional[float] = None
""",
    "TargetTrack velocity timestamp",
)

tracker = one(
    tracker,
    """    def rebase_velocity(self, now: float) -> None:
        self.velocity_reference_center = self.center
        self.velocity_reference_time = now
        self.velocity_sample_ms = 0
        self.velocity_valid = False
        self.vx = 0.0
        self.vy = 0.0

    def clear_velocity(self) -> None:
        self.velocity_reference_center = None
        self.velocity_reference_time = None
        self.velocity_sample_ms = 0
        self.velocity_valid = False
        self.vx = 0.0
        self.vy = 0.0
""",
    """    def rebase_velocity(self, now: float) -> None:
        self.velocity_reference_center = self.center
        self.velocity_reference_time = now
        self.velocity_sample_ms = 0
        self.velocity_valid = False
        self.velocity_estimate_time = None
        self.vx = 0.0
        self.vy = 0.0

    def clear_velocity(self) -> None:
        self.velocity_reference_center = None
        self.velocity_reference_time = None
        self.velocity_sample_ms = 0
        self.velocity_valid = False
        self.velocity_estimate_time = None
        self.vx = 0.0
        self.vy = 0.0
""",
    "TargetTrack velocity reset",
)

tracker = one(
    tracker,
    """        update_velocity: bool = True,
        min_velocity_sample_s: float = 0.10,
    ) -> None:
""",
    """        update_velocity: bool = True,
        min_velocity_sample_s: float = 0.10,
        preserve_velocity: bool = False,
    ) -> None:
""",
    "TargetTrack preserve velocity argument",
)

tracker = one(
    tracker,
    """            sample_s = max(0.0, now - self.velocity_reference_time)
            self.velocity_sample_ms = int(sample_s * 1000)
            if sample_s >= max(0.001, min_velocity_sample_s):
""",
    """            sample_s = max(0.0, now - self.velocity_reference_time)
            # Keep the last mature estimate usable while the next sample window
            # accumulates. Rev 3 accidentally rewrote velocity_sample_ms to
            # 50-70 ms on every intervening frame, making motion state alternate
            # between mature/immature at camera FPS.
            if sample_s >= max(0.001, min_velocity_sample_s):
""",
    "velocity maturity persistence",
)

tracker = one(
    tracker,
    """                self.velocity_valid = True
                self.velocity_reference_center = new_center
""",
    """                self.velocity_valid = True
                self.velocity_estimate_time = now
                self.velocity_reference_center = new_center
""",
    "velocity estimate timestamp update",
)

tracker = one(
    tracker,
    """        else:
            # Follow the bbox for association, but never learn global
            # image motion as target velocity while the PTZ is moving.
            self.clear_velocity()
""",
    """        elif not preserve_velocity:
            # PTZ motion invalidates image-space target velocity. A rejected
            # detector-geometry sample while the camera is stationary is
            # different: the last trusted estimate may be preserved briefly.
            self.clear_velocity()
""",
    "velocity preserve behavior",
)

tracker = one(
    tracker,
    """    def predicted_center(self, now: float) -> Tuple[float, float]:
        dt = min(0.6, max(0.0, now - self.last_update))
        return (self.center[0] + self.vx * dt, self.center[1] + self.vy * dt)
""",
    """    def predicted_center(self, now: float) -> Tuple[float, float]:
        if (
            not self.velocity_valid
            or self.velocity_estimate_time is None
            or (now - self.velocity_estimate_time) > 0.75
        ):
            return self.center
        dt = min(0.6, max(0.0, now - self.last_update))
        return (self.center[0] + self.vx * dt, self.center[1] + self.vy * dt)
""",
    "bounded predicted center",
)

# ---------------------------------------------------------------------------
# 2. Rev 4 control/retention knobs. Defaults are deliberately conservative.
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """        self._hybrid_confidence_grace = max(
            0.0, min(1.0, _env_float("TRACKER_HYBRID_CHASE_CONFIDENCE_GRACE", 0.25))
        )
""",
    """        self._hybrid_confidence_grace = max(
            0.0, min(1.0, _env_float("TRACKER_HYBRID_CHASE_CONFIDENCE_GRACE", 0.40))
        )
        self._retention_detection_conf = max(
            0.05,
            min(self.cfg.hold_conf, _env_float("TRACKER_RETENTION_DETECTION_CONF", 0.20)),
        )
        self._chase_detection_conf = max(
            0.05,
            min(self._retention_detection_conf, _env_float("TRACKER_CHASE_DETECTION_CONF", 0.18)),
        )
        self._target_retention_grace = max(
            0.20, min(1.50, _env_float("TRACKER_TARGET_RETENTION_GRACE", 0.65))
        )
        self._hybrid_missing_grace = max(
            self.cfg.hybrid_chase_miss_grace,
            max(0.20, min(0.80, _env_float("TRACKER_HYBRID_MISSING_GRACE", 0.35))),
        )
        self._motion_velocity_ttl = max(
            0.30, min(1.50, _env_float("TRACKER_MOTION_VELOCITY_TTL", 0.75))
        )
        self._precision_min_error = max(
            max(self.cfg.move_deadzone_x, self.cfg.move_deadzone_y),
            min(0.55, _env_float("TRACKER_PRECISION_MIN_ERROR", 0.28)),
        )
        self._precision_settle_s = max(
            0.20, min(1.50, _env_float("TRACKER_PRECISION_SETTLE_TIME", 0.45))
        )
        self._hybrid_axis_reverse_holdoff = max(
            0.10, min(0.75, _env_float("TRACKER_HYBRID_AXIS_REVERSE_HOLDOFF", 0.25))
        )
""",
    "Rev4 control settings",
)

tracker = one(
    tracker,
    """        self._precision_hold_until = 0.0
        self._precision_defer_reason: Optional[str] = None
        self._precision_defer_last_event_at = 0.0
""",
    """        self._precision_hold_until = 0.0
        self._precision_slow_since: Optional[float] = None
        self._precision_defer_reason: Optional[str] = None
        self._precision_defer_last_event_at = 0.0
        self._hybrid_pan_reverse_until = 0.0
        self._hybrid_tilt_reverse_until = 0.0
""",
    "Rev4 runtime state",
)

tracker = one(
    tracker,
    """        self._precision_hold_until = 0.0
        self._precision_defer_reason = None
        self._precision_defer_last_event_at = 0.0
""",
    """        self._precision_hold_until = 0.0
        self._precision_slow_since = None
        self._precision_defer_reason = None
        self._precision_defer_last_event_at = 0.0
        self._hybrid_pan_reverse_until = 0.0
        self._hybrid_tilt_reverse_until = 0.0
""",
    "Rev4 reset state",
)

# ---------------------------------------------------------------------------
# 3. Use lower-confidence detections only to retain an existing target.
#    Acquisition remains protected by acquire_conf.
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """                outcome, raw_detections, infer_ms = await self.inference_cb(
                    frame,
                    self.cfg.hold_conf,
                    self.cfg.model_name,
                    self.cfg.target_class_ids,
                )
""",
    """                inference_conf = self.cfg.hold_conf
                if self.target is not None:
                    inference_conf = min(inference_conf, self._retention_detection_conf)
                if self._hybrid_chase_active:
                    inference_conf = min(inference_conf, self._chase_detection_conf)
                outcome, raw_detections, infer_ms = await self.inference_cb(
                    frame,
                    inference_conf,
                    self.cfg.model_name,
                    self.cfg.target_class_ids,
                )
""",
    "dynamic tracking inference threshold",
)

tracker = one(
    tracker,
    """                    for d in raw_detections
                    if float(d.get("confidence", 0.0)) >= self.cfg.hold_conf
                ]
""",
    """                    for d in raw_detections
                    if float(d.get("confidence", 0.0)) >= inference_conf
                ]
""",
    "dynamic detection filter",
)

tracker = one(
    tracker,
    """        if (
            matched is not None
            and self.state in ("COAST", "REACQUIRE", "LOST")
            and matched.confidence < self.cfg.reacquire_conf
        ):
            matched = None
""",
    """        if matched is not None and self.state in ("COAST", "REACQUIRE", "LOST"):
            missing_for_match = max(0.0, now - self.target.last_seen)
            required_conf = (
                self._retention_detection_conf
                if missing_for_match <= self._target_retention_grace
                else self.cfg.reacquire_conf
            )
            if matched.confidence < required_conf:
                matched = None
""",
    "retention-aware reacquire confidence",
)

# ---------------------------------------------------------------------------
# 4. Preserve the last trusted velocity through detector-box deformation.
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """            velocity_learning_allowed = (
                self._ptz_operation is None
                and not self._hybrid_chase_active
                and ptz_ready
                and not self._velocity_rebase_required
                and self._frame_sharpness_ok
                and geometry_valid
            )
            self.target.update(
                matched,
                now,
                update_velocity=velocity_learning_allowed,
                min_velocity_sample_s=max(
                    self.cfg.velocity_min_sample_ms,
                    self._motion_control_min_sample_ms,
                ) / 1000.0,
            )
""",
    """            stationary_observation = (
                self._ptz_operation is None
                and not self._hybrid_chase_active
                and ptz_ready
                and not self._velocity_rebase_required
                and self._frame_sharpness_ok
            )
            velocity_learning_allowed = stationary_observation and geometry_valid
            velocity_age_s = (
                float("inf")
                if self.target.velocity_estimate_time is None
                else max(0.0, now - self.target.velocity_estimate_time)
            )
            preserve_velocity = bool(
                stationary_observation
                and not geometry_valid
                and self.target.velocity_valid
                and velocity_age_s <= self._motion_velocity_ttl
            )
            self.target.update(
                matched,
                now,
                update_velocity=velocity_learning_allowed,
                min_velocity_sample_s=max(
                    self.cfg.velocity_min_sample_ms,
                    self._motion_control_min_sample_ms,
                ) / 1000.0,
                preserve_velocity=preserve_velocity,
            )
""",
    "persistent velocity through bbox rejects",
)

# ---------------------------------------------------------------------------
# 5. Motion decisions only consume fresh velocity; precision moves require a
#    real stationary dwell and a meaningful error.
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """            handoff_eta_s = self._predict_move_eta(rough_move_distance)
            control_decision = _motion_control_decision(
                err_x=err_x,
                err_y=err_y,
                vx=self.target.vx,
                vy=self.target.vy,
                velocity_valid=self.target.velocity_valid,
                velocity_sample_ms=self.target.velocity_sample_ms,
""",
    """            handoff_eta_s = self._predict_move_eta(rough_move_distance)
            velocity_age_s = (
                float("inf")
                if self.target.velocity_estimate_time is None
                else max(0.0, now - self.target.velocity_estimate_time)
            )
            control_velocity_valid = bool(
                self.target.velocity_valid and velocity_age_s <= self._motion_velocity_ttl
            )
            control_vx = self.target.vx if control_velocity_valid else 0.0
            control_vy = self.target.vy if control_velocity_valid else 0.0
            control_decision = _motion_control_decision(
                err_x=err_x,
                err_y=err_y,
                vx=control_vx,
                vy=control_vy,
                velocity_valid=control_velocity_valid,
                velocity_sample_ms=self.target.velocity_sample_ms,
""",
    "fresh velocity control input",
)

tracker = one(
    tracker,
    """            if hybrid_entry:
                pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)
""",
    """            if hybrid_entry:
                self._precision_slow_since = None
                pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)
""",
    "reset stationary dwell on chase",
)

tracker = one(
    tracker,
    """            adaptive_hybrid_available = (
                self.cfg.hybrid_chase_enabled and not self._hybrid_disabled_for_session
            )
            precision_hold_reason = None
            if adaptive_hybrid_available and now < self._precision_hold_until:
                precision_hold_reason = "post_chase_holdoff"
            elif adaptive_hybrid_available and not bool(control_decision["precision_move_allowed"]):
                precision_hold_reason = str(control_decision["precision_hold_reason"] or "moving_target")

            if precision_hold_reason is not None:
                next_state = (
                    "VELOCITY_SAMPLE"
                    if precision_hold_reason == "velocity_sample"
                    else "PRECISION_HOLD"
                    if precision_hold_reason == "post_chase_holdoff"
                    else "MOTION_HOLD"
                )
""",
    """            adaptive_hybrid_available = (
                self.cfg.hybrid_chase_enabled and not self._hybrid_disabled_for_session
            )
            dominant_error = max(abs(err_x), abs(err_y))
            precision_hold_reason = None
            if adaptive_hybrid_available and now < self._precision_hold_until:
                self._precision_slow_since = None
                precision_hold_reason = "post_chase_holdoff"
            elif adaptive_hybrid_available and not bool(control_decision["precision_move_allowed"]):
                self._precision_slow_since = None
                precision_hold_reason = str(control_decision["precision_hold_reason"] or "moving_target")
            elif adaptive_hybrid_available and dominant_error < self._precision_min_error:
                self._precision_slow_since = None
                precision_hold_reason = "precision_deadband"
            elif adaptive_hybrid_available:
                if self._precision_slow_since is None:
                    self._precision_slow_since = now
                if (now - self._precision_slow_since) < self._precision_settle_s:
                    precision_hold_reason = "stationary_settle"

            if precision_hold_reason is not None:
                next_state = (
                    "VELOCITY_SAMPLE"
                    if precision_hold_reason == "velocity_sample"
                    else "PRECISION_HOLD"
                    if precision_hold_reason in (
                        "post_chase_holdoff",
                        "precision_deadband",
                        "stationary_settle",
                    )
                    else "MOTION_HOLD"
                )
""",
    "precision dwell and minimum error",
)

tracker = one(
    tracker,
    """            self._precision_defer_reason = None

            # moveDirectly centers the CURRENT measured point. In adaptive hybrid
""",
    """            self._precision_defer_reason = None

            # A precision move consumes the stationary dwell. A second positional
            # correction must earn a new stable observation window after settling.
            self._precision_slow_since = None

            # moveDirectly centers the CURRENT measured point. In adaptive hybrid
""",
    "consume precision dwell",
)

# ---------------------------------------------------------------------------
# 6. Frigate-style continuity: tolerate short detector dropouts during motion.
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """        if self._hybrid_chase_active:
            # Keep the bounded chase alive across one or two blurred YOLO misses.
            hybrid_missing_for = max(0.0, now - self.target.last_seen)
            if hybrid_missing_for <= self.cfg.hybrid_chase_miss_grace:
                self.state = "ESCAPE_CHASE"
                return
            await self._stop_hybrid_chase("target_missing", seq=seq)
""",
    """        if self._hybrid_chase_active:
            # Keep continuous steering alive through a short burst of motion blur.
            # The lower retention threshold is association-only; no new target can
            # be acquired at this confidence. The camera still stops at a bounded
            # grace interval if the detector does not recover.
            hybrid_missing_for = max(0.0, now - self.target.last_seen)
            if hybrid_missing_for <= self._hybrid_missing_grace:
                self.state = "ESCAPE_CHASE"
                return
            await self._stop_hybrid_chase("target_missing", seq=seq)
""",
    "bounded hybrid missing retention",
)

# ---------------------------------------------------------------------------
# 7. Per-axis reversal suppression: do not throw away useful pan tracking
#    because tilt crossed center (or vice versa).
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """        desired = (pan_speed, tilt_speed)
        current = (self._hybrid_pan_speed, self._hybrid_tilt_speed)

        # Escape chase never reverses through center. A sign flip means we have
        # recovered enough (or inertia carried us through); stop and let the
        # slower camera-managed moveDirectly controller take over after settling.
        def sign_flip(old: int, new: int) -> bool:
            return old != 0 and new != 0 and ((old > 0) != (new > 0))
        if self._hybrid_chase_active and (sign_flip(current[0], desired[0]) or sign_flip(current[1], desired[1])):
            return await self._stop_hybrid_chase("direction_reversal", seq=seq)

        elapsed = max(0.0, now - self._hybrid_last_command_at)
""",
    """        raw_desired = (pan_speed, tilt_speed)
        current = (self._hybrid_pan_speed, self._hybrid_tilt_speed)

        # Rev 4 handles center crossings independently per axis. Motor inertia on
        # one axis must not cancel useful tracking on the other. A reversing axis
        # is neutralized briefly, then may reverse only if the error still demands
        # it after the holdoff.
        def sign_flip(old: int, new: int) -> bool:
            return old != 0 and new != 0 and ((old > 0) != (new > 0))

        desired_pan, desired_tilt = raw_desired
        suppressed_axes = []
        if self._hybrid_chase_active:
            if sign_flip(current[0], desired_pan):
                self._hybrid_pan_reverse_until = max(
                    self._hybrid_pan_reverse_until,
                    now + self._hybrid_axis_reverse_holdoff,
                )
                desired_pan = 0
                suppressed_axes.append("pan")
            elif now < self._hybrid_pan_reverse_until:
                desired_pan = 0
                suppressed_axes.append("pan")

            if sign_flip(current[1], desired_tilt):
                self._hybrid_tilt_reverse_until = max(
                    self._hybrid_tilt_reverse_until,
                    now + self._hybrid_axis_reverse_holdoff,
                )
                desired_tilt = 0
                suppressed_axes.append("tilt")
            elif now < self._hybrid_tilt_reverse_until:
                desired_tilt = 0
                suppressed_axes.append("tilt")

        desired = (desired_pan, desired_tilt)
        if suppressed_axes:
            self._hybrid_divergence_count = 0
            self._record_event(
                "hybrid_axis_reversal_suppressed",
                axes=sorted(set(suppressed_axes)),
                requested=[raw_desired[0], raw_desired[1]],
                commanded=[desired[0], desired[1]],
                holdoff_ms=int(self._hybrid_axis_reverse_holdoff * 1000),
            )

        if desired == (0, 0):
            if raw_desired == (0, 0):
                return await self._stop_hybrid_chase("safe_inner_region", seq=seq)
            # Both axes are in reversal holdoff. Stop motor motion without ending
            # the chase session or rebasing the target; a subsequent fresh frame
            # can restart either axis in the new direction.
            if current != (0, 0):
                t0 = time.monotonic()
                ok = await asyncio.to_thread(self.ptz.continuous_stop)
                t1 = time.monotonic()
                if not ok:
                    return await self._stop_hybrid_chase("axis_pause_failed", seq=seq, force=True)
                self._hybrid_pan_speed = 0
                self._hybrid_tilt_speed = 0
                self._hybrid_last_command_at = t1
                self.ptz_commands += 1
                self._record_event(
                    "hybrid_axis_pause",
                    axes=sorted(set(suppressed_axes)),
                    http_ms=int((t1 - t0) * 1000),
                )
            self.state = "ESCAPE_CHASE"
            return True

        elapsed = max(0.0, now - self._hybrid_last_command_at)
""",
    "per-axis chase reversal",
)

tracker = one(
    tracker,
    """        self._hybrid_low_confidence_since = None
        if ok and was_active:
""",
    """        self._hybrid_low_confidence_since = None
        self._hybrid_pan_reverse_until = 0.0
        self._hybrid_tilt_reverse_until = 0.0
        self._precision_slow_since = None
        if ok and was_active:
""",
    "reset axis reversal state on stop",
)

# ---------------------------------------------------------------------------
# 8. Status/telemetry and revision markers.
# ---------------------------------------------------------------------------
tracker = one(
    tracker,
    """                "velocity_sample_ms": self.target.velocity_sample_ms,
                "velocity_valid": self.target.velocity_valid,
""",
    """                "velocity_sample_ms": self.target.velocity_sample_ms,
                "velocity_valid": self.target.velocity_valid,
                "velocity_age_ms": (
                    None
                    if self.target.velocity_estimate_time is None
                    else int(max(0.0, now - self.target.velocity_estimate_time) * 1000)
                ),
""",
    "velocity age status",
)

tracker = one(
    tracker,
    """                    "controller_revision": 3,
                    "strategy": "continuous_moving_precision_stationary",
""",
    """                    "controller_revision": 4,
                    "strategy": "persistent_velocity_continuous_servo_precision",
""",
    "Rev4 status marker",
)

tracker = one(
    tracker,
    """                    "confidence_grace_s": round(self._hybrid_confidence_grace, 3),
                    "precision_hold_until_ms": max(0, int((self._precision_hold_until - now) * 1000)),
                    "move_direct_predictive_lead": False,
""",
    """                    "confidence_grace_s": round(self._hybrid_confidence_grace, 3),
                    "missing_grace_s": round(self._hybrid_missing_grace, 3),
                    "retention_detection_conf": round(self._retention_detection_conf, 3),
                    "chase_detection_conf": round(self._chase_detection_conf, 3),
                    "target_retention_grace_s": round(self._target_retention_grace, 3),
                    "velocity_ttl_s": round(self._motion_velocity_ttl, 3),
                    "precision_min_error": round(self._precision_min_error, 3),
                    "precision_settle_s": round(self._precision_settle_s, 3),
                    "axis_reverse_holdoff_s": round(self._hybrid_axis_reverse_holdoff, 3),
                    "precision_hold_until_ms": max(0, int((self._precision_hold_until - now) * 1000)),
                    "move_direct_predictive_lead": False,
""",
    "Rev4 status details",
)

tracker = tracker.replace(
    "PTZ tracker STARTED (Rev3 continuous-moving + stationary-precision control)",
    "PTZ tracker STARTED (Rev4 persistent-motion + continuous-servo control)",
)
tracker = tracker.replace(
    "Rev 3 treats moveDirectly as a precision controller",
    "Rev 4 treats moveDirectly as a precision controller",
)
tracker = tracker.replace(
    "# Rev 3 never makes the precision deadzone smaller",
    "# Rev 4 never makes the precision deadzone smaller",
)

# ---------------------------------------------------------------------------
# 9. Geometry validator: reject deformation only when it dominates translation.
#    Walking toward/away from camera naturally changes bbox size quickly.
# ---------------------------------------------------------------------------
phase2 = one(
    phase2,
    """        elif abs(width_rate) > 2.5 or abs(height_rate) > 2.5:
            reason = "bbox_scale_jump"
        elif speed_a > 40.0 and speed_b > 40.0:
""",
    """        elif abs(width_rate) > 2.5 or abs(height_rate) > 2.5:
            # Rapid scale change by itself is not bad velocity evidence. A person
            # walking toward/away from the camera legitimately changes bbox size
            # several times per second. Reject only when edge deformation clearly
            # dominates coherent translation (Frigate-style edge consensus).
            center_vx = 0.5 * (vx1 + vx2)
            center_vy = 0.5 * (vy1 + vy2)
            deform_x = 0.5 * abs(vx2 - vx1)
            deform_y = 0.5 * abs(vy2 - vy1)
            width_bad = (
                abs(width_rate) > 2.5
                and deform_x > max(120.0, abs(center_vx) * 1.75)
            )
            height_bad = (
                abs(height_rate) > 2.5
                and deform_y > max(120.0, abs(center_vy) * 1.75)
            )
            extreme_scale = (
                (abs(width_rate) > 6.0 and deform_x > 80.0)
                or (abs(height_rate) > 6.0 and deform_y > 80.0)
            )
            if width_bad or height_bad or extreme_scale:
                reason = "bbox_scale_jump"
        elif speed_a > 40.0 and speed_b > 40.0:
""",
    "coherent bbox scale handling",
)

# Existing policy test should follow the active controller revision.
policy_test = one(
    policy_test,
    """    def test_rev3_precision_contracts_are_present(self):
        self.assertIn('"controller_revision": 3', TRACKER)
        self.assertIn('"continuous_moving_precision_stationary"', TRACKER)
""",
    """    def test_rev4_precision_contracts_are_present(self):
        self.assertIn('"controller_revision": 4', TRACKER)
        self.assertIn('"persistent_velocity_continuous_servo_precision"', TRACKER)
""",
    "motion policy revision test",
)

# Dedicated Rev 4 regression contracts.
rev4_test = r'''import ast
from pathlib import Path
import unittest

from ptz_phase2 import BBoxMotionValidator


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


class Rev4ControllerTests(unittest.TestCase):
    def test_velocity_maturity_is_not_rewritten_on_every_subwindow_frame(self):
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
        self.assertIn("velocity_estimate_time", source)
        self.assertIn("preserve_velocity", source)
        # There should be one assignment after a mature sample, not the Rev 3
        # pre-threshold assignment that toggled maturity every frame.
        self.assertEqual(source.count("self.velocity_sample_ms = int(sample_s * 1000)"), 1)

    def test_rev4_has_stationary_dwell_and_meaningful_precision_threshold(self):
        self.assertIn("TRACKER_PRECISION_MIN_ERROR", TRACKER)
        self.assertIn("TRACKER_PRECISION_SETTLE_TIME", TRACKER)
        self.assertIn('"precision_min_error"', TRACKER)
        self.assertIn('"precision_settle_s"', TRACKER)
        self.assertIn('"stationary_settle"', TRACKER)
        self.assertIn('"precision_deadband"', TRACKER)

    def test_rev4_uses_low_confidence_only_for_target_retention(self):
        self.assertIn("TRACKER_RETENTION_DETECTION_CONF", TRACKER)
        self.assertIn("TRACKER_CHASE_DETECTION_CONF", TRACKER)
        self.assertIn("inference_conf = self.cfg.hold_conf", TRACKER)
        self.assertIn("if self.target is not None:", TRACKER)
        # New acquisition remains independently gated at acquire_conf.
        self.assertIn("d.confidence >= self.cfg.acquire_conf", TRACKER)

    def test_axis_reversal_no_longer_ends_whole_chase(self):
        start = TRACKER.index("    async def _set_hybrid_chase_speed")
        end = TRACKER.index("    async def _drive_hybrid_chase", start)
        block = TRACKER[start:end]
        self.assertIn("hybrid_axis_reversal_suppressed", block)
        self.assertIn("hybrid_axis_pause", block)
        self.assertNotIn('_stop_hybrid_chase("direction_reversal"', block)

    def test_bbox_scale_change_with_coherent_translation_is_accepted(self):
        validator = BBoxMotionValidator()
        shape = (480, 640, 3)
        validator.reset((100, 100, 200, 300), 1.0)
        # Width grows 30% in 100 ms, but both x edges translate strongly in
        # the same direction. This is a plausible walking/perspective sample.
        result = validator.validate((120, 100, 250, 300), 1.1, shape)
        self.assertTrue(result.valid, result.reason)

    def test_extreme_bbox_deformation_is_still_rejected(self):
        validator = BBoxMotionValidator()
        shape = (480, 640, 3)
        validator.reset((200, 150, 300, 250), 1.0)
        result = validator.validate((20, 150, 610, 250), 1.1, shape)
        self.assertFalse(result.valid)

    def test_rev4_status_contract(self):
        self.assertIn('"controller_revision": 4', TRACKER)
        self.assertIn('"persistent_velocity_continuous_servo_precision"', TRACKER)
        self.assertIn('"missing_grace_s"', TRACKER)
        self.assertIn('"velocity_ttl_s"', TRACKER)
        self.assertIn('"axis_reverse_holdoff_s"', TRACKER)


if __name__ == "__main__":
    unittest.main()
'''
rev4_test_path.write_text(rev4_test)

# Hygiene guards: Rev 4 must not regress the already-fixed crash or live calibration.
if "_hybrid_axis_speed(err_x)" in tracker or "_hybrid_axis_speed(err_y)" in tracker:
    raise RuntimeError("hybrid chase arity regression: one-argument call remains")
quality_start = tracker.index("    def _complete_move_quality")
quality_end = tracker.index("    def _hybrid_axis_speed", quality_start)
if "set_spatial_scales" in tracker[quality_start:quality_end]:
    raise RuntimeError("live target spatial calibration mutation reintroduced")
if '"controller_revision": 4' not in tracker:
    raise RuntimeError("Rev 4 status marker missing")

tracker_path.write_text(tracker)
phase2_path.write_text(phase2)
policy_test_path.write_text(policy_test)
print("Rev 4 controller patch applied")
