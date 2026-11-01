from __future__ import annotations

import asyncio
import math
import time
from typing import Dict, List, Optional, Tuple

from tracker_models import Detection
from tracker_policy import (
    _native_edge_rescue_axis_requested,
    _native_edge_rescue_speed,
    _servo_axis_decision,
    _servo_axis_divergence_count,
    _servo_coast_scale,
    _servo_dynamic_brake_horizon,
    _servo_hold_action,
    _servo_large_error_gain,
    _servo_outward_catchup_gain,
)


class TrackerServoMixin:
    """Closed-loop ONVIF fractional servo, native rescue/handoff, and lockout routines."""

    def _reset_servo_feedback(self) -> None:
        self._servo_axis_state = {
            "pan": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
            "tilt": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
        }
        self._servo_feedforward = {"pan": 0.0, "tilt": 0.0}
        self._servo_feedforward_started_at = 0.0
        self._servo_last_telemetry_at = 0.0
        self._servo_last_decision = {}

    def _reset_loss_clock(self) -> None:
        self._loss_clock_pause_started_at = None
        self._loss_clock_paused_s = 0.0

    def _pause_loss_clock(self, now: float) -> None:
        if self._loss_clock_pause_started_at is None:
            self._loss_clock_pause_started_at = float(now)

    def _resume_loss_clock(self, now: float) -> None:
        if self._loss_clock_pause_started_at is None:
            return
        self._loss_clock_paused_s += max(
            0.0, float(now) - self._loss_clock_pause_started_at
        )
        self._loss_clock_pause_started_at = None

    def _target_missing_seconds(self, now: float) -> float:
        if self.target is None:
            return 0.0
        paused_s = self._loss_clock_paused_s
        if self._loss_clock_pause_started_at is not None:
            paused_s += max(
                0.0, float(now) - self._loss_clock_pause_started_at
            )
        return max(0.0, float(now) - self.target.last_seen - paused_s)

    def _clear_native_edge_rescue(
        self,
        now: Optional[float] = None,
        *,
        cooldown: bool = False,
    ) -> None:
        self._native_edge_rescue_until = 0.0
        self._native_edge_rescue_started_at = 0.0
        self._native_edge_rescue_axes.clear()
        self._native_edge_rescue_start_errors = {"pan": 0.0, "tilt": 0.0}
        self._native_edge_rescue_last_progress_at = 0.0
        if cooldown and now is not None:
            self._native_edge_rescue_cooldown_until = max(
                self._native_edge_rescue_cooldown_until,
                float(now) + self._native_edge_rescue_cooldown,
            )

    def _seed_servo_feedback(
        self,
        frame_shape: Tuple[int, ...],
        now: float,
        err_x: float,
        err_y: float,
    ) -> None:
        h, w = frame_shape[:2]
        self._reset_servo_feedback()
        self._servo_axis_state["pan"]["last_error"] = float(err_x)
        self._servo_axis_state["pan"]["last_time"] = float(now)
        self._servo_axis_state["tilt"]["last_error"] = float(err_y)
        self._servo_axis_state["tilt"]["last_time"] = float(now)
        if (
            self.target is not None
            and self.target.velocity_valid
            and self.target.velocity_estimate_time is not None
            and (now - self.target.velocity_estimate_time) <= self._motion_velocity_ttl
        ):
            self._servo_feedforward["pan"] = self.target.vx / max(1.0, w / 2.0)
            self._servo_feedforward["tilt"] = self.target.vy / max(1.0, h / 2.0)
        self._servo_feedforward_started_at = float(now)

    def _servo_axis_command(
        self,
        error: float,
        axis: str,
        now: float,
    ) -> Tuple[int, dict]:
        state = self._servo_axis_state[axis]
        previous_error = state.get("last_error")
        previous_time = float(state.get("last_time") or 0.0)
        filtered_rate = float(state.get("filtered_rate") or 0.0)
        dt = max(0.0, float(now) - previous_time)
        if previous_error is None or dt < 0.02 or dt > 0.75:
            raw_error_rate = 0.0
        else:
            raw_error_rate = (float(error) - float(previous_error)) / dt
        alpha = self._servo_derivative_alpha
        filtered_rate = (1.0 - alpha) * filtered_rate + alpha * raw_error_rate
        state["last_error"] = float(error)
        state["last_time"] = float(now)
        state["filtered_rate"] = filtered_rate

        ff = float(self._servo_feedforward.get(axis, 0.0))
        if self._servo_feedforward_started_at > 0.0:
            age = max(0.0, now - self._servo_feedforward_started_at)
            ff *= math.exp(-age / max(0.05, self._servo_feedforward_decay_s))

        axis_exit_error = min(
            self.cfg.hybrid_chase_exit_error,
            self._motion_control_continuous_exit_error,
        )
        error_gain = _servo_large_error_gain(
            error,
            start_error=self._servo_large_error_start,
            full_error=self._servo_large_error_full,
            max_gain=self._servo_large_error_gain,
        )
        effective_kp = self._servo_kp * error_gain
        current_fractional_velocity = 0.0
        if self._hybrid_actuator == "onvif_fractional":
            current_fractional_velocity = (
                self._hybrid_pan_velocity if axis == "pan" else self._hybrid_tilt_velocity
            )
        effective_brake_horizon = _servo_dynamic_brake_horizon(
            current_fractional_velocity,
            max_velocity=self._servo_onvif_max_velocity,
            base_horizon_s=self._servo_brake_horizon,
            velocity_extension_s=self._servo_brake_velocity_extension,
        )
        decision = _servo_axis_decision(
            error=error,
            error_rate=filtered_rate,
            feedforward_rate=ff,
            exit_error=axis_exit_error,
            kp=effective_kp,
            kd=self._servo_kd,
            feedforward_gain=self._servo_feedforward_gain,
            brake_horizon_s=effective_brake_horizon,
        )
        signed_desired_rate = float(decision["desired_rate"])
        outward_catchup_gain = 1.0
        if (
            abs(signed_desired_rate) > 1e-6
            and self._servo_onvif_available
            and self._servo_actuator_mode != "native"
        ):
            outward_catchup_gain = _servo_outward_catchup_gain(
                error,
                filtered_rate,
                start_error=self._servo_outward_catchup_start,
                full_error=self._servo_outward_catchup_full,
                max_gain=self._servo_outward_catchup_gain,
            )
            signed_desired_rate *= outward_catchup_gain
        desired_rate = abs(signed_desired_rate)
        requested_speed = 0
        selected_calibrated_rate = None
        command = 0
        onvif_velocity = 0.0 if desired_rate <= 1e-6 else None
        if desired_rate > 1e-6:
            zoom_factor = self._current_zoom_factor()
            calibrated_rates = self._active_calibration.continuous_rates(
                axis,
                zoom_factor,
                min_speed=self.cfg.hybrid_chase_min_speed,
                max_speed=self.cfg.hybrid_chase_max_speed,
            )
            calibrated = self._active_calibration.choose_continuous_speed_for_rate(
                axis,
                desired_rate,
                zoom_factor,
                min_speed=self.cfg.hybrid_chase_min_speed,
                max_speed=self.cfg.hybrid_chase_max_speed,
            )
            if calibrated is None:
                span = max(0.05, self.cfg.hybrid_chase_full_speed_error - axis_exit_error)
                ratio = max(0.0, min(1.0, (abs(error) - axis_exit_error) / span))
                calibrated = int(round(
                    self.cfg.hybrid_chase_min_speed
                    + ratio * (self.cfg.hybrid_chase_max_speed - self.cfg.hybrid_chase_min_speed)
                ))
            requested_speed = max(
                self.cfg.hybrid_chase_min_speed,
                min(self.cfg.hybrid_chase_max_speed, int(calibrated)),
            )
            selected_calibrated_rate = calibrated_rates.get(requested_speed)
            current_mag = abs(
                self._hybrid_pan_speed if axis == "pan" else self._hybrid_tilt_speed
            )
            allowed_mag = (
                self._servo_start_speed_max
                if not self._hybrid_chase_active
                else min(self.cfg.hybrid_chase_max_speed, current_mag + self._servo_accel_step)
            )
            command_mag = min(requested_speed, allowed_mag)
            if abs(error) <= axis_exit_error + 0.16:
                command_mag = min(command_mag, self.cfg.hybrid_chase_min_speed)
            command = command_mag if signed_desired_rate > 0.0 else -command_mag

            mapped_velocity = self._active_calibration.onvif_velocity_for_rate(
                axis,
                desired_rate,
                max_velocity=self._servo_onvif_max_velocity,
            )
            if mapped_velocity is not None:
                onvif_velocity = (
                    mapped_velocity if signed_desired_rate > 0.0 else -mapped_velocity
                )

        meta = {
            "axis": axis,
            "error": round(float(error), 4),
            "error_rate": round(filtered_rate, 4),
            "feedforward_rate": round(ff, 4),
            "predicted_error": round(float(decision["predicted_error"]), 4),
            "closing_rate": round(float(decision["closing_rate"]), 4),
            "desired_rate": round(signed_desired_rate, 4),
            "error_gain": round(error_gain, 3),
            "effective_kp": round(effective_kp, 3),
            "outward_catchup_gain": round(outward_catchup_gain, 3),
            "brake_horizon_s": round(effective_brake_horizon, 3),
            "requested_speed": requested_speed,
            "command_speed": command,
            "calibrated_rate": (
                None if selected_calibrated_rate is None
                else round(float(selected_calibrated_rate), 4)
            ),
            "onvif_velocity": (
                None if onvif_velocity is None else round(float(onvif_velocity), 5)
            ),
            "phase": str(decision["phase"]),
        }
        return command, meta

    def _servo_camera_command(
        self,
        pan_control: int,
        tilt_control: int,
        pan_meta: dict,
        tilt_meta: dict,
    ) -> Tuple[str, float, float]:
        pan_rate_active = abs(float(pan_meta.get("desired_rate") or 0.0)) > 1e-6
        tilt_rate_active = abs(float(tilt_meta.get("desired_rate") or 0.0)) > 1e-6
        pan_onvif = pan_meta.get("onvif_velocity")
        tilt_onvif = tilt_meta.get("onvif_velocity")
        onvif_mappable = (
            (not pan_rate_active or pan_onvif is not None)
            and (not tilt_rate_active or tilt_onvif is not None)
        )
        if (
            self._servo_actuator_mode != "native"
            and self._servo_onvif_available
            and onvif_mappable
        ):
            pan_sign = self._active_calibration.onvif_continuous_sign("pan", 1)
            tilt_sign = self._active_calibration.onvif_continuous_sign("tilt", 1)
            return (
                "onvif_fractional",
                pan_sign * float(pan_onvif or 0.0),
                tilt_sign * float(tilt_onvif or 0.0),
            )

        pan_sign = self._active_calibration.continuous_sign(
            "pan", self.cfg.hybrid_chase_pan_sign
        )
        tilt_sign = self._active_calibration.continuous_sign("tilt", -1)
        return (
            "native_discrete",
            float(pan_sign * pan_control),
            float(tilt_sign * tilt_control),
        )

    def _record_servo_telemetry(
        self,
        now: float,
        pan_meta: dict,
        tilt_meta: dict,
        actuator: str,
        pan_command: float,
        tilt_command: float,
        *,
        force: bool = False,
    ) -> None:
        payload = {
            "pan": pan_meta,
            "tilt": tilt_meta,
            "actuator": actuator,
            "camera_command": [
                round(float(pan_command), 5),
                round(float(tilt_command), 5),
            ],
        }
        self._servo_last_decision = payload
        if force or (now - self._servo_last_telemetry_at) >= self._servo_telemetry_interval:
            self._servo_last_telemetry_at = now
            self._record_event("hybrid_servo", **payload)

    def _class_mismatch_compatible(
        self,
        det: Detection,
        *,
        dist_norm: float,
        overlap: float,
        size_similarity: float,
    ) -> bool:
        if self.target is None or det.class_id == self.target.class_id:
            return True
        pair = {self.target.label, det.label}
        if not pair.issubset(self._class_continuity_labels):
            return False
        if "bird" in pair:
            return bool(
                size_similarity >= 0.45
                and (overlap >= 0.40 or dist_norm <= 0.07)
            )
        return bool(
            size_similarity >= 0.30
            and (overlap >= 0.25 or dist_norm <= 0.12)
        )

    def _observe_target_class(self, det: Detection, now: float) -> None:
        if self.target is None:
            return
        self._class_evidence.append((float(now), det.label, det.class_id, det.confidence))
        cutoff = now - self._class_vote_window
        while self._class_evidence and self._class_evidence[0][0] < cutoff:
            self._class_evidence.popleft()

        scores: Dict[str, float] = {}
        hits: Dict[str, int] = {}
        class_ids: Dict[str, int] = {}
        for _, label, class_id, confidence in self._class_evidence:
            scores[label] = scores.get(label, 0.0) + max(0.05, float(confidence))
            hits[label] = hits.get(label, 0) + 1
            class_ids[label] = int(class_id)

        current_label = self.target.label
        if current_label in scores:
            scores[current_label] += 0.25
        if not scores:
            return
        candidate = max(scores, key=lambda label: scores[label])
        if candidate == current_label:
            return
        current_score = max(0.05, scores.get(current_label, 0.0))
        if (
            hits.get(candidate, 0) >= self._class_switch_min_hits
            and scores[candidate] >= current_score * self._class_switch_ratio
        ):
            old_label = self.target.label
            self.target.label = candidate
            self.target.class_id = class_ids[candidate]
            self._record_event(
                "target_class_switched",
                from_label=old_label,
                to_label=candidate,
                candidate_score=round(scores[candidate], 3),
                previous_score=round(current_score, 3),
                hits=hits[candidate],
            )

    def _hybrid_axis_speed(self, error: float, axis: str) -> int:
        magnitude = abs(error)
        axis_exit_error = min(
            self.cfg.hybrid_chase_exit_error,
            self._motion_control_continuous_exit_error,
        )
        if magnitude <= axis_exit_error:
            return 0
        zoom_factor = self._current_zoom_factor()
        calibrated = self._active_calibration.choose_continuous_speed(
            axis,
            magnitude,
            zoom_factor,
            min_speed=self.cfg.hybrid_chase_min_speed,
            max_speed=self.cfg.hybrid_chase_max_speed,
            exit_error=axis_exit_error,
            full_speed_error=self.cfg.hybrid_chase_full_speed_error,
        )
        if calibrated is not None:
            return calibrated if error > 0 else -calibrated
        span = max(0.01, self.cfg.hybrid_chase_full_speed_error - axis_exit_error)
        ratio = max(0.0, min(1.0, (magnitude - axis_exit_error) / span))
        zoom_max = max(
            self.cfg.hybrid_chase_min_speed,
            int(round(self.cfg.hybrid_chase_max_speed / math.sqrt(zoom_factor))),
        )
        speed = int(round(self.cfg.hybrid_chase_min_speed + ratio * (zoom_max - self.cfg.hybrid_chase_min_speed)))
        speed = max(self.cfg.hybrid_chase_min_speed, min(zoom_max, speed))
        return speed if error > 0 else -speed

    def _enter_ptz_command_lockout(
        self,
        reason: str,
        *,
        operation: str,
        actuator: str,
        error: Optional[str],
    ) -> None:
        """Fail closed when continuous PTZ motion cannot be proven stopped."""
        self._ptz_command_lockout = True
        self._ptz_command_lockout_reason = reason
        self.active = False
        self.state = "PTZ_ERROR"
        self._stop_reason = reason
        self._record_event(
            "ptz_command_lockout",
            reason=reason,
            operation=operation,
            actuator=actuator,
            error=error,
        )
        self.logger.error("PTZ command lockout engaged: %s", reason)

    def _clear_ptz_command_lockout_from_status(
        self,
        status: Optional[Dict[str, str]],
        *,
        source: str,
    ) -> bool:
        if not self._ptz_command_lockout:
            return True
        if not status or self.ptz.pan_tilt_reported_idle(status) is not True:
            return False

        previous_reason = self._ptz_command_lockout_reason
        self._last_camera_status = status
        self._last_camera_status_at = time.monotonic()
        position = self.ptz.position_from_status(status)
        if position is not None:
            self._last_camera_position = position
            self._last_zoom_position = position[2]

        self._ptz_command_lockout = False
        self._ptz_command_lockout_reason = None
        self._stop_reason = None
        self._reset_tracking_state()
        self.state = "SEARCHING" if self.active else "OFF"
        self._record_event(
            "ptz_command_lockout_cleared",
            source=source,
            previous_reason=previous_reason,
            move_status=status.get("status.MoveStatus"),
            pan_tilt_status=status.get("status.PanTiltStatus"),
        )
        self.logger.warning("PTZ command lockout cleared after camera reported Idle (%s)", source)
        return True

    async def _ensure_ptz_command_unlocked(self, source: str) -> bool:
        if not self._ptz_command_lockout:
            return True
        status = await asyncio.to_thread(self.ptz.get_status)
        if self._clear_ptz_command_lockout_from_status(status, source=source):
            return True
        self._record_event(
            "ptz_command_blocked",
            source=source,
            reason=self._ptz_command_lockout_reason,
            status_available=bool(status),
            move_status=status.get("status.MoveStatus") if status else None,
            pan_tilt_status=status.get("status.PanTiltStatus") if status else None,
        )
        return False

    async def _pause_hybrid_actuator(self, reason: str) -> bool:
        actuator = self._hybrid_actuator
        moving = (
            self._hybrid_pan_speed != 0
            or self._hybrid_tilt_speed != 0
            or abs(self._hybrid_pan_velocity) > 1e-6
            or abs(self._hybrid_tilt_velocity) > 1e-6
        )
        if not moving:
            return True
        t0 = time.monotonic()
        if actuator == "onvif_fractional":
            ok = await self._onvif_motion.stop()
            if not ok:
                # Native CGI stop is a last-resort global PTZ brake on Dahua.
                ok = await asyncio.to_thread(self.ptz.continuous_stop)
        else:
            ok = await asyncio.to_thread(self.ptz.continuous_stop)
        t1 = time.monotonic()
        error = (
            self._onvif_motion.last_error
            if actuator == "onvif_fractional"
            else self.ptz.last_error
        )
        self._record_event(
            "hybrid_actuator_pause",
            reason=reason,
            actuator=actuator,
            success=bool(ok),
            http_ms=int((t1 - t0) * 1000),
            error=None if ok else error,
        )
        if not ok:
            self._enter_ptz_command_lockout(
                f"Continuous PTZ stop failed while pausing '{reason}'.",
                operation="continuous_pause",
                actuator=actuator,
                error=error,
            )
            return False

        if actuator == "native_discrete":
            self._archive_native_response_tail(t1, reason)
        self.ptz_commands += 1
        self._hybrid_pan_speed = 0
        self._hybrid_tilt_speed = 0
        self._hybrid_pan_velocity = 0.0
        self._hybrid_tilt_velocity = 0.0
        self._hybrid_last_command_at = t1
        self._reset_servo_response_monitor()
        self._reset_native_response_monitor()
        self._reset_servo_feedback()
        return True

    async def _stop_hybrid_chase(
        self,
        reason: str,
        *,
        seq: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        was_active = (
            self._hybrid_chase_active
            or self._hybrid_pan_speed != 0
            or self._hybrid_tilt_speed != 0
            or abs(self._hybrid_pan_velocity) > 1e-6
            or abs(self._hybrid_tilt_velocity) > 1e-6
        )
        if not was_active and not force:
            return True
        actuator = self._hybrid_actuator
        t0 = time.monotonic()
        if actuator == "onvif_fractional":
            ok = await self._onvif_motion.stop()
            if not ok:
                ok = await asyncio.to_thread(self.ptz.continuous_stop)
        else:
            ok = await asyncio.to_thread(self.ptz.continuous_stop)
        t1 = time.monotonic()

        if not ok:
            error = (
                self._onvif_motion.last_error
                if actuator == "onvif_fractional"
                else self.ptz.last_error
            )
            self._record_event(
                "hybrid_chase_stop_failed",
                reason=reason,
                actuator=actuator,
                error=error,
            )
            # A failed precautionary force-stop while no motion is believed active
            # is diagnostic only. If motion is active, however, the physical camera
            # state is unknown and every later PTZ command must be blocked.
            if was_active:
                self._enter_ptz_command_lockout(
                    f"Continuous PTZ stop failed while ending chase '{reason}'.",
                    operation="continuous_stop",
                    actuator=actuator,
                    error=error,
                )
                return False
            return True

        if actuator == "native_discrete":
            self._archive_native_response_tail(t1, reason)
        self._hybrid_chase_active = False
        self._hybrid_pan_speed = 0
        self._hybrid_tilt_speed = 0
        self._hybrid_pan_velocity = 0.0
        self._hybrid_tilt_velocity = 0.0
        self._hybrid_actuator = "none"
        self._hybrid_started_at = 0.0
        self._hybrid_divergence_grace_started_at = 0.0
        self._hybrid_last_command_at = t1
        self._hybrid_last_stopped_at = t1
        self._last_ptz_stopped_at = t1
        self._continuous_settle_until = max(
            self._continuous_settle_until,
            t1 + self._servo_post_stop_settle_s,
        )
        self._precision_hold_until = max(
            self._precision_hold_until,
            self._continuous_settle_until + self._post_chase_precision_holdoff,
        )
        self._hybrid_last_error = None
        self._hybrid_divergence_count = 0
        self._hybrid_axis_divergence = {
            "pan": {"last_error": None, "count": 0},
            "tilt": {"last_error": None, "count": 0},
        }
        self._clear_native_edge_rescue()
        self._native_edge_rescue_cooldown_until = 0.0
        self._hybrid_low_confidence_since = None
        self._hybrid_confidence_coast_velocity = (0.0, 0.0)
        self._hybrid_missing_coast_since = None
        self._hybrid_missing_coast_velocity = (0.0, 0.0)
        self._hybrid_fast_target = False
        self._hybrid_entry_speed_norm = 0.0
        self._hybrid_projected_travel_norm = 0.0
        self._native_fast_handoff_started_at = 0.0
        self._native_fast_handoff_takeup_deadline = 0.0
        self._native_fast_handoff_response_started_at = 0.0
        self._native_fast_handoff_until = 0.0
        self._native_fast_handoff_axes = set()
        self._active_acquire_gap_started_at = None
        self._servo_hold_since = None
        self._servo_hold_frames = 0
        self._servo_hold_error = (0.0, 0.0)
        self._servo_hold_motion_guard_logged = False
        self._hybrid_pan_reverse_until = 0.0
        self._hybrid_tilt_reverse_until = 0.0
        self._precision_slow_since = None
        self._reset_servo_response_monitor()
        self._reset_native_response_monitor()
        self._reset_servo_feedback()

        if was_active:
            if reason in ("safe_inner_region", "servo_hold_settled"):
                self._hybrid_divergence_strikes.clear()
            self._scene_stability.reset(t1)
            self._scene_stable_ready = False
            self._velocity_learning_ready = False
            self._velocity_stable_frames = 0
            self.ptz_commands += 1
            self.hybrid_chase_stops += 1
            self._record_event(
                "hybrid_chase_stop",
                reason=reason,
                actuator=actuator,
                http_ms=int((t1 - t0) * 1000),
                settle_ms=int(self._servo_post_stop_settle_s * 1000),
            )

            if self.target is not None:
                self.target.clear_velocity()
                self._velocity_rebase_required = True
            self._association_motion_start_center = None
            self._association_motion_end_center = None
            if seq is not None:
                self._post_motion_release_seq = max(
                    self._post_motion_release_seq,
                    seq + self.cfg.hybrid_chase_settle_frames,
                )
        return True

    @staticmethod
    def _servo_command_sign(value: float) -> int:
        if not math.isfinite(float(value)) or abs(float(value)) <= 1e-6:
            return 0
        return 1 if float(value) > 0.0 else -1

    def _reset_servo_response_monitor(self) -> None:
        for axis in ("pan", "tilt"):
            state = self._servo_axis_response[axis]
            state["correction_sign"] = 0
            state["changed_at"] = 0.0
            state["armed"] = False
            state["confirmed_at"] = 0.0

    def _note_servo_applied_command(
        self,
        axis: str,
        camera_command: float,
        now: float,
    ) -> int:
        state = self._servo_axis_response[axis]
        mapping_sign = self._active_calibration.onvif_continuous_sign(axis, 1)
        command_sign = self._servo_command_sign(camera_command)
        correction_sign = command_sign * mapping_sign
        previous_sign = int(state.get("correction_sign", 0) or 0)
        if correction_sign != previous_sign:
            state["correction_sign"] = correction_sign
            state["changed_at"] = float(now)
            state["armed"] = False
            state["confirmed_at"] = 0.0
            self._hybrid_axis_divergence[axis] = {"last_error": None, "count": 0}
            self._record_event(
                "servo_axis_command_direction",
                axis=axis,
                previous_correction_sign=previous_sign,
                correction_sign=correction_sign,
                camera_command=round(float(camera_command), 5),
                response_armed=False,
            )
        return correction_sign

    def _servo_axis_response_ready(
        self,
        axis: str,
        camera_shift: Optional[float],
        now: float,
    ) -> bool:
        state = self._servo_axis_response[axis]
        correction_sign = int(state.get("correction_sign", 0) or 0)
        if correction_sign == 0:
            return False
        if bool(state.get("armed")):
            return True
        if now < float(self._servo_response_attribution_block_until.get(axis, 0.0)):
            return False
        tail = self._native_response_tail.get(axis)
        if tail is not None and now < float(tail.get("expires_at", 0.0)):
            return False
        if camera_shift is None:
            return False
        shift = float(camera_shift)
        if (
            not math.isfinite(shift)
            or abs(shift) < self._servo_divergence_min_camera_shift_px
        ):
            return False

        # A correct camera response moves the background opposite the desired
        # image-space correction. After a reversal, residual inertia can keep
        # the scene moving in the old direction for several frames. Do not arm
        # divergence monitoring until the estimator proves the new command has
        # physically taken effect.
        if shift * correction_sign >= 0.0:
            return False

        state["armed"] = True
        state["confirmed_at"] = float(now)
        self._hybrid_axis_divergence[axis] = {"last_error": None, "count": 0}
        self._record_event(
            "servo_axis_response_armed",
            axis=axis,
            correction_sign=correction_sign,
            camera_shift_px=round(shift, 3),
            takeup_ms=int(
                max(
                    0.0,
                    float(now) - float(state.get("changed_at", now)),
                )
                * 1000
            ),
        )
        # The confirmation sample itself is not divergence evidence.
        return False

    def _reset_native_response_monitor(self) -> None:
        for axis in ("pan", "tilt"):
            state = self._native_axis_response[axis]
            state["correction_sign"] = 0
            state["changed_at"] = 0.0
            state["armed"] = False
            state["confirmed_at"] = 0.0

    def _archive_native_response_tail(self, stopped_at: float, reason: str) -> None:
        for axis in ("pan", "tilt"):
            state = self._native_axis_response[axis]
            correction_sign = int(state.get("correction_sign", 0) or 0)
            changed_at = float(state.get("changed_at", 0.0) or 0.0)
            if correction_sign == 0 or changed_at <= 0.0 or bool(state.get("armed")):
                continue
            self._native_response_tail[axis] = {
                "correction_sign": correction_sign,
                "changed_at": changed_at,
                "stopped_at": float(stopped_at),
                "expires_at": float(stopped_at) + self._native_response_tail_window_s,
                "reason": reason,
            }
            self._record_event(
                "native_axis_response_tail",
                axis=axis,
                correction_sign=correction_sign,
                reason=reason,
                remaining_ms=int(self._native_response_tail_window_s * 1000),
            )

    def _note_native_applied_command(
        self,
        axis: str,
        camera_command: float,
        now: float,
    ) -> int:
        fallback = self.cfg.hybrid_chase_pan_sign if axis == "pan" else -1
        mapping_sign = self._active_calibration.continuous_sign(axis, fallback)
        command_sign = self._servo_command_sign(camera_command)
        correction_sign = command_sign * mapping_sign
        state = self._native_axis_response[axis]
        self._native_response_tail[axis] = None
        previous_sign = int(state.get("correction_sign", 0) or 0)
        if correction_sign != previous_sign:
            state["correction_sign"] = correction_sign
            state["changed_at"] = float(now)
            state["armed"] = False
            state["confirmed_at"] = 0.0
            self._record_event(
                "native_axis_command_direction",
                axis=axis,
                previous_correction_sign=previous_sign,
                correction_sign=correction_sign,
                camera_command=int(camera_command),
                response_armed=False,
            )
        return correction_sign

    def _update_native_response_monitor(self, now: float) -> None:
        if self._camera_motion is None:
            return
        shifts = {
            "pan": float(self._camera_motion.dx),
            "tilt": float(self._camera_motion.dy),
        }
        if self._hybrid_actuator == "native_discrete":
            for axis in ("pan", "tilt"):
                state = self._native_axis_response[axis]
                correction_sign = int(state.get("correction_sign", 0) or 0)
                if correction_sign == 0 or bool(state.get("armed")):
                    continue
                shift = shifts[axis]
                if (
                    not math.isfinite(shift)
                    or abs(shift) < self._servo_divergence_min_camera_shift_px
                    or shift * correction_sign >= 0.0
                ):
                    continue
                state["armed"] = True
                state["confirmed_at"] = float(now)
                self._record_event(
                    "native_axis_response_armed",
                    axis=axis,
                    correction_sign=correction_sign,
                    camera_shift_px=round(shift, 3),
                    takeup_ms=int(
                        max(0.0, float(now) - float(state.get("changed_at", now))) * 1000
                    ),
                    fast_handoff=bool(self._native_fast_handoff_axes),
                )

        for axis in ("pan", "tilt"):
            tail = self._native_response_tail.get(axis)
            if tail is None:
                continue
            expires_at = float(tail.get("expires_at", 0.0))
            if now >= expires_at:
                self._record_event(
                    "native_axis_response_tail_expired",
                    axis=axis,
                    correction_sign=int(tail.get("correction_sign", 0) or 0),
                    waited_ms=int(
                        max(0.0, now - float(tail.get("stopped_at", now))) * 1000
                    ),
                )
                self._native_response_tail[axis] = None
                continue
            shift = shifts[axis]
            correction_sign = int(tail.get("correction_sign", 0) or 0)
            if (
                correction_sign == 0
                or not math.isfinite(shift)
                or abs(shift) < self._servo_divergence_min_camera_shift_px
                or shift * correction_sign >= 0.0
            ):
                continue
            self._record_event(
                "native_axis_response_delayed",
                axis=axis,
                correction_sign=correction_sign,
                camera_shift_px=round(shift, 3),
                takeup_ms=int(
                    max(0.0, now - float(tail.get("changed_at", now))) * 1000
                ),
                after_stop_ms=int(
                    max(0.0, now - float(tail.get("stopped_at", now))) * 1000
                ),
                stop_reason=tail.get("reason"),
            )
            self._servo_response_attribution_block_until[axis] = (
                now + self._native_response_attribution_guard_s
            )
            self._native_response_tail[axis] = None

    def _apply_axis_reversal_holdoff(
        self,
        current: Tuple[float, float],
        desired: Tuple[float, float],
        now: float,
    ) -> Tuple[Tuple[float, float], List[str]]:
        def sign_flip(old: float, new: float) -> bool:
            return abs(old) > 1e-6 and abs(new) > 1e-6 and ((old > 0) != (new > 0))

        desired_pan, desired_tilt = desired
        suppressed_axes: List[str] = []
        if self._hybrid_chase_active:
            if sign_flip(current[0], desired_pan):
                self._hybrid_pan_reverse_until = max(
                    self._hybrid_pan_reverse_until,
                    now + self._hybrid_axis_reverse_holdoff,
                )
                desired_pan = 0.0
                suppressed_axes.append("pan")
            elif now < self._hybrid_pan_reverse_until:
                desired_pan = 0.0
                suppressed_axes.append("pan")

            if sign_flip(current[1], desired_tilt):
                self._hybrid_tilt_reverse_until = max(
                    self._hybrid_tilt_reverse_until,
                    now + self._hybrid_axis_reverse_holdoff,
                )
                desired_tilt = 0.0
                suppressed_axes.append("tilt")
            elif now < self._hybrid_tilt_reverse_until:
                desired_tilt = 0.0
                suppressed_axes.append("tilt")
        return (desired_pan, desired_tilt), suppressed_axes

    async def _set_hybrid_chase_speed(
        self,
        pan_speed: int,
        tilt_speed: int,
        *,
        seq: int,
        now: float,
        error_x: float,
        error_y: float,
        target_span: float,
        force_command: bool = False,
    ) -> bool:
        if self._ptz_command_lockout:
            return False
        if self._hybrid_actuator == "onvif_fractional":
            if not await self._pause_hybrid_actuator("switch_to_native"):
                return False

        pan_speed = max(
            -self.cfg.hybrid_chase_max_speed,
            min(self.cfg.hybrid_chase_max_speed, int(pan_speed)),
        )
        tilt_speed = max(
            -self.cfg.hybrid_chase_max_speed,
            min(self.cfg.hybrid_chase_max_speed, int(tilt_speed)),
        )
        raw_desired = (float(pan_speed), float(tilt_speed))
        if raw_desired == (0.0, 0.0):
            return await self._stop_hybrid_chase("safe_inner_region", seq=seq)

        current = (
            float(self._hybrid_pan_speed),
            float(self._hybrid_tilt_speed),
        ) if self._hybrid_actuator == "native_discrete" else (0.0, 0.0)
        desired, suppressed_axes = self._apply_axis_reversal_holdoff(
            current, raw_desired, now
        )
        desired = (float(int(desired[0])), float(int(desired[1])))
        if suppressed_axes:
            self._hybrid_divergence_count = 0
            self._record_event(
                "hybrid_axis_reversal_suppressed",
                axes=sorted(set(suppressed_axes)),
                actuator="native_discrete",
                requested=[int(raw_desired[0]), int(raw_desired[1])],
                commanded=[int(desired[0]), int(desired[1])],
                holdoff_ms=int(self._hybrid_axis_reverse_holdoff * 1000),
            )

        if desired == (0.0, 0.0):
            if not await self._pause_hybrid_actuator("axis_reversal_holdoff"):
                return await self._stop_hybrid_chase(
                    "axis_pause_failed", seq=seq, force=True
                )
            self.state = "ESCAPE_CHASE"
            return True

        desired_int = (int(desired[0]), int(desired[1]))
        current_int = (int(current[0]), int(current[1]))
        elapsed = max(0.0, now - self._hybrid_last_command_at)
        same_speed = (
            self._hybrid_chase_active
            and self._hybrid_actuator == "native_discrete"
            and desired_int == current_int
        )
        if same_speed and elapsed < self.cfg.hybrid_chase_keepalive and not force_command:
            return True
        if (
            self._hybrid_chase_active
            and not same_speed
            and elapsed < self.cfg.hybrid_chase_command_interval
            and not force_command
        ):
            return True

        generation = self._session_generation
        t0 = time.monotonic()
        command_pan, command_tilt = desired_int
        ok = await asyncio.to_thread(
            self.ptz.continuous_move,
            command_pan,
            command_tilt,
            self.cfg.hybrid_chase_camera_timeout,
        )
        t1 = time.monotonic()
        if not self._session_valid(generation):
            if ok:
                await asyncio.to_thread(self.ptz.continuous_stop)
            return False
        if not ok:
            self._record_event(
                "hybrid_chase_move_failed",
                actuator="native_discrete",
                pan_speed=command_pan,
                tilt_speed=command_tilt,
                error=self.ptz.last_error,
            )
            if self._hybrid_chase_active:
                await self._stop_hybrid_chase("command_failed", seq=seq, force=True)
            return False

        entering = not self._hybrid_chase_active
        if entering:
            self._hybrid_started_at = t1
            self._hybrid_divergence_grace_started_at = t1
            self._hybrid_last_error = max(abs(error_x), abs(error_y))
            self._hybrid_divergence_count = 0
            self.hybrid_chase_entries += 1
        self._hybrid_chase_active = True
        self._hybrid_actuator = "native_discrete"
        self._hybrid_pan_speed = command_pan
        self._hybrid_tilt_speed = command_tilt
        self._hybrid_pan_velocity = 0.0
        self._hybrid_tilt_velocity = 0.0
        self._note_native_applied_command("pan", command_pan, t1)
        self._note_native_applied_command("tilt", command_tilt, t1)
        self._hybrid_last_command_at = t1
        self.ptz_commands += 1
        self.hybrid_chase_commands += 1
        self.state = "ESCAPE_CHASE"
        self._velocity_learning_ready = False
        self._velocity_stable_frames = 0
        if self.target is not None:
            self.target.clear_velocity()
            self._velocity_rebase_required = True
        self._association_motion_start_center = None
        self._association_motion_end_center = None
        self._record_event(
            "ptz_command_applied",
            source=(
                "native_fast_handoff"
                if self._native_fast_handoff_axes
                else "native_edge_rescue"
                if self._native_edge_rescue_axes
                else "hybrid_chase"
            ),
            actuator="native_discrete",
            command=[command_pan, command_tilt],
            error=[round(error_x, 3), round(error_y, 3)],
            http_ms=int((t1 - t0) * 1000),
        )
        if entering or desired_int != current_int:
            self._record_event(
                "hybrid_chase_move",
                entering=entering,
                actuator="native_discrete",
                label=None if self.target is None else self.target.label,
                pan_speed=command_pan,
                tilt_speed=command_tilt,
                error_x=round(error_x, 3),
                error_y=round(error_y, 3),
                target_span=round(target_span, 3),
                confidence=None if self.target is None else round(self.target.confidence, 3),
                http_ms=int((t1 - t0) * 1000),
            )
        return True

    async def _set_hybrid_chase_velocity(
        self,
        pan_velocity: float,
        tilt_velocity: float,
        *,
        seq: int,
        now: float,
        error_x: float,
        error_y: float,
        target_span: float,
        force_command: bool = False,
    ) -> bool:
        if self._ptz_command_lockout:
            return False
        if not self._servo_onvif_available:
            return False
        switching_from_native = self._hybrid_actuator == "native_discrete"
        if switching_from_native:
            if not await self._pause_hybrid_actuator("switch_to_onvif"):
                return False
            # The native->ONVIF rescue handoff must bypass the normal command
            # interval. The pause refreshed _hybrid_last_command_at; throttling
            # here would leave the chase marked native for one more frame and
            # allow the native max-duration watchdog to kill a fractional chase.
            force_command = True

        limit = self._servo_onvif_max_velocity
        raw_desired = (
            max(-limit, min(limit, float(pan_velocity))),
            max(-limit, min(limit, float(tilt_velocity))),
        )
        if abs(raw_desired[0]) <= 1e-6 and abs(raw_desired[1]) <= 1e-6:
            return await self._stop_hybrid_chase("safe_inner_region", seq=seq)

        current = (
            self._hybrid_pan_velocity,
            self._hybrid_tilt_velocity,
        ) if self._hybrid_actuator == "onvif_fractional" else (0.0, 0.0)
        desired, suppressed_axes = self._apply_axis_reversal_holdoff(
            current, raw_desired, now
        )
        if suppressed_axes:
            self._hybrid_divergence_count = 0
            self._record_event(
                "hybrid_axis_reversal_suppressed",
                axes=sorted(set(suppressed_axes)),
                actuator="onvif_fractional",
                requested=[round(raw_desired[0], 5), round(raw_desired[1], 5)],
                commanded=[round(desired[0], 5), round(desired[1], 5)],
                holdoff_ms=int(self._hybrid_axis_reverse_holdoff * 1000),
            )

        if abs(desired[0]) <= 1e-6 and abs(desired[1]) <= 1e-6:
            if not await self._pause_hybrid_actuator("axis_reversal_holdoff"):
                return await self._stop_hybrid_chase(
                    "axis_pause_failed", seq=seq, force=True
                )
            self.state = "ESCAPE_CHASE"
            return True

        elapsed = max(0.0, now - self._hybrid_last_command_at)
        same_velocity = (
            self._hybrid_chase_active
            and self._hybrid_actuator == "onvif_fractional"
            and abs(desired[0] - current[0]) <= 0.005
            and abs(desired[1] - current[1]) <= 0.005
        )
        keepalive = min(self.cfg.hybrid_chase_keepalive, 0.40)
        interval = min(self.cfg.hybrid_chase_command_interval, 0.12)
        if same_velocity and elapsed < keepalive and not force_command:
            return True
        if (
            self._hybrid_chase_active
            and not same_velocity
            and elapsed < interval
            and not force_command
        ):
            return True

        generation = self._session_generation
        t0 = time.monotonic()
        ok = await self._onvif_motion.continuous_move(desired[0], desired[1])
        t1 = time.monotonic()
        if not self._session_valid(generation):
            if ok:
                await self._onvif_motion.stop()
            return False
        if not ok:
            self._servo_onvif_available = False
            self._servo_onvif_capabilities = {
                **self._servo_onvif_capabilities,
                "available": False,
                "runtime_error": self._onvif_motion.last_error,
            }
            self._record_event(
                "hybrid_chase_move_failed",
                actuator="onvif_fractional",
                pan_velocity=round(desired[0], 5),
                tilt_velocity=round(desired[1], 5),
                error=self._onvif_motion.last_error,
            )
            await self._stop_hybrid_chase("onvif_command_failed", seq=seq, force=True)
            return False

        entering = not self._hybrid_chase_active
        if entering:
            self._hybrid_started_at = t1
            self._hybrid_divergence_grace_started_at = t1
            self._hybrid_last_error = max(abs(error_x), abs(error_y))
            self._hybrid_divergence_count = 0
            self.hybrid_chase_entries += 1
        self._hybrid_chase_active = True
        self._hybrid_actuator = "onvif_fractional"
        self._hybrid_pan_speed = 0
        self._hybrid_tilt_speed = 0
        self._hybrid_pan_velocity = float(desired[0])
        self._hybrid_tilt_velocity = float(desired[1])
        self._note_servo_applied_command("pan", desired[0], t1)
        self._note_servo_applied_command("tilt", desired[1], t1)
        self._hybrid_last_command_at = t1
        self.ptz_commands += 1
        self.hybrid_chase_commands += 1
        self.state = "ESCAPE_CHASE"
        self._velocity_learning_ready = False
        self._velocity_stable_frames = 0
        if self.target is not None:
            self.target.clear_velocity()
            self._velocity_rebase_required = True
        self._association_motion_start_center = None
        self._association_motion_end_center = None
        self._record_event(
            "ptz_command_applied",
            source="hybrid_servo",
            actuator="onvif_fractional",
            command=[round(desired[0], 5), round(desired[1], 5)],
            error=[round(error_x, 3), round(error_y, 3)],
            http_ms=int((t1 - t0) * 1000),
        )
        if entering or not same_velocity:
            self._record_event(
                "hybrid_chase_move",
                entering=entering,
                actuator="onvif_fractional",
                label=None if self.target is None else self.target.label,
                pan_velocity=round(desired[0], 5),
                tilt_velocity=round(desired[1], 5),
                error_x=round(error_x, 3),
                error_y=round(error_y, 3),
                target_span=round(target_span, 3),
                confidence=None if self.target is None else round(self.target.confidence, 3),
                http_ms=int((t1 - t0) * 1000),
            )
        return True

    async def _drive_hybrid_chase(
        self,
        frame_shape: Tuple[int, ...],
        seq: int,
        now: float,
        err_x: float,
        err_y: float,
        target_span: float,
    ) -> None:
        if not self._hybrid_chase_active:
            return
        dominant_error = max(abs(err_x), abs(err_y))
        self._update_native_response_monitor(now)

        if self._native_fast_handoff_until > 0.0:
            active_axes = sorted(self._native_fast_handoff_axes)
            armed = {
                axis: bool(self._native_axis_response[axis].get("armed"))
                for axis in active_axes
            }
            if (
                self._native_fast_handoff_response_started_at <= 0.0
                and any(armed.values())
            ):
                self._native_fast_handoff_response_started_at = now
                self._record_event(
                    "native_fast_handoff_response_confirmed",
                    axes=active_axes,
                    response_armed=armed,
                    takeup_ms=int(
                        max(0.0, now - self._native_fast_handoff_started_at) * 1000
                    ),
                    response_window_ms=int(
                        self._native_fast_handoff_response_seconds * 1000
                    ),
                )

            if dominant_error <= self._motion_control_continuous_exit_error:
                exit_reason = "inner_region"
            elif now >= self._native_fast_handoff_until:
                exit_reason = "hard_max"
            elif self._native_fast_handoff_response_started_at > 0.0:
                response_until = (
                    self._native_fast_handoff_response_started_at
                    + self._native_fast_handoff_response_seconds
                )
                if now < response_until:
                    self.state = "ESCAPE_CHASE"
                    return
                exit_reason = "response_window_complete"
            elif now < self._native_fast_handoff_takeup_deadline:
                self.state = "ESCAPE_CHASE"
                return
            else:
                exit_reason = "takeup_timeout"

            self._record_event(
                "native_fast_handoff_exit",
                reason=exit_reason,
                axes=active_axes,
                elapsed_ms=int(
                    max(0.0, now - self._native_fast_handoff_started_at) * 1000
                ),
                takeup_wait_ms=int(
                    max(
                        0.0,
                        min(
                            now,
                            self._native_fast_handoff_takeup_deadline,
                        )
                        - self._native_fast_handoff_started_at,
                    )
                    * 1000
                ),
                response_armed=armed,
                response_started=(
                    self._native_fast_handoff_response_started_at > 0.0
                ),
                error=[round(err_x, 3), round(err_y, 3)],
            )
            self._native_fast_handoff_started_at = 0.0
            self._native_fast_handoff_takeup_deadline = 0.0
            self._native_fast_handoff_response_started_at = 0.0
            self._native_fast_handoff_until = 0.0
            self._native_fast_handoff_axes.clear()

            if exit_reason == "inner_region":
                await self._stop_hybrid_chase(
                    "native_fast_handoff_inner_region", seq=seq
                )
                return

        if self._servo_hold_since is not None:
            self._servo_hold_frames += 1
            hold_elapsed = max(0.0, now - self._servo_hold_since)
            if self.target is not None and self.target.confidence < self._chase_detection_conf:
                await self._stop_hybrid_chase("confidence_floor", seq=seq)
                return

            # During hold the actuator is paused, so bbox drift becomes a useful
            # short-window estimate of subject motion. Use the same Rev 6.6
            # fast-follow horizon to ask whether a currently centered target is
            # likely to leave the inner region again before declaring it settled.
            hold_rate_x = 0.0
            hold_rate_y = 0.0
            hold_speed_norm = 0.0
            projected_hold_x = err_x
            projected_hold_y = err_y
            if hold_elapsed >= 0.10:
                hold_rate_x = (err_x - self._servo_hold_error[0]) / hold_elapsed
                hold_rate_y = (err_y - self._servo_hold_error[1]) / hold_elapsed
                h, w = frame_shape[:2]
                frame_diag = max(1.0, math.hypot(w, h))
                hold_speed_norm = math.hypot(
                    hold_rate_x * (float(w) / 2.0),
                    hold_rate_y * (float(h) / 2.0),
                ) / frame_diag
                if hold_speed_norm >= self._fast_follow_speed_norm:
                    projected_hold_x = err_x + hold_rate_x * self._fast_follow_horizon_s
                    projected_hold_y = err_y + hold_rate_y * self._fast_follow_horizon_s

            hold_action = _servo_hold_action(
                self._servo_hold_error[0],
                self._servo_hold_error[1],
                err_x,
                err_y,
                exit_error=self._motion_control_continuous_exit_error,
                resume_growth=self._servo_hold_resume_growth,
                elapsed_s=hold_elapsed,
                hold_seconds=self._servo_hold_seconds,
                frames=self._servo_hold_frames,
                min_frames=self._servo_hold_min_frames,
                projected_error_x=projected_hold_x,
                projected_error_y=projected_hold_y,
                max_hold_seconds=self._servo_hold_seconds + self._fast_follow_horizon_s,
            )
            if hold_action.startswith("resume_"):
                self._record_event(
                    "servo_hold_resume",
                    reason=hold_action,
                    hold_ms=int(hold_elapsed * 1000),
                    frames=self._servo_hold_frames,
                    held_error=[round(self._servo_hold_error[0], 3), round(self._servo_hold_error[1], 3)],
                    current_error=[round(err_x, 3), round(err_y, 3)],
                    projected_error=[round(projected_hold_x, 3), round(projected_hold_y, 3)],
                    hold_speed_norm=round(hold_speed_norm, 4),
                    divergence_grace_reset_ms=int(self._servo_divergence_grace * 1000),
                )
                self._servo_hold_since = None
                self._servo_hold_frames = 0
                self._servo_hold_error = (0.0, 0.0)
                self._servo_hold_motion_guard_logged = False
                self._seed_servo_feedback(frame_shape, now, err_x, err_y)
                self._hybrid_last_error = dominant_error
                self._hybrid_divergence_count = 0
                self._hybrid_axis_divergence = {
                    "pan": {"last_error": err_x, "count": 0},
                    "tilt": {"last_error": err_y, "count": 0},
                }
                self._hybrid_divergence_grace_started_at = now
            elif hold_action in ("wait", "wait_motion"):
                if hold_action == "wait_motion" and not self._servo_hold_motion_guard_logged:
                    self._servo_hold_motion_guard_logged = True
                    self._record_event(
                        "servo_hold_motion_guard",
                        hold_ms=int(hold_elapsed * 1000),
                        current_error=[round(err_x, 3), round(err_y, 3)],
                        projected_error=[round(projected_hold_x, 3), round(projected_hold_y, 3)],
                        hold_speed_norm=round(hold_speed_norm, 4),
                        max_hold_ms=int(
                            (self._servo_hold_seconds + self._fast_follow_horizon_s) * 1000
                        ),
                    )
                self.state = "ESCAPE_CHASE"
                return
            else:
                await self._stop_hybrid_chase("servo_hold_settled", seq=seq)
                return

        if (
            self._hybrid_actuator != "onvif_fractional"
            and self._native_edge_rescue_started_at <= 0.0
            and (now - self._hybrid_started_at) >= self.cfg.hybrid_chase_max_seconds
        ):
            await self._stop_hybrid_chase("max_duration", seq=seq)
            return
        if dominant_error <= self._motion_control_continuous_exit_error:
            await self._stop_hybrid_chase("safe_inner_region", seq=seq)
            return

        chase_confidence_threshold = self.cfg.hold_conf
        if self.target is not None and self.target.confidence < self._chase_detection_conf:
            self._record_event(
                "hybrid_chase_confidence_floor",
                confidence=round(self.target.confidence, 3),
                floor=round(self._chase_detection_conf, 3),
            )
            await self._stop_hybrid_chase("confidence_floor", seq=seq)
            return

        if self.target is not None and self.target.confidence < chase_confidence_threshold:
            if self._hybrid_low_confidence_since is None:
                self._hybrid_low_confidence_since = now
                self._hybrid_confidence_coast_velocity = (self._hybrid_pan_velocity, self._hybrid_tilt_velocity)
                self._record_event(
                    "hybrid_chase_confidence_grace",
                    confidence=round(self.target.confidence, 3),
                    threshold=round(chase_confidence_threshold, 3),
                    floor=round(self._chase_detection_conf, 3),
                    grace_ms=int(self._hybrid_confidence_grace * 1000),
                    snapshot_velocity=[
                        round(self._hybrid_confidence_coast_velocity[0], 5),
                        round(self._hybrid_confidence_coast_velocity[1], 5),
                    ],
                )
            confidence_elapsed = max(0.0, now - self._hybrid_low_confidence_since)
            if confidence_elapsed <= self._hybrid_confidence_grace:
                if self._hybrid_actuator == "onvif_fractional":
                    scale = _servo_coast_scale(
                        confidence_elapsed,
                        self._hybrid_confidence_grace,
                        start_scale=self._confidence_coast_start_scale,
                        end_scale=self._confidence_coast_end_scale,
                    )
                    await self._set_hybrid_chase_velocity(
                        self._hybrid_confidence_coast_velocity[0] * scale,
                        self._hybrid_confidence_coast_velocity[1] * scale,
                        seq=seq,
                        now=now,
                        error_x=err_x,
                        error_y=err_y,
                        target_span=target_span,
                    )
                else:
                    self._clear_native_edge_rescue(now, cooldown=True)
                    await self._pause_hybrid_actuator("confidence_grace_native")
                self.state = "ESCAPE_CHASE"
                return
            await self._stop_hybrid_chase("low_confidence", seq=seq)
            return

        if self._hybrid_low_confidence_since is not None:
            self._record_event(
                "hybrid_chase_confidence_recovered",
                grace_ms=int((now - self._hybrid_low_confidence_since) * 1000),
                confidence=None if self.target is None else round(self.target.confidence, 3),
            )
        self._hybrid_low_confidence_since = None
        self._hybrid_confidence_coast_velocity = (0.0, 0.0)

        pan_control, pan_meta = self._servo_axis_command(err_x, "pan", now)
        tilt_control, tilt_meta = self._servo_axis_command(err_y, "tilt", now)
        actuator, pan_command, tilt_command = self._servo_camera_command(
            pan_control, tilt_control, pan_meta, tilt_meta
        )
        self._record_servo_telemetry(now, pan_meta, tilt_meta, actuator, pan_command, tilt_command)

        if self._native_edge_rescue_until > 0.0 and now >= self._native_edge_rescue_until:
            self._record_event(
                "native_edge_rescue_exit",
                reason="duration_expired",
                axes=sorted(self._native_edge_rescue_axes),
                elapsed_ms=int(max(0.0, now - self._native_edge_rescue_started_at) * 1000),
                error=[round(err_x, 3), round(err_y, 3)],
            )
            self._clear_native_edge_rescue(now, cooldown=True)

        pan_rescue_requested = (
            self._native_edge_rescue_enabled
            and now >= self._native_edge_rescue_cooldown_until
            and actuator == "onvif_fractional"
            and _native_edge_rescue_axis_requested(
                err_x,
                float(pan_meta.get("error_rate") or 0.0),
                float(pan_meta.get("predicted_error") or err_x),
                pan_meta.get("onvif_velocity"),
                max_velocity=self._servo_onvif_max_velocity,
                start_error=self._native_edge_rescue_error,
                saturation=self._native_edge_rescue_saturation,
                full_error=self._native_edge_rescue_full_error,
            )
        )
        tilt_rescue_requested = (
            self._native_edge_rescue_enabled
            and now >= self._native_edge_rescue_cooldown_until
            and actuator == "onvif_fractional"
            and _native_edge_rescue_axis_requested(
                err_y,
                float(tilt_meta.get("error_rate") or 0.0),
                float(tilt_meta.get("predicted_error") or err_y),
                tilt_meta.get("onvif_velocity"),
                max_velocity=self._servo_onvif_max_velocity,
                start_error=self._native_edge_rescue_error,
                saturation=self._native_edge_rescue_saturation,
                full_error=self._native_edge_rescue_full_error,
            )
        )

        requested_axes = {
            axis for axis, requested in (
                ("pan", pan_rescue_requested),
                ("tilt", tilt_rescue_requested),
            ) if requested
        }
        axis_errors = {"pan": err_x, "tilt": err_y}
        rescue_active = (
            now < self._native_edge_rescue_until
            and bool(self._native_edge_rescue_axes)
        )
        rescue_entering = False
        added_axes = set()
        axes_changed = False

        if requested_axes and not rescue_active:
            rescue_entering = True
            rescue_active = True
            axes_changed = True
            self._native_edge_rescue_axes = set(requested_axes)
            self._native_edge_rescue_start_errors = {
                axis: float(axis_errors[axis]) for axis in ("pan", "tilt")
            }
            self._native_edge_rescue_started_at = now
            self._native_edge_rescue_last_progress_at = now
            self._native_edge_rescue_until = now + self._native_edge_rescue_seconds
            self._record_event(
                "native_edge_rescue_enter",
                axes=sorted(self._native_edge_rescue_axes),
                error=[round(err_x, 3), round(err_y, 3)],
                fractional_velocity=[
                    round(float(pan_meta.get("onvif_velocity") or 0.0), 5),
                    round(float(tilt_meta.get("onvif_velocity") or 0.0), 5),
                ],
                saturation=round(self._native_edge_rescue_saturation, 3),
                duration_ms=int(self._native_edge_rescue_seconds * 1000),
            )
        elif rescue_active:
            added_axes = requested_axes - self._native_edge_rescue_axes
            if added_axes:
                axes_changed = True
                for axis in added_axes:
                    self._native_edge_rescue_axes.add(axis)
                    self._native_edge_rescue_start_errors[axis] = float(axis_errors[axis])
                self._native_edge_rescue_last_progress_at = now
                self._record_event(
                    "native_edge_rescue_axis_added",
                    axes=sorted(added_axes),
                    active_axes=sorted(self._native_edge_rescue_axes),
                    error=[round(err_x, 3), round(err_y, 3)],
                )

        if rescue_active:
            settled_axes = {
                axis for axis in self._native_edge_rescue_axes
                if abs(float(axis_errors[axis])) <= self._native_edge_rescue_exit_error
            }
            if settled_axes:
                axes_changed = True
                self._native_edge_rescue_axes.difference_update(settled_axes)
                self._record_event(
                    "native_edge_rescue_axis_complete",
                    axes=sorted(settled_axes),
                    active_axes=sorted(self._native_edge_rescue_axes),
                    error=[round(err_x, 3), round(err_y, 3)],
                )

            if (
                self._native_edge_rescue_axes
                and (now - self._native_edge_rescue_last_progress_at)
                    >= self._native_edge_rescue_progress_check_s
            ):
                stalled_axes = []
                improvements = {}
                for axis in sorted(self._native_edge_rescue_axes):
                    start_error = float(self._native_edge_rescue_start_errors.get(axis, 0.0))
                    current_error = float(axis_errors[axis])
                    improvement = abs(start_error) - abs(current_error)
                    improvements[axis] = round(improvement, 4)
                    if improvement < self._native_edge_rescue_min_improvement:
                        stalled_axes.append(axis)
                    else:
                        self._native_edge_rescue_start_errors[axis] = current_error
                self._native_edge_rescue_last_progress_at = now
                if stalled_axes:
                    axes_changed = True
                    self._native_edge_rescue_axes.difference_update(stalled_axes)
                    self._record_event(
                        "native_edge_rescue_progress_abort",
                        axes=sorted(stalled_axes),
                        active_axes=sorted(self._native_edge_rescue_axes),
                        improvements=improvements,
                        required_improvement=round(
                            self._native_edge_rescue_min_improvement, 4
                        ),
                        elapsed_ms=int(
                            max(0.0, now - self._native_edge_rescue_started_at) * 1000
                        ),
                    )

            if not self._native_edge_rescue_axes:
                self._record_event(
                    "native_edge_rescue_exit",
                    reason="axes_complete_or_stalled",
                    elapsed_ms=int(
                        max(0.0, now - self._native_edge_rescue_started_at) * 1000
                    ),
                    error=[round(err_x, 3), round(err_y, 3)],
                )
                self._clear_native_edge_rescue(now, cooldown=True)
                rescue_active = False
            else:
                rescue_pan = (
                    _native_edge_rescue_speed(
                        err_x,
                        start_error=self._native_edge_rescue_exit_error,
                        full_error=self._native_edge_rescue_full_error,
                        min_speed=self._native_edge_rescue_min_speed,
                        max_speed=self._native_edge_rescue_max_speed,
                    )
                    if "pan" in self._native_edge_rescue_axes else 0
                )
                rescue_tilt = (
                    _native_edge_rescue_speed(
                        err_y,
                        start_error=self._native_edge_rescue_exit_error,
                        full_error=self._native_edge_rescue_full_error,
                        min_speed=self._native_edge_rescue_min_speed,
                        max_speed=self._native_edge_rescue_max_speed,
                    )
                    if "tilt" in self._native_edge_rescue_axes else 0
                )
                if rescue_pan != 0 or rescue_tilt != 0:
                    self._hybrid_divergence_count = 0
                    self._hybrid_axis_divergence = {
                        "pan": {"last_error": err_x, "count": 0},
                        "tilt": {"last_error": err_y, "count": 0},
                    }
                    pan_sign = self._active_calibration.continuous_sign(
                        "pan", self.cfg.hybrid_chase_pan_sign
                    )
                    tilt_sign = self._active_calibration.continuous_sign("tilt", -1)
                    await self._set_hybrid_chase_speed(
                        pan_sign * rescue_pan,
                        tilt_sign * rescue_tilt,
                        seq=seq,
                        now=now,
                        error_x=err_x,
                        error_y=err_y,
                        target_span=target_span,
                        force_command=rescue_entering or axes_changed,
                    )
                    return

        divergence_grace_anchor = max(
            self._hybrid_started_at,
            self._hybrid_divergence_grace_started_at,
        )
        if (now - divergence_grace_anchor) >= self._servo_divergence_grace:
            pan_state = self._hybrid_axis_divergence["pan"]
            tilt_state = self._hybrid_axis_divergence["tilt"]
            camera_pan_shift = (
                None if self._camera_motion is None else float(self._camera_motion.dx)
            )
            camera_tilt_shift = (
                None if self._camera_motion is None else float(self._camera_motion.dy)
            )

            pan_ready = self._servo_axis_response_ready(
                "pan", camera_pan_shift, now
            )
            tilt_ready = self._servo_axis_response_ready(
                "tilt", camera_tilt_shift, now
            )
            pan_applied_sign = int(
                self._servo_axis_response["pan"].get("correction_sign", 0) or 0
            )
            tilt_applied_sign = int(
                self._servo_axis_response["tilt"].get("correction_sign", 0) or 0
            )

            pan_state["count"] = (
                _servo_axis_divergence_count(
                    pan_state["last_error"],
                    err_x,
                    float(pan_applied_sign),
                    pan_state["count"],
                    self.cfg.hybrid_divergence_growth,
                    camera_shift=camera_pan_shift,
                    min_camera_shift=self._servo_divergence_min_camera_shift_px,
                )
                if pan_ready else 0
            )
            tilt_state["count"] = (
                _servo_axis_divergence_count(
                    tilt_state["last_error"],
                    err_y,
                    float(tilt_applied_sign),
                    tilt_state["count"],
                    self.cfg.hybrid_divergence_growth,
                    camera_shift=camera_tilt_shift,
                    min_camera_shift=self._servo_divergence_min_camera_shift_px,
                )
                if tilt_ready else 0
            )
            pan_state["last_error"] = err_x
            tilt_state["last_error"] = err_y
            self._hybrid_divergence_count = max(
                int(pan_state["count"]), int(tilt_state["count"])
            )
            if self._hybrid_divergence_count >= self.cfg.hybrid_divergence_frames:
                axis = (
                    "pan"
                    if int(pan_state["count"]) >= int(tilt_state["count"])
                    else "tilt"
                )
                count = int(self._hybrid_axis_divergence[axis]["count"])
                cutoff = now - self._servo_divergence_window_s
                while (
                    self._hybrid_divergence_strikes
                    and self._hybrid_divergence_strikes[0] < cutoff
                ):
                    self._hybrid_divergence_strikes.popleft()
                self._hybrid_divergence_strikes.append(now)
                strikes = len(self._hybrid_divergence_strikes)
                repeated = strikes >= self._servo_divergence_trip_limit
                cooldown_s = (
                    self._servo_divergence_escalated_cooldown_s
                    if repeated
                    else self._servo_divergence_cooldown_s
                )
                self._hybrid_recover_after = max(
                    self._hybrid_recover_after, now + cooldown_s
                )
                response_state = self._servo_axis_response[axis]
                self._record_event(
                    "hybrid_chase_diverging",
                    axis=axis,
                    error=round(err_x if axis == "pan" else err_y, 3),
                    axis_counts={
                        "pan": int(pan_state["count"]),
                        "tilt": int(tilt_state["count"]),
                    },
                    consecutive_growth_frames=count,
                    strikes_in_window=strikes,
                    trip_limit=self._servo_divergence_trip_limit,
                    repeated=repeated,
                    session_disabled=False,
                    recovery_cooldown_ms=int(cooldown_s * 1000),
                    actuator=self._hybrid_actuator,
                    pan_sign=self.cfg.hybrid_chase_pan_sign,
                    applied_correction_sign=int(
                        response_state.get("correction_sign", 0) or 0
                    ),
                    response_armed=bool(response_state.get("armed")),
                    command_age_ms=int(
                        max(
                            0.0,
                            now - float(response_state.get("changed_at", now)),
                        )
                        * 1000
                    ),
                    desired_rate=round(
                        float(
                            (pan_meta if axis == "pan" else tilt_meta).get(
                                "desired_rate"
                            )
                            or 0.0
                        ),
                        4,
                    ),
                    camera_shift_px=round(
                        float(
                            camera_pan_shift
                            if axis == "pan"
                            else camera_tilt_shift
                        ),
                        3,
                    ),
                    camera_motion=(
                        None
                        if self._camera_motion is None
                        else self._camera_motion.public_dict()
                    ),
                    divergence_evidence="armed_wrong_direction_camera_motion",
                )
                await self._stop_hybrid_chase(
                    "repeated_divergence_cooldown"
                    if repeated
                    else "diverging",
                    seq=seq,
                    force=True,
                )
                return
        else:
            self._hybrid_axis_divergence["pan"] = {
                "last_error": err_x,
                "count": 0,
            }
            self._hybrid_axis_divergence["tilt"] = {
                "last_error": err_y,
                "count": 0,
            }
            self._hybrid_divergence_count = 0

        self._hybrid_last_error = dominant_error

        if abs(pan_command) <= 1e-6 and abs(tilt_command) <= 1e-6:
            if self._servo_hold_since is None:
                paused = await self._pause_hybrid_actuator("servo_hold")
                if not paused:
                    return
                self._servo_hold_since = now
                self._servo_hold_frames = 1
                self._servo_hold_error = (err_x, err_y)
                self._servo_hold_motion_guard_logged = False
                self._hybrid_divergence_count = 0
                self._hybrid_axis_divergence = {
                    "pan": {"last_error": err_x, "count": 0},
                    "tilt": {"last_error": err_y, "count": 0},
                }
                self._record_event(
                    "servo_hold_enter",
                    error_x=round(err_x, 3),
                    error_y=round(err_y, 3),
                    hold_ms=int(self._servo_hold_seconds * 1000),
                    min_frames=self._servo_hold_min_frames,
                )
            self.state = "ESCAPE_CHASE"
            return

        if actuator == "onvif_fractional":
            ok = await self._set_hybrid_chase_velocity(
                pan_command, tilt_command, seq=seq, now=now,
                error_x=err_x, error_y=err_y, target_span=target_span,
            )
            if ok:
                return
            if dominant_error < 0.85:
                return

        pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)
        tilt_sign = self._active_calibration.continuous_sign("tilt", -1)
        await self._set_hybrid_chase_speed(
            pan_sign * pan_control, tilt_sign * tilt_control,
            seq=seq, now=now, error_x=err_x, error_y=err_y, target_span=target_span,
        )

