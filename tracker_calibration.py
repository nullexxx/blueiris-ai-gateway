from __future__ import annotations

import asyncio
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ptz_active_calibration import OnvifPanTiltProbe, estimate_static_frame_shift
from ptz_enhancements import frame_sharpness, zoom_bucket
from tracker_config import _env_int


class TrackerCalibrationMixin:
    """Startup and manual PTZ zoom/motion/ONVIF calibration routines for DogTracker."""

    def _startup_calibration_requirements(self) -> Tuple[bool, bool, bool]:
        policy = self._startup_calibration_policy
        if policy == "off":
            return False, False, False
        zoom_missing = self.cfg.autozoom and len(self._zoom_map.points()) < 2
        saved_move = self._active_calibration.move_directly()
        try:
            motion_revision = int(saved_move.get("controller_revision", 0) or 0)
        except (TypeError, ValueError):
            motion_revision = 0
        motion_missing = (
            not self._active_calibration.has_motion_calibration()
            or motion_revision < 2
        )
        # Rev 6 adds a calibrated fractional ONVIF ContinuousMove profile. Older
        # Rev 2-5 motion calibration is still valid for moveDirectly/native PTZ,
        # but it cannot safely drive the fractional servo until this profile has
        # been measured on the actual camera. Treat that as a one-time migration
        # requirement instead of forcing a full motion recalibration.
        fractional_profile = self._active_calibration.onvif_fractional_continuous()
        fractional_missing = bool(
            self._calibration_onvif_benchmark
            and self._servo_actuator_mode in ("auto", "onvif")
            and "usable" not in fractional_profile
        )
        motion_stale = not self._active_calibration.is_fresh(self._calibration_max_age_days)
        if policy == "always":
            zoom_needed = self.cfg.autozoom
            motion_needed = True
            fractional_needed = self._servo_actuator_mode in ("auto", "onvif")
        elif policy == "if_stale":
            zoom_needed = zoom_missing
            motion_needed = motion_missing or motion_stale
            fractional_needed = fractional_missing or motion_stale
        else:  # if_missing
            zoom_needed = zoom_missing
            motion_needed = motion_missing
            fractional_needed = fractional_missing

        scope = self._calibration_scope
        if scope == "zoom":
            motion_needed = False
            fractional_needed = False
        elif scope == "movedirectly":
            zoom_needed = False
            fractional_needed = False
        elif scope in ("continuous", "onvif"):
            zoom_needed = False
            motion_needed = False
        elif scope == "motion":
            zoom_needed = False
        return zoom_needed, motion_needed, fractional_needed

    async def _startup_calibration_sequence(self) -> None:
        self._startup_calibration_state = "checking"
        safe_to_track = True
        try:
            # Let RTSP produce several fresh frames before a calibration starts.
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline and not self._shutdown:
                frame, _, frame_time = self.capture.latest()
                if frame is not None and frame_time > 0 and (time.monotonic() - frame_time) < 0.5:
                    break
                await asyncio.sleep(0.10)

            zoom_needed, motion_needed, fractional_needed = self._startup_calibration_requirements()
            if not zoom_needed and not motion_needed and not fractional_needed:
                self._startup_calibration_state = "skipped"
                self._startup_calibration_last_result = {
                    "success": True,
                    "skipped": True,
                    "reason": "calibration policy considers persisted data current",
                }
            else:
                if zoom_needed and motion_needed:
                    mode = "all"
                elif zoom_needed:
                    mode = "zoom"
                elif motion_needed:
                    if self._calibration_scope in ("movedirectly", "continuous", "onvif"):
                        mode = self._calibration_scope
                    else:
                        mode = "motion"
                elif fractional_needed:
                    # Upgrade a valid Rev 2-5 motion calibration in place by
                    # measuring native continuous response + fractional ONVIF
                    # response only. This preserves moveDirectly/zoom calibration
                    # and avoids a long full-camera exercise after a Rev 6 update.
                    mode = "continuous"
                else:
                    mode = self._calibration_scope
                self._startup_calibration_state = "running"
                self._record_event("startup_calibration_started", mode=mode, policy=self._startup_calibration_policy)
                result = await self.calibrate(mode=mode)
                self._startup_calibration_last_result = result
                safe_to_track = bool(result.get("safe_to_track", True))
                self._startup_calibration_state = "success" if result.get("success") else "failed"
                self._record_event(
                    "startup_calibration_finished",
                    mode=mode,
                    success=bool(result.get("success")),
                    safe_to_track=safe_to_track,
                )
        except asyncio.CancelledError:
            self._startup_calibration_state = "cancelled"
            raise
        except Exception as exc:
            self._startup_calibration_state = "failed"
            self._startup_calibration_last_result = {"success": False, "error": str(exc), "safe_to_track": False}
            safe_to_track = False
            self.logger.error("Startup PTZ calibration failed: %s", exc, exc_info=True)
        finally:
            if self.cfg.autostart and not self._shutdown and safe_to_track and not self.active:
                await self.start()

    async def _calibration_wait_pan_tilt_idle(self, timeout_s: Optional[float] = None) -> Optional[dict]:
        timeout_s = max(1.0, float(timeout_s or self.cfg.ptz_operation_timeout))
        started = time.monotonic()
        deadline = started + timeout_s
        initial_position = None
        previous_position = None
        stable_polls = 0
        seen_motion = False
        motion_started_at = None
        last_status = None
        while time.monotonic() < deadline:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if status:
                last_status = status
                position = self.ptz.position_from_status(status)
                idle = self.ptz.pan_tilt_reported_idle(status)
                if initial_position is None and position is not None:
                    initial_position = position
                if idle is False and not seen_motion:
                    seen_motion = True
                    motion_started_at = checked_at
                if initial_position is not None and position is not None:
                    if abs(position[0] - initial_position[0]) > 0.01 or abs(position[1] - initial_position[1]) > 0.01:
                        if not seen_motion:
                            motion_started_at = checked_at
                        seen_motion = True
                stable = self._position_stable(previous_position, position, "move") if previous_position is not None else None
                if position is not None:
                    previous_position = position
                elapsed = checked_at - started
                if (idle is True or (idle is None and stable is True)) and stable is not False and (seen_motion or elapsed >= 0.35):
                    stable_polls += 1
                else:
                    stable_polls = 0
                if stable_polls >= 2 and position is not None:
                    return {
                        "position": position,
                        "elapsed_s": checked_at - started,
                        "motion_start_s": None if motion_started_at is None else motion_started_at - started,
                        "seen_motion": seen_motion,
                        "status": last_status,
                    }
            await asyncio.sleep(self.cfg.ptz_status_poll_interval)
        return None

    async def _calibration_capture_frame(self, after_seq: int = -1, timeout_s: float = 1.5) -> Optional[Tuple[np.ndarray, int]]:
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        best = None
        while time.monotonic() < deadline:
            frame, seq, frame_time = self.capture.latest()
            if frame is not None and seq > after_seq and frame_time > 0 and (time.monotonic() - frame_time) <= 0.40:
                score = frame_sharpness(frame)
                if score >= 35.0:
                    return frame.copy(), seq
                best = (frame.copy(), seq)
            await asyncio.sleep(0.04)
        return best

    async def _calibration_home(self) -> bool:
        ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
        if not ok:
            return False
        result = await self._calibration_wait_pan_tilt_idle(max(self.cfg.ptz_operation_timeout, 5.0))
        if result is None:
            return False
        await asyncio.sleep(0.15)
        return True

    async def _calibration_set_zoom(self, factor: float) -> bool:
        factor = max(self.cfg.zoom_min_factor, min(self.cfg.zoom_max_factor, float(factor)))
        if not self.cfg.autozoom:
            return factor <= 1.01
        if not self._onvif_zoom.available and not await self._recover_onvif_once():
            return factor <= 1.01
        if factor <= 1.01:
            normalized = 0.0
        else:
            if len(self._zoom_map.points()) < 2:
                return False
            normalized = self._zoom_map.estimate_normalized(factor, self._camera_max_optical_zoom)
        try:
            ok = await asyncio.wait_for(self._onvif_zoom.set_normalized(normalized), timeout=1.5)
        except asyncio.TimeoutError:
            return False
        if not ok:
            return False
        position = await self._wait_zoom_idle(max(self.cfg.ptz_operation_timeout, 4.0))
        if position is not None:
            self._last_zoom_position = position[2]
            return True
        return False

    async def _calibration_prepare_scene(self, factor: float) -> Optional[Tuple[np.ndarray, int]]:
        if not await self._calibration_home():
            return None
        if not await self._calibration_set_zoom(factor):
            return None
        await asyncio.sleep(0.18)
        return await self._calibration_capture_frame(timeout_s=2.0)

    async def _calibration_move_directly_sample(self, factor: float, err_x: float, err_y: float) -> Optional[dict]:
        prepared = await self._calibration_prepare_scene(factor)
        if prepared is None:
            return None
        before, seq = prepared
        h, w = before.shape[:2]
        point = (w / 2.0 + err_x * (w / 2.0), h / 2.0 + err_y * (h / 2.0))
        started = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.move_directly_point, point, before.shape)
        command_done = time.monotonic()
        if not ok:
            return None
        wait = await self._calibration_wait_pan_tilt_idle(max(self.cfg.ptz_operation_timeout, 5.0))
        if wait is None:
            return None
        await asyncio.sleep(0.15)
        captured = await self._calibration_capture_frame(after_seq=seq, timeout_s=2.0)
        if captured is None:
            return None
        after, _ = captured
        shift = estimate_static_frame_shift(before, after)
        if shift is None:
            return None
        nx, ny = shift.normalized(before.shape)
        observed_x, observed_y = -nx, -ny
        requested = math.hypot(err_x, err_y)
        command_s = command_done - started
        settled_s = command_s + float(wait["elapsed_s"])
        row = {
            "zoom_factor": round(factor, 3),
            "requested_error": [round(err_x, 4), round(err_y, 4)],
            "observed_correction": [round(observed_x, 4), round(observed_y, 4)],
            "move_distance": round(requested, 4),
            "command_http_ms": int((command_done - started) * 1000),
            "motion_start_ms": None if wait["motion_start_s"] is None else int((command_s + float(wait["motion_start_s"])) * 1000),
            "settled_ms": int(settled_s * 1000),
            "shift": shift.public_dict(),
        }
        if requested > 0.01 and math.hypot(observed_x, observed_y) > 0.03:
            self._update_move_timing_model(requested, settled_s)
        self._record_event("active_calibration_move_directly", **row)
        return row

    async def _calibration_continuous_sample(self, factor: float, axis: str, speed: int, raw_sign: int) -> Optional[dict]:
        prepared = await self._calibration_prepare_scene(factor)
        if prepared is None:
            return None
        before, seq = prepared
        pan = raw_sign * speed if axis == "pan" else 0
        tilt = raw_sign * speed if axis == "tilt" else 0
        started = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.continuous_move, pan, tilt, 1)
        command_done = time.monotonic()
        if not ok:
            return None
        try:
            await asyncio.sleep(self._calibration_continuous_duration)
        finally:
            await asyncio.to_thread(self.ptz.continuous_stop)
        wait = await self._calibration_wait_pan_tilt_idle(max(self.cfg.ptz_operation_timeout, 4.0))
        if wait is None:
            return None
        await asyncio.sleep(0.12)
        captured = await self._calibration_capture_frame(after_seq=seq, timeout_s=2.0)
        if captured is None:
            return None
        after, _ = captured
        shift = estimate_static_frame_shift(before, after)
        if shift is None:
            return None
        nx, ny = shift.normalized(before.shape)
        correction = -nx if axis == "pan" else -ny
        rate = abs(correction) / max(0.05, self._calibration_continuous_duration)
        desired_sign = raw_sign if correction >= 0 else -raw_sign
        settled_s = (command_done - started) + self._calibration_continuous_duration + float(wait["elapsed_s"])
        row = {
            "zoom_factor": round(factor, 3),
            "axis": axis,
            "speed": speed,
            "raw_sign": raw_sign,
            "correction_sign": desired_sign,
            "normalized_rate_per_s": round(rate, 4),
            "command_http_ms": int((command_done - started) * 1000),
            "settled_ms": int(settled_s * 1000),
            "shift": shift.public_dict(),
        }
        self._record_event("active_calibration_continuous", **row)
        return row

    async def _calibration_onvif_continuous_sample(
        self,
        probe: OnvifPanTiltProbe,
        axis: str,
        velocity: float,
    ) -> Optional[dict]:
        prepared = await self._calibration_prepare_scene(1.0)
        if prepared is None:
            return None
        before, seq = prepared
        raw_sign = 1 if float(velocity) >= 0.0 else -1
        pan = float(velocity) if axis == "pan" else 0.0
        tilt = float(velocity) if axis == "tilt" else 0.0
        started = time.monotonic()
        ok = await probe.continuous_move(pan, tilt)
        command_done = time.monotonic()
        if not ok:
            return None
        try:
            await asyncio.sleep(self._servo_onvif_calibration_duration)
        finally:
            await probe.stop()
        wait = await self._calibration_wait_pan_tilt_idle(
            max(self.cfg.ptz_operation_timeout, 4.0)
        )
        if wait is None:
            return None
        await asyncio.sleep(0.12)
        captured = await self._calibration_capture_frame(after_seq=seq, timeout_s=2.0)
        if captured is None:
            return None
        after, _ = captured
        shift = estimate_static_frame_shift(before, after)
        if shift is None:
            return None
        nx, ny = shift.normalized(before.shape)
        correction = -nx if axis == "pan" else -ny
        rate = abs(correction) / max(0.05, self._servo_onvif_calibration_duration)
        correction_sign = raw_sign if correction >= 0.0 else -raw_sign
        settled_s = (
            (command_done - started)
            + self._servo_onvif_calibration_duration
            + float(wait["elapsed_s"])
        )
        row = {
            "axis": axis,
            "velocity": round(abs(float(velocity)), 5),
            "raw_sign": raw_sign,
            "correction_sign": correction_sign,
            "normalized_rate_per_s": round(rate, 5),
            "command_http_ms": int((command_done - started) * 1000),
            "settled_ms": int(settled_s * 1000),
            "shift": shift.public_dict(),
        }
        self._record_event("active_calibration_onvif_continuous", **row)
        return row

    async def _calibration_onvif_benchmark_run(
        self,
        native_samples: Optional[List[dict]] = None,
    ) -> dict:
        probe = OnvifPanTiltProbe(
            self.cfg.camera_ip,
            _env_int("TRACKER_ONVIF_PORT", 80),
            self.cfg.camera_user,
            self.cfg.camera_password,
        )
        result = await probe.initialize()
        try:
            if not self._calibration_onvif_benchmark:
                return result

            if result.get("relative_fov_supported"):
                prepared = await self._calibration_prepare_scene(1.0)
                if prepared is None:
                    result["benchmark_error"] = "unable to prepare stable home scene"
                else:
                    before, seq = prepared
                    started = time.monotonic()
                    ok = await probe.relative_move(0.12, 0.0)
                    command_done = time.monotonic()
                    if not ok:
                        result["benchmark_error"] = probe.last_error or "RelativeMove failed"
                    else:
                        wait = await self._calibration_wait_pan_tilt_idle(
                            max(self.cfg.ptz_operation_timeout, 5.0)
                        )
                        if wait is None:
                            result["benchmark_error"] = "RelativeMove did not settle"
                        else:
                            await asyncio.sleep(0.15)
                            captured = await self._calibration_capture_frame(
                                after_seq=seq, timeout_s=2.0
                            )
                            if captured is None:
                                result["benchmark_error"] = "no stable post-ONVIF frame"
                            else:
                                after, _ = captured
                                shift = estimate_static_frame_shift(before, after)
                                result["relative_move"] = {
                                    "success": shift is not None,
                                    "command_http_ms": int(
                                        (command_done - started) * 1000
                                    ),
                                    "settled_ms": int(
                                        (time.monotonic() - started) * 1000
                                    ),
                                    "shift": (
                                        None if shift is None else shift.public_dict()
                                    ),
                                }

            fractional_rows: List[dict] = []
            if result.get("continuous_supported"):
                velocities = sorted(self._servo_onvif_calibration_velocities)
                midpoint = velocities[len(velocities) // 2] if velocities else None
                for axis in ("pan", "tilt"):
                    for velocity in velocities:
                        raw_signs = (1, -1) if velocity == midpoint else (1,)
                        for raw_sign in raw_signs:
                            row = await self._calibration_onvif_continuous_sample(
                                probe, axis, raw_sign * velocity
                            )
                            if (
                                row is not None
                                and float(row.get("normalized_rate_per_s", 0.0)) > 0.003
                            ):
                                fractional_rows.append(row)

            signs: Dict[str, int] = {}
            axis_metrics: Dict[str, dict] = {}
            usable_axes = []
            native_samples = native_samples or []
            for axis in ("pan", "tilt"):
                rows = [
                    row for row in fractional_rows
                    if str(row.get("axis")) == axis
                ]
                votes = [int(row.get("correction_sign", 1)) for row in rows]
                signs[axis] = 1 if not votes or sum(votes) >= 0 else -1
                sign_agreement = (
                    abs(sum(votes)) / max(1, len(votes))
                    if votes else 0.0
                )

                by_velocity: Dict[float, List[float]] = {}
                for row in rows:
                    velocity = abs(float(row["velocity"]))
                    rate = abs(float(row["normalized_rate_per_s"]))
                    if velocity > 0.0 and rate > 0.0:
                        by_velocity.setdefault(velocity, []).append(rate)
                rates = sorted(
                    (
                        velocity,
                        float(np.median(values)),
                    )
                    for velocity, values in by_velocity.items()
                    if values
                )
                rate_values = [rate for _, rate in rates]
                spread = (
                    max(rate_values) / max(1e-6, min(rate_values))
                    if len(rate_values) >= 2 else 0.0
                )
                monotonic_steps = sum(
                    1
                    for (_, r0), (_, r1) in zip(rates, rates[1:])
                    if r1 >= r0 * 0.80
                )
                monotonic_fraction = (
                    monotonic_steps / max(1, len(rates) - 1)
                    if rates else 0.0
                )

                native_rate_values = []
                for row in native_samples:
                    try:
                        if (
                            str(row.get("axis")) == axis
                            and int(row.get("speed", 0)) == self.cfg.hybrid_chase_min_speed
                            and abs(
                                float(row.get("zoom_factor", 1.0))
                                - float(self._calibration_zoom_levels[0])
                            ) <= 0.10
                        ):
                            native_rate_values.append(
                                abs(float(row.get("normalized_rate_per_s", 0.0)))
                            )
                    except (TypeError, ValueError):
                        continue
                native_speed1_rate = (
                    float(np.median(native_rate_values))
                    if native_rate_values else None
                )
                if native_speed1_rate is None:
                    native_speed1_rate = self._active_calibration.continuous_rates(
                        axis,
                        1.0,
                        min_speed=self.cfg.hybrid_chase_min_speed,
                        max_speed=self.cfg.hybrid_chase_min_speed,
                    ).get(self.cfg.hybrid_chase_min_speed)

                lowest_rate = min(rate_values) if rate_values else None
                below_native_ratio = (
                    None
                    if lowest_rate is None or not native_speed1_rate
                    else lowest_rate / max(1e-6, float(native_speed1_rate))
                )
                sub_native_proven = (
                    True
                    if below_native_ratio is None
                    else below_native_ratio <= 0.80
                )
                axis_usable = bool(
                    len(rates) >= 3
                    and spread >= 1.35
                    and monotonic_fraction >= 0.50
                    and sign_agreement >= 0.50
                    and sub_native_proven
                )
                if axis_usable:
                    usable_axes.append(axis)
                axis_metrics[axis] = {
                    "samples": len(rows),
                    "unique_velocities": len(rates),
                    "rate_spread": round(spread, 3),
                    "monotonic_fraction": round(monotonic_fraction, 3),
                    "sign_agreement": round(sign_agreement, 3),
                    "native_speed1_rate": (
                        None
                        if native_speed1_rate is None
                        else round(float(native_speed1_rate), 5)
                    ),
                    "lowest_fractional_rate": (
                        None if lowest_rate is None else round(lowest_rate, 5)
                    ),
                    "lowest_to_native_ratio": (
                        None
                        if below_native_ratio is None
                        else round(float(below_native_ratio), 3)
                    ),
                    "sub_native_proven": sub_native_proven,
                    "usable": axis_usable,
                }

            fractional_usable = set(usable_axes) == {"pan", "tilt"}
            result["fractional_continuous"] = {
                "usable": fractional_usable,
                "reason": (
                    "calibrated_fractional_response"
                    if fractional_usable
                    else "quantized_or_insufficient_response"
                ),
                "duration_s": self._servo_onvif_calibration_duration,
                "requested_velocities": list(self._servo_onvif_calibration_velocities),
                "samples": fractional_rows,
                "signs": signs,
                "axis_metrics": axis_metrics,
            }
            return result
        finally:
            try:
                await probe.stop()
            except Exception:
                pass
            await probe.close()
            await self._calibration_home()

    async def _calibrate_motion(self, mode: str) -> dict:
        if self.active:
            return {"success": False, "error": "Stop tracking before calibration.", "safe_to_track": True}
        if self._calibrating:
            return {"success": False, "error": "Calibration is already running.", "safe_to_track": True}
        self._calibrating = True
        self.state = "CALIBRATING"
        move_samples: List[dict] = []
        continuous_samples: List[dict] = []
        onvif_result = self._active_calibration.onvif_benchmark()
        run_move = mode in ("movedirectly", "motion", "all")
        run_continuous = mode in ("continuous", "motion", "all")
        # Rev 6 continuous calibration also characterizes fractional ONVIF
        # ContinuousMove so the runtime can operate below native speed 1.
        run_onvif = mode in ("continuous", "onvif", "motion", "all")
        safe_to_track = True
        result_payload: Optional[dict] = None
        try:
            if run_move:
                self._move_timing_samples.clear()
                self._move_eta_intercept = max(
                    self.cfg.move_eta_min,
                    min(self.cfg.move_eta_max, self.cfg.lead_time),
                )
                self._move_eta_slope = 0.0
                self._move_eta_model_ready = False
                self._last_predicted_move_eta = self._move_eta_intercept
                for zoom_index, factor in enumerate(self._calibration_zoom_levels):
                    offsets = self._calibration_offsets if zoom_index == 0 else [max(self._calibration_offsets)]
                    for offset in offsets:
                        for axis in ("pan", "tilt"):
                            for direction in (-1, 1):
                                err_x = direction * offset if axis == "pan" else 0.0
                                err_y = direction * offset if axis == "tilt" else 0.0
                                row = await self._calibration_move_directly_sample(factor, err_x, err_y)
                                if row is not None:
                                    move_samples.append(row)

                # Convert measured full-command response into the multiplicative
                # correction already consumed by the runtime controller.
                bucket_ratios: Dict[str, Dict[str, List[float]]] = {}
                for row in move_samples:
                    factor = float(row["zoom_factor"])
                    bucket = zoom_bucket(factor)
                    requested_x, requested_y = row["requested_error"]
                    observed_x, observed_y = row["observed_correction"]
                    entry = bucket_ratios.setdefault(bucket, {"pan": [], "tilt": []})
                    if abs(requested_x) >= 0.08 and requested_x * observed_x > 0:
                        entry["pan"].append(abs(observed_x / requested_x))
                    if abs(requested_y) >= 0.08 and requested_y * observed_y > 0:
                        entry["tilt"].append(abs(observed_y / requested_y))
                for bucket, values in bucket_ratios.items():
                    pan_ratio = float(np.median(values["pan"])) if values["pan"] else 1.0
                    tilt_ratio = float(np.median(values["tilt"])) if values["tilt"] else 1.0
                    # Runtime moveDirectly applies cfg.move_gain first. Seed the
                    # learned multiplier so a stationary target receives about an
                    # 80% one-step correction. Live moving targets never tune this.
                    target_fraction = 0.80
                    pan_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * pan_ratio)))
                    tilt_scale = max(0.75, min(1.35, target_fraction / max(0.20, self.cfg.move_gain * tilt_ratio)))
                    self._calibration.set_spatial_scales(bucket, pan_scale, tilt_scale)

            if run_continuous:
                zooms = [self._calibration_zoom_levels[0]]
                if self._calibration_zoom_levels[-1] - self._calibration_zoom_levels[0] >= 0.25:
                    zooms.append(self._calibration_zoom_levels[-1])
                mid_speed = self._calibration_continuous_speeds[len(self._calibration_continuous_speeds) // 2]
                for factor in sorted(set(zooms)):
                    for axis in ("pan", "tilt"):
                        for speed in self._calibration_continuous_speeds:
                            signs = (1, -1) if speed == mid_speed else (1,)
                            for raw_sign in signs:
                                row = await self._calibration_continuous_sample(factor, axis, speed, raw_sign)
                                if row is not None and row["normalized_rate_per_s"] > 0.005:
                                    continuous_samples.append(row)

            if run_onvif:
                onvif_result = await self._calibration_onvif_benchmark_run(
                    continuous_samples
                )

            existing_move = self._active_calibration.move_directly()
            existing_continuous = self._active_calibration.continuous()
            move_result = existing_move
            continuous_result = existing_continuous

            if run_move and move_samples:
                move_result = {
                    "controller_revision": 2,
                    "samples": move_samples,
                    "timing": {
                        "intercept_s": round(self._move_eta_intercept, 4),
                        "slope_s_per_norm": round(self._move_eta_slope, 4),
                        "ready": self._move_eta_model_ready,
                    },
                    "spatial_response": self._calibration.public_dict().get("spatial_response", {}),
                }

            if run_continuous and continuous_samples:
                sign_votes: Dict[str, List[int]] = {"pan": [], "tilt": []}
                rates: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
                for row in continuous_samples:
                    axis = str(row["axis"])
                    sign_votes[axis].append(int(row["correction_sign"]))
                    zkey = f'{float(row["zoom_factor"]):.2f}'
                    speed_key = str(int(row["speed"]))
                    rates.setdefault(zkey, {}).setdefault(axis, {}).setdefault(speed_key, []).append(
                        float(row["normalized_rate_per_s"])
                    )
                by_zoom = {}
                for zkey, axes in rates.items():
                    by_zoom[zkey] = {}
                    for axis, speeds in axes.items():
                        by_zoom[zkey][axis] = {
                            speed: round(float(np.median(values)), 5)
                            for speed, values in speeds.items() if values
                        }
                signs = {}
                for axis, votes in sign_votes.items():
                    fallback = self.cfg.hybrid_chase_pan_sign if axis == "pan" else -1
                    signs[axis] = fallback if not votes else (1 if sum(votes) >= 0 else -1)
                continuous_result = {
                    "samples": continuous_samples,
                    "duration_s": self._calibration_continuous_duration,
                    "signs": signs,
                    "by_zoom": by_zoom,
                }

            move_ok = (not run_move) or len(move_samples) >= 4
            continuous_ok = (not run_continuous) or len(continuous_samples) >= 4
            # ONVIF remains optional for native motion/all calibration. An
            # explicit mode=onvif request succeeds only if the service is reachable.
            onvif_ok = mode != "onvif" or bool(isinstance(onvif_result, dict) and onvif_result.get("available"))
            success = move_ok and continuous_ok and onvif_ok
            if success or move_samples or continuous_samples:
                self._active_calibration.replace_motion_calibration(
                    move_directly=move_result,
                    continuous=continuous_result,
                    onvif_benchmark=onvif_result,
                )
            result_payload = {
                "success": success,
                "mode": mode,
                "move_directly_samples": len(move_samples),
                "continuous_samples": len(continuous_samples),
                "move_timing_model": {
                    "samples": len(self._move_timing_samples),
                    "ready": self._move_eta_model_ready,
                    "intercept_s": round(self._move_eta_intercept, 4),
                    "slope_s_per_norm": round(self._move_eta_slope, 4),
                },
                "active_calibration": self._active_calibration.public_dict(),
                "onvif_benchmark": onvif_result,
                "safe_to_track": True,
            }
            return result_payload
        except asyncio.CancelledError:
            safe_to_track = False
            raise
        except Exception as exc:
            self.logger.error("Active PTZ calibration failed: %s", exc, exc_info=True)
            result_payload = {"success": False, "mode": mode, "error": str(exc), "safe_to_track": False}
            return result_payload
        finally:
            await asyncio.to_thread(self.ptz.continuous_stop)
            home_ok = await self._calibration_home()
            safe_to_track = bool(home_ok)
            if result_payload is not None:
                result_payload["safe_to_track"] = safe_to_track
                if not home_ok:
                    result_payload["success"] = False
                    result_payload["home_error"] = "Camera did not safely return to the configured home preset."
            self.state = "OFF"
            self._calibrating = False

    async def calibrate(self, mode: str = "zoom") -> dict:
        mode = (mode or "zoom").strip().lower()
        aliases = {"move": "movedirectly", "move_directly": "movedirectly", "pan_tilt": "motion"}
        mode = aliases.get(mode, mode)
        allowed = ("zoom", "movedirectly", "continuous", "motion", "onvif", "all")
        if mode not in allowed:
            return {"success": False, "error": f"Unknown calibration mode '{mode}'. Allowed: {', '.join(allowed)}"}
        if self._ptz_command_lockout and not await self._ensure_ptz_command_unlocked("calibrate"):
            return {
                "success": False,
                "error": "PTZ command lockout is active because continuous motion could not be proven stopped.",
                "safe_to_track": False,
            }
        if self.active:
            return {"success": False, "error": "Stop tracking before calibration.", "safe_to_track": True}
        if self._calibrating:
            return {"success": False, "error": "Calibration is already running.", "safe_to_track": True}
        if mode == "zoom":
            result = await self._calibrate_zoom()
            result.setdefault("safe_to_track", True)
            return result
        if mode == "all":
            zoom_result = {"success": True, "skipped": True}
            if self.cfg.autozoom:
                zoom_result = await self._calibrate_zoom()
            motion_result = await self._calibrate_motion("all")
            return {
                "success": bool(zoom_result.get("success")) and bool(motion_result.get("success")),
                "mode": "all",
                "zoom": zoom_result,
                "motion": motion_result,
                "safe_to_track": bool(motion_result.get("safe_to_track", True)),
            }
        return await self._calibrate_motion(mode)

    async def _wait_zoom_idle(self, timeout_s: float = 4.0) -> Optional[Tuple[float, float, float]]:
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        previous = None
        stable_polls = 0
        while time.monotonic() < deadline:
            status = await asyncio.to_thread(self.ptz.get_status)
            if status:
                position = self.ptz.position_from_status(status)
                idle = self.ptz.zoom_reported_idle(status)
                stable = self._position_stable(previous, position, "zoom") if previous is not None else None
                if position is not None:
                    previous = position
                if idle is True and stable is not False:
                    stable_polls += 1
                elif idle is None and stable is True:
                    stable_polls += 1
                else:
                    stable_polls = 0
                if stable_polls >= 2 and position is not None:
                    return position
            await asyncio.sleep(self.cfg.ptz_status_poll_interval)
        return None

    async def _calibrate_zoom(self) -> dict:
        """Manual bounded zoom calibration using independent ONVIF probes.

        The calibration sweep deliberately does not consult the existing learned
        map. Each requested optical factor is bracketed using the physical Dahua
        zoom position returned by getStatus, then the complete map is replaced
        only after the sweep succeeds. This prevents calibration from learning
        from its own partially-written results.
        """
        if self.active:
            return {"success": False, "error": "Stop tracking before calibration."}
        if self._calibrating:
            return {"success": False, "error": "Calibration is already running."}
        if not self.cfg.autozoom:
            return {"success": False, "error": "Autozoom is disabled."}

        self._calibrating = True
        chosen_samples = []
        probe_rows = {}

        async def probe(normalized: float):
            normalized = max(0.0, min(1.0, float(normalized)))
            key = round(normalized, 6)
            if key in probe_rows:
                return probe_rows[key]
            try:
                ok = await asyncio.wait_for(self._onvif_zoom.set_normalized(normalized), timeout=1.5)
            except asyncio.TimeoutError:
                self._schedule_onvif_retry("ONVIF absolute zoom calibration command timed out")
                return None
            if not ok:
                self._schedule_onvif_retry(self._onvif_zoom.last_error)
                return None
            position = await self._wait_zoom_idle(self.cfg.ptz_operation_timeout)
            if position is None:
                self._record_event(
                    "zoom_calibration_probe_timeout",
                    onvif_normalized=round(normalized, 6),
                )
                return None
            actual_factor = max(1.0, position[2] / max(0.001, self.cfg.zoom_wide_position))
            row = {
                "actual_factor": float(actual_factor),
                "onvif_normalized": float(normalized),
                "cgi_zoom_position": float(position[2]),
            }
            probe_rows[key] = row
            self._record_event(
                "zoom_calibration_probe",
                actual_factor=round(actual_factor, 3),
                onvif_normalized=round(normalized, 6),
                cgi_zoom_position=round(position[2], 3),
            )
            await asyncio.sleep(0.10)
            return row

        try:
            if not self._onvif_zoom.available and not await self._recover_onvif_once():
                return {
                    "success": False,
                    "error": f"ONVIF absolute zoom unavailable: {self._onvif_zoom.last_error}",
                }

            wide = await probe(0.0)
            if wide is None:
                return {"success": False, "error": "Unable to establish the ONVIF wide-angle anchor."}

            target_factors = [
                f for f in (1.25, 1.5, 2.0, 2.5, 3.0)
                if self.cfg.zoom_min_factor < f <= self.cfg.zoom_max_factor + 1e-6
            ]
            if self.cfg.zoom_max_factor > self.cfg.zoom_min_factor and not target_factors:
                target_factors = [self.cfg.zoom_max_factor]

            for desired_factor in target_factors:
                desired_factor = float(desired_factor)

                # Expand outward until a physical zoom sample brackets the desired
                # factor. Reuse all prior probes so later targets need few moves.
                for _ in range(10):
                    ordered = sorted(probe_rows.values(), key=lambda row: row["onvif_normalized"])
                    upper = next(
                        (row for row in ordered if row["actual_factor"] >= desired_factor),
                        None,
                    )
                    if upper is not None:
                        break
                    last_norm = ordered[-1]["onvif_normalized"]
                    if last_norm >= 0.999:
                        break
                    next_norm = 0.015 if last_norm <= 0.0001 else min(1.0, last_norm * 1.55 + 0.010)
                    if await probe(next_norm) is None:
                        break

                ordered = sorted(probe_rows.values(), key=lambda row: row["onvif_normalized"])
                lower_candidates = [row for row in ordered if row["actual_factor"] < desired_factor]
                upper_candidates = [row for row in ordered if row["actual_factor"] >= desired_factor]
                if not upper_candidates:
                    self._record_event(
                        "zoom_calibration_unreachable",
                        desired_factor=round(desired_factor, 3),
                        highest_actual_factor=round(max(row["actual_factor"] for row in ordered), 3),
                    )
                    continue

                low = lower_candidates[-1] if lower_candidates else ordered[0]
                high = upper_candidates[0]
                low_norm = float(low["onvif_normalized"])
                high_norm = float(high["onvif_normalized"])

                # Refine the transition without assuming zoom is linear. Five
                # iterations give sub-0.5% normalized-position resolution even for
                # a fairly wide initial bracket. Plateaus/quantized Dahua zoom are
                # expected; the nearest physically observed factor wins.
                for _ in range(5):
                    if (high_norm - low_norm) <= 0.0015:
                        break
                    mid = (low_norm + high_norm) / 2.0
                    row = await probe(mid)
                    if row is None:
                        break
                    if row["actual_factor"] < desired_factor:
                        low_norm = mid
                    else:
                        high_norm = mid

                candidates = list(probe_rows.values())
                best = min(
                    candidates,
                    key=lambda row: (
                        abs(row["actual_factor"] - desired_factor),
                        abs(row["onvif_normalized"] - ((low_norm + high_norm) / 2.0)),
                    ),
                )
                chosen = {
                    "desired_factor": round(desired_factor, 3),
                    "actual_factor": round(best["actual_factor"], 3),
                    "onvif_normalized": round(best["onvif_normalized"], 6),
                    "cgi_zoom_position": round(best["cgi_zoom_position"], 3),
                }
                chosen_samples.append(chosen)
                self._record_event("zoom_calibration_target", **chosen)

            physical_points = [
                (row["actual_factor"], row["onvif_normalized"])
                for row in probe_rows.values()
                if row["actual_factor"] <= (self.cfg.zoom_max_factor * 1.20 + 0.10)
            ]
            # Explicitly preserve the measured wide-angle minimum.
            physical_points.append((1.0, 0.0))

            # Require at least two distinct physical zoom factors before replacing
            # an existing map. A failed/flat sweep must never destroy known data.
            distinct = []
            for factor, _ in sorted(physical_points):
                if not distinct or abs(factor - distinct[-1]) > 0.04:
                    distinct.append(factor)
            success = len(distinct) >= 2
            if success:
                self._zoom_map.replace_points(physical_points)

            await self.home()
            self._record_event(
                "zoom_calibration_complete",
                success=success,
                targets=len(chosen_samples),
                probes=len(probe_rows),
            )
            return {
                "success": success,
                "samples": chosen_samples,
                "probe_count": len(probe_rows),
                "probes": [
                    {
                        "actual_factor": round(row["actual_factor"], 3),
                        "onvif_normalized": round(row["onvif_normalized"], 6),
                        "cgi_zoom_position": round(row["cgi_zoom_position"], 3),
                    }
                    for row in sorted(probe_rows.values(), key=lambda item: item["onvif_normalized"])
                ],
                "zoom_mapping": self._zoom_map.public_dict(),
                "note": "Calibration probes are independent of the existing map; pan/tilt response remains continuously self-calibrating during normal tracking.",
            }
        finally:
            self._calibrating = False

