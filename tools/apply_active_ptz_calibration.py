from pathlib import Path
import re


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


tracker = read("tracker.py")

tracker = replace_once(
    tracker,
    "from ptz_phase2 import (\n    AcquisitionZonePolicy, BBoxMotionValidator, MotionMaskPolicy,\n    OnvifRetryState, SceneStabilityGate, ZoomCalibrationMap,\n)\n",
    "from ptz_phase2 import (\n    AcquisitionZonePolicy, BBoxMotionValidator, MotionMaskPolicy,\n    OnvifRetryState, SceneStabilityGate, ZoomCalibrationMap,\n)\nfrom ptz_active_calibration import (\n    ActivePtzCalibrationStore, OnvifPanTiltProbe, estimate_static_frame_shift,\n)\n",
    "active calibration import",
)

tracker = replace_once(
    tracker,
    '''def _env_float(name: str, default: float) -> float:\n    try:\n        return float(os.getenv(name, str(default)))\n    except (TypeError, ValueError):\n        return default\n''',
    '''def _env_float(name: str, default: float) -> float:\n    try:\n        return float(os.getenv(name, str(default)))\n    except (TypeError, ValueError):\n        return default\n\n\ndef _env_float_list(name: str, default: List[float], low: float, high: float) -> List[float]:\n    raw = os.getenv(name, "").strip()\n    values: List[float] = []\n    for token in raw.split(",") if raw else []:\n        try:\n            value = float(token.strip())\n        except (TypeError, ValueError):\n            continue\n        if math.isfinite(value):\n            values.append(max(low, min(high, value)))\n    result = values or [max(low, min(high, float(v))) for v in default]\n    return sorted(set(round(v, 6) for v in result))\n\n\ndef _env_int_list(name: str, default: List[int], low: int, high: int) -> List[int]:\n    raw = os.getenv(name, "").strip()\n    values: List[int] = []\n    for token in raw.split(",") if raw else []:\n        try:\n            value = int(token.strip())\n        except (TypeError, ValueError):\n            continue\n        values.append(max(low, min(high, value)))\n    return sorted(set(values or [max(low, min(high, int(v))) for v in default]))\n''',
    "list env parsers",
)

tracker = replace_once(
    tracker,
    '''        self._zoom_map = ZoomCalibrationMap(\n            os.getenv("TRACKER_ZOOM_CALIBRATION_PATH", "/app/models/tracker_zoom_calibration.json"),\n            cfg.camera_ip,\n        )\n        self._zoom_operation_target_normalized: Optional[float] = None\n''',
    '''        self._zoom_map = ZoomCalibrationMap(\n            os.getenv("TRACKER_ZOOM_CALIBRATION_PATH", "/app/models/tracker_zoom_calibration.json"),\n            cfg.camera_ip,\n        )\n        self._active_calibration = ActivePtzCalibrationStore(\n            os.getenv("TRACKER_ACTIVE_CALIBRATION_PATH", "/app/models/tracker_ptz_active_calibration.json"),\n            cfg.camera_ip,\n        )\n        self._startup_calibration_policy = os.getenv("TRACKER_CALIBRATE_ON_START", "if_missing").strip().lower()\n        if self._startup_calibration_policy not in ("off", "if_missing", "if_stale", "always"):\n            self._startup_calibration_policy = "if_missing"\n        self._calibration_scope = os.getenv("TRACKER_CALIBRATION_SCOPE", "all").strip().lower()\n        if self._calibration_scope not in ("zoom", "movedirectly", "continuous", "motion", "onvif", "all"):\n            self._calibration_scope = "all"\n        self._calibration_max_age_days = max(0.0, _env_float("TRACKER_CALIBRATION_MAX_AGE_DAYS", 30.0))\n        self._calibration_zoom_levels = _env_float_list(\n            "TRACKER_CALIBRATION_ZOOM_LEVELS", [1.0, 1.75, 2.25, 3.0], 1.0, max(1.0, cfg.zoom_max_factor)\n        )\n        self._calibration_offsets = _env_float_list(\n            "TRACKER_CALIBRATION_OFFSETS", [0.18, 0.35], 0.08, 0.60\n        )\n        self._calibration_continuous_speeds = _env_int_list(\n            "TRACKER_CALIBRATION_CONTINUOUS_SPEEDS", [1, 3, 6], 1, 8\n        )\n        self._calibration_continuous_duration = max(0.10, min(0.50, _env_float("TRACKER_CALIBRATION_CONTINUOUS_DURATION", 0.22)))\n        self._calibration_onvif_benchmark = _env_bool("TRACKER_CALIBRATION_ONVIF_BENCHMARK", True)\n        self._startup_calibration_task: Optional[asyncio.Task] = None\n        self._startup_calibration_state = "idle"\n        self._startup_calibration_last_result: Optional[dict] = None\n        self._zoom_operation_target_normalized: Optional[float] = None\n''',
    "active calibration init",
)

tracker = replace_once(
    tracker,
    '''        if self.cfg.autostart:\n            await self.start()\n\n    async def shutdown(self) -> None:\n        self._shutdown = True\n''',
    '''        if self.cfg.autostart:\n            if self._startup_calibration_policy == "off":\n                await self.start()\n            else:\n                self._startup_calibration_task = asyncio.create_task(\n                    self._startup_calibration_sequence(), name="ptz-startup-calibration"\n                )\n\n    async def shutdown(self) -> None:\n        self._shutdown = True\n        if self._startup_calibration_task is not None and not self._startup_calibration_task.done():\n            self._startup_calibration_task.cancel()\n            try:\n                await self._startup_calibration_task\n            except asyncio.CancelledError:\n                pass\n            self._startup_calibration_task = None\n        await asyncio.to_thread(self.ptz.continuous_stop)\n''',
    "startup calibration scheduling",
)

startup_helpers = r'''
    def _startup_calibration_requirements(self) -> Tuple[bool, bool]:
        policy = self._startup_calibration_policy
        if policy == "off":
            return False, False
        zoom_missing = self.cfg.autozoom and len(self._zoom_map.points()) < 2
        motion_missing = not self._active_calibration.has_motion_calibration()
        motion_stale = not self._active_calibration.is_fresh(self._calibration_max_age_days)
        if policy == "always":
            zoom_needed = self.cfg.autozoom
            motion_needed = True
        elif policy == "if_stale":
            zoom_needed = zoom_missing
            motion_needed = motion_missing or motion_stale
        else:  # if_missing
            zoom_needed = zoom_missing
            motion_needed = motion_missing

        scope = self._calibration_scope
        if scope == "zoom":
            motion_needed = False
        elif scope in ("movedirectly", "continuous", "motion", "onvif"):
            zoom_needed = False
        return zoom_needed, motion_needed

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

            zoom_needed, motion_needed = self._startup_calibration_requirements()
            if not zoom_needed and not motion_needed:
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
                elif self._calibration_scope in ("movedirectly", "continuous", "onvif"):
                    mode = self._calibration_scope
                else:
                    mode = "motion"
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
        elapsed = time.monotonic() - started
        row = {
            "zoom_factor": round(factor, 3),
            "requested_error": [round(err_x, 4), round(err_y, 4)],
            "observed_correction": [round(observed_x, 4), round(observed_y, 4)],
            "move_distance": round(requested, 4),
            "command_http_ms": int((command_done - started) * 1000),
            "motion_start_ms": None if wait["motion_start_s"] is None else int(wait["motion_start_s"] * 1000),
            "settled_ms": int(elapsed * 1000),
            "shift": shift.public_dict(),
        }
        if requested > 0.01 and math.hypot(observed_x, observed_y) > 0.03:
            self._update_move_timing_model(requested, elapsed)
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
        row = {
            "zoom_factor": round(factor, 3),
            "axis": axis,
            "speed": speed,
            "raw_sign": raw_sign,
            "correction_sign": desired_sign,
            "normalized_rate_per_s": round(rate, 4),
            "command_http_ms": int((command_done - started) * 1000),
            "settled_ms": int((time.monotonic() - started) * 1000),
            "shift": shift.public_dict(),
        }
        self._record_event("active_calibration_continuous", **row)
        return row

    async def _calibration_onvif_benchmark_run(self) -> dict:
        probe = OnvifPanTiltProbe(
            self.cfg.camera_ip,
            _env_int("TRACKER_ONVIF_PORT", 80),
            self.cfg.camera_user,
            self.cfg.camera_password,
        )
        result = await probe.initialize()
        try:
            if not self._calibration_onvif_benchmark or not result.get("relative_fov_supported"):
                return result
            prepared = await self._calibration_prepare_scene(1.0)
            if prepared is None:
                result["benchmark_error"] = "unable to prepare stable home scene"
                return result
            before, seq = prepared
            started = time.monotonic()
            ok = await probe.relative_move(0.12, 0.0)
            command_done = time.monotonic()
            if not ok:
                result["benchmark_error"] = probe.last_error or "RelativeMove failed"
                return result
            wait = await self._calibration_wait_pan_tilt_idle(max(self.cfg.ptz_operation_timeout, 5.0))
            if wait is None:
                result["benchmark_error"] = "RelativeMove did not settle"
                return result
            await asyncio.sleep(0.15)
            captured = await self._calibration_capture_frame(after_seq=seq, timeout_s=2.0)
            if captured is None:
                result["benchmark_error"] = "no stable post-ONVIF frame"
                return result
            after, _ = captured
            shift = estimate_static_frame_shift(before, after)
            result["relative_move"] = {
                "success": shift is not None,
                "command_http_ms": int((command_done - started) * 1000),
                "settled_ms": int((time.monotonic() - started) * 1000),
                "shift": None if shift is None else shift.public_dict(),
            }
            return result
        finally:
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
        run_onvif = mode in ("onvif", "motion", "all")
        safe_to_track = True
        try:
            if run_move:
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
                    pan_scale = max(0.75, min(1.25, 1.0 / max(0.20, pan_ratio)))
                    tilt_scale = max(0.75, min(1.25, 1.0 / max(0.20, tilt_ratio)))
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
                onvif_result = await self._calibration_onvif_benchmark_run()

            existing_move = self._active_calibration.move_directly()
            existing_continuous = self._active_calibration.continuous()
            move_result = existing_move
            continuous_result = existing_continuous

            if run_move and move_samples:
                move_result = {
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
            onvif_ok = (not run_onvif) or isinstance(onvif_result, dict)
            success = move_ok and continuous_ok and onvif_ok
            if success or move_samples or continuous_samples:
                self._active_calibration.replace_motion_calibration(
                    move_directly=move_result,
                    continuous=continuous_result,
                    onvif_benchmark=onvif_result,
                )
            return {
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
                "safe_to_track": safe_to_track,
            }
        except asyncio.CancelledError:
            safe_to_track = False
            raise
        except Exception as exc:
            self.logger.error("Active PTZ calibration failed: %s", exc, exc_info=True)
            return {"success": False, "mode": mode, "error": str(exc), "safe_to_track": False}
        finally:
            await asyncio.to_thread(self.ptz.continuous_stop)
            home_ok = await self._calibration_home()
            if not home_ok:
                safe_to_track = False
            self.state = "OFF"
            self._calibrating = False

    async def calibrate(self, mode: str = "zoom") -> dict:
        mode = (mode or "zoom").strip().lower()
        aliases = {"move": "movedirectly", "move_directly": "movedirectly", "pan_tilt": "motion"}
        mode = aliases.get(mode, mode)
        allowed = ("zoom", "movedirectly", "continuous", "motion", "onvif", "all")
        if mode not in allowed:
            return {"success": False, "error": f"Unknown calibration mode '{mode}'. Allowed: {', '.join(allowed)}"}
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

'''

tracker = replace_once(
    tracker,
    "    async def _wait_zoom_idle(self, timeout_s: float = 4.0) -> Optional[Tuple[float, float, float]]:\n",
    startup_helpers + "    async def _wait_zoom_idle(self, timeout_s: float = 4.0) -> Optional[Tuple[float, float, float]]:\n",
    "startup and active calibration helpers",
)

tracker = replace_once(
    tracker,
    "    async def calibrate(self) -> dict:\n        \"\"\"Manual bounded zoom calibration using independent ONVIF probes.\n",
    "    async def _calibrate_zoom(self) -> dict:\n        \"\"\"Manual bounded zoom calibration using independent ONVIF probes.\n",
    "rename zoom calibrator",
)

tracker = replace_once(
    tracker,
    '''    async def start(self) -> dict:\n        self._session_generation += 1\n''',
    '''    async def start(self) -> dict:\n        if self._calibrating:\n            return {"success": False, "error": "Calibration is in progress.", "status": self.status()}\n        self._session_generation += 1\n''',
    "start calibration guard",
)

tracker = replace_once(
    tracker,
    '''    def _hybrid_axis_speed(self, error: float) -> int:\n        magnitude = abs(error)\n        if magnitude <= self.cfg.hybrid_chase_exit_error:\n            return 0\n        span = max(0.01, self.cfg.hybrid_chase_full_speed_error - self.cfg.hybrid_chase_exit_error)\n        ratio = max(0.0, min(1.0, (magnitude - self.cfg.hybrid_chase_exit_error) / span))\n        zoom_max = max(self.cfg.hybrid_chase_min_speed, int(round(self.cfg.hybrid_chase_max_speed / math.sqrt(self._current_zoom_factor()))))\n        speed = int(round(self.cfg.hybrid_chase_min_speed + ratio * (zoom_max - self.cfg.hybrid_chase_min_speed)))\n        speed = max(self.cfg.hybrid_chase_min_speed, min(zoom_max, speed))\n        return speed if error > 0 else -speed\n''',
    '''    def _hybrid_axis_speed(self, error: float, axis: str) -> int:\n        magnitude = abs(error)\n        if magnitude <= self.cfg.hybrid_chase_exit_error:\n            return 0\n        zoom_factor = self._current_zoom_factor()\n        calibrated = self._active_calibration.choose_continuous_speed(\n            axis,\n            magnitude,\n            zoom_factor,\n            min_speed=self.cfg.hybrid_chase_min_speed,\n            max_speed=self.cfg.hybrid_chase_max_speed,\n            exit_error=self.cfg.hybrid_chase_exit_error,\n            full_speed_error=self.cfg.hybrid_chase_full_speed_error,\n        )\n        if calibrated is not None:\n            return calibrated if error > 0 else -calibrated\n        span = max(0.01, self.cfg.hybrid_chase_full_speed_error - self.cfg.hybrid_chase_exit_error)\n        ratio = max(0.0, min(1.0, (magnitude - self.cfg.hybrid_chase_exit_error) / span))\n        zoom_max = max(self.cfg.hybrid_chase_min_speed, int(round(self.cfg.hybrid_chase_max_speed / math.sqrt(zoom_factor))))\n        speed = int(round(self.cfg.hybrid_chase_min_speed + ratio * (zoom_max - self.cfg.hybrid_chase_min_speed)))\n        speed = max(self.cfg.hybrid_chase_min_speed, min(zoom_max, speed))\n        return speed if error > 0 else -speed\n''',
    "calibrated hybrid speed",
)

tracker = replace_once(
    tracker,
    '''        pan_speed = self.cfg.hybrid_chase_pan_sign * self._hybrid_axis_speed(err_x)\n        tilt_speed = -self._hybrid_axis_speed(err_y)\n''',
    '''        pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)\n        tilt_sign = self._active_calibration.continuous_sign("tilt", -1)\n        pan_speed = pan_sign * self._hybrid_axis_speed(err_x, "pan")\n        tilt_speed = tilt_sign * self._hybrid_axis_speed(err_y, "tilt")\n''',
    "calibrated hybrid signs",
)

tracker = replace_once(
    tracker,
    '''                "zoom_calibration": self._zoom_map.public_dict(),\n                "onvif_retry": self._onvif_retry.public_dict(now),\n''',
    '''                "zoom_calibration": self._zoom_map.public_dict(),\n                "active_calibration": self._active_calibration.public_dict(),\n                "startup_calibration": {\n                    "policy": self._startup_calibration_policy,\n                    "scope": self._calibration_scope,\n                    "state": self._startup_calibration_state,\n                    "task_running": bool(self._startup_calibration_task is not None and not self._startup_calibration_task.done()),\n                    "last_result": self._startup_calibration_last_result,\n                },\n                "onvif_retry": self._onvif_retry.public_dict(now),\n''',
    "calibration status",
)

tracker = replace_once(
    tracker,
    '''                "history_size": self.cfg.history_size,\n            },\n''',
    '''                "history_size": self.cfg.history_size,\n                "calibrate_on_start": self._startup_calibration_policy,\n                "calibration_scope": self._calibration_scope,\n                "calibration_max_age_days": self._calibration_max_age_days,\n                "calibration_zoom_levels": self._calibration_zoom_levels,\n                "calibration_offsets": self._calibration_offsets,\n                "calibration_continuous_speeds": self._calibration_continuous_speeds,\n                "calibration_continuous_duration": self._calibration_continuous_duration,\n                "calibration_onvif_benchmark": self._calibration_onvif_benchmark,\n            },\n''',
    "calibration config status",
)

write("tracker.py", tracker)

app = read("app.py")
app = replace_once(
    app,
    '''@app.post("/v1/tracker/calibrate")\nasync def tracker_calibrate():\n    return await _require_tracker().calibrate()\n''',
    '''@app.post("/v1/tracker/calibrate")\nasync def tracker_calibrate(mode: str = "zoom"):\n    return await _require_tracker().calibrate(mode=mode)\n''',
    "calibration endpoint mode",
)
write("app.py", app)

dockerfile = read("Dockerfile")
dockerfile = replace_once(
    dockerfile,
    "COPY ptz_phase2.py /app/ptz_phase2.py\n",
    "COPY ptz_phase2.py /app/ptz_phase2.py\nCOPY ptz_active_calibration.py /app/ptz_active_calibration.py\n",
    "Dockerfile active calibration copy",
)
write("Dockerfile", dockerfile)

compose = read("docker-compose.example.yml")
compose = replace_once(
    compose,
    '''      TRACKER_ONVIF_RETRY_BASE: "15.0"\n      TRACKER_ONVIF_RETRY_MAX: "300.0"\n''',
    '''      TRACKER_ONVIF_RETRY_BASE: "15.0"\n      TRACKER_ONVIF_RETRY_MAX: "300.0"\n\n      # Active camera calibration. The default performs the motion exercise only\n      # when persisted calibration is missing; "if_stale" and "always" are also supported.\n      TRACKER_CALIBRATE_ON_START: "if_missing"\n      TRACKER_CALIBRATION_SCOPE: "all"\n      TRACKER_ACTIVE_CALIBRATION_PATH: "/app/models/tracker_ptz_active_calibration.json"\n      TRACKER_CALIBRATION_MAX_AGE_DAYS: "30"\n      TRACKER_CALIBRATION_ZOOM_LEVELS: "1.0,1.75,2.25,3.0"\n      TRACKER_CALIBRATION_OFFSETS: "0.18,0.35"\n      TRACKER_CALIBRATION_CONTINUOUS_SPEEDS: "1,3,6"\n      TRACKER_CALIBRATION_CONTINUOUS_DURATION: "0.22"\n      TRACKER_CALIBRATION_ONVIF_BENCHMARK: "true"\n''',
    "compose active calibration settings",
)
write("docker-compose.example.yml", compose)

readme = read("README.md")
needle = "| `TRACKER_AUTOSTART` | `false` | Start tracking automatically when the app starts. |\n"
if needle not in readme:
    raise RuntimeError("README autostart row missing")
readme = readme.replace(
    needle,
    needle
    + "| `TRACKER_CALIBRATE_ON_START` | `if_missing` | Startup calibration policy: `off`, `if_missing`, `if_stale`, or `always`. Full calibration moves the camera before autotracking begins. |\n"
    + "| `TRACKER_CALIBRATION_SCOPE` | `all` | Startup/manual active calibration scope: `zoom`, `movedirectly`, `continuous`, `motion`, `onvif`, or `all`. |\n"
    + "| `TRACKER_ACTIVE_CALIBRATION_PATH` | `/app/models/tracker_ptz_active_calibration.json` | Persistent native/ONVIF motion-response calibration. |\n"
    + "| `TRACKER_CALIBRATION_MAX_AGE_DAYS` | `30` | Age threshold used by the `if_stale` startup policy. |\n"
    + "| `TRACKER_CALIBRATION_ZOOM_LEVELS` | `1.0,1.75,2.25,3.0` | Optical zoom factors sampled by active pan/tilt calibration. |\n"
    + "| `TRACKER_CALIBRATION_OFFSETS` | `0.18,0.35` | Normalized moveDirectly offsets used to learn response and timing. |\n"
    + "| `TRACKER_CALIBRATION_CONTINUOUS_SPEEDS` | `1,3,6` | Native continuous PTZ speeds sampled during calibration. |\n"
    + "| `TRACKER_CALIBRATION_CONTINUOUS_DURATION` | `0.22` | Seconds each bounded continuous test pulse runs. |\n"
    + "| `TRACKER_CALIBRATION_ONVIF_BENCHMARK` | `true` | Probe ONVIF PTZ spaces and benchmark a tiny FOV-relative move when supported. |\n",
    1,
)
readme += '''\n\n### PTZ active calibration\n\n`POST /v1/tracker/calibrate?mode=all` now calibrates the optical zoom map, native Dahua `moveDirectly` response/timing, native continuous pan/tilt speed response, and ONVIF pan/tilt capabilities. The motion calibration always runs with tracking stopped, repeatedly returns to the configured home preset, uses settled video frames to measure actual scene displacement, and returns home before releasing the camera.\n\nWith `TRACKER_AUTOSTART=true` and the default `TRACKER_CALIBRATE_ON_START=if_missing`, the API and health endpoint come up normally while a guarded startup calibration runs. Autotracking starts only after calibration completes. Persisted calibration prevents that full camera exercise from repeating on ordinary restarts. Use `if_stale` to refresh it after `TRACKER_CALIBRATION_MAX_AGE_DAYS`, `always` to recalibrate every startup, or `off` to disable startup calibration.\n'''
write("README.md", readme)

contract = '''from pathlib import Path\nimport unittest\n\nROOT = Path(__file__).resolve().parents[1]\nTRACKER = (ROOT / "tracker.py").read_text()\nAPP = (ROOT / "app.py").read_text()\nDOCKER = (ROOT / "Dockerfile").read_text()\nCOMPOSE = (ROOT / "docker-compose.example.yml").read_text()\n\n\nclass ActiveCalibrationContracts(unittest.TestCase):\n    def test_active_calibration_runtime_is_wired(self):\n        self.assertIn("ActivePtzCalibrationStore", TRACKER)\n        self.assertIn("estimate_static_frame_shift", TRACKER)\n        self.assertIn("_calibration_move_directly_sample", TRACKER)\n        self.assertIn("_calibration_continuous_sample", TRACKER)\n        self.assertIn("OnvifPanTiltProbe", TRACKER)\n        self.assertIn("choose_continuous_speed", TRACKER)\n        self.assertIn("continuous_sign", TRACKER)\n\n    def test_startup_policy_is_nonblocking_and_guarded(self):\n        self.assertIn('TRACKER_CALIBRATE_ON_START', TRACKER)\n        self.assertIn('"if_missing"', TRACKER)\n        self.assertIn('asyncio.create_task(\\n                    self._startup_calibration_sequence()', TRACKER)\n        self.assertIn('if self._calibrating:', TRACKER)\n        self.assertIn('Calibration is in progress.', TRACKER)\n\n    def test_api_and_container_include_active_calibration(self):\n        self.assertIn('async def tracker_calibrate(mode: str = "zoom")', APP)\n        self.assertIn('COPY ptz_active_calibration.py /app/ptz_active_calibration.py', DOCKER)\n        for name in (\n            "TRACKER_CALIBRATE_ON_START",\n            "TRACKER_CALIBRATION_SCOPE",\n            "TRACKER_ACTIVE_CALIBRATION_PATH",\n            "TRACKER_CALIBRATION_MAX_AGE_DAYS",\n            "TRACKER_CALIBRATION_ZOOM_LEVELS",\n            "TRACKER_CALIBRATION_OFFSETS",\n            "TRACKER_CALIBRATION_CONTINUOUS_SPEEDS",\n            "TRACKER_CALIBRATION_CONTINUOUS_DURATION",\n            "TRACKER_CALIBRATION_ONVIF_BENCHMARK",\n        ):\n            self.assertIn(name, TRACKER)\n            self.assertIn(name, COMPOSE)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'''
write("tests/test_active_ptz_contracts.py", contract)

print("active PTZ calibration integration applied")
