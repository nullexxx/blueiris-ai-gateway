from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def insert_before(text: str, marker: str, addition: str, label: str) -> str:
    return replace_once(text, marker, addition + marker, label)


def replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    si = text.find(start)
    if si < 0:
        raise RuntimeError(f"{label}: start marker missing")
    ei = text.find(end, si + len(start))
    if ei < 0:
        raise RuntimeError(f"{label}: end marker missing")
    if text.find(start, si + 1) >= 0 and text.find(start, si + 1) < ei:
        raise RuntimeError(f"{label}: ambiguous start marker")
    return text[:si] + replacement + text[ei:]


# ---------------------------------------------------------------------------
# ptz_enhancements.py: expose exact normalized ONVIF absolute zoom.
# ---------------------------------------------------------------------------
helper = read("ptz_enhancements.py")
start = "    async def set_factor(self, factor: float, max_optical_zoom: float, speed: float = 1.0) -> bool:\n"
end = "    async def close(self) -> None:\n"
replacement = '''    async def set_normalized(self, normalized: float, speed: float = 1.0) -> bool:\n        \"\"\"Move to an exact normalized point in the camera's advertised absolute zoom range.\"\"\"\n        if not self.available or self.ptz is None or self.profile_token is None:\n            return False\n        if self.zoom_min is None or self.zoom_max is None:\n            return False\n        try:\n            normalized = _bounded(float(normalized), 0.0, 1.0)\n            target = self.zoom_min + normalized * (self.zoom_max - self.zoom_min)\n            request = self.ptz.create_type(\"AbsoluteMove\")\n            request.ProfileToken = self.profile_token\n            request.Speed = {\"Zoom\": float(speed)}\n            request.Position = {\"Zoom\": float(target)}\n            await self.ptz.AbsoluteMove(request)\n            self.last_error = None\n            self.failures = 0\n            return True\n        except Exception as exc:\n            self.last_error = str(exc)\n            self.failures += 1\n            if self.failures >= 2:\n                self.available = False\n            return False\n\n    async def set_factor(self, factor: float, max_optical_zoom: float, speed: float = 1.0) -> bool:\n        max_optical_zoom = max(1.01, float(max_optical_zoom))\n        factor = _bounded(float(factor), 1.0, max_optical_zoom)\n        normalized = (factor - 1.0) / (max_optical_zoom - 1.0)\n        return await self.set_normalized(normalized, speed=speed)\n\n'''
helper = replace_between(helper, start, end, replacement, "ONVIF normalized zoom")
write("ptz_enhancements.py", helper)


# ---------------------------------------------------------------------------
# tracker.py integration.
# ---------------------------------------------------------------------------
tracker = read("tracker.py")
tracker = replace_once(
    tracker,
    '''from ptz_enhancements import (\n    CalibrationStore, CameraMotionEstimator, OnvifAbsoluteZoom,\n    TargetHistory, frame_sharpness, zoom_bucket,\n)\n''',
    '''from ptz_enhancements import (\n    CalibrationStore, CameraMotionEstimator, OnvifAbsoluteZoom,\n    TargetHistory, frame_sharpness, zoom_bucket,\n)\nfrom ptz_phase2 import (\n    AcquisitionZonePolicy, BBoxMotionValidator, MotionMaskPolicy,\n    OnvifRetryState, SceneStabilityGate, ZoomCalibrationMap,\n)\n''',
    "phase2 imports",
)

tracker = replace_once(
    tracker,
    "        self._smart_motion = CameraMotionEstimator(120, 12)\n",
    '''        self._smart_motion = CameraMotionEstimator(120, 12)\n        self._acquisition_zones = AcquisitionZonePolicy(\n            os.getenv(\"TRACKER_ACQUIRE_ZONES_JSON\", \"\"),\n            os.getenv(\"TRACKER_IGNORE_ZONES_JSON\", \"\"),\n        )\n        self._motion_masks = MotionMaskPolicy(os.getenv(\"TRACKER_MOTION_MASKS_JSON\", \"\"))\n        self._bbox_motion = BBoxMotionValidator()\n        self._scene_stability = SceneStabilityGate(\n            stable_frames=max(1, _env_int(\"TRACKER_SCENE_STABLE_FRAMES\", 2)),\n            max_wait_s=_env_float(\"TRACKER_SCENE_STABLE_MAX_WAIT\", 0.60),\n            flow_threshold_norm=_env_float(\"TRACKER_SCENE_STABLE_FLOW\", 0.012),\n        )\n        self._scene_stable_ready = True\n        self._last_velocity_geometry = None\n''',
    "phase2 init perception",
)

tracker = replace_once(
    tracker,
    '''        self._camera_max_optical_zoom = max(\n            cfg.zoom_max_factor, _env_float(\"TRACKER_CAMERA_MAX_OPTICAL_ZOOM\", 25.0)\n        )\n        self._zoom_control_active = \"cgi_timed_fallback\"\n''',
    '''        self._camera_max_optical_zoom = max(\n            cfg.zoom_max_factor, _env_float(\"TRACKER_CAMERA_MAX_OPTICAL_ZOOM\", 25.0)\n        )\n        self._onvif_retry = OnvifRetryState(\n            _env_float(\"TRACKER_ONVIF_RETRY_BASE\", 15.0),\n            _env_float(\"TRACKER_ONVIF_RETRY_MAX\", 300.0),\n        )\n        self._onvif_recovery_task: Optional[asyncio.Task] = None\n        self._zoom_map = ZoomCalibrationMap(\n            os.getenv(\"TRACKER_ZOOM_CALIBRATION_PATH\", \"/app/models/tracker_zoom_calibration.json\"),\n            cfg.camera_ip,\n        )\n        self._zoom_operation_target_normalized: Optional[float] = None\n        self._zoom_operation_target_factor: Optional[float] = None\n        self._calibrating = False\n        self._zoom_control_active = \"cgi_timed_fallback\"\n''',
    "phase2 init onvif",
)

tracker = replace_once(
    tracker,
    '''        self._smart_history.clear()\n        self._smart_motion.reset()\n        self._camera_motion = None\n''',
    '''        self._smart_history.clear()\n        self._smart_motion.reset()\n        self._bbox_motion.reset()\n        self._scene_stability.clear()\n        self._scene_stable_ready = True\n        self._last_velocity_geometry = None\n        self._camera_motion = None\n''',
    "phase2 reset",
)

tracker = replace_once(
    tracker,
    '''            if onvif_ok:\n                self.logger.info(\"PTZ tracker zoom: ONVIF AbsoluteMove enabled\")\n            else:\n                self.logger.warning(\"ONVIF absolute zoom unavailable; CGI fallback remains active: %s\", self._onvif_zoom.last_error)\n''',
    '''            if onvif_ok:\n                self._onvif_retry.success()\n                self.logger.info(\"PTZ tracker zoom: ONVIF AbsoluteMove enabled\")\n            else:\n                delay = self._onvif_retry.failure(time.monotonic(), self._onvif_zoom.last_error)\n                self.logger.warning(\n                    \"ONVIF absolute zoom unavailable; CGI fallback remains active and re-probe is scheduled in %.1fs: %s\",\n                    delay, self._onvif_zoom.last_error,\n                )\n''',
    "initial ONVIF retry",
)

methods_marker = "    async def start(self) -> dict:\n"
methods = '''    def _onvif_optional_enabled(self) -> bool:\n        return self.cfg.autozoom and os.getenv(\"TRACKER_ZOOM_CONTROL_MODE\", \"auto\").strip().lower() != \"cgi\"\n\n    def _schedule_onvif_retry(self, error: Optional[str]) -> None:\n        if not self._onvif_optional_enabled():\n            return\n        delay = self._onvif_retry.failure(time.monotonic(), error)\n        self._zoom_control_active = \"cgi_timed_fallback\"\n        self._record_event(\"onvif_zoom_retry_scheduled\", retry_in_s=round(delay, 1), error=error)\n\n    async def _recover_onvif_once(self) -> bool:\n        if not self._onvif_optional_enabled():\n            return False\n        try:\n            await self._onvif_zoom.close()\n            ok = await asyncio.wait_for(self._onvif_zoom.initialize(), timeout=5.0)\n        except Exception as exc:\n            ok = False\n            self._onvif_zoom.last_error = str(exc)\n        if ok:\n            self._onvif_retry.success()\n            self._zoom_control_active = \"onvif_absolute\"\n            self._record_event(\"onvif_zoom_recovered\")\n            self.logger.info(\"ONVIF absolute zoom recovered; exact zoom control restored\")\n            return True\n        self._schedule_onvif_retry(self._onvif_zoom.last_error)\n        return False\n\n    async def _onvif_recovery_worker(self) -> None:\n        try:\n            await self._recover_onvif_once()\n        finally:\n            self._onvif_recovery_task = None\n\n    def _maybe_start_onvif_recovery(self, now: float) -> None:\n        if not self._onvif_optional_enabled() or self._onvif_zoom.available:\n            return\n        if self._onvif_recovery_task is not None and not self._onvif_recovery_task.done():\n            return\n        if self._onvif_retry.due(now):\n            self._onvif_recovery_task = asyncio.create_task(\n                self._onvif_recovery_worker(), name=\"ptz-onvif-recovery\"\n            )\n\n    async def _wait_zoom_idle(self, timeout_s: float = 4.0) -> Optional[Tuple[float, float, float]]:\n        deadline = time.monotonic() + max(0.5, float(timeout_s))\n        previous = None\n        stable_polls = 0\n        while time.monotonic() < deadline:\n            status = await asyncio.to_thread(self.ptz.get_status)\n            if status:\n                position = self.ptz.position_from_status(status)\n                idle = self.ptz.zoom_reported_idle(status)\n                stable = self._position_stable(previous, position, \"zoom\") if previous is not None else None\n                if position is not None:\n                    previous = position\n                if idle is True and stable is not False:\n                    stable_polls += 1\n                elif idle is None and stable is True:\n                    stable_polls += 1\n                else:\n                    stable_polls = 0\n                if stable_polls >= 2 and position is not None:\n                    return position\n            await asyncio.sleep(self.cfg.ptz_status_poll_interval)\n        return None\n\n    async def calibrate(self) -> dict:\n        \"\"\"Manual, bounded zoom calibration. Pan/tilt response continues to learn passively.\"\"\"\n        if self.active:\n            return {\"success\": False, \"error\": \"Stop tracking before calibration.\"}\n        if self._calibrating:\n            return {\"success\": False, \"error\": \"Calibration is already running.\"}\n        if not self.cfg.autozoom:\n            return {\"success\": False, \"error\": \"Autozoom is disabled.\"}\n        self._calibrating = True\n        samples = []\n        try:\n            if not self._onvif_zoom.available and not await self._recover_onvif_once():\n                return {\"success\": False, \"error\": f\"ONVIF absolute zoom unavailable: {self._onvif_zoom.last_error}\"}\n            factors = [f for f in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0) if self.cfg.zoom_min_factor <= f <= self.cfg.zoom_max_factor]\n            if self.cfg.zoom_min_factor not in factors:\n                factors.insert(0, self.cfg.zoom_min_factor)\n            if self.cfg.zoom_max_factor not in factors:\n                factors.append(self.cfg.zoom_max_factor)\n            for desired_factor in sorted(set(round(float(f), 3) for f in factors)):\n                normalized = self._zoom_map.estimate_normalized(desired_factor, self._camera_max_optical_zoom)\n                try:\n                    ok = await asyncio.wait_for(self._onvif_zoom.set_normalized(normalized), timeout=1.5)\n                except asyncio.TimeoutError:\n                    self._schedule_onvif_retry(\"ONVIF absolute zoom calibration command timed out\")\n                    break\n                if not ok:\n                    self._schedule_onvif_retry(self._onvif_zoom.last_error)\n                    break\n                position = await self._wait_zoom_idle(self.cfg.ptz_operation_timeout)\n                if position is None:\n                    self._record_event(\"zoom_calibration_timeout\", desired_factor=desired_factor, normalized=round(normalized, 6))\n                    continue\n                actual_factor = max(1.0, position[2] / max(0.001, self.cfg.zoom_wide_position))\n                self._zoom_map.record(actual_factor, normalized)\n                sample = {\n                    \"desired_factor\": desired_factor,\n                    \"actual_factor\": round(actual_factor, 3),\n                    \"onvif_normalized\": round(normalized, 6),\n                    \"cgi_zoom_position\": round(position[2], 3),\n                }\n                samples.append(sample)\n                self._record_event(\"zoom_calibration_sample\", **sample)\n                await asyncio.sleep(0.15)\n            await self.home()\n            success = len(samples) >= 2\n            self._record_event(\"zoom_calibration_complete\", success=success, samples=len(samples))\n            return {\n                \"success\": success,\n                \"samples\": samples,\n                \"zoom_mapping\": self._zoom_map.public_dict(),\n                \"note\": \"Pan/tilt spatial response remains continuously self-calibrating during normal tracking.\",\n            }\n        finally:\n            self._calibrating = False\n\n'''
tracker = insert_before(tracker, methods_marker, methods, "phase2 methods")

tracker = replace_once(
    tracker,
    '''                if not self.active:\n                    await asyncio.sleep(0.05)\n                    continue\n''',
    '''                self._maybe_start_onvif_recovery(time.monotonic())\n                if not self.active:\n                    await asyncio.sleep(0.05)\n                    continue\n''',
    "background ONVIF recovery",
)

tracker = replace_once(
    tracker,
    '''        motion_active = self._ptz_operation is not None or self._hybrid_chase_active or seq < self._post_motion_release_seq\n        self._camera_motion = self._smart_motion.update(\n            frame, [d.bbox for d in detections], active=motion_active, use_homography=self._ptz_operation == \"zoom\"\n        )\n        self._update_frame_quality(frame, now)\n''',
    '''        motion_active = (\n            self._ptz_operation is not None\n            or self._hybrid_chase_active\n            or seq < self._post_motion_release_seq\n            or not self._scene_stable_ready\n        )\n        motion_boxes = [d.bbox for d in detections] + self._motion_masks.boxes(frame.shape)\n        self._camera_motion = self._smart_motion.update(\n            frame, motion_boxes, active=motion_active, use_homography=self._ptz_operation == \"zoom\"\n        )\n        self._update_frame_quality(frame, now)\n        if self._ptz_operation is None and not self._scene_stable_ready:\n            was_ready = self._scene_stable_ready\n            self._scene_stable_ready = self._scene_stability.observe(\n                now, self._camera_motion, self._frame_sharpness_ok, frame.shape\n            )\n            if self._scene_stable_ready and not was_ready:\n                self._record_event(\n                    \"post_move_scene_stable\",\n                    reason=self._scene_stability.last_reason,\n                    flow_norm=self._scene_stability.last_flow_norm,\n                    timed_out=self._scene_stability.timed_out,\n                )\n''',
    "scene stability observation",
)

tracker = replace_once(
    tracker,
    '''            candidates = [d for d in detections if d.confidence >= self.cfg.acquire_conf]\n''',
    '''            candidates = [\n                d for d in detections\n                if d.confidence >= self.cfg.acquire_conf\n                and self._acquisition_zones.allows(d.center, frame.shape)\n            ]\n''',
    "acquisition zones",
)

tracker = replace_once(
    tracker,
    '''                acquire_hits=1,\n            )\n            self._smart_history.clear()\n''',
    '''                acquire_hits=1,\n            )\n            self._bbox_motion.reset(chosen.bbox, now)\n            self._last_velocity_geometry = None\n            self._smart_history.clear()\n''',
    "bbox validator acquisition reset",
)

old_velocity = '''            ptz_ready = self._ptz_action_ready(seq)\n            rebasing_velocity = (\n                self._ptz_operation is None\n                and not self._hybrid_chase_active\n                and ptz_ready\n                and self._velocity_rebase_required\n                and self._frame_sharpness_ok\n            )\n            velocity_learning_allowed = (\n                self._ptz_operation is None\n                and not self._hybrid_chase_active\n                and ptz_ready\n                and not self._velocity_rebase_required\n                and self._frame_sharpness_ok\n            )\n            self.target.update(\n                matched,\n                now,\n                update_velocity=velocity_learning_allowed,\n                min_velocity_sample_s=self.cfg.velocity_min_sample_ms / 1000.0,\n            )\n'''
new_velocity = '''            ptz_ready = self._ptz_action_ready(seq)\n            rebasing_velocity = (\n                self._ptz_operation is None\n                and not self._hybrid_chase_active\n                and ptz_ready\n                and self._velocity_rebase_required\n                and self._frame_sharpness_ok\n            )\n            geometry_valid = True\n            if (\n                self._ptz_operation is None\n                and not self._hybrid_chase_active\n                and ptz_ready\n                and not self._velocity_rebase_required\n                and self._frame_sharpness_ok\n            ):\n                geometry = self._bbox_motion.validate(matched.bbox, now, frame.shape)\n                self._last_velocity_geometry = geometry.public_dict()\n                geometry_valid = geometry.valid\n                if not geometry_valid:\n                    self._record_event(\n                        \"velocity_geometry_rejected\",\n                        reason=geometry.reason,\n                        geometry=self._last_velocity_geometry,\n                    )\n            else:\n                self._bbox_motion.reset(matched.bbox, now)\n                self._last_velocity_geometry = None\n            velocity_learning_allowed = (\n                self._ptz_operation is None\n                and not self._hybrid_chase_active\n                and ptz_ready\n                and not self._velocity_rebase_required\n                and self._frame_sharpness_ok\n                and geometry_valid\n            )\n            self.target.update(\n                matched,\n                now,\n                update_velocity=velocity_learning_allowed,\n                min_velocity_sample_s=self.cfg.velocity_min_sample_ms / 1000.0,\n            )\n'''
tracker = replace_once(tracker, old_velocity, new_velocity, "bbox velocity validation")

tracker = replace_once(
    tracker,
    '''        self._last_ptz_stopped_at = now\n        self._sharpness_wait_logged = False\n''',
    '''        self._last_ptz_stopped_at = now\n        self._sharpness_wait_logged = False\n        self._scene_stability.reset(now)\n        self._scene_stable_ready = False\n''',
    "scene gate reset",
)

tracker = replace_once(
    tracker,
    '''            if zoom_noop:\n                self._record_event(\n                    \"zoom_step_noop\",\n                    zoom_before=round(zoom_before, 3),\n                    zoom_after=round(zoom_after, 3),\n                    backoff_ms=int(self.cfg.zoom_noop_backoff * 1000),\n                )\n''',
    '''            if (\n                not zoom_noop\n                and zoom_after is not None\n                and self._zoom_operation_target_normalized is not None\n                and self._zoom_control_active == \"onvif_absolute\"\n            ):\n                actual_factor = max(1.0, zoom_after / max(0.001, self.cfg.zoom_wide_position))\n                self._zoom_map.record(actual_factor, self._zoom_operation_target_normalized)\n                event_fields.update(\n                    zoom_calibrated_factor=round(actual_factor, 3),\n                    onvif_normalized=round(self._zoom_operation_target_normalized, 6),\n                )\n            if zoom_noop:\n                self._record_event(\n                    \"zoom_step_noop\",\n                    zoom_before=round(zoom_before, 3),\n                    zoom_after=round(zoom_after, 3),\n                    backoff_ms=int(self.cfg.zoom_noop_backoff * 1000),\n                )\n''',
    "passive zoom calibration",
)

tracker = replace_once(
    tracker,
    '''        self._zoom_operation_start_position = None\n''',
    '''        self._zoom_operation_start_position = None\n        self._zoom_operation_target_normalized = None\n        self._zoom_operation_target_factor = None\n''',
    "clear zoom target calibration state",
)

tracker = replace_once(
    tracker,
    '''    def _ptz_action_ready(self, seq: int) -> bool:\n        return self._ptz_operation is None and seq >= self._post_motion_release_seq\n''',
    '''    def _ptz_action_ready(self, seq: int) -> bool:\n        return (\n            self._ptz_operation is None\n            and seq >= self._post_motion_release_seq\n            and self._scene_stable_ready\n        )\n''',
    "scene-aware PTZ ready",
)

old_zoom = '''        t0 = time.monotonic()\n        zoom_control = \"cgi_timed_fallback\"\n        ok = False\n        if self._onvif_zoom.available:\n            ok = await self._onvif_zoom.set_factor(desired_factor, self._camera_max_optical_zoom)\n            if ok:\n                zoom_control = \"onvif_absolute\"\n        if not ok:\n            ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)\n        self._zoom_control_active = zoom_control\n'''
new_zoom = '''        t0 = time.monotonic()\n        zoom_control = \"cgi_timed_fallback\"\n        ok = False\n        onvif_normalized = None\n        if self._onvif_zoom.available:\n            onvif_normalized = self._zoom_map.estimate_normalized(\n                desired_factor, self._camera_max_optical_zoom\n            )\n            try:\n                ok = await asyncio.wait_for(\n                    self._onvif_zoom.set_normalized(onvif_normalized), timeout=1.5\n                )\n            except asyncio.TimeoutError:\n                self._schedule_onvif_retry(\"ONVIF absolute zoom command timed out\")\n                self._record_event(\"zoom_step_deferred\", direction=direction, reason=\"onvif_timeout\")\n                return\n            if ok:\n                zoom_control = \"onvif_absolute\"\n                self._onvif_retry.success()\n                self._zoom_operation_target_normalized = onvif_normalized\n                self._zoom_operation_target_factor = desired_factor\n            else:\n                self._schedule_onvif_retry(self._onvif_zoom.last_error)\n        if not ok:\n            self._zoom_operation_target_normalized = None\n            self._zoom_operation_target_factor = None\n            ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)\n        self._zoom_control_active = zoom_control\n'''
tracker = replace_once(tracker, old_zoom, new_zoom, "calibrated zoom command")

tracker = replace_once(
    tracker,
    '''                control=zoom_control, desired_factor=round(desired_factor, 2),\n''',
    '''                control=zoom_control, desired_factor=round(desired_factor, 2),\n                onvif_normalized=None if onvif_normalized is None else round(onvif_normalized, 6),\n''',
    "zoom event normalized target",
)

tracker = replace_once(
    tracker,
    '''                \"calibration\": self._calibration.public_dict(),\n''',
    '''                \"calibration\": self._calibration.public_dict(),\n                \"zoom_calibration\": self._zoom_map.public_dict(),\n                \"onvif_retry\": self._onvif_retry.public_dict(now),\n                \"acquisition_zones\": self._acquisition_zones.public_dict(),\n                \"motion_masks\": self._motion_masks.public_dict(),\n                \"scene_stability\": self._scene_stability.public_dict(),\n                \"velocity_geometry\": self._last_velocity_geometry,\n                \"calibrating\": self._calibrating,\n''',
    "phase2 status",
)

camera_status_marker = '''    async def camera_status(self) -> dict:\n        return await asyncio.to_thread(self.ptz.get_status)\n'''
# Methods were inserted earlier, but keep calibration endpoint method discoverable next to status API semantics.
if camera_status_marker not in tracker:
    raise RuntimeError("camera status marker missing")

write("tracker.py", tracker)


# ---------------------------------------------------------------------------
# app.py: manual calibration endpoint.
# ---------------------------------------------------------------------------
app = read("app.py")
app = replace_once(
    app,
    '''@app.get(\"/v1/tracker/camera-status\")\nasync def tracker_camera_status():\n    return await _require_tracker().camera_status()\n\n\n''',
    '''@app.get(\"/v1/tracker/camera-status\")\nasync def tracker_camera_status():\n    return await _require_tracker().camera_status()\n\n\n@app.post(\"/v1/tracker/calibrate\")\nasync def tracker_calibrate():\n    return await _require_tracker().calibrate()\n\n\n''',
    "calibration endpoint",
)
write("app.py", app)


# ---------------------------------------------------------------------------
# Dockerfile / workflows / compose / README.
# ---------------------------------------------------------------------------
dockerfile = read("Dockerfile")
dockerfile = replace_once(
    dockerfile,
    "COPY ptz_enhancements.py /app/ptz_enhancements.py\n",
    "COPY ptz_enhancements.py /app/ptz_enhancements.py\nCOPY ptz_phase2.py /app/ptz_phase2.py\n",
    "Docker phase2 copy",
)
write("Dockerfile", dockerfile)

build = read(".github/workflows/build.yml")
build = replace_once(
    build,
    '      - "ptz_enhancements.py"\n',
    '      - "ptz_enhancements.py"\n      - "ptz_phase2.py"\n',
    "GHCR phase2 trigger",
)
write(".github/workflows/build.yml", build)

validation = read(".github/workflows/test.yml")
validation = replace_once(
    validation,
    "      - name: Compile Python sources\n        run: python -m py_compile app.py tracker.py ptz_enhancements.py\n",
    "      - name: Compile Python sources\n        run: python -m py_compile app.py tracker.py ptz_enhancements.py ptz_phase2.py\n",
    "validation compile phase2",
)
write(".github/workflows/test.yml", validation)

compose = read("docker-compose.example.yml")
compose = replace_once(
    compose,
    '''      TRACKER_CALIBRATION_PATH: \"/app/models/tracker_calibration.json\"\n\n      # Target-loss / home behavior\n''',
    '''      TRACKER_CALIBRATION_PATH: \"/app/models/tracker_calibration.json\"\n      TRACKER_ZOOM_CALIBRATION_PATH: \"/app/models/tracker_zoom_calibration.json\"\n      TRACKER_ONVIF_RETRY_BASE: \"15.0\"\n      TRACKER_ONVIF_RETRY_MAX: \"300.0\"\n\n      # Frigate-inspired acquisition/motion filtering. JSON polygon coordinates are normalized 0..1.\n      # Empty arrays preserve the existing behavior. These gates apply only to initial target acquisition.\n      TRACKER_ACQUIRE_ZONES_JSON: \"[]\"\n      TRACKER_IGNORE_ZONES_JSON: \"[]\"\n      TRACKER_MOTION_MASKS_JSON: \"[]\"\n\n      # Do not trust the first decoded frame just because the PTZ reports IDLE.\n      # Require sharp, low-global-motion frames, but release after a bounded timeout.\n      TRACKER_SCENE_STABLE_FRAMES: \"2\"\n      TRACKER_SCENE_STABLE_MAX_WAIT: \"0.60\"\n      TRACKER_SCENE_STABLE_FLOW: \"0.012\"\n\n      # Target-loss / home behavior\n''',
    "compose phase2 vars",
)
write("docker-compose.example.yml", compose)

readme = read("README.md")
readme = replace_once(
    readme,
    "- `GET /v1/tracker/camera-status` — Raw camera PTZ status from the Dahua/Amcrest CGI.\n",
    "- `GET /v1/tracker/camera-status` — Raw camera PTZ status from the Dahua/Amcrest CGI.\n- `POST /v1/tracker/calibrate` — Manual bounded ONVIF zoom calibration; requires tracking to be stopped.\n",
    "README calibrate endpoint",
)
write("README.md", readme)


# ---------------------------------------------------------------------------
# Regression/contract tests.
# ---------------------------------------------------------------------------
test_path = Path("tests/test_ptz_phase2_contracts.py")
test_path.write_text('''import ast\nfrom pathlib import Path\nimport unittest\n\nROOT = Path(__file__).resolve().parents[1]\nTRACKER = (ROOT / "tracker.py").read_text()\nHELPER = (ROOT / "ptz_enhancements.py").read_text()\nPHASE2 = (ROOT / "ptz_phase2.py").read_text()\nAPP = (ROOT / "app.py").read_text()\nCOMPOSE = (ROOT / "docker-compose.example.yml").read_text()\nDOCKER = (ROOT / "Dockerfile").read_text()\nBUILD = (ROOT / ".github/workflows/build.yml").read_text()\n\nclass PtzPhase2Contracts(unittest.TestCase):\n    def test_sources_parse(self):\n        for source in (TRACKER, HELPER, PHASE2, APP):\n            ast.parse(source)\n\n    def test_onvif_is_optional_and_self_healing(self):\n        self.assertIn("class OnvifRetryState", PHASE2)\n        self.assertIn("_maybe_start_onvif_recovery", TRACKER)\n        self.assertIn("cgi_timed_fallback", TRACKER)\n        self.assertIn("async def set_normalized", HELPER)\n\n    def test_zoom_mapping_is_persisted_and_used(self):\n        self.assertIn("class ZoomCalibrationMap", PHASE2)\n        self.assertIn("estimate_normalized", TRACKER)\n        self.assertIn("self._zoom_map.record", TRACKER)\n        self.assertIn("TRACKER_ZOOM_CALIBRATION_PATH", COMPOSE)\n\n    def test_bbox_geometry_and_scene_stability_guard_control(self):\n        self.assertIn("class BBoxMotionValidator", PHASE2)\n        self.assertIn("velocity_geometry_rejected", TRACKER)\n        self.assertIn("class SceneStabilityGate", PHASE2)\n        self.assertIn("post_move_scene_stable", TRACKER)\n        self.assertIn("and self._scene_stable_ready", TRACKER)\n\n    def test_zones_and_motion_masks_are_optional(self):\n        self.assertIn("class AcquisitionZonePolicy", PHASE2)\n        self.assertIn("class MotionMaskPolicy", PHASE2)\n        self.assertIn("self._acquisition_zones.allows", TRACKER)\n        self.assertIn("self._motion_masks.boxes", TRACKER)\n        for key in ("TRACKER_ACQUIRE_ZONES_JSON", "TRACKER_IGNORE_ZONES_JSON", "TRACKER_MOTION_MASKS_JSON"):\n            self.assertIn(key, COMPOSE)\n\n    def test_manual_calibration_is_bounded_and_not_automatic(self):\n        self.assertIn("async def calibrate", TRACKER)\n        self.assertIn('/v1/tracker/calibrate', APP)\n        self.assertIn("Stop tracking before calibration", TRACKER)\n        self.assertNotIn("await self.calibrate()", TRACKER)\n\n    def test_container_and_ci_track_phase2_module(self):\n        self.assertIn("COPY ptz_phase2.py /app/ptz_phase2.py", DOCKER)\n        self.assertIn('      - "ptz_phase2.py"', BUILD)\n\nif __name__ == "__main__":\n    unittest.main()\n''')

print("PTZ phase 2 patch applied")
