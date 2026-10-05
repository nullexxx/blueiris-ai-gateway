from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


tracker_path = Path("tracker.py")
tracker = tracker_path.read_text()

tracker = replace_once(
    tracker,
    '''    # Bounded, deliberately conservative auto-zoom. Camera reports 5.12 at
    # full-wide and 128 at full-tele (25x). Default auto-tracking ceiling is 6x.
    autozoom: bool = True
    zoom_wide_position: float = 5.12
    zoom_min_factor: float = 1.0
    zoom_max_factor: float = 6.0
    zoom_target_min: float = 0.18
    zoom_target_max: float = 0.42
    zoom_in_step_ms: int = 90
    zoom_out_step_ms: int = 140
    zoom_cooldown: float = 1.0
''',
    '''    # Evidence-gated auto-zoom. Zoom-in is intentionally rare: the subject
    # must be genuinely small, confidently detected, close to center, and remain
    # eligible for several consecutive observations. Zoom-out is deliberately
    # easier so the tracker can recover context quickly.
    autozoom: bool = True
    zoom_wide_position: float = 5.12
    zoom_min_factor: float = 1.0
    zoom_max_factor: float = 3.0
    zoom_target_min: float = 0.12
    zoom_target_max: float = 0.42
    zoom_in_min_conf: float = 0.65
    zoom_in_confirm_frames: int = 8
    zoom_in_max_error: float = 0.18
    zoom_in_step_ms: int = 90
    zoom_out_step_ms: int = 140
    zoom_cooldown: float = 3.0
    zoom_out_cooldown: float = 0.75
    zoom_noop_backoff: float = 5.0
    zoom_noop_epsilon: float = 0.03
''',
    "zoom config defaults",
)

tracker = replace_once(
    tracker,
    '''            autozoom=_env_bool("TRACKER_AUTOZOOM", True),
            zoom_wide_position=_env_float("TRACKER_ZOOM_WIDE_POSITION", 5.12),
            zoom_min_factor=_env_float("TRACKER_ZOOM_MIN_FACTOR", 1.0),
            zoom_max_factor=_env_float("TRACKER_ZOOM_MAX_FACTOR", 6.0),
            zoom_target_min=_env_float("TRACKER_ZOOM_TARGET_MIN", 0.18),
            zoom_target_max=_env_float("TRACKER_ZOOM_TARGET_MAX", 0.42),
            zoom_in_step_ms=_env_int("TRACKER_ZOOM_IN_STEP_MS", 90),
            zoom_out_step_ms=_env_int("TRACKER_ZOOM_OUT_STEP_MS", 140),
            zoom_cooldown=_env_float("TRACKER_ZOOM_COOLDOWN", 1.0),
''',
    '''            autozoom=_env_bool("TRACKER_AUTOZOOM", True),
            zoom_wide_position=_env_float("TRACKER_ZOOM_WIDE_POSITION", 5.12),
            zoom_min_factor=_env_float("TRACKER_ZOOM_MIN_FACTOR", 1.0),
            zoom_max_factor=_env_float("TRACKER_ZOOM_MAX_FACTOR", 3.0),
            zoom_target_min=_env_float("TRACKER_ZOOM_TARGET_MIN", 0.12),
            zoom_target_max=_env_float("TRACKER_ZOOM_TARGET_MAX", 0.42),
            zoom_in_min_conf=_env_float("TRACKER_ZOOM_IN_MIN_CONF", 0.65),
            zoom_in_confirm_frames=_env_int("TRACKER_ZOOM_IN_CONFIRM_FRAMES", 8),
            zoom_in_max_error=_env_float("TRACKER_ZOOM_IN_MAX_ERROR", 0.18),
            zoom_in_step_ms=_env_int("TRACKER_ZOOM_IN_STEP_MS", 90),
            zoom_out_step_ms=_env_int("TRACKER_ZOOM_OUT_STEP_MS", 140),
            zoom_cooldown=_env_float("TRACKER_ZOOM_COOLDOWN", 3.0),
            zoom_out_cooldown=_env_float("TRACKER_ZOOM_OUT_COOLDOWN", 0.75),
            zoom_noop_backoff=_env_float("TRACKER_ZOOM_NOOP_BACKOFF", 5.0),
            zoom_noop_epsilon=_env_float("TRACKER_ZOOM_NOOP_EPSILON", 0.03),
''',
    "zoom env parsing",
)

tracker = replace_once(
    tracker,
    '''        cfg.zoom_target_min = max(0.03, min(0.80, cfg.zoom_target_min))
        cfg.zoom_target_max = max(cfg.zoom_target_min + 0.02, min(0.95, cfg.zoom_target_max))
        cfg.zoom_in_step_ms = max(30, min(500, cfg.zoom_in_step_ms))
        cfg.zoom_out_step_ms = max(30, min(700, cfg.zoom_out_step_ms))
        cfg.zoom_cooldown = max(0.25, min(10.0, cfg.zoom_cooldown))
''',
    '''        cfg.zoom_target_min = max(0.03, min(0.80, cfg.zoom_target_min))
        cfg.zoom_target_max = max(cfg.zoom_target_min + 0.02, min(0.95, cfg.zoom_target_max))
        cfg.zoom_in_min_conf = max(cfg.hold_conf, min(0.99, cfg.zoom_in_min_conf))
        cfg.zoom_in_confirm_frames = max(1, min(60, cfg.zoom_in_confirm_frames))
        cfg.zoom_in_max_error = max(0.02, min(0.50, cfg.zoom_in_max_error))
        cfg.zoom_in_step_ms = max(30, min(500, cfg.zoom_in_step_ms))
        cfg.zoom_out_step_ms = max(30, min(700, cfg.zoom_out_step_ms))
        cfg.zoom_cooldown = max(0.25, min(30.0, cfg.zoom_cooldown))
        cfg.zoom_out_cooldown = max(0.10, min(10.0, cfg.zoom_out_cooldown))
        cfg.zoom_noop_backoff = max(0.5, min(30.0, cfg.zoom_noop_backoff))
        cfg.zoom_noop_epsilon = max(0.001, min(1.0, cfg.zoom_noop_epsilon))
''',
    "zoom clamps",
)

tracker = replace_once(
    tracker,
    '''            "zoom_target_min": self.zoom_target_min,
            "zoom_target_max": self.zoom_target_max,
            "zoom_in_step_ms": self.zoom_in_step_ms,
            "zoom_out_step_ms": self.zoom_out_step_ms,
            "zoom_cooldown": self.zoom_cooldown,
''',
    '''            "zoom_target_min": self.zoom_target_min,
            "zoom_target_max": self.zoom_target_max,
            "zoom_in_min_conf": self.zoom_in_min_conf,
            "zoom_in_confirm_frames": self.zoom_in_confirm_frames,
            "zoom_in_max_error": self.zoom_in_max_error,
            "zoom_in_step_ms": self.zoom_in_step_ms,
            "zoom_out_step_ms": self.zoom_out_step_ms,
            "zoom_cooldown": self.zoom_cooldown,
            "zoom_out_cooldown": self.zoom_out_cooldown,
            "zoom_noop_backoff": self.zoom_noop_backoff,
            "zoom_noop_epsilon": self.zoom_noop_epsilon,
''',
    "zoom public config",
)

tracker = replace_once(
    tracker,
    '''        self._last_target_span: Optional[float] = None
        self._last_zoom_position: Optional[float] = None
        self._last_zoom_command_at = 0.0
''',
    '''        self._last_target_span: Optional[float] = None
        self._last_zoom_position: Optional[float] = None
        self._last_zoom_command_at = 0.0
        self._zoom_in_candidate_frames = 0
        self._zoom_in_candidate_last_at = 0.0
        self._zoom_suppressed_until = 0.0
        self._zoom_operation_start_position: Optional[float] = None
''',
    "zoom runtime state init",
)

tracker = replace_once(
    tracker,
    '''        self._last_target_span = None
        self._last_move_point = None
        self._last_zoom_position = None
        self._last_zoom_command_at = 0.0
        self._ptz_operation = None
''',
    '''        self._last_target_span = None
        self._last_move_point = None
        self._last_zoom_position = None
        self._last_zoom_command_at = 0.0
        self._zoom_in_candidate_frames = 0
        self._zoom_in_candidate_last_at = 0.0
        self._zoom_suppressed_until = 0.0
        self._zoom_operation_start_position = None
        self._ptz_operation = None
''',
    "zoom runtime state reset",
)

tracker = replace_once(
    tracker,
    '''        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        if self.target is not None:
            self.target.clear_velocity()
''',
    '''        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        # Any physical camera action breaks the run of stationary observations
        # required before another zoom-in is allowed.
        self._zoom_in_candidate_frames = 0
        self._zoom_in_candidate_last_at = 0.0
        if self.target is not None:
            self.target.clear_velocity()
''',
    "reset zoom evidence on ptz operation",
)

tracker = replace_once(
    tracker,
    '''        elapsed_ms = int(max(0.0, now - self._ptz_operation_started_at) * 1000)
        move_distance = self._pending_move_distance if kind == "move" else None
''',
    '''        elapsed_ms = int(max(0.0, now - self._ptz_operation_started_at) * 1000)
        move_distance = self._pending_move_distance if kind == "move" else None
        zoom_before = self._zoom_operation_start_position if kind == "zoom" else None
        zoom_after = position[2] if kind == "zoom" and position is not None else None
        zoom_delta = (
            None
            if zoom_before is None or zoom_after is None
            else zoom_after - zoom_before
        )
        zoom_noop = bool(
            kind == "zoom"
            and zoom_delta is not None
            and abs(zoom_delta) <= self.cfg.zoom_noop_epsilon
        )
        if zoom_noop:
            self._zoom_suppressed_until = max(
                self._zoom_suppressed_until,
                now + self.cfg.zoom_noop_backoff,
            )
''',
    "zoom noop detection",
)

tracker = replace_once(
    tracker,
    '''        if kind == "move":
            event_fields.update(
                move_distance=None if move_distance is None else round(move_distance, 3),
                move_timing_samples=len(self._move_timing_samples),
                move_eta_model_ready=self._move_eta_model_ready,
                move_eta_intercept_s=round(self._move_eta_intercept, 3),
                move_eta_slope_s_per_norm=round(self._move_eta_slope, 3),
            )
        self._record_event(
''',
    '''        if kind == "move":
            event_fields.update(
                move_distance=None if move_distance is None else round(move_distance, 3),
                move_timing_samples=len(self._move_timing_samples),
                move_eta_model_ready=self._move_eta_model_ready,
                move_eta_intercept_s=round(self._move_eta_intercept, 3),
                move_eta_slope_s_per_norm=round(self._move_eta_slope, 3),
            )
        elif kind == "zoom":
            event_fields.update(
                zoom_before=None if zoom_before is None else round(zoom_before, 3),
                zoom_after=None if zoom_after is None else round(zoom_after, 3),
                zoom_delta=None if zoom_delta is None else round(zoom_delta, 3),
                zoom_noop=zoom_noop,
            )
            if zoom_noop:
                self._record_event(
                    "zoom_step_noop",
                    zoom_before=round(zoom_before, 3),
                    zoom_after=round(zoom_after, 3),
                    backoff_ms=int(self.cfg.zoom_noop_backoff * 1000),
                )
        self._record_event(
''',
    "zoom noop history",
)

tracker = replace_once(
    tracker,
    '''        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        self._pending_move_distance = None
''',
    '''        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        self._pending_move_distance = None
        self._zoom_operation_start_position = None
''',
    "clear zoom operation start",
)

old_zoom_block = '''        # Zoom is intentionally secondary to pan/tilt and happens only while the
        # subject is already centered and no camera operation is in flight.
        if not self.cfg.autozoom:
            return
        if (now - self._last_zoom_command_at) < self.cfg.zoom_cooldown:
            return

        if self._last_zoom_position is None:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not self._session_valid(generation):
                return
            if not status:
                return
            self._last_camera_status = status
            self._last_camera_status_at = checked_at
            position = self.ptz.position_from_status(status)
            if position is not None:
                self._last_camera_position = position
                self._last_zoom_position = position[2]
            if self.ptz.pan_tilt_reported_idle(status) is False:
                return

        if self._last_zoom_position is None:
            return

        direction: Optional[str] = None
        duration_ms = 0
        zoom_in_guard = max(0.50, self.cfg.zoom_max_position * 0.05)

        # If a timed zoom step ever lands beyond a configured bound, correct it
        # before making any target-size-based decision.
        if self._last_zoom_position > (self.cfg.zoom_max_position + 0.05):
            direction = "out"
            duration_ms = self.cfg.zoom_out_step_ms
        elif self._last_zoom_position < (self.cfg.zoom_min_position - 0.05):
            direction = "in"
            duration_ms = self.cfg.zoom_in_step_ms
        elif (
            target_span < self.cfg.zoom_target_min
            and self._last_zoom_position < (self.cfg.zoom_max_position - zoom_in_guard)
        ):
            direction = "in"
            duration_ms = self.cfg.zoom_in_step_ms
        elif (
            target_span > self.cfg.zoom_target_max
            and self._last_zoom_position > (self.cfg.zoom_min_position + 0.25)
        ):
            direction = "out"
            duration_ms = self.cfg.zoom_out_step_ms

        if direction is None:
            return

        before = self._last_zoom_position
        t0 = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)
        t1 = time.monotonic()
        if not self._session_valid(generation):
            return
        self._last_zoom_command_at = t1
        if ok:
            self.ptz_commands += 1
            self.zoom_commands += 1
            self._last_zoom_position = None
            self._begin_ptz_operation("zoom", seq, t1)
            self.state = "PTZ_MOVING"
            self._record_event(
                "zoom_step",
                label=self.target.label,
                direction=direction,
                duration_ms=duration_ms,
                target_span=round(target_span, 3),
                zoom_before=round(before, 3),
                min_position=round(self.cfg.zoom_min_position, 3),
                max_position=round(self.cfg.zoom_max_position, 3),
                http_ms=int((t1 - t0) * 1000),
            )
        else:
            self._record_event("zoom_step_failed", direction=direction, error=self.ptz.last_error)
'''

new_zoom_block = '''        # Zoom is subordinate to tracking. Zoom-in is evidence-gated so the lens
        # only moves when the current framing is clearly wasting useful pixels.
        # Zoom-out stays intentionally easier so the camera can regain context.
        if not self.cfg.autozoom:
            return

        zoom_in_eligible = (
            target_span < self.cfg.zoom_target_min
            and self.target.confidence >= self.cfg.zoom_in_min_conf
            and abs(err_x) <= self.cfg.zoom_in_max_error
            and abs(err_y) <= self.cfg.zoom_in_max_error
        )
        if (now - self._zoom_in_candidate_last_at) > 0.20:
            self._zoom_in_candidate_frames = 0
        if zoom_in_eligible:
            self._zoom_in_candidate_frames += 1
            self._zoom_in_candidate_last_at = now
        else:
            self._zoom_in_candidate_frames = 0
            self._zoom_in_candidate_last_at = 0.0

        if self._last_zoom_position is None:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not self._session_valid(generation):
                return
            if not status:
                return
            self._last_camera_status = status
            self._last_camera_status_at = checked_at
            position = self.ptz.position_from_status(status)
            if position is not None:
                self._last_camera_position = position
                self._last_zoom_position = position[2]
            if self.ptz.pan_tilt_reported_idle(status) is False:
                return

        if self._last_zoom_position is None:
            return

        since_zoom = now - self._last_zoom_command_at
        direction: Optional[str] = None
        duration_ms = 0
        zoom_in_guard = max(0.50, self.cfg.zoom_max_position * 0.05)

        # Bounds and context recovery take precedence over target-size zoom-in.
        if (
            self._last_zoom_position > (self.cfg.zoom_max_position + 0.05)
            and since_zoom >= self.cfg.zoom_out_cooldown
        ):
            direction = "out"
            duration_ms = self.cfg.zoom_out_step_ms
        elif (
            self._last_zoom_position < (self.cfg.zoom_min_position - 0.05)
            and since_zoom >= self.cfg.zoom_cooldown
            and now >= self._zoom_suppressed_until
        ):
            direction = "in"
            duration_ms = self.cfg.zoom_in_step_ms
        elif (
            self._zoom_in_candidate_frames >= self.cfg.zoom_in_confirm_frames
            and target_span < self.cfg.zoom_target_min
            and since_zoom >= self.cfg.zoom_cooldown
            and now >= self._zoom_suppressed_until
            and self._last_zoom_position < (self.cfg.zoom_max_position - zoom_in_guard)
        ):
            direction = "in"
            duration_ms = self.cfg.zoom_in_step_ms
        elif (
            target_span > self.cfg.zoom_target_max
            and since_zoom >= self.cfg.zoom_out_cooldown
            and self._last_zoom_position > (self.cfg.zoom_min_position + 0.25)
        ):
            direction = "out"
            duration_ms = self.cfg.zoom_out_step_ms

        if direction is None:
            return

        before = self._last_zoom_position
        if direction == "in":
            self._zoom_in_candidate_frames = 0
            self._zoom_in_candidate_last_at = 0.0
        self._zoom_operation_start_position = before
        t0 = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)
        t1 = time.monotonic()
        if not self._session_valid(generation):
            return
        self._last_zoom_command_at = t1
        if ok:
            self.ptz_commands += 1
            self.zoom_commands += 1
            self._last_zoom_position = None
            self._begin_ptz_operation("zoom", seq, t1)
            self.state = "PTZ_MOVING"
            self._record_event(
                "zoom_step",
                label=self.target.label,
                direction=direction,
                duration_ms=duration_ms,
                target_span=round(target_span, 3),
                confidence=round(self.target.confidence, 3),
                zoom_before=round(before, 3),
                min_position=round(self.cfg.zoom_min_position, 3),
                max_position=round(self.cfg.zoom_max_position, 3),
                http_ms=int((t1 - t0) * 1000),
            )
        else:
            self._zoom_operation_start_position = None
            self._record_event("zoom_step_failed", direction=direction, error=self.ptz.last_error)
'''
tracker = replace_once(tracker, old_zoom_block, new_zoom_block, "conservative zoom controller")
tracker_path.write_text(tracker)

compose_path = Path("docker-compose.example.yml")
compose = compose_path.read_text()
compose = replace_once(
    compose,
    '''      # Conservative bounded auto-zoom
      TRACKER_AUTOZOOM: "true"
      TRACKER_ZOOM_WIDE_POSITION: "5.12"
      TRACKER_ZOOM_MIN_FACTOR: "1.0"
      TRACKER_ZOOM_MAX_FACTOR: "6.0"
      TRACKER_ZOOM_TARGET_MIN: "0.18"
      TRACKER_ZOOM_TARGET_MAX: "0.42"
      TRACKER_ZOOM_IN_STEP_MS: "90"
      TRACKER_ZOOM_OUT_STEP_MS: "140"
      TRACKER_ZOOM_COOLDOWN: "1.0"
''',
    '''      # Evidence-gated auto-zoom: prefer stable framing over lens activity.
      TRACKER_AUTOZOOM: "true"
      TRACKER_ZOOM_WIDE_POSITION: "5.12"
      TRACKER_ZOOM_MIN_FACTOR: "1.0"
      TRACKER_ZOOM_MAX_FACTOR: "3.0"
      TRACKER_ZOOM_TARGET_MIN: "0.12"
      TRACKER_ZOOM_TARGET_MAX: "0.42"
      TRACKER_ZOOM_IN_MIN_CONF: "0.65"
      TRACKER_ZOOM_IN_CONFIRM_FRAMES: "8"
      TRACKER_ZOOM_IN_MAX_ERROR: "0.18"
      TRACKER_ZOOM_IN_STEP_MS: "90"
      TRACKER_ZOOM_OUT_STEP_MS: "140"
      TRACKER_ZOOM_COOLDOWN: "3.0"
      TRACKER_ZOOM_OUT_COOLDOWN: "0.75"
      TRACKER_ZOOM_NOOP_BACKOFF: "5.0"
      TRACKER_ZOOM_NOOP_EPSILON: "0.03"
''',
    "compose zoom block",
)
compose_path.write_text(compose)

readme_path = Path("README.md")
readme = readme_path.read_text()
readme = readme.replace(
    "- **Conservative auto-zoom** with configurable minimum/maximum optical zoom bounds.",
    "- **Evidence-gated conservative auto-zoom** that requires a small, confident, centered target across multiple observations before zooming in.",
)
readme = readme.replace(
    "| `TRACKER_PTZ_STATUS_POLL_INTERVAL` | `0.12` | Seconds between camera PTZ status polls. Example Compose uses `0.06`. |",
    "| `TRACKER_PTZ_STATUS_POLL_INTERVAL` | `0.12` | Seconds between camera PTZ status polls. |",
)
readme = readme.replace(
    "| `TRACKER_ZOOM_MAX_FACTOR` | `6.0` | Maximum tracking zoom factor. |\n"
    "| `TRACKER_ZOOM_TARGET_MIN` | `0.18` | Zoom in when a centered target is smaller than this span. |\n"
    "| `TRACKER_ZOOM_TARGET_MAX` | `0.42` | Zoom out when a centered target is larger than this span. |\n"
    "| `TRACKER_ZOOM_IN_STEP_MS` | `90` | Timed zoom-in pulse length. |\n"
    "| `TRACKER_ZOOM_OUT_STEP_MS` | `140` | Timed zoom-out pulse length. |\n"
    "| `TRACKER_ZOOM_COOLDOWN` | `1.0` | Minimum seconds between zoom commands. |",
    "| `TRACKER_ZOOM_MAX_FACTOR` | `3.0` | Maximum tracking zoom factor; deliberately capped to preserve context. |\n"
    "| `TRACKER_ZOOM_TARGET_MIN` | `0.12` | A target must be smaller than this span before zoom-in can even qualify. |\n"
    "| `TRACKER_ZOOM_TARGET_MAX` | `0.42` | Zoom out when the target is larger than this span. |\n"
    "| `TRACKER_ZOOM_IN_MIN_CONF` | `0.65` | Minimum confidence required for zoom-in. |\n"
    "| `TRACKER_ZOOM_IN_CONFIRM_FRAMES` | `8` | Consecutive qualifying observations required before zoom-in. |\n"
    "| `TRACKER_ZOOM_IN_MAX_ERROR` | `0.18` | Maximum normalized center error per axis allowed for zoom-in. |\n"
    "| `TRACKER_ZOOM_IN_STEP_MS` | `90` | Timed zoom-in pulse length. |\n"
    "| `TRACKER_ZOOM_OUT_STEP_MS` | `140` | Timed zoom-out pulse length. |\n"
    "| `TRACKER_ZOOM_COOLDOWN` | `3.0` | Minimum seconds between zoom-in commands. |\n"
    "| `TRACKER_ZOOM_OUT_COOLDOWN` | `0.75` | Faster cooldown for zoom-out/context recovery. |\n"
    "| `TRACKER_ZOOM_NOOP_BACKOFF` | `5.0` | Suppress further zoom-in after a completed zoom pulse that changed no reported zoom position. |\n"
    "| `TRACKER_ZOOM_NOOP_EPSILON` | `0.03` | Maximum reported zoom-position delta treated as a no-op. |",
)
readme_path.write_text(readme)

test_path = Path("tests/test_static_contracts.py")
test = test_path.read_text()
needle = '''    def test_face_and_debug_optimizations(self):
'''
addition = '''    def test_conservative_zoom_policy(self):
        tracker = (ROOT / "tracker.py").read_text()
        compose = (ROOT / "docker-compose.example.yml").read_text()
        self.assertIn('zoom_max_factor: float = 3.0', tracker)
        self.assertIn('zoom_target_min: float = 0.12', tracker)
        self.assertIn('TRACKER_ZOOM_IN_MIN_CONF', tracker)
        self.assertIn('TRACKER_ZOOM_IN_CONFIRM_FRAMES', tracker)
        self.assertIn('TRACKER_ZOOM_IN_MAX_ERROR', tracker)
        self.assertIn('zoom_step_noop', tracker)
        self.assertIn('_zoom_suppressed_until', tracker)
        self.assertIn('TRACKER_ZOOM_MAX_FACTOR: "3.0"', compose)
        self.assertIn('TRACKER_ZOOM_TARGET_MIN: "0.12"', compose)
        self.assertIn('TRACKER_ZOOM_COOLDOWN: "3.0"', compose)
        self.assertIn('TRACKER_ZOOM_NOOP_BACKOFF: "5.0"', compose)

'''
if addition.strip() not in test:
    if needle not in test:
        raise RuntimeError("test insertion point not found")
    test = test.replace(needle, addition + needle, 1)
test_path.write_text(test)
