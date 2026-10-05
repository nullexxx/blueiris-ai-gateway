from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing expected block: {label}")
    return text.replace(old, new, 1)


def replace_regex(text: str, pattern: str, repl: str, label: str, flags=re.S) -> str:
    out, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise SystemExit(f"expected one regex match for {label}, got {count}")
    return out


# ---------------------------------------------------------------------------
# app.py
# ---------------------------------------------------------------------------
p = Path("app.py")
app = p.read_text()

app = replace_once(
    app,
    'RECOVERY_GRACE_PERIOD = float(os.getenv("RECOVERY_GRACE_PERIOD", "4.0"))\n',
    'RECOVERY_GRACE_PERIOD = float(os.getenv("RECOVERY_GRACE_PERIOD", "4.0"))\nMAX_FACE_EMBEDDINGS_PER_USER = max(1, min(100, int(os.getenv("MAX_FACE_EMBEDDINGS_PER_USER", "20"))))\n',
    "face embedding cap config",
)

app = replace_once(
    app,
    '    if entry.is_pt and HALF_PRECISION:\n        kwargs["half"] = True\n',
    '    if entry.is_pt and HALF_PRECISION:\n        kwargs["quantize"] = 16\n',
    "tracker quantize",
)

app = replace_regex(
    app,
    r'def sync_face_register_embedding\(img: Image\.Image\).*?\n\ndef sync_face_recognize',
    '''def sync_face_register_embedding(img: Image.Image) -> Tuple[Optional[torch.Tensor], int]:
    """Extract exactly one normalized face embedding without repeating MTCNN detection."""
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    face_count = 0 if boxes is None else len(boxes)
    if face_count != 1:
        return None, face_count

    faces = face_detector.extract(img, boxes, None)
    if faces is None or len(faces) != 1:
        return None, face_count

    face_tensor = faces[0].unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        emb = face_recognizer(face_tensor)
        return F.normalize(emb, p=2, dim=1).cpu(), face_count


def sync_face_recognize''',
    "single-pass face registration",
)

app = replace_regex(
    app,
    r'def sync_face_recognize\(img: Image\.Image, min_conf: float\) -> List\[dict\]:.*?\n\n    return predictions',
    '''def sync_face_recognize(img: Image.Image, min_conf: float) -> List[dict]:
    """Detect faces once, extract crops, and compare embeddings against enrollment data."""
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    faces = face_detector.extract(img, boxes, None)
    if faces is None or len(faces) == 0:
        return []

    faces = faces.to("cuda:0")
    with torch.no_grad():
        embeddings = face_recognizer(faces)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    if _face_cache_dirty:
        _rebuild_face_matrix()

    if _face_matrix_cache is not None:
        sims = embeddings @ _face_matrix_cache.T
        best_sims, best_idx = sims.max(dim=1)
    else:
        best_sims = best_idx = None

    predictions = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        if best_sims is not None:
            sim = float(best_sims[i].item())
            matched_id = _face_labels_cache[int(best_idx[i].item())] if sim >= min_conf else "unknown"
        else:
            sim = 0.0
            matched_id = "unknown"

        predictions.append({
            "confidence": round(sim, 2),
            "userid": matched_id,
            "label": matched_id,
            "x_min": int(max(0, x1)),
            "y_min": int(max(0, y1)),
            "x_max": int(x2),
            "y_max": int(y2),
        })

    return predictions''',
    "single-pass face recognition",
)

app = replace_once(
    app,
    '''                embedding = await run_gpu_job(
                    sync_face_register_embedding,
                    img,
                    label=f"Face registration for '{target_id}'",
                    timeout=INFERENCE_TIMEOUT,
                    enqueue=True,
                )
''',
    '''                embedding, face_count = await run_gpu_job(
                    sync_face_register_embedding,
                    img,
                    label=f"Face registration for '{target_id}'",
                    timeout=INFERENCE_TIMEOUT,
                    enqueue=True,
                )
''',
    "face registration result tuple",
)

app = replace_once(
    app,
    '''            if embedding is None:
                return {"success": False, "error": "No face detected in provided image."}

            async with face_state_lock:
                registered_faces.setdefault(target_id, []).append(embedding)
                _face_cache_dirty = True
                await asyncio.to_thread(save_faces_db, _face_db_snapshot())
''',
    '''            if embedding is None:
                if face_count == 0:
                    return {"success": False, "error": "No face detected in provided image."}
                return {"success": False, "error": f"Expected exactly one face for enrollment; detected {face_count}."}

            async with face_state_lock:
                embeddings = registered_faces.setdefault(target_id, [])
                embeddings.append(embedding)
                if len(embeddings) > MAX_FACE_EMBEDDINGS_PER_USER:
                    del embeddings[:-MAX_FACE_EMBEDDINGS_PER_USER]
                _face_cache_dirty = True
                await asyncio.to_thread(save_faces_db, _face_db_snapshot())
''',
    "bounded face enrollment",
)

app = app.replace('debug = tracker_service.debug_jpeg()', 'debug = await tracker_service.debug_jpeg()')
p.write_text(app)


# ---------------------------------------------------------------------------
# tracker.py
# ---------------------------------------------------------------------------
p = Path("tracker.py")
tracker = p.read_text()

tracker = replace_once(
    tracker,
    '    rtsp_url: str = ""\n\n    model_name: str = "yolo11l"\n',
    '    rtsp_url: str = ""\n    rtsp_open_timeout: float = 5.0\n    rtsp_read_timeout: float = 5.0\n\n    model_name: str = "yolo11l"\n',
    "RTSP timeout config fields",
)

tracker = replace_once(
    tracker,
    '    hybrid_chase_settle_frames: int = 2\n\n    # Native PTZ operation tracking.',
    '    hybrid_chase_settle_frames: int = 2\n    hybrid_divergence_frames: int = 3\n    hybrid_divergence_growth: float = 0.05\n\n    # Native PTZ operation tracking.',
    "hybrid divergence fields",
)

tracker = replace_once(
    tracker,
    '    ptz_http_timeout: float = 0.75\n    frame_stale_timeout: float = 1.0\n',
    '    ptz_http_timeout: float = 0.75\n    move_failure_backoff_base: float = 0.25\n    move_failure_backoff_max: float = 1.0\n    move_failure_stop_after: int = 5\n    home_retry_attempts: int = 3\n    home_retry_delay: float = 0.75\n    frame_stale_timeout: float = 1.0\n',
    "PTZ retry fields",
)

tracker = replace_once(
    tracker,
    '            rtsp_url=os.getenv("TRACKER_RTSP_URL", "").strip(),\n',
    '            rtsp_url=os.getenv("TRACKER_RTSP_URL", "").strip(),\n            rtsp_open_timeout=max(1.0, _env_float("TRACKER_RTSP_OPEN_TIMEOUT", 5.0)),\n            rtsp_read_timeout=max(1.0, _env_float("TRACKER_RTSP_READ_TIMEOUT", 5.0)),\n',
    "RTSP env parsing",
)

tracker = replace_once(
    tracker,
    '            hybrid_chase_settle_frames=_env_int("TRACKER_HYBRID_CHASE_SETTLE_FRAMES", 2),\n',
    '            hybrid_chase_settle_frames=_env_int("TRACKER_HYBRID_CHASE_SETTLE_FRAMES", 2),\n            hybrid_divergence_frames=max(2, _env_int("TRACKER_HYBRID_DIVERGENCE_FRAMES", 3)),\n            hybrid_divergence_growth=max(0.01, _env_float("TRACKER_HYBRID_DIVERGENCE_GROWTH", 0.05)),\n',
    "hybrid divergence env parsing",
)

tracker = replace_once(
    tracker,
    '            ptz_http_timeout=_env_float("TRACKER_PTZ_HTTP_TIMEOUT", 0.75),\n',
    '            ptz_http_timeout=_env_float("TRACKER_PTZ_HTTP_TIMEOUT", 0.75),\n            move_failure_backoff_base=max(0.05, _env_float("TRACKER_MOVE_FAILURE_BACKOFF_BASE", 0.25)),\n            move_failure_backoff_max=max(0.10, _env_float("TRACKER_MOVE_FAILURE_BACKOFF_MAX", 1.0)),\n            move_failure_stop_after=max(1, _env_int("TRACKER_MOVE_FAILURE_STOP_AFTER", 5)),\n            home_retry_attempts=max(1, _env_int("TRACKER_HOME_RETRY_ATTEMPTS", 3)),\n            home_retry_delay=max(0.10, _env_float("TRACKER_HOME_RETRY_DELAY", 0.75)),\n',
    "PTZ retry env parsing",
)

tracker = replace_once(
    tracker,
    '            "rtsp_subtype": self.rtsp_subtype,\n',
    '            "rtsp_subtype": self.rtsp_subtype,\n            "rtsp_open_timeout": self.rtsp_open_timeout,\n            "rtsp_read_timeout": self.rtsp_read_timeout,\n',
    "RTSP public config",
)

tracker = replace_once(
    tracker,
    '            "hybrid_chase_settle_frames": self.hybrid_chase_settle_frames,\n',
    '            "hybrid_chase_settle_frames": self.hybrid_chase_settle_frames,\n            "hybrid_divergence_frames": self.hybrid_divergence_frames,\n            "hybrid_divergence_growth": self.hybrid_divergence_growth,\n',
    "hybrid divergence public config",
)

tracker = replace_once(
    tracker,
    '            "ptz_http_timeout": self.ptz_http_timeout,\n',
    '            "ptz_http_timeout": self.ptz_http_timeout,\n            "move_failure_backoff_base": self.move_failure_backoff_base,\n            "move_failure_backoff_max": self.move_failure_backoff_max,\n            "move_failure_stop_after": self.move_failure_stop_after,\n            "home_retry_attempts": self.home_retry_attempts,\n            "home_retry_delay": self.home_retry_delay,\n',
    "PTZ retry public config",
)

tracker = replace_regex(
    tracker,
    r'class LatestFrameCapture:.*?\n\nclass AmcrestPTZ:',
    '''class LatestFrameCapture:
    """Continuously drains RTSP and retains only the newest decoded frame."""

    def __init__(self, url: str, open_timeout_s: float = 5.0, read_timeout_s: float = 5.0):
        self.url = url
        self.open_timeout_s = max(1.0, float(open_timeout_s))
        self.read_timeout_s = max(1.0, float(read_timeout_s))
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.last_error: Optional[str] = None
        self.reconnects = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ptz-tracker-rtsp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            return self._frame, self._seq, self._frame_time

    def _open_capture(self):
        params = []
        open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_prop is not None:
            params.extend([open_prop, int(self.open_timeout_s * 1000)])
        if read_prop is not None:
            params.extend([read_prop, int(self.read_timeout_s * 1000)])
        if params:
            try:
                return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            except TypeError:
                pass
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = None
            try:
                cap = self._open_capture()
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    raise RuntimeError("OpenCV/FFmpeg could not open RTSP stream")

                self.connected = True
                self.last_error = None
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("RTSP read failed")
                    now = time.monotonic()
                    with self._lock:
                        self._frame = frame
                        self._seq += 1
                        self._frame_time = now
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
                self.reconnects += 1
                if not self._stop.wait(1.0):
                    continue
            finally:
                if cap is not None:
                    cap.release()
        self.connected = False


class AmcrestPTZ:''',
    "RTSP capture timeouts",
)

tracker = replace_once(
    tracker,
    '            if text and "ERROR" in text.upper():\n                raise RuntimeError(text)\n',
    '            if text and text.lstrip().lower().startswith("error"):\n                raise RuntimeError(text)\n',
    "CGI error detection",
)

tracker = replace_once(
    tracker,
    '        self.capture = LatestFrameCapture(cfg.resolved_rtsp_url())\n',
    '        self.capture = LatestFrameCapture(\n            cfg.resolved_rtsp_url(),\n            open_timeout_s=cfg.rtsp_open_timeout,\n            read_timeout_s=cfg.rtsp_read_timeout,\n        )\n',
    "capture timeout wiring",
)

tracker = tracker.replace('        self._debug_jpeg: Optional[bytes] = None\n', '        self._last_debug_detections: List[Detection] = []\n')

tracker = replace_once(
    tracker,
    '        self._last_task_error: Optional[str] = None\n',
    '        self._last_task_error: Optional[str] = None\n        self._session_generation = 0\n        self._move_failure_count = 0\n        self._move_retry_after = 0.0\n        self._hybrid_disabled_for_session = False\n        self._hybrid_last_error: Optional[float] = None\n        self._hybrid_divergence_count = 0\n',
    "tracker safety state",
)

tracker = replace_once(
    tracker,
    '        self._hybrid_last_command_at = 0.0\n',
    '        self._hybrid_last_command_at = 0.0\n        self._hybrid_last_error = None\n        self._hybrid_divergence_count = 0\n        self._move_failure_count = 0\n        self._move_retry_after = 0.0\n',
    "reset transient safety state",
)

tracker = replace_once(
    tracker,
    '    async def start(self) -> dict:\n        self._history.clear()\n',
    '    def _session_valid(self, generation: int) -> bool:\n        return self.active and generation == self._session_generation and not self._shutdown\n\n    async def start(self) -> dict:\n        self._session_generation += 1\n        self._history.clear()\n',
    "session generation helper/start",
)

tracker = replace_once(
    tracker,
    '        self._last_task_error = None\n        self._shutdown = False\n',
    '        self._last_task_error = None\n        self._hybrid_disabled_for_session = False\n        self._shutdown = False\n',
    "reset hybrid session disable",
)

tracker = replace_once(
    tracker,
    '    async def stop(self) -> dict:\n        self._record_event("tracker_stopping")\n',
    '    async def stop(self) -> dict:\n        self._session_generation += 1\n        self._record_event("tracker_stopping")\n',
    "stop invalidates in-flight work",
)

tracker = replace_regex(
    tracker,
    r'    async def home\(self\) -> dict:.*?\n    def debug_jpeg\(self\) -> Optional\[bytes\]:\n        return self\._debug_jpeg\n',
    '''    async def _goto_home_with_retry(self, reason: str, generation: int) -> bool:
        attempts = max(1, self.cfg.home_retry_attempts)
        for attempt in range(1, attempts + 1):
            ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
            if generation != self._session_generation or self._shutdown:
                return False
            if ok:
                if attempt > 1:
                    self._record_event("home_retry_recovered", reason=reason, attempt=attempt)
                return True
            self._record_event(
                "home_command_failed",
                reason=reason,
                attempt=attempt,
                attempts=attempts,
                error=self.ptz.last_error,
            )
            if attempt < attempts:
                await asyncio.sleep(self.cfg.home_retry_delay)
                if generation != self._session_generation or self._shutdown:
                    return False
        return False

    async def home(self) -> dict:
        self._session_generation += 1
        generation = self._session_generation
        await self._stop_hybrid_chase("home", force=True)
        self._reset_tracking_state()
        self._home_sent = True
        self.state = "HOME" if self.active else "OFF"
        ok = await self._goto_home_with_retry("manual_home", generation)
        now = time.monotonic()
        _, seq, _ = self.capture.latest()
        self._last_zoom_position = None
        if ok:
            self.home_returns += 1
            if self.active and generation == self._session_generation:
                self._begin_ptz_operation("home", seq, now)
        self._record_event("home_command", preset=self.cfg.home_preset, success=bool(ok))
        return self.status()

    async def debug_jpeg(self) -> Optional[bytes]:
        frame, _, _ = self.capture.latest()
        if frame is None:
            return None
        detections = [
            Detection(d.class_id, d.label, d.confidence, tuple(d.bbox))
            for d in self._last_debug_detections
        ]
        target = None
        if self.target is not None:
            target = Detection(
                self.target.class_id,
                self.target.label,
                self.target.confidence,
                tuple(self.target.bbox),
            )
        return await asyncio.to_thread(
            self._render_debug_frame,
            frame.copy(),
            detections,
            target,
            self.state,
            self._last_error_x,
            self._last_error_y,
            self._last_zoom_position,
            self._ptz_operation,
            self._last_inference_ms,
        )
''',
    "home retry and lazy debug endpoint",
)

tracker = replace_once(
    tracker,
    '            "tracker_task_error": self._last_task_error,\n',
    '            "tracker_task_error": self._last_task_error,\n            "session_generation": self._session_generation,\n            "hybrid_chase_disabled_for_session": self._hybrid_disabled_for_session,\n            "move_failure_count": self._move_failure_count,\n',
    "status safety fields",
)

tracker = replace_once(
    tracker,
    '        self._hybrid_last_stopped_at = t1\n',
    '        self._hybrid_last_stopped_at = t1\n        self._hybrid_last_error = None\n        self._hybrid_divergence_count = 0\n',
    "reset divergence on chase stop",
)

tracker = replace_once(
    tracker,
    '        entering = not self._hybrid_chase_active\n        if entering:\n            self._hybrid_started_at = t1\n',
    '        entering = not self._hybrid_chase_active\n        if entering:\n            self._hybrid_started_at = t1\n            self._hybrid_last_error = max(abs(error_x), abs(error_y))\n            self._hybrid_divergence_count = 0\n',
    "initialize divergence guard",
)

tracker = replace_once(
    tracker,
    '        t0 = time.monotonic()\n        ok = await asyncio.to_thread(\n            self.ptz.continuous_move,\n',
    '        generation = self._session_generation\n        t0 = time.monotonic()\n        ok = await asyncio.to_thread(\n            self.ptz.continuous_move,\n',
    "hybrid chase generation capture",
)

tracker = replace_once(
    tracker,
    '        t1 = time.monotonic()\n        if not ok:\n            self._record_event(\n                "hybrid_chase_move_failed",\n',
    '        t1 = time.monotonic()\n        if not self._session_valid(generation):\n            if ok:\n                await asyncio.to_thread(self.ptz.continuous_stop)\n            return False\n        if not ok:\n            self._record_event(\n                "hybrid_chase_move_failed",\n',
    "hybrid stale-result guard",
)

tracker = replace_once(
    tracker,
    '        if (now - self._hybrid_started_at) >= self.cfg.hybrid_chase_max_seconds:\n',
    '''        dominant_error = max(abs(err_x), abs(err_y))
        if self._hybrid_last_error is not None:
            if dominant_error > (self._hybrid_last_error + self.cfg.hybrid_divergence_growth):
                self._hybrid_divergence_count += 1
            elif dominant_error < self._hybrid_last_error:
                self._hybrid_divergence_count = 0
            self._hybrid_last_error = dominant_error
            if self._hybrid_divergence_count >= self.cfg.hybrid_divergence_frames:
                count = self._hybrid_divergence_count
                self._hybrid_disabled_for_session = True
                self._record_event(
                    "hybrid_chase_diverging",
                    error=round(dominant_error, 3),
                    consecutive_growth_frames=count,
                    pan_sign=self.cfg.hybrid_chase_pan_sign,
                )
                await self._stop_hybrid_chase("diverging", seq=seq, force=True)
                return
        else:
            self._hybrid_last_error = dominant_error

        if (now - self._hybrid_started_at) >= self.cfg.hybrid_chase_max_seconds:
''',
    "hybrid divergence failsafe",
)

# Replace the final association gate with a hard candidate gate before scoring.
tracker = replace_once(
    tracker,
    '        best: Optional[Detection] = None\n        best_score = -1.0\n        best_dist_norm = 999.0\n        for det in detections:\n',
    '        max_dist = self.cfg.association_moving_distance if camera_recently_moved else self.cfg.association_idle_distance\n        best: Optional[Detection] = None\n        best_score = -1.0\n        for det in detections:\n',
    "association max distance placement",
)

tracker = replace_once(
    tracker,
    '            overlap = _iou(self.target.bbox, det.bbox)\n            size_similarity = min(old_area, det.area) / max(old_area, det.area)\n',
    '            overlap = _iou(self.target.bbox, det.bbox)\n            if dist_norm > max_dist and overlap < 0.30:\n                continue\n            size_similarity = min(old_area, det.area) / max(old_area, det.area)\n',
    "association hard gate",
)

tracker = replace_once(
    tracker,
    '                best = det\n                best_dist_norm = dist_norm\n\n        max_dist = self.cfg.association_moving_distance if camera_recently_moved else self.cfg.association_idle_distance\n        if best is not None and (best_score >= 0.20 or best_dist_norm <= max_dist):\n            return best\n',
    '                best = det\n\n        if best is not None and best_score >= 0.20:\n            return best\n',
    "association final gate",
)

# Hybrid chase must remain disabled for the rest of a session after divergence.
tracker = replace_once(
    tracker,
    '                self.cfg.hybrid_chase_enabled\n                and (now - self._hybrid_last_stopped_at) >= self.cfg.hybrid_chase_cooldown\n',
    '                self.cfg.hybrid_chase_enabled\n                and not self._hybrid_disabled_for_session\n                and (now - self._hybrid_last_stopped_at) >= self.cfg.hybrid_chase_cooldown\n',
    "hybrid session disable gate",
)

# Add move failure backoff and fail-closed threshold.
tracker = replace_once(
    tracker,
    '        if outside_deadzone:\n            if not self.cfg.move_directly_enabled:\n                return\n',
    '        if outside_deadzone:\n            if not self.cfg.move_directly_enabled:\n                return\n            if now < self._move_retry_after:\n                self.state = "PTZ_BACKOFF"\n                return\n',
    "move failure backoff precheck",
)

tracker = replace_once(
    tracker,
    '            t0 = time.monotonic()\n            ok = await asyncio.to_thread(self.ptz.move_directly_point, command_center, frame_shape)\n            t1 = time.monotonic()\n            if ok:\n',
    '            generation = self._session_generation\n            t0 = time.monotonic()\n            ok = await asyncio.to_thread(self.ptz.move_directly_point, command_center, frame_shape)\n            t1 = time.monotonic()\n            if not self._session_valid(generation):\n                return\n            if ok:\n                self._move_failure_count = 0\n                self._move_retry_after = 0.0\n',
    "move stale-result/backoff reset",
)

tracker = replace_once(
    tracker,
    '            else:\n                self._record_event("move_directly_failed", error=self.ptz.last_error)\n            return\n',
    '''            else:
                self._move_failure_count += 1
                backoff = min(
                    self.cfg.move_failure_backoff_max,
                    self.cfg.move_failure_backoff_base * (2 ** max(0, self._move_failure_count - 1)),
                )
                self._move_retry_after = t1 + backoff
                self._record_event(
                    "move_directly_failed",
                    error=self.ptz.last_error,
                    consecutive_failures=self._move_failure_count,
                    retry_after_ms=int(backoff * 1000),
                )
                if self._move_failure_count >= self.cfg.move_failure_stop_after:
                    await self._halt_tracking_for_ptz_failure(
                        f"moveDirectly failed {self._move_failure_count} consecutive times; last error: {self.ptz.last_error}",
                        operation="move",
                    )
            return
''',
    "move failure backoff/fail closed",
)

# Stop/home/session race guards around long awaits.
tracker = replace_once(
    tracker,
    '                if not self.active:\n                    await asyncio.sleep(0.05)\n                    continue\n\n                frame, seq, frame_time = self.capture.latest()\n',
    '                if not self.active:\n                    await asyncio.sleep(0.05)\n                    continue\n\n                generation = self._session_generation\n                frame, seq, frame_time = self.capture.latest()\n',
    "run loop generation capture",
)

tracker = replace_once(
    tracker,
    '                if not self.active:\n                    continue\n\n                if frame is None or frame_time <= 0:\n',
    '                if not self._session_valid(generation):\n                    continue\n\n                if frame is None or frame_time <= 0:\n',
    "post-poll generation check",
)

tracker = replace_once(
    tracker,
    '                self._last_inference_outcome = outcome\n                self._last_inference_ms = infer_ms\n\n                if outcome == "busy":\n',
    '                if not self._session_valid(generation):\n                    continue\n                self._last_inference_outcome = outcome\n                self._last_inference_ms = infer_ms\n\n                if outcome == "busy":\n',
    "post-inference generation check",
)

tracker = replace_once(
    tracker,
    '                self._last_detection_count = len(detections)\n                await self._process_observation(frame, seq, detections, time.monotonic())\n                self._make_debug_frame(frame, detections)\n',
    '                self._last_detection_count = len(detections)\n                self._last_debug_detections = detections\n                await self._process_observation(frame, seq, detections, time.monotonic(), generation)\n',
    "lazy debug and observation generation",
)

tracker = replace_once(
    tracker,
    '        now: float,\n    ) -> None:\n        # Native 3D positioning',
    '        now: float,\n        generation: int,\n    ) -> None:\n        if not self._session_valid(generation):\n            return\n        # Native 3D positioning',
    "observation generation parameter",
)

tracker = replace_once(
    tracker,
    '            await self._drive_to_target(frame.shape, seq, now)\n            return\n',
    '            await self._drive_to_target(frame.shape, seq, now, generation)\n            return\n',
    "drive generation propagation",
)

tracker = replace_once(
    tracker,
    '        if self.cfg.return_home_on_lost and not self._home_sent:\n            ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)\n            now2 = time.monotonic()\n            if ok:\n',
    '        if self.cfg.return_home_on_lost and not self._home_sent:\n            ok = await self._goto_home_with_retry("target_lost", generation)\n            now2 = time.monotonic()\n            if not self._session_valid(generation):\n                return\n            if ok:\n',
    "lost-target home retry",
)

tracker = replace_once(
    tracker,
    '    async def _drive_to_target(self, frame_shape: Tuple[int, ...], seq: int, now: float) -> None:\n        if self.target is None:\n',
    '    async def _drive_to_target(self, frame_shape: Tuple[int, ...], seq: int, now: float, generation: int) -> None:\n        if self.target is None or not self._session_valid(generation):\n',
    "drive generation signature",
)

tracker = replace_once(
    tracker,
    '            status = await asyncio.to_thread(self.ptz.get_status)\n            checked_at = time.monotonic()\n            if not status:\n',
    '            status = await asyncio.to_thread(self.ptz.get_status)\n            checked_at = time.monotonic()\n            if not self._session_valid(generation):\n                return\n            if not status:\n',
    "zoom status stale-result guard",
)

tracker = replace_once(
    tracker,
    '        ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)\n        t1 = time.monotonic()\n        self._last_zoom_command_at = t1\n',
    '        ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)\n        t1 = time.monotonic()\n        if not self._session_valid(generation):\n            return\n        self._last_zoom_command_at = t1\n',
    "zoom stale-result guard",
)

# Replace eager debug renderer at EOF with a pure lazy renderer.
tracker = replace_regex(
    tracker,
    r'    def _make_debug_frame\(self, frame: np\.ndarray, detections: List\[Detection\]\) -> None:.*\Z',
    '''    def _render_debug_frame(
        self,
        debug: np.ndarray,
        detections: List[Detection],
        target: Optional[Detection],
        state: str,
        error_x: Optional[float],
        error_y: Optional[float],
        zoom_position: Optional[float],
        ptz_operation: Optional[str],
        inference_ms: int,
    ) -> Optional[bytes]:
        try:
            h, w = debug.shape[:2]
            cx, cy = w // 2, h // 2
            left = int(cx - self.cfg.move_deadzone_x * (w / 2.0))
            right = int(cx + self.cfg.move_deadzone_x * (w / 2.0))
            top = int(cy - self.cfg.move_deadzone_y * (h / 2.0))
            bottom = int(cy + self.cfg.move_deadzone_y * (h / 2.0))
            cv2.rectangle(debug, (left, top), (right, bottom), (160, 160, 160), 1)
            cv2.drawMarker(debug, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 190, 255), 1)
                cv2.putText(debug, f"{det.label} {det.confidence:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 190, 255), 1, cv2.LINE_AA)

            if target is not None:
                x1, y1, x2, y2 = [int(v) for v in target.bbox]
                tx, ty = [int(v) for v in target.center]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(debug, (cx, cy), (tx, ty), (0, 255, 0), 1)
                cv2.circle(debug, (tx, ty), 4, (0, 255, 0), -1)

            target_text = "" if target is None else f" {target.label} {target.confidence:.2f}"
            zoom_text = "?" if zoom_position is None else f"{zoom_position / self.cfg.zoom_wide_position:.1f}x"
            op_text = ptz_operation or "idle"
            text = (
                f"{state}{target_text} mode=3D op={op_text} "
                f"err={error_x if error_x is not None else 0:+.2f},"
                f"{error_y if error_y is not None else 0:+.2f} "
                f"zoom={zoom_text} infer={inference_ms}ms"
            )
            cv2.putText(debug, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            ok, encoded = cv2.imencode(".jpg", debug, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            return encoded.tobytes() if ok else None
        except Exception:
            return None
''',
    "lazy debug renderer",
)

p.write_text(tracker)


# ---------------------------------------------------------------------------
# docker-compose.example.yml
# ---------------------------------------------------------------------------
p = Path("docker-compose.example.yml")
compose = p.read_text()
compose = replace_once(
    compose,
    '      RECOVERY_GRACE_PERIOD: "4.0"\n',
    '      RECOVERY_GRACE_PERIOD: "4.0"\n      MAX_FACE_EMBEDDINGS_PER_USER: "20"\n      OPENCV_FFMPEG_CAPTURE_OPTIONS: "rtsp_transport;tcp"\n',
    "compose face/RTSP options",
)
compose = replace_once(
    compose,
    '      TRACKER_RTSP_SUBTYPE: "1"\n',
    '      TRACKER_RTSP_SUBTYPE: "1"\n      TRACKER_RTSP_OPEN_TIMEOUT: "5.0"\n      TRACKER_RTSP_READ_TIMEOUT: "5.0"\n',
    "compose RTSP timeouts",
)
compose = replace_once(
    compose,
    '      TRACKER_HYBRID_CHASE_SETTLE_FRAMES: "2"\n',
    '      TRACKER_HYBRID_CHASE_SETTLE_FRAMES: "2"\n      TRACKER_HYBRID_DIVERGENCE_FRAMES: "3"\n      TRACKER_HYBRID_DIVERGENCE_GROWTH: "0.05"\n',
    "compose divergence guard",
)
compose = replace_once(
    compose,
    '      TRACKER_PTZ_STATUS_POLL_INTERVAL: "0.06"\n',
    '      TRACKER_PTZ_STATUS_POLL_INTERVAL: "0.12"\n',
    "compose PTZ polling",
)
compose = replace_once(
    compose,
    '      TRACKER_PTZ_HTTP_TIMEOUT: "0.75"\n',
    '      TRACKER_PTZ_HTTP_TIMEOUT: "0.75"\n      TRACKER_MOVE_FAILURE_BACKOFF_BASE: "0.25"\n      TRACKER_MOVE_FAILURE_BACKOFF_MAX: "1.0"\n      TRACKER_MOVE_FAILURE_STOP_AFTER: "5"\n',
    "compose move failure backoff",
)
compose = replace_once(
    compose,
    '      TRACKER_RETURN_HOME_ON_LOST: "true"\n',
    '      TRACKER_RETURN_HOME_ON_LOST: "true"\n      TRACKER_HOME_RETRY_ATTEMPTS: "3"\n      TRACKER_HOME_RETRY_DELAY: "0.75"\n',
    "compose home retry",
)
p.write_text(compose)


# ---------------------------------------------------------------------------
# README.md (concise docs for the new safety/perf knobs)
# ---------------------------------------------------------------------------
p = Path("README.md")
readme = p.read_text()
readme = readme.replace(
    '| `RECOVERY_GRACE_PERIOD` | `4.0` | Grace period for an orphaned CUDA worker before the container self-terminates. |',
    '| `RECOVERY_GRACE_PERIOD` | `4.0` | Grace period for an orphaned CUDA worker before the container self-terminates. |\n| `MAX_FACE_EMBEDDINGS_PER_USER` | `20` | Maximum saved FaceNet embeddings per enrolled identity; oldest samples are discarded first. |',
)
readme = readme.replace(
    '| `TRACKER_HYBRID_CHASE_SETTLE_FRAMES` | `2` | Fresh frames required after chase stops. |',
    '| `TRACKER_HYBRID_CHASE_SETTLE_FRAMES` | `2` | Fresh frames required after chase stops. |\n| `TRACKER_HYBRID_DIVERGENCE_FRAMES` | `3` | Consecutive materially-worsening chase frames before the chase is aborted. |\n| `TRACKER_HYBRID_DIVERGENCE_GROWTH` | `0.05` | Minimum normalized error growth that counts toward divergence. |',
)
readme = readme.replace(
    '| `TRACKER_PTZ_OPERATION_TIMEOUT` | `4.0` | Maximum seconds a native PTZ operation may remain in flight before final status validation. |',
    '| `TRACKER_PTZ_OPERATION_TIMEOUT` | `4.0` | Maximum seconds a native PTZ operation may remain in flight before final status validation. |\n| `TRACKER_MOVE_FAILURE_BACKOFF_BASE` | `0.25` | Initial delay after a failed `moveDirectly` request. |\n| `TRACKER_MOVE_FAILURE_BACKOFF_MAX` | `1.0` | Maximum exponential retry delay for failed `moveDirectly` requests. |\n| `TRACKER_MOVE_FAILURE_STOP_AFTER` | `5` | Consecutive `moveDirectly` failures that stop tracking fail-closed. |',
)
readme += '\n\n### Additional tracker reliability notes\n\nThe RTSP reader uses OpenCV/FFmpeg open and read timeouts (`TRACKER_RTSP_OPEN_TIMEOUT` / `TRACKER_RTSP_READ_TIMEOUT`) so a stalled stream can reconnect instead of blocking forever. Debug JPEGs are rendered only when `/v1/tracker/debug.jpg` is requested. Tracker sessions use a generation token so results from inference or PTZ calls that complete after `/stop` or `/home` are discarded. Lost-target home commands are retried a bounded number of times (`TRACKER_HOME_RETRY_ATTEMPTS`, `TRACKER_HOME_RETRY_DELAY`). Hybrid chase disables itself for the remainder of the session if tracking error grows materially for several consecutive frames, then falls back to normal `moveDirectly`.\n'
p.write_text(readme)


# ---------------------------------------------------------------------------
# tests/test_static_contracts.py
# ---------------------------------------------------------------------------
p = Path("tests/test_static_contracts.py")
tests = p.read_text()
insert = '''\n    def test_second_pass_tracker_safety_contracts(self):\n        tracker = (ROOT / "tracker.py").read_text()\n        compose = (ROOT / "docker-compose.example.yml").read_text()\n        self.assertIn("if dist_norm > max_dist and overlap < 0.30", tracker)\n        self.assertNotIn("best_score >= 0.20 or best_dist_norm <= max_dist", tracker)\n        self.assertIn("if not self._session_valid(generation):", tracker)\n        self.assertIn("hybrid_chase_diverging", tracker)\n        self.assertIn("moveDirectly failed", tracker)\n        self.assertIn("CAP_PROP_READ_TIMEOUT_MSEC", tracker)\n        self.assertIn('TRACKER_PTZ_STATUS_POLL_INTERVAL: "0.12"', compose)\n\n    def test_face_and_debug_optimizations(self):\n        app = (ROOT / "app.py").read_text()\n        tracker = (ROOT / "tracker.py").read_text()\n        self.assertIn("face_detector.extract(img, boxes, None)", app)\n        self.assertNotIn("faces = face_detector(img)", app)\n        self.assertIn('kwargs["quantize"] = 16', app)\n        self.assertNotIn('kwargs["half"] = True', app)\n        self.assertIn("MAX_FACE_EMBEDDINGS_PER_USER", app)\n        self.assertNotIn("self._make_debug_frame(frame, detections)", tracker)\n        self.assertIn("async def debug_jpeg", tracker)\n        self.assertIn("await tracker_service.debug_jpeg()", app)\n'''
marker = '\n\nif __name__ == "__main__":\n'
if marker not in tests:
    raise SystemExit("missing test insertion marker")
tests = tests.replace(marker, insert + marker, 1)
p.write_text(tests)

print("second-pass hardening patch applied")
