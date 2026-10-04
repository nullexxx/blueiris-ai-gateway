from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import os
import threading
import time
from typing import Awaitable, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth


InferenceCallback = Callable[[np.ndarray, float, str, List[int]], Awaitable[Tuple[str, List[dict], int]]]


COCO_TRACKER_CLASSES: Dict[str, int] = {
    "person": 0,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
}


def _parse_class_list(value: str, default: List[str]) -> List[str]:
    raw = [item.strip().lower() for item in value.split(",") if item.strip()]
    result: List[str] = []
    for token in (raw or default):
        if token.isdigit():
            cid = int(token)
            name = next((n for n, i in COCO_TRACKER_CLASSES.items() if i == cid), str(cid))
        else:
            if token not in COCO_TRACKER_CLASSES:
                raise ValueError(
                    f"Unsupported TRACKER target class '{token}'. "
                    f"Supported names: {', '.join(COCO_TRACKER_CLASSES)}"
                )
            name = token
        if name not in result:
            result.append(name)
    return result


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class TrackerConfig:
    enabled: bool = False
    autostart: bool = False

    camera_ip: str = ""
    camera_user: str = "admin"
    camera_password: str = ""
    camera_channel: int = 0
    rtsp_channel: int = 1
    rtsp_subtype: int = 1
    rtsp_url: str = ""

    model_name: str = "yolo11l"
    target_classes: List[str] = field(default_factory=lambda: ["person", "dog", "cat", "bird", "bear"])
    target_priority: List[str] = field(default_factory=lambda: ["dog", "person", "bear", "cat", "bird"])
    fps: float = 10.0
    acquire_conf: float = 0.40
    hold_conf: float = 0.22
    reacquire_conf: float = 0.35
    acquire_frames: int = 2

    association_idle_distance: float = 0.22
    association_moving_distance: float = 0.40

    # 3D positioning: point-only moveDirectly (startPoint == endPoint).
    # On this camera that re-centers the point while preserving optical zoom.
    move_directly_enabled: bool = True
    move_deadzone_x: float = 0.14
    move_deadzone_y: float = 0.18
    move_cooldown: float = 0.30
    move_min_frame_advance: int = 2

    # Bounded, deliberately slow auto-zoom. Camera reports 5.12 at full-wide
    # and 128 at full-tele, i.e. exactly 25x relative to the wide position.
    autozoom: bool = True
    zoom_wide_position: float = 5.12
    zoom_min_factor: float = 1.0
    zoom_max_factor: float = 8.0
    zoom_target_min: float = 0.18
    zoom_target_max: float = 0.42
    zoom_in_step_ms: int = 90
    zoom_out_step_ms: int = 140
    zoom_cooldown: float = 1.0
    zoom_settle_time: float = 0.45
    zoom_status_interval: float = 0.75

    coast_time: float = 0.20
    reacquire_time: float = 1.25
    home_timeout: float = 3.0
    home_preset: int = 5
    goto_home_on_start: bool = True
    return_home_on_lost: bool = True
    home_settle_time: float = 2.0

    ptz_http_timeout: float = 0.75
    frame_stale_timeout: float = 1.0
    history_size: int = 300

    @classmethod
    def from_env(cls) -> "TrackerConfig":
        cfg = cls(
            enabled=_env_bool("TRACKER_ENABLED", False),
            autostart=_env_bool("TRACKER_AUTOSTART", False),
            camera_ip=os.getenv("TRACKER_CAMERA_IP", "").strip(),
            camera_user=os.getenv("TRACKER_CAMERA_USER", "admin").strip(),
            camera_password=os.getenv("TRACKER_CAMERA_PASSWORD", ""),
            camera_channel=_env_int("TRACKER_CAMERA_CHANNEL", 0),
            rtsp_channel=_env_int("TRACKER_RTSP_CHANNEL", 1),
            rtsp_subtype=_env_int("TRACKER_RTSP_SUBTYPE", 1),
            rtsp_url=os.getenv("TRACKER_RTSP_URL", "").strip(),
            model_name=os.getenv("TRACKER_MODEL", os.getenv("DEFAULT_MODEL", "yolo11l")).strip(),
            target_classes=_parse_class_list(
                os.getenv("TRACKER_TARGET_CLASSES", "person,dog,cat,bird,bear"),
                ["person", "dog", "cat", "bird", "bear"],
            ),
            target_priority=_parse_class_list(
                os.getenv("TRACKER_TARGET_PRIORITY", "dog,person,bear,cat,bird"),
                ["dog", "person", "bear", "cat", "bird"],
            ),
            fps=_env_float("TRACKER_FPS", 10.0),
            acquire_conf=_env_float("TRACKER_ACQUIRE_CONF", 0.40),
            hold_conf=_env_float("TRACKER_HOLD_CONF", 0.22),
            reacquire_conf=_env_float("TRACKER_REACQUIRE_CONF", 0.35),
            acquire_frames=max(1, _env_int("TRACKER_ACQUIRE_FRAMES", 2)),
            association_idle_distance=_env_float("TRACKER_ASSOCIATION_IDLE_DISTANCE", 0.22),
            association_moving_distance=_env_float("TRACKER_ASSOCIATION_MOVING_DISTANCE", 0.40),
            move_directly_enabled=_env_bool("TRACKER_MOVE_DIRECTLY_ENABLED", True),
            move_deadzone_x=_env_float("TRACKER_MOVE_DEADZONE_X", 0.14),
            move_deadzone_y=_env_float("TRACKER_MOVE_DEADZONE_Y", 0.18),
            move_cooldown=_env_float("TRACKER_MOVE_COOLDOWN", 0.30),
            move_min_frame_advance=_env_int("TRACKER_MOVE_MIN_FRAME_ADVANCE", 2),
            autozoom=_env_bool("TRACKER_AUTOZOOM", True),
            zoom_wide_position=_env_float("TRACKER_ZOOM_WIDE_POSITION", 5.12),
            zoom_min_factor=_env_float("TRACKER_ZOOM_MIN_FACTOR", 1.0),
            zoom_max_factor=_env_float("TRACKER_ZOOM_MAX_FACTOR", 8.0),
            zoom_target_min=_env_float("TRACKER_ZOOM_TARGET_MIN", 0.18),
            zoom_target_max=_env_float("TRACKER_ZOOM_TARGET_MAX", 0.42),
            zoom_in_step_ms=_env_int("TRACKER_ZOOM_IN_STEP_MS", 90),
            zoom_out_step_ms=_env_int("TRACKER_ZOOM_OUT_STEP_MS", 140),
            zoom_cooldown=_env_float("TRACKER_ZOOM_COOLDOWN", 1.0),
            zoom_settle_time=_env_float("TRACKER_ZOOM_SETTLE_TIME", 0.45),
            zoom_status_interval=_env_float("TRACKER_ZOOM_STATUS_INTERVAL", 0.75),
            coast_time=_env_float("TRACKER_COAST_TIME", 0.20),
            reacquire_time=_env_float("TRACKER_REACQUIRE_TIME", 1.25),
            home_timeout=_env_float("TRACKER_HOME_TIMEOUT", 3.0),
            home_preset=max(1, _env_int("TRACKER_HOME_PRESET", 5)),
            goto_home_on_start=_env_bool("TRACKER_GOTO_HOME_ON_START", True),
            return_home_on_lost=_env_bool("TRACKER_RETURN_HOME_ON_LOST", True),
            home_settle_time=_env_float("TRACKER_HOME_SETTLE_TIME", 2.0),
            ptz_http_timeout=_env_float("TRACKER_PTZ_HTTP_TIMEOUT", 0.75),
            frame_stale_timeout=_env_float("TRACKER_FRAME_STALE_TIMEOUT", 1.0),
            history_size=_env_int("TRACKER_HISTORY_SIZE", 300),
        )

        cfg.target_priority = [name for name in cfg.target_priority if name in cfg.target_classes]
        for name in cfg.target_classes:
            if name not in cfg.target_priority:
                cfg.target_priority.append(name)

        cfg.fps = max(1.0, min(30.0, cfg.fps))
        cfg.acquire_conf = max(0.01, min(0.99, cfg.acquire_conf))
        cfg.hold_conf = max(0.01, min(cfg.acquire_conf, cfg.hold_conf))
        cfg.reacquire_conf = max(cfg.hold_conf, min(cfg.acquire_conf, cfg.reacquire_conf))
        cfg.association_idle_distance = max(0.05, min(0.80, cfg.association_idle_distance))
        cfg.association_moving_distance = max(
            cfg.association_idle_distance,
            min(0.90, cfg.association_moving_distance),
        )
        cfg.move_deadzone_x = max(0.02, min(0.80, cfg.move_deadzone_x))
        cfg.move_deadzone_y = max(0.02, min(0.80, cfg.move_deadzone_y))
        cfg.move_cooldown = max(0.10, min(2.0, cfg.move_cooldown))
        cfg.move_min_frame_advance = max(1, min(20, cfg.move_min_frame_advance))

        cfg.zoom_wide_position = max(0.01, cfg.zoom_wide_position)
        cfg.zoom_min_factor = max(1.0, cfg.zoom_min_factor)
        cfg.zoom_max_factor = max(cfg.zoom_min_factor, min(25.0, cfg.zoom_max_factor))
        cfg.zoom_target_min = max(0.03, min(0.80, cfg.zoom_target_min))
        cfg.zoom_target_max = max(cfg.zoom_target_min + 0.02, min(0.95, cfg.zoom_target_max))
        cfg.zoom_in_step_ms = max(30, min(500, cfg.zoom_in_step_ms))
        cfg.zoom_out_step_ms = max(30, min(700, cfg.zoom_out_step_ms))
        cfg.zoom_cooldown = max(0.25, min(10.0, cfg.zoom_cooldown))
        cfg.zoom_settle_time = max(0.10, min(5.0, cfg.zoom_settle_time))
        cfg.zoom_status_interval = max(0.25, min(10.0, cfg.zoom_status_interval))

        cfg.coast_time = max(0.0, min(2.0, cfg.coast_time))
        cfg.reacquire_time = max(cfg.coast_time, min(10.0, cfg.reacquire_time))
        cfg.home_timeout = max(cfg.reacquire_time, min(60.0, cfg.home_timeout))
        cfg.home_settle_time = max(0.0, min(10.0, cfg.home_settle_time))
        cfg.ptz_http_timeout = max(0.10, min(5.0, cfg.ptz_http_timeout))
        cfg.frame_stale_timeout = max(0.25, min(10.0, cfg.frame_stale_timeout))
        cfg.history_size = max(50, min(2000, cfg.history_size))
        return cfg

    @property
    def target_class_ids(self) -> List[int]:
        ids: List[int] = []
        for name in self.target_classes:
            if name.isdigit():
                ids.append(int(name))
            else:
                ids.append(COCO_TRACKER_CLASSES[name])
        return ids

    def class_priority_rank(self, label: str) -> int:
        try:
            return self.target_priority.index(label)
        except ValueError:
            return len(self.target_priority) + 100

    def resolved_rtsp_url(self) -> str:
        if self.rtsp_url:
            return self.rtsp_url
        user = quote(self.camera_user, safe="")
        password = quote(self.camera_password, safe="")
        return (
            f"rtsp://{user}:{password}@{self.camera_ip}:554/"
            f"cam/realmonitor?channel={self.rtsp_channel}&subtype={self.rtsp_subtype}"
        )

    @property
    def zoom_min_position(self) -> float:
        return self.zoom_wide_position * self.zoom_min_factor

    @property
    def zoom_max_position(self) -> float:
        return self.zoom_wide_position * self.zoom_max_factor

    def public_dict(self) -> dict:
        return {
            "camera_ip": self.camera_ip,
            "camera_user": self.camera_user,
            "camera_channel": self.camera_channel,
            "rtsp_channel": self.rtsp_channel,
            "rtsp_subtype": self.rtsp_subtype,
            "model": self.model_name,
            "target_classes": list(self.target_classes),
            "target_class_ids": list(self.target_class_ids),
            "target_priority": list(self.target_priority),
            "fps": self.fps,
            "acquire_conf": self.acquire_conf,
            "hold_conf": self.hold_conf,
            "reacquire_conf": self.reacquire_conf,
            "acquire_frames": self.acquire_frames,
            "association_idle_distance": self.association_idle_distance,
            "association_moving_distance": self.association_moving_distance,
            "move_directly_enabled": self.move_directly_enabled,
            "move_deadzone_x": self.move_deadzone_x,
            "move_deadzone_y": self.move_deadzone_y,
            "move_cooldown": self.move_cooldown,
            "move_min_frame_advance": self.move_min_frame_advance,
            "autozoom": self.autozoom,
            "zoom_wide_position": self.zoom_wide_position,
            "zoom_min_factor": self.zoom_min_factor,
            "zoom_max_factor": self.zoom_max_factor,
            "zoom_min_position": round(self.zoom_min_position, 3),
            "zoom_max_position": round(self.zoom_max_position, 3),
            "zoom_target_min": self.zoom_target_min,
            "zoom_target_max": self.zoom_target_max,
            "zoom_in_step_ms": self.zoom_in_step_ms,
            "zoom_out_step_ms": self.zoom_out_step_ms,
            "zoom_cooldown": self.zoom_cooldown,
            "zoom_settle_time": self.zoom_settle_time,
            "zoom_status_interval": self.zoom_status_interval,
            "coast_time": self.coast_time,
            "reacquire_time": self.reacquire_time,
            "home_timeout": self.home_timeout,
            "home_preset": self.home_preset,
            "goto_home_on_start": self.goto_home_on_start,
            "return_home_on_lost": self.return_home_on_lost,
            "home_settle_time": self.home_settle_time,
            "ptz_http_timeout": self.ptz_http_timeout,
            "frame_stale_timeout": self.frame_stale_timeout,
            "history_size": self.history_size,
        }


class LatestFrameCapture:
    """Continuously drains RTSP and retains only the newest decoded frame."""

    def __init__(self, url: str):
        self.url = url
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

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
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


class AmcrestPTZ:
    """Dahua/Amcrest CGI client using a persistent Digest-auth session."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.base = f"http://{cfg.camera_ip}"
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(cfg.camera_user, cfg.camera_password)
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None
        self.last_http_ms: Optional[int] = None
        self.last_move_point: Optional[Tuple[int, int]] = None

    def close(self) -> None:
        self.session.close()

    def _get(self, path: str, params: Dict[str, object]) -> str:
        start = time.perf_counter()
        try:
            with self._lock:
                response = self.session.get(
                    self.base + path,
                    params=params,
                    timeout=self.cfg.ptz_http_timeout,
                )
            response.raise_for_status()
            self.last_http_ms = int((time.perf_counter() - start) * 1000)
            text = response.text.strip()
            if text and "ERROR" in text.upper():
                raise RuntimeError(text)
            self.last_error = None
            return text
        except Exception as exc:
            self.last_http_ms = int((time.perf_counter() - start) * 1000)
            self.last_error = str(exc)
            raise

    def goto_preset(self, preset: int) -> bool:
        try:
            self._get(
                "/cgi-bin/ptz.cgi",
                {
                    "action": "start",
                    "channel": self.cfg.camera_channel,
                    "code": "GotoPreset",
                    "arg1": 0,
                    "arg2": int(preset),
                    "arg3": 0,
                },
            )
            self.last_move_point = None
            return True
        except Exception:
            return False

    def get_status(self) -> Dict[str, str]:
        try:
            text = self._get(
                "/cgi-bin/ptz.cgi",
                {"action": "getStatus", "channel": self.cfg.camera_channel},
            )
        except Exception:
            return {}

        result: Dict[str, str] = {}
        for line in text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
        return result

    @staticmethod
    def zoom_position_from_status(status: Dict[str, str]) -> Optional[float]:
        raw = status.get("status.Postion[2]")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _scale_point(center: Tuple[float, float], frame_shape: Tuple[int, ...]) -> Tuple[int, int]:
        h, w = frame_shape[:2]
        cx, cy = center
        # Match the proven manual test: 20% of a 640px frame -> 1638 of 8192.
        sx = int(round((max(0.0, min(float(w), cx)) / max(1.0, float(w))) * 8192.0))
        sy = int(round((max(0.0, min(float(h), cy)) / max(1.0, float(h))) * 8192.0))
        return max(0, min(8192, sx)), max(0, min(8192, sy))

    def move_directly_point(self, center: Tuple[float, float], frame_shape: Tuple[int, ...]) -> bool:
        """Use Dahua 3D positioning in point-only mode, preserving current zoom."""
        sx, sy = self._scale_point(center, frame_shape)
        try:
            self._get(
                "/cgi-bin/ptzBase.cgi",
                {
                    "action": "moveDirectly",
                    "channel": self.cfg.camera_channel,
                    "startPoint[0]": sx,
                    "startPoint[1]": sy,
                    "endPoint[0]": sx,
                    "endPoint[1]": sy,
                },
            )
            self.last_move_point = (sx, sy)
            return True
        except Exception:
            return False

    def zoom_step(self, direction: str, duration_ms: int) -> bool:
        """Perform one bounded optical-zoom step using ZoomTele/ZoomWide."""
        if direction not in ("in", "out"):
            raise ValueError("direction must be 'in' or 'out'")
        code = "ZoomTele" if direction == "in" else "ZoomWide"
        duration_ms = max(20, min(1000, int(duration_ms)))
        started = False
        try:
            self._get(
                "/cgi-bin/ptz.cgi",
                {
                    "action": "start",
                    "channel": self.cfg.camera_channel,
                    "code": code,
                    "arg1": 0,
                    "arg2": 1,
                    "arg3": 0,
                    "arg4": 0,
                },
            )
            started = True
            time.sleep(duration_ms / 1000.0)
            return True
        except Exception:
            return False
        finally:
            if started:
                try:
                    self._get(
                        "/cgi-bin/ptz.cgi",
                        {
                            "action": "stop",
                            "channel": self.cfg.camera_channel,
                            "code": code,
                            "arg1": 0,
                            "arg2": 1,
                            "arg3": 0,
                            "arg4": 0,
                        },
                    )
                except Exception:
                    pass


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(1.0, (x2 - x1) * (y2 - y1))


@dataclass
class TargetTrack:
    class_id: int
    label: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    center: Tuple[float, float]
    last_update: float
    last_seen: float
    vx: float = 0.0
    vy: float = 0.0
    acquire_hits: int = 1

    def update(self, det: Detection, now: float) -> None:
        if det.class_id != self.class_id:
            raise ValueError("TargetTrack.update received a different object class")
        new_center = det.center
        dt = max(0.001, now - self.last_update)
        inst_vx = (new_center[0] - self.center[0]) / dt
        inst_vy = (new_center[1] - self.center[1]) / dt
        alpha = 0.35
        self.vx = (1.0 - alpha) * self.vx + alpha * inst_vx
        self.vy = (1.0 - alpha) * self.vy + alpha * inst_vy
        self.bbox = det.bbox
        self.confidence = det.confidence
        self.center = new_center
        self.last_update = now
        self.last_seen = now
        self.acquire_hits += 1

    def predicted_center(self, now: float) -> Tuple[float, float]:
        dt = min(0.6, max(0.0, now - self.last_update))
        return (self.center[0] + self.vx * dt, self.center[1] + self.vy * dt)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)


class DogTracker:
    """Multi-class YOLO tracker using Dahua 3D moveDirectly + bounded autozoom."""

    def __init__(self, cfg: TrackerConfig, inference_cb: InferenceCallback, logger):
        self.cfg = cfg
        self.inference_cb = inference_cb
        self.logger = logger
        self.capture = LatestFrameCapture(cfg.resolved_rtsp_url())
        self.ptz = AmcrestPTZ(cfg)

        self.active = False
        self.state = "OFF"
        self.target: Optional[TargetTrack] = None
        self._task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._last_processed_seq = -1
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        self._debug_jpeg: Optional[bytes] = None

        self._home_sent = False
        self._home_hold_until = 0.0
        self._move_hold_until = 0.0
        self._zoom_hold_until = 0.0
        self._last_move_frame_seq = -10_000
        self._last_move_point: Optional[Tuple[int, int]] = None
        self._last_error_x: Optional[float] = None
        self._last_error_y: Optional[float] = None
        self._last_target_span: Optional[float] = None
        self._last_zoom_position: Optional[float] = None
        self._last_zoom_status_at = 0.0
        self._last_zoom_command_at = 0.0

        self._last_detection_count = 0
        self._last_inference_ms = 0
        self._last_inference_outcome = "never"
        self._inference_times: Deque[float] = deque(maxlen=128)
        self._history: Deque[dict] = deque(maxlen=self.cfg.history_size)
        self._session_started_wall: Optional[str] = None
        self._session_started_mono: Optional[float] = None

        self.total_inferences = 0
        self.frames_skipped_gpu_busy = 0
        self.frames_skipped_duplicate = 0
        self.frames_stale = 0
        self.ptz_commands = 0
        self.move_direct_commands = 0
        self.zoom_commands = 0
        self.targets_acquired = 0
        self.home_returns = 0

    def _record_event(self, event: str, **fields) -> None:
        item = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "state": self.state,
        }
        if self._session_started_mono is not None:
            item["session_ms"] = int((time.monotonic() - self._session_started_mono) * 1000)
        item.update(fields)
        self._history.append(item)

    def history(self, limit: int = 200) -> dict:
        limit = max(1, min(int(limit), self.cfg.history_size))
        events = list(self._history)[-limit:]
        return {
            "active": self.active,
            "state": self.state,
            "session_started": self._session_started_wall,
            "event_count": len(self._history),
            "returned": len(events),
            "events": events,
        }

    def clear_history(self) -> dict:
        self._history.clear()
        return {"success": True, "event_count": 0}

    async def initialize(self) -> None:
        if not self.cfg.camera_ip:
            raise RuntimeError("TRACKER_CAMERA_IP is required when TRACKER_ENABLED=true")
        if not self.cfg.camera_password:
            self.logger.warning("PTZ tracker camera password is empty.")
        self.capture.start()
        self._task = asyncio.create_task(self._run(), name="direct-3d-ptz-tracker")
        self.logger.info(
            "PTZ tracker initialized: camera=%s model=%s fps=%.1f mode=moveDirectly autozoom=%s autostart=%s",
            self.cfg.camera_ip,
            self.cfg.model_name,
            self.cfg.fps,
            self.cfg.autozoom,
            self.cfg.autostart,
        )
        self._record_event(
            "initialized",
            camera=self.cfg.camera_ip,
            model=self.cfg.model_name,
            fps=self.cfg.fps,
            move_directly=self.cfg.move_directly_enabled,
            autozoom=self.cfg.autozoom,
        )
        if self.cfg.autostart:
            await self.start()

    async def shutdown(self) -> None:
        self._shutdown = True
        self.active = False
        self.state = "SHUTDOWN"
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self.capture.stop)
        self.ptz.close()

    def _reset_tracking_state(self) -> None:
        self.target = None
        self._last_error_x = None
        self._last_error_y = None
        self._last_target_span = None
        self._move_hold_until = 0.0
        self._zoom_hold_until = 0.0
        self._last_move_frame_seq = -10_000
        self._last_move_point = None
        self._last_zoom_command_at = 0.0

    async def start(self) -> dict:
        self._history.clear()
        self._session_started_wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._session_started_mono = time.monotonic()
        self.active = True
        self.state = "SEARCHING"
        self._home_sent = False
        self._home_hold_until = 0.0
        self._reset_tracking_state()
        if self.cfg.goto_home_on_start:
            await self.home()
        self._record_event(
            "tracker_started",
            goto_home_on_start=self.cfg.goto_home_on_start,
            move_directly=self.cfg.move_directly_enabled,
            autozoom=self.cfg.autozoom,
        )
        self.logger.info("PTZ tracker STARTED (3D moveDirectly)")
        return self.status()

    async def stop(self) -> dict:
        self._record_event("tracker_stopping")
        self.active = False
        self.state = "OFF"
        self._reset_tracking_state()
        self._home_hold_until = 0.0
        self._record_event("tracker_stopped")
        self.logger.info("PTZ tracker STOPPED")
        return self.status()

    async def home(self) -> dict:
        self._reset_tracking_state()
        self._home_sent = True
        self.state = "HOME" if self.active else "OFF"
        ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
        now = time.monotonic()
        self._home_hold_until = now + self.cfg.home_settle_time if ok else 0.0
        self._move_hold_until = self._home_hold_until
        self._zoom_hold_until = self._home_hold_until
        self._last_zoom_status_at = 0.0
        self._last_zoom_position = None
        if ok:
            self.home_returns += 1
        self._record_event("home_command", preset=self.cfg.home_preset, success=bool(ok))
        return self.status()

    def debug_jpeg(self) -> Optional[bytes]:
        return self._debug_jpeg

    def status(self) -> dict:
        now = time.monotonic()
        _, seq, frame_time = self.capture.latest()
        frame_age_ms = None if frame_time <= 0 else int(max(0.0, now - frame_time) * 1000)
        last_seen_ms = None
        target_info = None
        if self.target is not None:
            last_seen_ms = int(max(0.0, now - self.target.last_seen) * 1000)
            target_info = {
                "class_id": self.target.class_id,
                "label": self.target.label,
                "confidence": round(self.target.confidence, 3),
                "bbox": [round(v, 1) for v in self.target.bbox],
                "center": [round(v, 1) for v in self.target.center],
                "velocity_px_s": [round(self.target.vx, 1), round(self.target.vy, 1)],
                "acquire_hits": self.target.acquire_hits,
                "span": None if self._last_target_span is None else round(self._last_target_span, 3),
            }

        cutoff = now - 5.0
        while self._inference_times and self._inference_times[0] < cutoff:
            self._inference_times.popleft()
        infer_fps = len(self._inference_times) / 5.0

        zoom_factor = None
        if self._last_zoom_position is not None and self.cfg.zoom_wide_position > 0:
            zoom_factor = self._last_zoom_position / self.cfg.zoom_wide_position

        return {
            "enabled": self.cfg.enabled,
            "active": self.active,
            "session_started": self._session_started_wall,
            "history_events": len(self._history),
            "state": self.state,
            "control_mode": "moveDirectly",
            "home_settle_remaining_ms": max(0, int((self._home_hold_until - now) * 1000)),
            "move_cooldown_remaining_ms": max(0, int((self._move_hold_until - now) * 1000)),
            "zoom_settle_remaining_ms": max(0, int((self._zoom_hold_until - now) * 1000)),
            "camera_connected": self.capture.connected,
            "capture_reconnects": self.capture.reconnects,
            "capture_error": self.capture.last_error,
            "frame_seq": seq,
            "frame_age_ms": frame_age_ms,
            "frame_shape": list(self._last_frame_shape) if self._last_frame_shape else None,
            "model": self.cfg.model_name,
            "target_classes": list(self.cfg.target_classes),
            "target_class_ids": list(self.cfg.target_class_ids),
            "target_priority": list(self.cfg.target_priority),
            "target": target_info,
            "last_seen_ms": last_seen_ms,
            "last_error_x": None if self._last_error_x is None else round(self._last_error_x, 3),
            "last_error_y": None if self._last_error_y is None else round(self._last_error_y, 3),
            "last_move_point_8192": list(self._last_move_point) if self._last_move_point else None,
            "zoom_position": None if self._last_zoom_position is None else round(self._last_zoom_position, 3),
            "zoom_factor": None if zoom_factor is None else round(zoom_factor, 2),
            "last_detection_count": self._last_detection_count,
            "ptz_last_http_ms": self.ptz.last_http_ms,
            "ptz_error": self.ptz.last_error,
            "last_inference_outcome": self._last_inference_outcome,
            "last_inference_ms": self._last_inference_ms,
            "rolling_inference_fps": round(infer_fps, 2),
            "counters": {
                "total_inferences": self.total_inferences,
                "frames_skipped_gpu_busy": self.frames_skipped_gpu_busy,
                "frames_skipped_duplicate": self.frames_skipped_duplicate,
                "frames_stale": self.frames_stale,
                "ptz_commands": self.ptz_commands,
                "move_direct_commands": self.move_direct_commands,
                "zoom_commands": self.zoom_commands,
                "targets_acquired": self.targets_acquired,
                "home_returns": self.home_returns,
            },
            "config": self.cfg.public_dict(),
        }

    async def camera_status(self) -> dict:
        return await asyncio.to_thread(self.ptz.get_status)

    async def _run(self) -> None:
        period = 1.0 / self.cfg.fps
        next_tick = time.monotonic()
        try:
            while not self._shutdown:
                now = time.monotonic()
                if now < next_tick:
                    await asyncio.sleep(next_tick - now)
                next_tick = max(next_tick + period, time.monotonic())

                if not self.active:
                    await asyncio.sleep(0.05)
                    continue

                frame, seq, frame_time = self.capture.latest()
                now = time.monotonic()
                if frame is None or frame_time <= 0:
                    self.state = "WAITING_FRAME"
                    continue
                if (now - frame_time) > self.cfg.frame_stale_timeout:
                    self.frames_stale += 1
                    self.state = "STALE_FRAME"
                    continue
                if seq == self._last_processed_seq:
                    self.frames_skipped_duplicate += 1
                    continue

                self._last_processed_seq = seq
                self._last_frame_shape = frame.shape[:2]
                outcome, raw_detections, infer_ms = await self.inference_cb(
                    frame,
                    self.cfg.hold_conf,
                    self.cfg.model_name,
                    self.cfg.target_class_ids,
                )
                self._last_inference_outcome = outcome
                self._last_inference_ms = infer_ms

                if outcome == "busy":
                    self.frames_skipped_gpu_busy += 1
                    continue
                if outcome != "ok":
                    self.state = "INFERENCE_ERROR"
                    continue

                self.total_inferences += 1
                self._inference_times.append(time.monotonic())
                detections = [
                    Detection(
                        class_id=int(d["class_id"]),
                        label=str(d["label"]),
                        confidence=float(d["confidence"]),
                        bbox=(float(d["x_min"]), float(d["y_min"]), float(d["x_max"]), float(d["y_max"])),
                    )
                    for d in raw_detections
                    if float(d.get("confidence", 0.0)) >= self.cfg.hold_conf
                ]
                self._last_detection_count = len(detections)
                await self._process_observation(frame, seq, detections, time.monotonic())
                self._make_debug_frame(frame, detections)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state = "TRACKER_ERROR"
            self.logger.exception("PTZ tracker loop crashed: %s", exc)

    async def _process_observation(
        self,
        frame: np.ndarray,
        seq: int,
        detections: List[Detection],
        now: float,
    ) -> None:
        if now < self._home_hold_until:
            self.state = "HOME"
            self.target = None
            self._last_error_x = None
            self._last_error_y = None
            return

        if self.target is None:
            candidates = [d for d in detections if d.confidence >= self.cfg.acquire_conf]
            if not candidates:
                self.state = "HOME" if self._home_sent else "SEARCHING"
                self._last_error_x = None
                self._last_error_y = None
                return

            h, w = frame.shape[:2]
            frame_area = max(1.0, float(h * w))

            def acquisition_key(det: Detection) -> Tuple[int, float]:
                rank = self.cfg.class_priority_rank(det.label)
                quality = det.confidence + 0.10 * math.sqrt(det.area / frame_area)
                return (-rank, quality)

            chosen = max(candidates, key=acquisition_key)
            self.target = TargetTrack(
                class_id=chosen.class_id,
                label=chosen.label,
                bbox=chosen.bbox,
                confidence=chosen.confidence,
                center=chosen.center,
                last_update=now,
                last_seen=now,
                acquire_hits=1,
            )
            self.state = "ACQUIRE"
            self._record_event(
                "acquire_candidate",
                label=chosen.label,
                confidence=round(chosen.confidence, 3),
                bbox=[round(v, 1) for v in chosen.bbox],
            )
            self._home_sent = False
            return

        matched = self._associate(detections, frame.shape, now)
        if (
            matched is not None
            and self.state in ("COAST", "REACQUIRE", "LOST")
            and matched.confidence < self.cfg.reacquire_conf
        ):
            matched = None

        if matched is not None:
            was_acquiring = self.target.acquire_hits < self.cfg.acquire_frames
            self.target.update(matched, now)
            self._home_sent = False

            if was_acquiring and self.target.acquire_hits < self.cfg.acquire_frames:
                self.state = "ACQUIRE"
                return

            if was_acquiring and self.target.acquire_hits == self.cfg.acquire_frames:
                self.targets_acquired += 1
                self._record_event(
                    "target_acquired",
                    label=self.target.label,
                    confidence=round(self.target.confidence, 3),
                    bbox=[round(v, 1) for v in self.target.bbox],
                )
                self.logger.info(
                    "%s target acquired: conf=%.2f bbox=%s",
                    self.target.label,
                    self.target.confidence,
                    tuple(round(v, 1) for v in self.target.bbox),
                )

            self.state = "TRACK"
            await self._drive_to_target(frame.shape, seq, now)
            return

        if self.target.acquire_hits < self.cfg.acquire_frames:
            self._record_event("acquire_dropped", label=self.target.label, hits=self.target.acquire_hits)
            self.target = None
            self.state = "SEARCHING"
            return

        missing_for = now - self.target.last_seen
        if missing_for <= self.cfg.coast_time:
            if self.state != "COAST":
                self._record_event(
                    "target_missing",
                    phase="coast",
                    missing_ms=int(missing_for * 1000),
                    label=self.target.label,
                )
            self.state = "COAST"
            return

        if missing_for <= self.cfg.reacquire_time:
            if self.state != "REACQUIRE":
                self._record_event(
                    "target_missing",
                    phase="reacquire",
                    missing_ms=int(missing_for * 1000),
                    label=self.target.label,
                )
            self.state = "REACQUIRE"
            return

        if missing_for < self.cfg.home_timeout:
            if self.state != "LOST":
                self._record_event(
                    "target_missing",
                    phase="lost",
                    missing_ms=int(missing_for * 1000),
                    label=self.target.label,
                )
            self.state = "LOST"
            return

        old_label = self.target.label
        if self.cfg.return_home_on_lost and not self._home_sent:
            ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
            now2 = time.monotonic()
            if ok:
                self.home_returns += 1
                self._home_hold_until = now2 + self.cfg.home_settle_time
                self._move_hold_until = self._home_hold_until
                self._zoom_hold_until = self._home_hold_until
                self.logger.info(
                    "%s lost for %.1fs; returned to preset %d",
                    old_label,
                    missing_for,
                    self.cfg.home_preset,
                )
            self._home_sent = bool(ok)

        self._record_event(
            "target_released",
            label=old_label,
            missing_ms=int(missing_for * 1000),
            returned_home=bool(self.cfg.return_home_on_lost and self._home_sent),
        )
        self.target = None
        self._last_error_x = None
        self._last_error_y = None
        self._last_target_span = None
        self._last_move_frame_seq = -10_000
        self._last_move_point = None
        self._last_zoom_position = None
        self._last_zoom_status_at = 0.0
        self.state = "HOME" if self._home_sent else "SEARCHING"

    def _associate(self, detections: List[Detection], frame_shape: Tuple[int, ...], now: float) -> Optional[Detection]:
        if self.target is None or not detections:
            return None
        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))
        px, py = self.target.predicted_center(now)
        old_area = max(1.0, (self.target.bbox[2] - self.target.bbox[0]) * (self.target.bbox[3] - self.target.bbox[1]))
        camera_recently_moved = now < max(self._move_hold_until, self._zoom_hold_until)

        best: Optional[Detection] = None
        best_score = -1.0
        best_dist_norm = 999.0
        for det in detections:
            if det.class_id != self.target.class_id:
                continue
            cx, cy = det.center
            dist_norm = math.hypot(cx - px, cy - py) / diag
            proximity_span = 0.70 if camera_recently_moved else 0.45
            proximity = max(0.0, 1.0 - (dist_norm / proximity_span))
            overlap = _iou(self.target.bbox, det.bbox)
            size_similarity = min(old_area, det.area) / max(old_area, det.area)
            score = 0.40 * overlap + 0.40 * proximity + 0.10 * size_similarity + 0.10 * det.confidence
            if score > best_score:
                best_score = score
                best = det
                best_dist_norm = dist_norm

        max_dist = self.cfg.association_moving_distance if camera_recently_moved else self.cfg.association_idle_distance
        if best is not None and (best_score >= 0.20 or best_dist_norm <= max_dist):
            return best
        return None

    async def _drive_to_target(self, frame_shape: Tuple[int, ...], seq: int, now: float) -> None:
        if self.target is None:
            return

        h, w = frame_shape[:2]
        cx, cy = self.target.center
        err_x = (cx - (w / 2.0)) / (w / 2.0)
        err_y = (cy - (h / 2.0)) / (h / 2.0)
        self._last_error_x = err_x
        self._last_error_y = err_y

        x1, y1, x2, y2 = self.target.bbox
        width_ratio = max(0.0, (x2 - x1) / max(1.0, float(w)))
        height_ratio = max(0.0, (y2 - y1) / max(1.0, float(h)))
        target_span = max(width_ratio, height_ratio)
        self._last_target_span = target_span

        outside_deadzone = (
            abs(err_x) > self.cfg.move_deadzone_x
            or abs(err_y) > self.cfg.move_deadzone_y
        )

        # Tracking movement has priority over zoom. A single point-only moveDirectly
        # request replaces the old start/sleep/stop PTZ control loop.
        if (
            self.cfg.move_directly_enabled
            and outside_deadzone
            and now >= self._move_hold_until
            and now >= self._zoom_hold_until
            and (seq - self._last_move_frame_seq) >= self.cfg.move_min_frame_advance
        ):
            t0 = time.monotonic()
            center = self.target.center
            scaled = self.ptz._scale_point(center, frame_shape)
            ok = await asyncio.to_thread(self.ptz.move_directly_point, center, frame_shape)
            t1 = time.monotonic()
            if ok:
                self.ptz_commands += 1
                self.move_direct_commands += 1
                self._last_move_frame_seq = seq
                self._last_move_point = scaled
                self._move_hold_until = t1 + self.cfg.move_cooldown
                self._record_event(
                    "move_directly",
                    label=self.target.label,
                    point_8192=[scaled[0], scaled[1]],
                    center_px=[round(center[0], 1), round(center[1], 1)],
                    error_x=round(err_x, 3),
                    error_y=round(err_y, 3),
                    target_span=round(target_span, 3),
                    confidence=round(self.target.confidence, 3),
                    http_ms=int((t1 - t0) * 1000),
                    cooldown_ms=int(self.cfg.move_cooldown * 1000),
                )
            else:
                self._record_event("move_directly_failed", error=self.ptz.last_error)
            return

        # Do not zoom while the subject is far off-center. Center first, then make
        # infrequent bounded zoom adjustments. This keeps pan/tilt behavior simple.
        if not self.cfg.autozoom or outside_deadzone:
            return
        if now < max(self._move_hold_until, self._zoom_hold_until):
            return
        if (now - self._last_zoom_command_at) < self.cfg.zoom_cooldown:
            return

        if (
            self._last_zoom_position is None
            or (now - self._last_zoom_status_at) >= self.cfg.zoom_status_interval
        ):
            status = await asyncio.to_thread(self.ptz.get_status)
            self._last_zoom_status_at = time.monotonic()
            pos = self.ptz.zoom_position_from_status(status)
            if pos is not None:
                self._last_zoom_position = pos

        if self._last_zoom_position is None:
            return

        direction: Optional[str] = None
        duration_ms = 0
        # Leave a 5% position guard below the configured max before starting a
        # tele step. Timed CGI zoom steps are intentionally conservative, and
        # this extra headroom makes the configured max a practical hard ceiling
        # even though the camera does not expose an atomic absolute-zoom command
        # through the path we are using here.
        zoom_in_guard = max(0.50, self.cfg.zoom_max_position * 0.05)
        if (
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
        self._last_zoom_command_at = t1
        self._zoom_hold_until = t1 + self.cfg.zoom_settle_time
        self._move_hold_until = max(self._move_hold_until, self._zoom_hold_until)
        self._last_zoom_status_at = 0.0
        self._last_zoom_position = None

        if ok:
            self.ptz_commands += 1
            self.zoom_commands += 1
            self._record_event(
                "zoom_step",
                label=self.target.label,
                direction=direction,
                duration_ms=duration_ms,
                target_span=round(target_span, 3),
                zoom_before=None if before is None else round(before, 3),
                min_position=round(self.cfg.zoom_min_position, 3),
                max_position=round(self.cfg.zoom_max_position, 3),
                http_ms=int((t1 - t0) * 1000),
                settle_ms=int(self.cfg.zoom_settle_time * 1000),
            )
        else:
            self._record_event("zoom_step_failed", direction=direction, error=self.ptz.last_error)

    def _make_debug_frame(self, frame: np.ndarray, detections: List[Detection]) -> None:
        try:
            debug = frame.copy()
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
                cv2.putText(
                    debug,
                    f"{det.label} {det.confidence:.2f}",
                    (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 190, 255),
                    1,
                    cv2.LINE_AA,
                )

            if self.target is not None:
                x1, y1, x2, y2 = [int(v) for v in self.target.bbox]
                tx, ty = [int(v) for v in self.target.center]
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(debug, (cx, cy), (tx, ty), (0, 255, 0), 1)
                cv2.circle(debug, (tx, ty), 4, (0, 255, 0), -1)

            target_text = ""
            if self.target is not None:
                target_text = f" {self.target.label} {self.target.confidence:.2f}"
            zoom_text = "?"
            if self._last_zoom_position is not None:
                zoom_text = f"{self._last_zoom_position / self.cfg.zoom_wide_position:.1f}x"
            text = (
                f"{self.state}{target_text} mode=3D "
                f"err={self._last_error_x if self._last_error_x is not None else 0:+.2f},"
                f"{self._last_error_y if self._last_error_y is not None else 0:+.2f} "
                f"zoom={zoom_text} infer={self._last_inference_ms}ms"
            )
            cv2.putText(debug, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            ok, encoded = cv2.imencode(".jpg", debug, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                self._debug_jpeg = encoded.tobytes()
        except Exception:
            pass
