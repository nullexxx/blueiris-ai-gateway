from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
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


# Standard Ultralytics YOLO11 COCO class IDs used by the tracker.
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
    fps: float = 8.0
    acquire_conf: float = 0.40
    hold_conf: float = 0.22
    acquire_frames: int = 2

    deadzone_x_in: float = 0.12
    deadzone_x_out: float = 0.16
    deadzone_y_in: float = 0.10
    deadzone_y_out: float = 0.14
    max_pan_speed: int = 5
    max_tilt_speed: int = 4
    command_keepalive: float = 0.55

    coast_time: float = 0.25
    reacquire_time: float = 1.50
    home_timeout: float = 6.0
    home_preset: int = 5
    goto_home_on_start: bool = False
    return_home_on_lost: bool = True

    ptz_http_timeout: float = 0.35
    ptz_command_timeout: int = 1
    frame_stale_timeout: float = 1.0

    # Kept off for the first tuning pass. The camera supports this and the
    # method is implemented below, but smooth continuous tracking comes first.
    move_directly_enabled: bool = False

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
            fps=_env_float("TRACKER_FPS", 8.0),
            acquire_conf=_env_float("TRACKER_ACQUIRE_CONF", 0.40),
            hold_conf=_env_float("TRACKER_HOLD_CONF", 0.22),
            acquire_frames=max(1, _env_int("TRACKER_ACQUIRE_FRAMES", 2)),
            deadzone_x_in=_env_float("TRACKER_DEADZONE_X_IN", 0.12),
            deadzone_x_out=_env_float("TRACKER_DEADZONE_X_OUT", 0.16),
            deadzone_y_in=_env_float("TRACKER_DEADZONE_Y_IN", 0.10),
            deadzone_y_out=_env_float("TRACKER_DEADZONE_Y_OUT", 0.14),
            max_pan_speed=max(1, min(8, _env_int("TRACKER_MAX_PAN_SPEED", 5))),
            max_tilt_speed=max(1, min(8, _env_int("TRACKER_MAX_TILT_SPEED", 4))),
            command_keepalive=max(0.15, _env_float("TRACKER_COMMAND_KEEPALIVE", 0.55)),
            coast_time=max(0.0, _env_float("TRACKER_COAST_TIME", 0.25)),
            reacquire_time=max(0.1, _env_float("TRACKER_REACQUIRE_TIME", 1.50)),
            home_timeout=max(0.5, _env_float("TRACKER_HOME_TIMEOUT", 6.0)),
            home_preset=max(1, _env_int("TRACKER_HOME_PRESET", 5)),
            goto_home_on_start=_env_bool("TRACKER_GOTO_HOME_ON_START", False),
            return_home_on_lost=_env_bool("TRACKER_RETURN_HOME_ON_LOST", True),
            ptz_http_timeout=max(0.05, _env_float("TRACKER_PTZ_HTTP_TIMEOUT", 0.35)),
            ptz_command_timeout=max(1, _env_int("TRACKER_PTZ_COMMAND_TIMEOUT", 1)),
            frame_stale_timeout=max(0.25, _env_float("TRACKER_FRAME_STALE_TIMEOUT", 1.0)),
            move_directly_enabled=_env_bool("TRACKER_MOVE_DIRECTLY_ENABLED", False),
        )

        # Keep priority limited to enabled classes, then append any enabled
        # classes omitted from the priority list so every target remains selectable.
        cfg.target_priority = [name for name in cfg.target_priority if name in cfg.target_classes]
        for name in cfg.target_classes:
            if name not in cfg.target_priority:
                cfg.target_priority.append(name)

        # Sanity constraints for hysteresis.
        cfg.deadzone_x_in = max(0.0, min(0.8, cfg.deadzone_x_in))
        cfg.deadzone_x_out = max(cfg.deadzone_x_in, min(0.9, cfg.deadzone_x_out))
        cfg.deadzone_y_in = max(0.0, min(0.8, cfg.deadzone_y_in))
        cfg.deadzone_y_out = max(cfg.deadzone_y_in, min(0.9, cfg.deadzone_y_out))
        cfg.fps = max(1.0, min(30.0, cfg.fps))
        cfg.acquire_conf = max(0.01, min(0.99, cfg.acquire_conf))
        cfg.hold_conf = max(0.01, min(cfg.acquire_conf, cfg.hold_conf))
        cfg.reacquire_time = max(cfg.coast_time, cfg.reacquire_time)
        cfg.home_timeout = max(cfg.reacquire_time, cfg.home_timeout)
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
            "deadzone_x_in": self.deadzone_x_in,
            "deadzone_x_out": self.deadzone_x_out,
            "deadzone_y_in": self.deadzone_y_in,
            "deadzone_y_out": self.deadzone_y_out,
            "max_pan_speed": self.max_pan_speed,
            "max_tilt_speed": self.max_tilt_speed,
            "coast_time": self.coast_time,
            "reacquire_time": self.reacquire_time,
            "home_timeout": self.home_timeout,
            "home_preset": self.home_preset,
            "move_directly_enabled": self.move_directly_enabled,
        }


class LatestFrameCapture:
    """Continuously drains RTSP and keeps only the newest decoded frame."""

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
        self._thread = threading.Thread(target=self._run, name="dog-tracker-rtsp", daemon=True)
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
    """Small Dahua/Amcrest CGI client using persistent Digest auth."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.base = f"http://{cfg.camera_ip}"
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(cfg.camera_user, cfg.camera_password)
        self._lock = threading.Lock()
        self.current_vector = (0, 0, 0)
        self.last_command_time = 0.0
        self.last_error: Optional[str] = None
        self.last_http_ms: Optional[int] = None

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

    def continuous(self, pan: int, tilt: int, zoom: int = 0, force: bool = False) -> bool:
        pan = max(-8, min(8, int(pan)))
        tilt = max(-8, min(8, int(tilt)))
        zoom = max(-100, min(100, int(zoom)))
        vector = (pan, tilt, zoom)
        now = time.monotonic()

        if vector == (0, 0, 0):
            return self.stop(force=force)

        if (
            not force
            and vector == self.current_vector
            and (now - self.last_command_time) < self.cfg.command_keepalive
        ):
            return True

        try:
            self._get(
                "/cgi-bin/ptz.cgi",
                {
                    "action": "start",
                    "channel": self.cfg.camera_channel,
                    "code": "Continuously",
                    "arg1": pan,
                    "arg2": tilt,
                    "arg3": zoom,
                    "arg4": self.cfg.ptz_command_timeout,
                },
            )
            self.current_vector = vector
            self.last_command_time = now
            return True
        except Exception:
            return False

    def stop(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and self.current_vector == (0, 0, 0):
            return True
        try:
            self._get(
                "/cgi-bin/ptz.cgi",
                {
                    "action": "stop",
                    "channel": self.cfg.camera_channel,
                    "code": "Continuously",
                    "arg1": 0,
                    "arg2": 0,
                    "arg3": 0,
                    "arg4": 0,
                },
            )
            self.current_vector = (0, 0, 0)
            self.last_command_time = now
            return True
        except Exception:
            # Even if the explicit stop fails, the 1-second timeout on every
            # continuous command is the second safety layer.
            self.current_vector = (0, 0, 0)
            self.last_command_time = now
            return False

    def goto_preset(self, preset: int) -> bool:
        try:
            self.stop(force=True)
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
            self.current_vector = (0, 0, 0)
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

    def move_directly(self, bbox: Tuple[float, float, float, float], frame_shape: Tuple[int, int, int], padding: float = 2.5) -> bool:
        """3D-position to a padded target rectangle. Not enabled by default."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(4.0, (x2 - x1) * padding)
        bh = max(4.0, (y2 - y1) * padding)
        px1 = max(0.0, cx - bw / 2.0)
        py1 = max(0.0, cy - bh / 2.0)
        px2 = min(float(w - 1), cx + bw / 2.0)
        py2 = min(float(h - 1), cy + bh / 2.0)

        def scale_x(x: float) -> int:
            return max(0, min(8192, int(round((x / max(1, w - 1)) * 8192))))

        def scale_y(y: float) -> int:
            return max(0, min(8192, int(round((y / max(1, h - 1)) * 8192))))

        try:
            self.stop(force=True)
            self._get(
                "/cgi-bin/ptzBase.cgi",
                {
                    "action": "moveDirectly",
                    "channel": self.cfg.camera_channel,
                    "startPoint[0]": scale_x(px1),
                    "startPoint[1]": scale_y(py1),
                    "endPoint[0]": scale_x(px2),
                    "endPoint[1]": scale_y(py2),
                },
            )
            return True
        except Exception:
            return False


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
        dt = min(0.5, max(0.0, now - self.last_update))
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
        self._x_active = False
        self._y_active = False
        self._last_pan = 0
        self._last_tilt = 0
        self._home_sent = False
        self._last_error_x: Optional[float] = None
        self._last_error_y: Optional[float] = None
        self._last_detection_count = 0
        self._last_inference_ms = 0
        self._last_inference_outcome = "never"
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        self._debug_jpeg: Optional[bytes] = None
        self._inference_times: Deque[float] = deque(maxlen=128)

        self.total_inferences = 0
        self.frames_skipped_gpu_busy = 0
        self.frames_skipped_duplicate = 0
        self.frames_stale = 0
        self.ptz_commands = 0
        self.targets_acquired = 0
        self.home_returns = 0

    async def initialize(self) -> None:
        if not self.cfg.camera_ip:
            raise RuntimeError("TRACKER_CAMERA_IP is required when TRACKER_ENABLED=true")
        if not self.cfg.camera_password:
            self.logger.warning("PTZ tracker camera password is empty.")
        self.capture.start()
        self._task = asyncio.create_task(self._run(), name="multi-ptz-tracker")
        self.logger.info(
            "PTZ tracker initialized: camera=%s model=%s fps=%.1f autostart=%s",
            self.cfg.camera_ip,
            self.cfg.model_name,
            self.cfg.fps,
            self.cfg.autostart,
        )
        if self.cfg.autostart:
            await self.start()

    async def shutdown(self) -> None:
        self._shutdown = True
        self.active = False
        self.state = "SHUTDOWN"
        try:
            await asyncio.to_thread(self.ptz.stop, True)
        except Exception:
            pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self.capture.stop)
        self.ptz.close()

    async def start(self) -> dict:
        self.active = True
        self.state = "SEARCHING"
        self.target = None
        self._home_sent = False
        self._x_active = False
        self._y_active = False
        self._last_pan = 0
        self._last_tilt = 0
        if self.cfg.goto_home_on_start:
            await self.home()
        self.logger.info("PTZ tracker STARTED")
        return self.status()

    async def stop(self) -> dict:
        self.active = False
        self.state = "OFF"
        self.target = None
        self._x_active = False
        self._y_active = False
        self._last_pan = 0
        self._last_tilt = 0
        await asyncio.to_thread(self.ptz.stop, True)
        self.logger.info("PTZ tracker STOPPED")
        return self.status()

    async def home(self) -> dict:
        self.target = None
        self._home_sent = True
        self._x_active = False
        self._y_active = False
        self._last_pan = 0
        self._last_tilt = 0
        self.state = "HOME" if self.active else "OFF"
        ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
        if ok:
            self.home_returns += 1
        return self.status()

    def debug_jpeg(self) -> Optional[bytes]:
        return self._debug_jpeg

    def status(self) -> dict:
        now = time.monotonic()
        frame, seq, frame_time = self.capture.latest()
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
            }

        # 5-second rolling inference rate.
        cutoff = now - 5.0
        while self._inference_times and self._inference_times[0] < cutoff:
            self._inference_times.popleft()
        infer_fps = len(self._inference_times) / 5.0

        return {
            "enabled": self.cfg.enabled,
            "active": self.active,
            "state": self.state,
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
            "last_detection_count": self._last_detection_count,
            "ptz_vector": list(self.ptz.current_vector),
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
                    await self._stop_ptz_if_needed()
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
                    # Do not treat a skipped inference as a negative observation.
                    # The camera's 1-second command timeout remains the safety net.
                    continue
                if outcome != "ok":
                    self.state = "INFERENCE_ERROR"
                    await self._stop_ptz_if_needed()
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
                await self._process_observation(frame, detections, time.monotonic())
                self._make_debug_frame(frame, detections)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state = "TRACKER_ERROR"
            self.logger.exception("PTZ tracker loop crashed: %s", exc)
            try:
                await self._stop_ptz_if_needed(force=True)
            except Exception:
                pass

    async def _process_observation(self, frame: np.ndarray, detections: List[Detection], now: float) -> None:
        if self.target is None:
            candidates = [d for d in detections if d.confidence >= self.cfg.acquire_conf]
            if not candidates:
                self.state = "HOME" if self._home_sent else "SEARCHING"
                self._last_error_x = None
                self._last_error_y = None
                await self._stop_ptz_if_needed()
                return

            # Priority is applied only while acquiring a new target. Once a target
            # is locked, another class cannot steal the lock until the current target
            # is genuinely lost. Within the same priority class, favor confidence
            # with a mild preference for a larger/closer detection.
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
            self._home_sent = False
            await self._stop_ptz_if_needed()
            return

        matched = self._associate(detections, frame.shape, now)
        if matched is not None:
            was_acquiring = self.target.acquire_hits < self.cfg.acquire_frames
            self.target.update(matched, now)
            self._home_sent = False

            if was_acquiring and self.target.acquire_hits < self.cfg.acquire_frames:
                self.state = "ACQUIRE"
                await self._stop_ptz_if_needed()
                return

            if was_acquiring and self.target.acquire_hits == self.cfg.acquire_frames:
                self.targets_acquired += 1
                self.logger.info(
                    "%s target acquired: conf=%.2f bbox=%s",
                    self.target.label,
                    self.target.confidence,
                    tuple(round(v, 1) for v in self.target.bbox),
                )

            self.state = "TRACK"
            await self._drive_to_target(frame.shape)
            return

        # If the tentative acquisition did not survive the very next observations,
        # drop it quickly instead of holding a one-frame false positive for the
        # full lost-target timeout.
        if self.target.acquire_hits < self.cfg.acquire_frames:
            self.target = None
            self.state = "SEARCHING"
            self._x_active = False
            self._y_active = False
            await self._stop_ptz_if_needed()
            return

        # Negative observation: nothing matched our sticky target.
        missing_for = now - self.target.last_seen
        if missing_for <= self.cfg.coast_time:
            self.state = "COAST"
            # Intentionally leave the last PTZ vector active for a very brief
            # period. The hardware timeout prevents runaway motion.
            return

        if missing_for <= self.cfg.reacquire_time:
            self.state = "REACQUIRE"
            await self._stop_ptz_if_needed()
            return

        if missing_for < self.cfg.home_timeout:
            self.state = "LOST"
            await self._stop_ptz_if_needed()
            return

        await self._stop_ptz_if_needed(force=True)
        if self.cfg.return_home_on_lost and not self._home_sent:
            ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
            if ok:
                self.home_returns += 1
                self.logger.info(
                    "%s lost for %.1fs; returned to preset %d",
                    self.target.label if self.target is not None else "target",
                    missing_for,
                    self.cfg.home_preset,
                )
            self._home_sent = True
        self.target = None
        self._x_active = False
        self._y_active = False
        self._last_pan = 0
        self._last_tilt = 0
        self._last_error_x = None
        self._last_error_y = None
        self.state = "HOME" if self._home_sent else "SEARCHING"

    def _associate(self, detections: List[Detection], frame_shape: Tuple[int, int, int], now: float) -> Optional[Detection]:
        if self.target is None or not detections:
            return None
        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))
        px, py = self.target.predicted_center(now)
        old_area = max(1.0, (self.target.bbox[2] - self.target.bbox[0]) * (self.target.bbox[3] - self.target.bbox[1]))

        best: Optional[Detection] = None
        best_score = -1.0
        best_dist_norm = 999.0
        for det in detections:
            # Sticky multi-class lock: while tracking a dog, person, cat, etc.,
            # detections from other classes are ignored until that target is lost.
            if det.class_id != self.target.class_id:
                continue
            cx, cy = det.center
            dist_norm = math.hypot(cx - px, cy - py) / diag
            proximity = max(0.0, 1.0 - (dist_norm / 0.45))
            overlap = _iou(self.target.bbox, det.bbox)
            size_similarity = min(old_area, det.area) / max(old_area, det.area)
            score = 0.45 * overlap + 0.35 * proximity + 0.10 * size_similarity + 0.10 * det.confidence
            if score > best_score:
                best_score = score
                best = det
                best_dist_norm = dist_norm

        # IoU can be near zero while the PTZ itself is moving, so allow a
        # proximity-based match as long as the candidate is not implausibly far away.
        if best is not None and (best_score >= 0.22 or best_dist_norm <= 0.22):
            return best
        return None

    async def _drive_to_target(self, frame_shape: Tuple[int, int, int]) -> None:
        if self.target is None:
            return
        h, w = frame_shape[:2]
        cx, cy = self.target.center

        # +X is right on this specific camera. Image Y increases downward while
        # +Y PTZ means up, therefore vertical error is intentionally inverted.
        err_x = (cx - (w / 2.0)) / (w / 2.0)
        err_y = ((h / 2.0) - cy) / (h / 2.0)
        self._last_error_x = err_x
        self._last_error_y = err_y

        desired_pan, self._x_active = self._axis_velocity(
            err_x,
            self._x_active,
            self.cfg.deadzone_x_in,
            self.cfg.deadzone_x_out,
            self.cfg.max_pan_speed,
        )
        desired_tilt, self._y_active = self._axis_velocity(
            err_y,
            self._y_active,
            self.cfg.deadzone_y_in,
            self.cfg.deadzone_y_out,
            self.cfg.max_tilt_speed,
        )

        pan = self._slew(self._last_pan, desired_pan)
        tilt = self._slew(self._last_tilt, desired_tilt)
        self._last_pan, self._last_tilt = pan, tilt

        changed = (pan, tilt, 0) != self.ptz.current_vector
        ok = await asyncio.to_thread(self.ptz.continuous, pan, tilt, 0, False)
        if ok and changed:
            self.ptz_commands += 1

    @staticmethod
    def _axis_velocity(error: float, active: bool, inner: float, outer: float, max_speed: int) -> Tuple[int, bool]:
        magnitude = abs(error)
        if active:
            if magnitude <= inner:
                return 0, False
        else:
            if magnitude <= outer:
                return 0, False
            active = True

        usable = max(0.001, 1.0 - inner)
        normalized = max(0.0, min(1.0, (magnitude - inner) / usable))
        speed = max(1, min(max_speed, int(math.ceil(normalized * max_speed))))
        return (speed if error > 0 else -speed), active

    @staticmethod
    def _slew(previous: int, desired: int) -> int:
        if previous == desired:
            return desired
        if previous != 0 and desired != 0 and (previous > 0) != (desired > 0):
            return 0
        if desired > previous:
            return previous + 1
        return previous - 1

    async def _stop_ptz_if_needed(self, force: bool = False) -> None:
        if force or self.ptz.current_vector != (0, 0, 0):
            await asyncio.to_thread(self.ptz.stop, force)
            self._last_pan = 0
            self._last_tilt = 0

    def _make_debug_frame(self, frame: np.ndarray, detections: List[Detection]) -> None:
        try:
            debug = frame.copy()
            h, w = debug.shape[:2]
            cx, cy = w // 2, h // 2

            # Outer deadzone = threshold that starts movement.
            left = int(cx - self.cfg.deadzone_x_out * (w / 2.0))
            right = int(cx + self.cfg.deadzone_x_out * (w / 2.0))
            top = int(cy - self.cfg.deadzone_y_out * (h / 2.0))
            bottom = int(cy + self.cfg.deadzone_y_out * (h / 2.0))
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
            text = (
                f"{self.state}{target_text} PTZ={self.ptz.current_vector[0]},{self.ptz.current_vector[1]} "
                f"err={self._last_error_x if self._last_error_x is not None else 0:+.2f},"
                f"{self._last_error_y if self._last_error_y is not None else 0:+.2f} "
                f"infer={self._last_inference_ms}ms"
            )
            cv2.putText(debug, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            ok, encoded = cv2.imencode(".jpg", debug, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                self._debug_jpeg = encoded.tobytes()
        except Exception:
            # Debug rendering must never affect tracking.
            pass
