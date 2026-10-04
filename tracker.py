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

    # Native 3D moves take roughly 1.5-2.1 seconds on this camera. Correct only
    # part of the current error and add a bounded subject-velocity lead so we do
    # not chase a point that is already stale by the time the PTZ arrives.
    move_gain: float = 0.60
    lead_time: float = 0.75

    # Predictive lead is only trusted after a real stationary-camera sample
    # window and while the detection is large/clean enough to measure reliably.
    velocity_min_sample_ms: int = 100
    lead_min_conf: float = 0.50
    lead_min_span: float = 0.08
    lead_edge_margin: float = 0.02

    # Bounded in-flight retargeting. moveDirectly can take ~1.5 s on this camera,
    # so allow a small number of destination updates while it is already slewing
    # when the target reverses or is escaping toward a frame edge. Predictive lead
    # remains disabled during PTZ motion because image motion is not subject motion.
    inflight_retarget_enabled: bool = True
    inflight_retarget_interval: float = 0.30
    inflight_retarget_max: int = 2
    inflight_reversal_error: float = 0.22
    escape_error: float = 0.72
    escape_gain: float = 0.85

    # High-speed continuous chase controller. Continuously exposes signed motor
    # speeds (-8..8), unlike moveDirectly where firmware chooses the positional
    # slew. Start at speed 5 because this Dahua API family documents movement
    # outside the +/-4 soft region, then scale to speed 8 as frame error grows.
    continuous_chase_enabled: bool = True
    continuous_min_speed: int = 5
    continuous_max_speed: int = 8
    continuous_full_speed_error: float = 0.60
    continuous_command_interval: float = 0.15
    continuous_keepalive: float = 0.45
    continuous_camera_timeout: int = 1

    # Native PTZ operation tracking. Instead of guessing how long a 3D move takes,
    # poll getStatus until the camera reports idle and its reported position is stable.
    ptz_status_poll_interval: float = 0.12
    ptz_operation_timeout: float = 4.0
    post_move_frames: int = 1

    # Bounded, deliberately conservative auto-zoom. Camera reports 5.12 at
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

    coast_time: float = 0.20
    reacquire_time: float = 1.25
    home_timeout: float = 3.0
    home_preset: int = 5
    goto_home_on_start: bool = True
    return_home_on_lost: bool = True

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
            move_gain=_env_float("TRACKER_MOVE_GAIN", 0.60),
            lead_time=_env_float("TRACKER_LEAD_TIME", 0.75),
            velocity_min_sample_ms=_env_int("TRACKER_VELOCITY_MIN_SAMPLE_MS", 100),
            lead_min_conf=_env_float("TRACKER_LEAD_MIN_CONF", 0.50),
            lead_min_span=_env_float("TRACKER_LEAD_MIN_SPAN", 0.08),
            lead_edge_margin=_env_float("TRACKER_LEAD_EDGE_MARGIN", 0.02),
            inflight_retarget_enabled=_env_bool("TRACKER_INFLIGHT_RETARGET_ENABLED", True),
            inflight_retarget_interval=_env_float("TRACKER_INFLIGHT_RETARGET_INTERVAL", 0.30),
            inflight_retarget_max=_env_int("TRACKER_INFLIGHT_RETARGET_MAX", 2),
            inflight_reversal_error=_env_float("TRACKER_INFLIGHT_REVERSAL_ERROR", 0.22),
            escape_error=_env_float("TRACKER_ESCAPE_ERROR", 0.72),
            escape_gain=_env_float("TRACKER_ESCAPE_GAIN", 0.85),
            continuous_chase_enabled=_env_bool("TRACKER_CONTINUOUS_CHASE_ENABLED", True),
            continuous_min_speed=_env_int("TRACKER_CONTINUOUS_MIN_SPEED", 5),
            continuous_max_speed=_env_int("TRACKER_CONTINUOUS_MAX_SPEED", 8),
            continuous_full_speed_error=_env_float("TRACKER_CONTINUOUS_FULL_SPEED_ERROR", 0.60),
            continuous_command_interval=_env_float("TRACKER_CONTINUOUS_COMMAND_INTERVAL", 0.15),
            continuous_keepalive=_env_float("TRACKER_CONTINUOUS_KEEPALIVE", 0.45),
            continuous_camera_timeout=_env_int("TRACKER_CONTINUOUS_CAMERA_TIMEOUT", 1),
            ptz_status_poll_interval=_env_float("TRACKER_PTZ_STATUS_POLL_INTERVAL", 0.12),
            ptz_operation_timeout=_env_float("TRACKER_PTZ_OPERATION_TIMEOUT", 4.0),
            post_move_frames=_env_int("TRACKER_POST_MOVE_FRAMES", 1),
            autozoom=_env_bool("TRACKER_AUTOZOOM", True),
            zoom_wide_position=_env_float("TRACKER_ZOOM_WIDE_POSITION", 5.12),
            zoom_min_factor=_env_float("TRACKER_ZOOM_MIN_FACTOR", 1.0),
            zoom_max_factor=_env_float("TRACKER_ZOOM_MAX_FACTOR", 6.0),
            zoom_target_min=_env_float("TRACKER_ZOOM_TARGET_MIN", 0.18),
            zoom_target_max=_env_float("TRACKER_ZOOM_TARGET_MAX", 0.42),
            zoom_in_step_ms=_env_int("TRACKER_ZOOM_IN_STEP_MS", 90),
            zoom_out_step_ms=_env_int("TRACKER_ZOOM_OUT_STEP_MS", 140),
            zoom_cooldown=_env_float("TRACKER_ZOOM_COOLDOWN", 1.0),
            coast_time=_env_float("TRACKER_COAST_TIME", 0.20),
            reacquire_time=_env_float("TRACKER_REACQUIRE_TIME", 1.25),
            home_timeout=_env_float("TRACKER_HOME_TIMEOUT", 3.0),
            home_preset=max(1, _env_int("TRACKER_HOME_PRESET", 5)),
            goto_home_on_start=_env_bool("TRACKER_GOTO_HOME_ON_START", True),
            return_home_on_lost=_env_bool("TRACKER_RETURN_HOME_ON_LOST", True),
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
        cfg.move_gain = max(0.20, min(1.00, cfg.move_gain))
        cfg.lead_time = max(0.0, min(1.50, cfg.lead_time))
        cfg.velocity_min_sample_ms = max(50, min(500, cfg.velocity_min_sample_ms))
        cfg.lead_min_conf = max(cfg.hold_conf, min(0.95, cfg.lead_min_conf))
        cfg.lead_min_span = max(0.02, min(0.50, cfg.lead_min_span))
        cfg.lead_edge_margin = max(0.0, min(0.10, cfg.lead_edge_margin))
        cfg.inflight_retarget_interval = max(0.20, min(1.0, cfg.inflight_retarget_interval))
        cfg.inflight_retarget_max = max(0, min(4, cfg.inflight_retarget_max))
        cfg.inflight_reversal_error = max(0.10, min(0.80, cfg.inflight_reversal_error))
        cfg.escape_error = max(0.50, min(0.95, cfg.escape_error))
        cfg.escape_gain = max(cfg.move_gain, min(1.0, cfg.escape_gain))
        cfg.continuous_min_speed = max(1, min(8, cfg.continuous_min_speed))
        cfg.continuous_max_speed = max(cfg.continuous_min_speed, min(8, cfg.continuous_max_speed))
        cfg.continuous_full_speed_error = max(
            max(cfg.move_deadzone_x, cfg.move_deadzone_y) + 0.05,
            min(0.95, cfg.continuous_full_speed_error),
        )
        cfg.continuous_command_interval = max(0.05, min(0.50, cfg.continuous_command_interval))
        cfg.continuous_camera_timeout = max(1, min(5, cfg.continuous_camera_timeout))
        cfg.continuous_keepalive = max(
            cfg.continuous_command_interval,
            min(cfg.continuous_camera_timeout * 0.75, cfg.continuous_keepalive),
        )
        cfg.ptz_status_poll_interval = max(0.05, min(1.0, cfg.ptz_status_poll_interval))
        cfg.ptz_operation_timeout = max(0.75, min(15.0, cfg.ptz_operation_timeout))
        cfg.post_move_frames = max(1, min(20, cfg.post_move_frames))

        cfg.zoom_wide_position = max(0.01, cfg.zoom_wide_position)
        cfg.zoom_min_factor = max(1.0, cfg.zoom_min_factor)
        cfg.zoom_max_factor = max(cfg.zoom_min_factor, min(25.0, cfg.zoom_max_factor))
        cfg.zoom_target_min = max(0.03, min(0.80, cfg.zoom_target_min))
        cfg.zoom_target_max = max(cfg.zoom_target_min + 0.02, min(0.95, cfg.zoom_target_max))
        cfg.zoom_in_step_ms = max(30, min(500, cfg.zoom_in_step_ms))
        cfg.zoom_out_step_ms = max(30, min(700, cfg.zoom_out_step_ms))
        cfg.zoom_cooldown = max(0.25, min(10.0, cfg.zoom_cooldown))

        cfg.coast_time = max(0.0, min(2.0, cfg.coast_time))
        cfg.reacquire_time = max(cfg.coast_time, min(10.0, cfg.reacquire_time))
        cfg.home_timeout = max(cfg.reacquire_time, min(60.0, cfg.home_timeout))
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
            "move_gain": self.move_gain,
            "lead_time": self.lead_time,
            "velocity_min_sample_ms": self.velocity_min_sample_ms,
            "lead_min_conf": self.lead_min_conf,
            "lead_min_span": self.lead_min_span,
            "lead_edge_margin": self.lead_edge_margin,
            "inflight_retarget_enabled": self.inflight_retarget_enabled,
            "inflight_retarget_interval": self.inflight_retarget_interval,
            "inflight_retarget_max": self.inflight_retarget_max,
            "inflight_reversal_error": self.inflight_reversal_error,
            "escape_error": self.escape_error,
            "escape_gain": self.escape_gain,
            "continuous_chase_enabled": self.continuous_chase_enabled,
            "continuous_min_speed": self.continuous_min_speed,
            "continuous_max_speed": self.continuous_max_speed,
            "continuous_full_speed_error": self.continuous_full_speed_error,
            "continuous_command_interval": self.continuous_command_interval,
            "continuous_keepalive": self.continuous_keepalive,
            "continuous_camera_timeout": self.continuous_camera_timeout,
            "ptz_status_poll_interval": self.ptz_status_poll_interval,
            "ptz_operation_timeout": self.ptz_operation_timeout,
            "post_move_frames": self.post_move_frames,
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
            "coast_time": self.coast_time,
            "reacquire_time": self.reacquire_time,
            "home_timeout": self.home_timeout,
            "home_preset": self.home_preset,
            "goto_home_on_start": self.goto_home_on_start,
            "return_home_on_lost": self.return_home_on_lost,
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
    def position_from_status(status: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
        try:
            return (
                float(status["status.Postion[0]"]),
                float(status["status.Postion[1]"]),
                float(status["status.Postion[2]"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _idle_value(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in ("idle", "stop", "stopped"):
            return True
        if normalized in ("moving", "move", "running", "busy"):
            return False
        return None

    @classmethod
    def pan_tilt_reported_idle(cls, status: Dict[str, str]) -> Optional[bool]:
        values = [
            cls._idle_value(status.get("status.MoveStatus")),
            cls._idle_value(status.get("status.PanTiltStatus")),
        ]
        known = [value for value in values if value is not None]
        return None if not known else all(known)

    @classmethod
    def zoom_reported_idle(cls, status: Dict[str, str]) -> Optional[bool]:
        value = cls._idle_value(status.get("status.ZoomStatus"))
        if value is not None:
            return value
        return cls._idle_value(status.get("status.MoveStatus"))

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

    def continuous_move(self, pan_speed: int, tilt_speed: int, timeout_s: int = 1) -> bool:
        """Drive pan/tilt continuously with signed Dahua speeds (-8..8)."""
        pan_speed = max(-8, min(8, int(pan_speed)))
        tilt_speed = max(-8, min(8, int(tilt_speed)))
        timeout_s = max(1, min(5, int(timeout_s)))
        if pan_speed == 0 and tilt_speed == 0:
            return self.continuous_stop()
        try:
            self._get(
                "/cgi-bin/ptz.cgi",
                {
                    "action": "start",
                    "channel": self.cfg.camera_channel,
                    "code": "Continuously",
                    "arg1": pan_speed,
                    "arg2": tilt_speed,
                    "arg3": 0,
                    "arg4": timeout_s,
                },
            )
            self.last_move_point = None
            return True
        except Exception:
            return False

    def continuous_stop(self) -> bool:
        """Immediately stop continuous pan/tilt movement."""
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
    velocity_reference_center: Optional[Tuple[float, float]] = None
    velocity_reference_time: Optional[float] = None
    velocity_sample_ms: int = 0
    velocity_valid: bool = False

    def rebase_velocity(self, now: float) -> None:
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

    def update(
        self,
        det: Detection,
        now: float,
        *,
        update_velocity: bool = True,
        min_velocity_sample_s: float = 0.10,
    ) -> None:
        if det.class_id != self.class_id:
            raise ValueError("TargetTrack.update received a different object class")

        new_center = det.center
        if update_velocity:
            if self.velocity_reference_center is None or self.velocity_reference_time is None:
                self.velocity_reference_center = self.center
                self.velocity_reference_time = self.last_update

            sample_s = max(0.0, now - self.velocity_reference_time)
            self.velocity_sample_ms = int(sample_s * 1000)
            if sample_s >= max(0.001, min_velocity_sample_s):
                ref_x, ref_y = self.velocity_reference_center
                inst_vx = (new_center[0] - ref_x) / sample_s
                inst_vy = (new_center[1] - ref_y) / sample_s
                alpha = 0.35
                if self.velocity_valid:
                    self.vx = (1.0 - alpha) * self.vx + alpha * inst_vx
                    self.vy = (1.0 - alpha) * self.vy + alpha * inst_vy
                else:
                    self.vx = inst_vx
                    self.vy = inst_vy
                self.velocity_valid = True
                self.velocity_reference_center = new_center
                self.velocity_reference_time = now
                self.velocity_sample_ms = int(sample_s * 1000)
        else:
            # Follow the bbox for association, but never learn global
            # image motion as target velocity while the PTZ is moving.
            self.clear_velocity()

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
    """Multi-class YOLO tracker using Dahua continuous chase + bounded autozoom."""

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
        self._last_move_point: Optional[Tuple[int, int]] = None
        self._last_error_x: Optional[float] = None
        self._last_error_y: Optional[float] = None
        self._last_target_span: Optional[float] = None
        self._last_zoom_position: Optional[float] = None
        self._last_zoom_command_at = 0.0

        # One native camera operation may be in flight at a time.  We do not
        # guess when it is finished: getStatus is polled until the relevant
        # status reports idle and the reported PTZ position is stable.
        self._ptz_operation: Optional[str] = None
        self._ptz_operation_started_at = 0.0
        self._ptz_operation_deadline = 0.0
        self._ptz_next_status_poll_at = 0.0
        self._ptz_idle_polls = 0
        self._ptz_seen_motion = False
        self._ptz_last_poll_position: Optional[Tuple[float, float, float]] = None
        self._last_camera_position: Optional[Tuple[float, float, float]] = None
        self._last_camera_status: Dict[str, str] = {}
        self._last_camera_status_at = 0.0
        self._post_motion_release_seq = -1
        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        self._velocity_rebase_required = False

        # A moveDirectly chain may be retargeted a small number of times while the
        # camera is physically moving. The operation timer tracks the latest command;
        # the chain timer tracks the whole mechanical chase for diagnostics.
        self._ptz_chain_started_at = 0.0
        self._inflight_retarget_count = 0
        self._last_inflight_retarget_at = 0.0
        self._last_move_command_error: Optional[Tuple[float, float]] = None

        self._continuous_active = False
        self._continuous_pan_speed = 0
        self._continuous_tilt_speed = 0
        self._continuous_last_command_at = 0.0

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
        self.inflight_retargets = 0
        self.continuous_commands = 0
        self.continuous_stops = 0
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
        self._task = asyncio.create_task(self._run(), name="continuous-ptz-tracker")
        self.logger.info(
            "PTZ tracker initialized: camera=%s model=%s fps=%.1f mode=continuous autozoom=%s autostart=%s",
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
            continuous_chase=self.cfg.continuous_chase_enabled,
            autozoom=self.cfg.autozoom,
        )
        if self.cfg.autostart:
            await self.start()

    async def shutdown(self) -> None:
        self._shutdown = True
        self.active = False
        self.state = "SHUTDOWN"
        await self._stop_continuous("shutdown", force=True)
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
        self._last_move_point = None
        self._last_zoom_position = None
        self._last_zoom_command_at = 0.0
        self._ptz_operation = None
        self._ptz_operation_started_at = 0.0
        self._ptz_operation_deadline = 0.0
        self._ptz_next_status_poll_at = 0.0
        self._ptz_idle_polls = 0
        self._ptz_seen_motion = False
        self._ptz_last_poll_position = None
        self._post_motion_release_seq = -1
        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        self._velocity_rebase_required = False
        self._ptz_chain_started_at = 0.0
        self._inflight_retarget_count = 0
        self._last_inflight_retarget_at = 0.0
        self._last_move_command_error = None
        self._continuous_active = False
        self._continuous_pan_speed = 0
        self._continuous_tilt_speed = 0
        self._continuous_last_command_at = 0.0

    async def start(self) -> dict:
        self._history.clear()
        self._session_started_wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._session_started_mono = time.monotonic()
        self.active = True
        self.state = "SEARCHING"
        self._home_sent = False
        self._reset_tracking_state()
        if self.cfg.goto_home_on_start:
            await self.home()
        self._record_event(
            "tracker_started",
            goto_home_on_start=self.cfg.goto_home_on_start,
            continuous_chase=self.cfg.continuous_chase_enabled,
            autozoom=self.cfg.autozoom,
        )
        self.logger.info("PTZ tracker STARTED (continuous chase + status-driven home/zoom)")
        return self.status()

    async def stop(self) -> dict:
        self._record_event("tracker_stopping")
        self.active = False
        self.state = "OFF"
        await self._stop_continuous("tracker_stop", force=True)
        self._reset_tracking_state()
        self._record_event("tracker_stopped")
        self.logger.info("PTZ tracker STOPPED")
        return self.status()

    async def home(self) -> dict:
        await self._stop_continuous("home", force=True)
        self._reset_tracking_state()
        self._home_sent = True
        self.state = "HOME" if self.active else "OFF"
        ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
        now = time.monotonic()
        _, seq, _ = self.capture.latest()
        self._last_zoom_position = None
        if ok:
            self.home_returns += 1
            if self.active:
                self._begin_ptz_operation("home", seq, now)
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
                "velocity_sample_ms": self.target.velocity_sample_ms,
                "velocity_valid": self.target.velocity_valid,
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

        op_elapsed_ms = None
        op_chain_elapsed_ms = None
        op_timeout_remaining_ms = 0
        if self._ptz_operation is not None:
            op_elapsed_ms = int(max(0.0, now - self._ptz_operation_started_at) * 1000)
            if self._ptz_chain_started_at > 0:
                op_chain_elapsed_ms = int(max(0.0, now - self._ptz_chain_started_at) * 1000)
            op_timeout_remaining_ms = max(0, int((self._ptz_operation_deadline - now) * 1000))

        post_frames_remaining = max(0, self._post_motion_release_seq - seq)
        return {
            "enabled": self.cfg.enabled,
            "active": self.active,
            "session_started": self._session_started_wall,
            "history_events": len(self._history),
            "state": self.state,
            "control_mode": "continuous",
            "ptz_operation": self._ptz_operation,
            "ptz_operation_elapsed_ms": op_elapsed_ms,
            "ptz_operation_chain_elapsed_ms": op_chain_elapsed_ms,
            "ptz_operation_timeout_remaining_ms": op_timeout_remaining_ms,
            "continuous_active": self._continuous_active,
            "continuous_speed": [self._continuous_pan_speed, self._continuous_tilt_speed],
            "continuous_command_age_ms": (
                None
                if self._continuous_last_command_at <= 0
                else int(max(0.0, now - self._continuous_last_command_at) * 1000)
            ),
            "post_move_frames_remaining": post_frames_remaining,
            "camera_motion_status": {
                "move": self._last_camera_status.get("status.MoveStatus"),
                "pan_tilt": self._last_camera_status.get("status.PanTiltStatus"),
                "zoom": self._last_camera_status.get("status.ZoomStatus"),
            },
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
                "inflight_retargets": self.inflight_retargets,
                "continuous_commands": self.continuous_commands,
                "continuous_stops": self.continuous_stops,
                "targets_acquired": self.targets_acquired,
                "home_returns": self.home_returns,
            },
            "config": self.cfg.public_dict(),
        }

    async def camera_status(self) -> dict:
        return await asyncio.to_thread(self.ptz.get_status)

    def _begin_ptz_operation(self, kind: str, seq: int, now: float) -> None:
        self._ptz_operation = kind
        self._ptz_operation_started_at = now
        self._ptz_chain_started_at = now
        self._ptz_operation_deadline = now + self.cfg.ptz_operation_timeout
        self._ptz_next_status_poll_at = now + self.cfg.ptz_status_poll_interval
        self._ptz_idle_polls = 0
        self._ptz_seen_motion = False
        self._ptz_last_poll_position = self._last_camera_position
        self._post_motion_release_seq = max(self._post_motion_release_seq, seq + 1)
        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        self._inflight_retarget_count = 0
        self._last_inflight_retarget_at = now if kind == "move" else 0.0
        self._last_move_command_error = None
        if self.target is not None:
            self.target.clear_velocity()

    def _refresh_move_after_retarget(
        self,
        seq: int,
        now: float,
        error: Tuple[float, float],
    ) -> None:
        # The camera treats a new moveDirectly as a new destination. Restart the
        # latest-command settle timer, but retain the original chain start so the
        # history shows total chase time. Reset seen-motion so an immediate stale
        # Idle response cannot falsely complete the replacement command.
        self._ptz_operation = "move"
        self._ptz_operation_started_at = now
        self._ptz_operation_deadline = now + self.cfg.ptz_operation_timeout
        self._ptz_next_status_poll_at = now + self.cfg.ptz_status_poll_interval
        self._ptz_idle_polls = 0
        self._ptz_seen_motion = False
        self._ptz_last_poll_position = self._last_camera_position
        self._post_motion_release_seq = max(self._post_motion_release_seq, seq + 1)
        self._inflight_retarget_count += 1
        self._last_inflight_retarget_at = now
        self._last_move_command_error = error
        if self.target is not None:
            self.target.clear_velocity()

    @staticmethod
    def _position_stable(
        previous: Optional[Tuple[float, float, float]],
        current: Optional[Tuple[float, float, float]],
        kind: str,
    ) -> Optional[bool]:
        if previous is None or current is None:
            return None
        if kind == "zoom":
            return abs(current[2] - previous[2]) <= 0.03
        return abs(current[0] - previous[0]) <= 0.05 and abs(current[1] - previous[1]) <= 0.05

    def _finish_ptz_operation(
        self,
        seq: int,
        now: float,
        *,
        timed_out: bool,
        status: Optional[Dict[str, str]] = None,
    ) -> None:
        kind = self._ptz_operation
        if kind is None:
            return
        status = status or self._last_camera_status
        position = self.ptz.position_from_status(status) if status else self._last_camera_position
        if position is not None:
            self._last_camera_position = position
            self._last_zoom_position = position[2]

        elapsed_ms = int(max(0.0, now - self._ptz_operation_started_at) * 1000)
        chain_elapsed_ms = (
            int(max(0.0, now - self._ptz_chain_started_at) * 1000)
            if self._ptz_chain_started_at > 0
            else elapsed_ms
        )
        retargets = self._inflight_retarget_count if kind == "move" else 0
        self._record_event(
            "ptz_operation_timeout" if timed_out else "ptz_operation_complete",
            operation=kind,
            elapsed_ms=elapsed_ms,
            chain_elapsed_ms=chain_elapsed_ms,
            retargets=retargets,
            seen_motion=self._ptz_seen_motion,
            idle_polls=self._ptz_idle_polls,
            move_status=status.get("status.MoveStatus") if status else None,
            pan_tilt_status=status.get("status.PanTiltStatus") if status else None,
            zoom_status=status.get("status.ZoomStatus") if status else None,
            position=None if position is None else [round(v, 3) for v in position],
        )

        # PTZ movement itself must never count as target-loss time. Always restart
        # the loss clock when the camera finishes, even if YOLO briefly saw the
        # target during the slew and then lost it again before PTZ became idle.
        # The first post-move matched detection becomes a clean velocity baseline;
        # only subsequent stationary detections are allowed to learn subject motion.
        if self.target is not None:
            self.target.last_seen = now
            self.target.last_update = now
            self.target.clear_velocity()
            self._velocity_rebase_required = True

        self._ptz_operation = None
        self._ptz_operation_started_at = 0.0
        self._ptz_operation_deadline = 0.0
        self._ptz_next_status_poll_at = 0.0
        self._ptz_idle_polls = 0
        self._ptz_last_poll_position = position
        self._post_motion_release_seq = max(self._post_motion_release_seq, seq + self.cfg.post_move_frames)
        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        self._ptz_chain_started_at = 0.0
        self._inflight_retarget_count = 0
        self._last_inflight_retarget_at = 0.0
        self._last_move_command_error = None

    async def _poll_ptz_operation(self, seq: int, now: float) -> None:
        kind = self._ptz_operation
        if kind is None:
            return

        if now >= self._ptz_operation_deadline:
            self._finish_ptz_operation(seq, now, timed_out=True)
            return
        if now < self._ptz_next_status_poll_at:
            return

        status = await asyncio.to_thread(self.ptz.get_status)
        polled_at = time.monotonic()
        self._ptz_next_status_poll_at = polled_at + self.cfg.ptz_status_poll_interval
        if not status:
            if polled_at >= self._ptz_operation_deadline:
                self._finish_ptz_operation(seq, polled_at, timed_out=True)
            return

        self._last_camera_status = status
        self._last_camera_status_at = polled_at
        position = self.ptz.position_from_status(status)
        if position is not None:
            self._last_camera_position = position
            self._last_zoom_position = position[2]

        stable = self._position_stable(self._ptz_last_poll_position, position, kind)
        if stable is False:
            self._ptz_seen_motion = True

        if kind == "zoom":
            reported_idle = self.ptz.zoom_reported_idle(status)
        else:
            reported_idle = self.ptz.pan_tilt_reported_idle(status)
        if reported_idle is False:
            self._ptz_seen_motion = True

        self._ptz_last_poll_position = position

        # Avoid accepting an immediate stale Idle response directly after a command.
        # If real movement has already been observed, one subsequent idle/stable
        # poll is enough. If movement was never observed, retain the conservative
        # two-poll confirmation to protect against a stale immediate Idle response.
        if (polled_at - self._ptz_operation_started_at) < 0.20:
            self._ptz_idle_polls = 0
            return

        status_ok = reported_idle is True or (reported_idle is None and stable is True)
        stable_ok = stable is not False
        if status_ok and stable_ok:
            self._ptz_idle_polls += 1
        else:
            self._ptz_idle_polls = 0

        required_idle_polls = 1 if self._ptz_seen_motion else 2
        if self._ptz_idle_polls >= required_idle_polls:
            self._finish_ptz_operation(seq, polled_at, timed_out=False, status=status)

    def _ptz_action_ready(self, seq: int) -> bool:
        return self._ptz_operation is None and seq >= self._post_motion_release_seq

    def _escape_zone(
        self,
        err_x: float,
        err_y: float,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, ...],
    ) -> bool:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        edge_margin_x = w * 0.05
        edge_margin_y = h * 0.05
        near_edge = (
            x1 <= edge_margin_x
            or y1 <= edge_margin_y
            or x2 >= (w - edge_margin_x)
            or y2 >= (h - edge_margin_y)
        )
        return near_edge or max(abs(err_x), abs(err_y)) >= self.cfg.escape_error

    def _inflight_retarget_reason(
        self,
        err_x: float,
        err_y: float,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, ...],
        now: float,
    ) -> Optional[str]:
        if not self.cfg.inflight_retarget_enabled or self._ptz_operation != "move":
            return None
        if self.cfg.inflight_retarget_max <= 0:
            return None
        if self._inflight_retarget_count >= self.cfg.inflight_retarget_max:
            return None
        if (now - self._last_inflight_retarget_at) < self.cfg.inflight_retarget_interval:
            return None

        previous = self._last_move_command_error
        current_mag = max(abs(err_x), abs(err_y))
        if previous is not None:
            prev_x, prev_y = previous
            reversed_x = (
                err_x * prev_x < 0
                and abs(err_x) >= self.cfg.inflight_reversal_error
                and abs(prev_x) >= self.cfg.move_deadzone_x
            )
            reversed_y = (
                err_y * prev_y < 0
                and abs(err_y) >= self.cfg.inflight_reversal_error
                and abs(prev_y) >= self.cfg.move_deadzone_y
            )
            if reversed_x or reversed_y:
                return "reversal"

        escaping = self._escape_zone(err_x, err_y, bbox, frame_shape)
        if escaping:
            if previous is None:
                return "escape"
            previous_mag = max(abs(previous[0]), abs(previous[1]))
            # Do not keep rewriting a destination if the camera is clearly catching
            # up. Retarget if the subject is clipped/near-edge or has not improved
            # by at least ~5% of half-frame since the previous command.
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = bbox
            hard_edge = (x1 <= 1 or y1 <= 1 or x2 >= (w - 1) or y2 >= (h - 1))
            if hard_edge or current_mag >= (previous_mag - 0.05):
                return "escape"

        if previous is not None:
            previous_mag = max(abs(previous[0]), abs(previous[1]))
            growth_floor = max(0.55, self.cfg.escape_error - 0.15)
            if current_mag >= growth_floor and current_mag >= (previous_mag + 0.20):
                return "error_growth"

        return None

    async def _maybe_inflight_retarget(
        self,
        frame_shape: Tuple[int, ...],
        seq: int,
        now: float,
        err_x: float,
        err_y: float,
        target_span: float,
    ) -> bool:
        if self.target is None:
            return False
        reason = self._inflight_retarget_reason(
            err_x, err_y, self.target.bbox, frame_shape, now
        )
        if reason is None:
            return False

        h, w = frame_shape[:2]
        target_cx, target_cy = self.target.center
        escaping = self._escape_zone(err_x, err_y, self.target.bbox, frame_shape)
        gain = self.cfg.escape_gain if escaping else self.cfg.move_gain
        frame_cx = w / 2.0
        frame_cy = h / 2.0
        command_center = (
            frame_cx + (target_cx - frame_cx) * gain,
            frame_cy + (target_cy - frame_cy) * gain,
        )
        command_center = (
            max(w * 0.05, min(w * 0.95, command_center[0])),
            max(h * 0.05, min(h * 0.95, command_center[1])),
        )
        scaled = self.ptz._scale_point(command_center, frame_shape)

        t0 = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.move_directly_point, command_center, frame_shape)
        t1 = time.monotonic()
        if not ok:
            self._record_event(
                "move_directly_retarget_failed",
                reason=reason,
                error=self.ptz.last_error,
            )
            return False

        self.ptz_commands += 1
        self.move_direct_commands += 1
        self.inflight_retargets += 1
        self._last_move_point = scaled
        chain_elapsed_ms = (
            int(max(0.0, t1 - self._ptz_chain_started_at) * 1000)
            if self._ptz_chain_started_at > 0
            else None
        )
        self._refresh_move_after_retarget(seq, t1, (err_x, err_y))
        self.state = "PTZ_MOVING"
        self._record_event(
            "move_directly_retarget",
            label=self.target.label,
            reason=reason,
            retarget_index=self._inflight_retarget_count,
            point_8192=[scaled[0], scaled[1]],
            center_px=[round(target_cx, 1), round(target_cy, 1)],
            command_center_px=[round(command_center[0], 1), round(command_center[1], 1)],
            gain=round(gain, 3),
            error_x=round(err_x, 3),
            error_y=round(err_y, 3),
            target_span=round(target_span, 3),
            confidence=round(self.target.confidence, 3),
            predictive_lead_used=False,
            chain_elapsed_ms=chain_elapsed_ms,
            http_ms=int((t1 - t0) * 1000),
        )
        return True

    def _continuous_axis_speed(self, error: float, deadzone: float) -> int:
        magnitude = abs(error)
        if magnitude <= deadzone:
            return 0
        full_error = max(deadzone + 0.01, self.cfg.continuous_full_speed_error)
        ratio = max(0.0, min(1.0, (magnitude - deadzone) / (full_error - deadzone)))
        speed = int(round(
            self.cfg.continuous_min_speed
            + ratio * (self.cfg.continuous_max_speed - self.cfg.continuous_min_speed)
        ))
        speed = max(self.cfg.continuous_min_speed, min(self.cfg.continuous_max_speed, speed))
        return speed if error > 0 else -speed

    async def _stop_continuous(
        self,
        reason: str,
        *,
        seq: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        was_active = (
            self._continuous_active
            or self._continuous_pan_speed != 0
            or self._continuous_tilt_speed != 0
        )
        if not was_active and not force:
            return True
        t0 = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.continuous_stop)
        t1 = time.monotonic()
        self._continuous_active = False
        self._continuous_pan_speed = 0
        self._continuous_tilt_speed = 0
        self._continuous_last_command_at = t1
        if ok and was_active:
            self.ptz_commands += 1
            self.continuous_stops += 1
            self._record_event("continuous_stop", reason=reason, http_ms=int((t1 - t0) * 1000))
        elif not ok and was_active:
            self._record_event("continuous_stop_failed", reason=reason, error=self.ptz.last_error)
        if was_active:
            if self.target is not None:
                self.target.clear_velocity()
                self._velocity_rebase_required = True
            if seq is not None:
                self._post_motion_release_seq = max(
                    self._post_motion_release_seq, seq + self.cfg.post_move_frames
                )
        return ok

    async def _set_continuous_speed(
        self,
        pan_speed: int,
        tilt_speed: int,
        *,
        seq: int,
        now: float,
        reason: str,
        error_x: float,
        error_y: float,
        target_span: float,
    ) -> bool:
        pan_speed = max(-self.cfg.continuous_max_speed, min(self.cfg.continuous_max_speed, int(pan_speed)))
        tilt_speed = max(-self.cfg.continuous_max_speed, min(self.cfg.continuous_max_speed, int(tilt_speed)))
        if pan_speed == 0 and tilt_speed == 0:
            return await self._stop_continuous("deadzone", seq=seq)
        desired = (pan_speed, tilt_speed)
        current = (self._continuous_pan_speed, self._continuous_tilt_speed)
        same_speed = self._continuous_active and desired == current
        elapsed = max(0.0, now - self._continuous_last_command_at)
        if same_speed and elapsed < self.cfg.continuous_keepalive:
            return True
        def sign_changed(old: int, new: int) -> bool:
            return old != 0 and new != 0 and ((old > 0) != (new > 0))
        urgent = (
            not self._continuous_active
            or sign_changed(current[0], desired[0])
            or sign_changed(current[1], desired[1])
            or (current[0] != 0 and desired[0] == 0)
            or (current[1] != 0 and desired[1] == 0)
        )
        if not same_speed and not urgent and elapsed < self.cfg.continuous_command_interval:
            return True
        t0 = time.monotonic()
        ok = await asyncio.to_thread(
            self.ptz.continuous_move, pan_speed, tilt_speed, self.cfg.continuous_camera_timeout
        )
        t1 = time.monotonic()
        if not ok:
            self._record_event(
                "continuous_move_failed", reason=reason, pan_speed=pan_speed,
                tilt_speed=tilt_speed, error=self.ptz.last_error
            )
            if self._continuous_active:
                await self._stop_continuous("continuous_command_failed", seq=seq, force=True)
            return False
        changed = not same_speed
        self._continuous_active = True
        self._continuous_pan_speed = pan_speed
        self._continuous_tilt_speed = tilt_speed
        self._continuous_last_command_at = t1
        self.ptz_commands += 1
        self.continuous_commands += 1
        self.state = "CHASE"
        if self.target is not None:
            self.target.clear_velocity()
            self._velocity_rebase_required = True
        if changed:
            self._record_event(
                "continuous_move", label=None if self.target is None else self.target.label,
                reason=reason, pan_speed=pan_speed, tilt_speed=tilt_speed,
                error_x=round(error_x, 3), error_y=round(error_y, 3),
                target_span=round(target_span, 3),
                confidence=None if self.target is None else round(self.target.confidence, 3),
                http_ms=int((t1 - t0) * 1000),
                camera_timeout_s=self.cfg.continuous_camera_timeout,
            )
        return True

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

                # PTZ completion is independent of RTSP/GPU health. Keep polling
                # the camera while an operation is active even if the next video
                # frame is duplicate/stale or tracker inference gets skipped.
                await self._poll_ptz_operation(seq, now)
                now = time.monotonic()

                if frame is None or frame_time <= 0:
                    if self._continuous_active:
                        await self._stop_continuous("frame_unavailable", seq=seq, force=True)
                    self.state = "WAITING_FRAME"
                    continue
                if (now - frame_time) > self.cfg.frame_stale_timeout:
                    if self._continuous_active:
                        await self._stop_continuous("stale_frame", seq=seq, force=True)
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
                    if self._continuous_active:
                        await self._stop_continuous("inference_error", seq=seq, force=True)
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
            try:
                await self._stop_continuous("tracker_error", force=True)
            except Exception:
                pass
            self.logger.exception("PTZ tracker loop crashed: %s", exc)

    async def _process_observation(
        self,
        frame: np.ndarray,
        seq: int,
        detections: List[Detection],
        now: float,
    ) -> None:
        # Native 3D positioning is asynchronous inside the camera. The run loop
        # polls PTZ status before inference. Tracking continues while the camera
        # moves, and a bounded in-flight retarget may replace the destination when
        # the subject reverses or is escaping; predictive velocity is never learned
        # from those moving-camera frames.
        now = time.monotonic()

        # Never acquire a new target while a home preset is still moving, or from
        # the first couple of frames that were already buffered before it settled.
        if self._ptz_operation == "home" or (
            self.target is None and self._home_sent and not self._ptz_action_ready(seq)
        ):
            self.state = "HOME"
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

        if (
            matched is not None
            and self.target.acquire_hits < self.cfg.acquire_frames
            and matched.confidence < self.cfg.acquire_conf
        ):
            matched = None

        if matched is not None:
            was_acquiring = self.target.acquire_hits < self.cfg.acquire_frames

            ptz_ready = self._ptz_action_ready(seq)
            rebasing_velocity = (
                self._ptz_operation is None
                and not self._continuous_active
                and ptz_ready
                and self._velocity_rebase_required
            )
            velocity_learning_allowed = (
                self._ptz_operation is None
                and not self._continuous_active
                and ptz_ready
                and not self._velocity_rebase_required
            )
            self.target.update(
                matched,
                now,
                update_velocity=velocity_learning_allowed,
                min_velocity_sample_s=self.cfg.velocity_min_sample_ms / 1000.0,
            )

            # Consume one post-move match as the stationary image-space reference.
            # Later bbox updates do not move this reference until the sample window
            # is mature, eliminating the noisy 30-60 ms velocity estimates.
            if rebasing_velocity:
                self.target.rebase_velocity(now)
                self._velocity_rebase_required = False

            self._home_sent = False
            if self._ptz_operation is None and ptz_ready:
                self._loss_pause_logged = False
            if self._ptz_operation is not None:
                self._target_seen_during_ptz_operation = True

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

            self.state = "CHASE" if self._continuous_active else (
                "PTZ_MOVING" if self._ptz_operation is not None else (
                    "PTZ_SETTLING" if not self._ptz_action_ready(seq) else "TRACK"
                )
            )
            if rebasing_velocity:
                self._record_event(
                    "velocity_rebased",
                    label=self.target.label,
                    center_px=[round(self.target.center[0], 1), round(self.target.center[1], 1)],
                )
                return

            # Wait only for one robust stationary-camera sample after a PTZ move.
            # At 15 FPS the 100 ms default is about two frames, negligible beside
            # the camera's ~1.5 s mechanical slew.
            if (
                not self.cfg.continuous_chase_enabled
                and self._ptz_operation is None
                and ptz_ready
                and self.target.velocity_reference_time is not None
                and not self.target.velocity_valid
                and (now - self.target.velocity_reference_time)
                    < (self.cfg.velocity_min_sample_ms / 1000.0)
            ):
                self.state = "VELOCITY_SAMPLE"
                return

            await self._drive_to_target(frame.shape, seq, now)
            return

        if self.target.acquire_hits < self.cfg.acquire_frames:
            self._record_event("acquire_dropped", label=self.target.label, hits=self.target.acquire_hits)
            self.target = None
            self.state = "SEARCHING"
            return

        if self._continuous_active:
            await self._stop_continuous("target_missing", seq=seq)

        # A temporary detector miss while a discrete home/zoom operation is moving is
        # not evidence that the subject is gone. Pause the loss state machine until PTZ idle + fresh
        # post-move frames, with ptz_operation_timeout as the hard safety bound.
        if self._ptz_operation in ("move", "zoom") or not self._ptz_action_ready(seq):
            self.state = "PTZ_MOVING" if self._ptz_operation is not None else "PTZ_SETTLING"
            if not self._loss_pause_logged:
                self._record_event(
                    "target_loss_paused",
                    operation=self._ptz_operation,
                    post_move_frames_remaining=max(0, self._post_motion_release_seq - seq),
                    label=self.target.label,
                )
                self._loss_pause_logged = True
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
        self.target = None
        self._last_error_x = None
        self._last_error_y = None
        self._last_target_span = None
        self._last_move_point = None
        self._last_zoom_position = None

        returned_home = False
        if self.cfg.return_home_on_lost and not self._home_sent:
            ok = await asyncio.to_thread(self.ptz.goto_preset, self.cfg.home_preset)
            now2 = time.monotonic()
            if ok:
                self.home_returns += 1
                self._home_sent = True
                returned_home = True
                self._begin_ptz_operation("home", seq, now2)
                self.logger.info(
                    "%s lost for %.1fs; returned to preset %d",
                    old_label,
                    missing_for,
                    self.cfg.home_preset,
                )

        self._record_event(
            "target_released",
            label=old_label,
            missing_ms=int(missing_for * 1000),
            returned_home=returned_home,
        )
        self.state = "HOME" if returned_home else "SEARCHING"

    def _associate(self, detections: List[Detection], frame_shape: Tuple[int, ...], now: float) -> Optional[Detection]:
        if self.target is None or not detections:
            return None
        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))
        camera_recently_moved = (
            self._continuous_active
            or self._ptz_operation is not None
            or self._last_processed_seq < self._post_motion_release_seq
        )
        if camera_recently_moved:
            # PTZ motion is global image motion, not subject velocity. Do not
            # extrapolate the bbox across a camera movement.
            px, py = self.target.center
        else:
            px, py = self.target.predicted_center(now)
        old_area = max(1.0, (self.target.bbox[2] - self.target.bbox[0]) * (self.target.bbox[3] - self.target.bbox[1]))

        best: Optional[Detection] = None
        best_score = -1.0
        best_dist_norm = 999.0
        for det in detections:
            if det.class_id != self.target.class_id:
                continue
            cx, cy = det.center
            dist_norm = math.hypot(cx - px, cy - py) / diag
            proximity_span = 0.75 if camera_recently_moved else 0.45
            proximity = max(0.0, 1.0 - (dist_norm / proximity_span))
            overlap = _iou(self.target.bbox, det.bbox)
            size_similarity = min(old_area, det.area) / max(old_area, det.area)
            score = 0.35 * overlap + 0.45 * proximity + 0.10 * size_similarity + 0.10 * det.confidence
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

        if self._ptz_operation is not None or seq < self._post_motion_release_seq:
            return

        if self.cfg.continuous_chase_enabled:
            pan_speed = self._continuous_axis_speed(err_x, self.cfg.move_deadzone_x)
            tilt_speed = -self._continuous_axis_speed(err_y, self.cfg.move_deadzone_y)
            if pan_speed != 0 or tilt_speed != 0:
                magnitude = max(abs(err_x), abs(err_y))
                reason = "escape" if magnitude >= self.cfg.continuous_full_speed_error else "track"
                await self._set_continuous_speed(
                    pan_speed, tilt_speed, seq=seq, now=now, reason=reason,
                    error_x=err_x, error_y=err_y, target_span=target_span,
                )
                return
            if self._continuous_active:
                await self._stop_continuous("deadzone", seq=seq)
                return
        elif self._continuous_active:
            await self._stop_continuous("continuous_disabled", seq=seq)
            return

        if not self.cfg.autozoom:
            return
        if (now - self._last_zoom_command_at) < self.cfg.zoom_cooldown:
            return
        if self._last_zoom_position is None:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
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
        if self._last_zoom_position > (self.cfg.zoom_max_position + 0.05):
            direction = "out"
            duration_ms = self.cfg.zoom_out_step_ms
        elif self._last_zoom_position < (self.cfg.zoom_min_position - 0.05):
            direction = "in"
            duration_ms = self.cfg.zoom_in_step_ms
        elif target_span < self.cfg.zoom_target_min and self._last_zoom_position < (self.cfg.zoom_max_position - zoom_in_guard):
            direction = "in"
            duration_ms = self.cfg.zoom_in_step_ms
        elif target_span > self.cfg.zoom_target_max and self._last_zoom_position > (self.cfg.zoom_min_position + 0.25):
            direction = "out"
            duration_ms = self.cfg.zoom_out_step_ms
        if direction is None:
            return
        before = self._last_zoom_position
        t0 = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)
        t1 = time.monotonic()
        self._last_zoom_command_at = t1
        if ok:
            self.ptz_commands += 1
            self.zoom_commands += 1
            self._last_zoom_position = None
            self._begin_ptz_operation("zoom", seq, t1)
            self.state = "PTZ_MOVING"
            self._record_event(
                "zoom_step", label=self.target.label, direction=direction,
                duration_ms=duration_ms, target_span=round(target_span, 3),
                zoom_before=round(before, 3), min_position=round(self.cfg.zoom_min_position, 3),
                max_position=round(self.cfg.zoom_max_position, 3),
                http_ms=int((t1 - t0) * 1000),
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
            op_text = self._ptz_operation or "idle"
            text = (
                f"{self.state}{target_text} mode=CONT op={op_text} "
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
