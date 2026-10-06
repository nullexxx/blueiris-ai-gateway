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

from ptz_enhancements import (
    CalibrationStore, CameraMotionEstimator, OnvifAbsoluteZoom,
    TargetHistory, frame_sharpness, zoom_bucket,
)
from ptz_phase2 import (
    AcquisitionZonePolicy, BBoxMotionValidator, MotionMaskPolicy,
    OnvifRetryState, SceneStabilityGate, ZoomCalibrationMap,
)
from ptz_active_calibration import (
    ActivePtzCalibrationStore, OnvifPanTiltProbe, estimate_static_frame_shift,
)


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


def _env_float_list(name: str, default: List[float], low: float, high: float) -> List[float]:
    raw = os.getenv(name, "").strip()
    values: List[float] = []
    for token in raw.split(",") if raw else []:
        try:
            value = float(token.strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(max(low, min(high, value)))
    result = values or [max(low, min(high, float(v))) for v in default]
    return sorted(set(round(v, 6) for v in result))


def _env_int_list(name: str, default: List[int], low: int, high: int) -> List[int]:
    raw = os.getenv(name, "").strip()
    values: List[int] = []
    for token in raw.split(",") if raw else []:
        try:
            value = int(token.strip())
        except (TypeError, ValueError):
            continue
        values.append(max(low, min(high, value)))
    return sorted(set(values or [max(low, min(high, int(v))) for v in default]))


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
    rtsp_open_timeout: float = 5.0
    rtsp_read_timeout: float = 5.0

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

    # Frigate-inspired predictive control: learn this camera's actual moveDirectly
    # duration from completed moves and predict where the subject will be when the
    # camera arrives, rather than assuming one fixed movement duration forever.
    adaptive_lead: bool = True
    move_eta_min_samples: int = 3
    move_eta_history: int = 24
    move_eta_min: float = 0.50
    move_eta_max: float = 2.00

    # Reject abrupt/noisy velocity changes for one correction instead of turning
    # a questionable sample into a large predictive lead.
    velocity_consistency_cosine: float = 0.25
    velocity_jump_ratio: float = 4.0

    # Measured escape behavior: keep the known-good partial correction for normal
    # tracking, but use a stronger one-shot moveDirectly correction when the bbox
    # is already clipped or the target is close to escaping the frame. No command
    # overlap or continuous steering is introduced.
    edge_rescue_enabled: bool = True
    edge_rescue_error: float = 0.75
    edge_rescue_gain: float = 0.85

    # Hybrid fast-escape chase. Normal tracking remains camera-managed moveDirectly.
    # Continuous movement is used only when a target is already in genuine danger
    # of leaving the frame, and stops well before center so motor inertia cannot
    # create the oscillation seen when continuous control was used as the main loop.
    hybrid_chase_enabled: bool = True
    hybrid_chase_pan_sign: int = -1
    hybrid_chase_entry_error: float = 0.82
    hybrid_chase_exit_error: float = 0.50
    # Fast-moving targets may enter chase before the hard edge threshold.
    hybrid_chase_motion_error: float = 0.55
    hybrid_chase_motion_speed_norm: float = 0.03
    # Coast through very short detector dropouts caused by PTZ motion blur.
    hybrid_chase_miss_grace: float = 0.15
    hybrid_chase_min_speed: int = 1
    hybrid_chase_max_speed: int = 6
    hybrid_chase_full_speed_error: float = 0.90
    hybrid_chase_command_interval: float = 0.18
    hybrid_chase_keepalive: float = 0.45
    hybrid_chase_camera_timeout: int = 1
    hybrid_chase_max_seconds: float = 2.50
    hybrid_chase_cooldown: float = 0.35
    hybrid_chase_settle_frames: int = 2
    hybrid_divergence_frames: int = 3
    hybrid_divergence_growth: float = 0.05

    # Native PTZ operation tracking. Instead of guessing how long a 3D move takes,
    # poll getStatus until the camera reports idle and its reported position is stable.
    ptz_status_poll_interval: float = 0.12
    ptz_operation_timeout: float = 4.0
    post_move_frames: int = 1

    # Evidence-gated auto-zoom. Zoom-in is intentionally rare: the subject
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

    coast_time: float = 0.20
    reacquire_time: float = 1.25
    home_timeout: float = 3.0
    home_preset: int = 5
    goto_home_on_start: bool = True
    return_home_on_lost: bool = True

    ptz_http_timeout: float = 0.75
    move_failure_backoff_base: float = 0.25
    move_failure_backoff_max: float = 1.0
    move_failure_stop_after: int = 5
    home_retry_attempts: int = 3
    home_retry_delay: float = 0.75
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
            rtsp_open_timeout=max(1.0, _env_float("TRACKER_RTSP_OPEN_TIMEOUT", 5.0)),
            rtsp_read_timeout=max(1.0, _env_float("TRACKER_RTSP_READ_TIMEOUT", 5.0)),
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
            adaptive_lead=_env_bool("TRACKER_ADAPTIVE_LEAD", True),
            move_eta_min_samples=_env_int("TRACKER_MOVE_ETA_MIN_SAMPLES", 3),
            move_eta_history=_env_int("TRACKER_MOVE_ETA_HISTORY", 24),
            move_eta_min=_env_float("TRACKER_MOVE_ETA_MIN", 0.50),
            move_eta_max=_env_float("TRACKER_MOVE_ETA_MAX", 2.00),
            velocity_consistency_cosine=_env_float("TRACKER_VELOCITY_CONSISTENCY_COSINE", 0.25),
            velocity_jump_ratio=_env_float("TRACKER_VELOCITY_JUMP_RATIO", 4.0),
            edge_rescue_enabled=_env_bool("TRACKER_EDGE_RESCUE_ENABLED", True),
            edge_rescue_error=_env_float("TRACKER_EDGE_RESCUE_ERROR", 0.75),
            edge_rescue_gain=_env_float("TRACKER_EDGE_RESCUE_GAIN", 0.85),
            hybrid_chase_enabled=_env_bool("TRACKER_HYBRID_CHASE_ENABLED", True),
            hybrid_chase_pan_sign=_env_int("TRACKER_HYBRID_CHASE_PAN_SIGN", -1),
            hybrid_chase_entry_error=_env_float("TRACKER_HYBRID_CHASE_ENTRY_ERROR", 0.82),
            hybrid_chase_exit_error=_env_float("TRACKER_HYBRID_CHASE_EXIT_ERROR", 0.50),
            hybrid_chase_motion_error=_env_float("TRACKER_HYBRID_CHASE_MOTION_ERROR", 0.55),
            hybrid_chase_motion_speed_norm=_env_float("TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM", 0.03),
            hybrid_chase_miss_grace=_env_float("TRACKER_HYBRID_CHASE_MISS_GRACE", 0.15),
            hybrid_chase_min_speed=_env_int("TRACKER_HYBRID_CHASE_MIN_SPEED", 1),
            hybrid_chase_max_speed=_env_int("TRACKER_HYBRID_CHASE_MAX_SPEED", 6),
            hybrid_chase_full_speed_error=_env_float("TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR", 0.90),
            hybrid_chase_command_interval=_env_float("TRACKER_HYBRID_CHASE_COMMAND_INTERVAL", 0.18),
            hybrid_chase_keepalive=_env_float("TRACKER_HYBRID_CHASE_KEEPALIVE", 0.45),
            hybrid_chase_camera_timeout=_env_int("TRACKER_HYBRID_CHASE_CAMERA_TIMEOUT", 1),
            hybrid_chase_max_seconds=_env_float("TRACKER_HYBRID_CHASE_MAX_SECONDS", 2.50),
            hybrid_chase_cooldown=_env_float("TRACKER_HYBRID_CHASE_COOLDOWN", 0.35),
            hybrid_chase_settle_frames=_env_int("TRACKER_HYBRID_CHASE_SETTLE_FRAMES", 2),
            hybrid_divergence_frames=max(2, _env_int("TRACKER_HYBRID_DIVERGENCE_FRAMES", 3)),
            hybrid_divergence_growth=max(0.01, _env_float("TRACKER_HYBRID_DIVERGENCE_GROWTH", 0.05)),
            ptz_status_poll_interval=_env_float("TRACKER_PTZ_STATUS_POLL_INTERVAL", 0.12),
            ptz_operation_timeout=_env_float("TRACKER_PTZ_OPERATION_TIMEOUT", 4.0),
            post_move_frames=_env_int("TRACKER_POST_MOVE_FRAMES", 1),
            autozoom=_env_bool("TRACKER_AUTOZOOM", True),
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
            coast_time=_env_float("TRACKER_COAST_TIME", 0.20),
            reacquire_time=_env_float("TRACKER_REACQUIRE_TIME", 1.25),
            home_timeout=_env_float("TRACKER_HOME_TIMEOUT", 3.0),
            home_preset=max(1, _env_int("TRACKER_HOME_PRESET", 5)),
            goto_home_on_start=_env_bool("TRACKER_GOTO_HOME_ON_START", True),
            return_home_on_lost=_env_bool("TRACKER_RETURN_HOME_ON_LOST", True),
            ptz_http_timeout=_env_float("TRACKER_PTZ_HTTP_TIMEOUT", 0.75),
            move_failure_backoff_base=max(0.05, _env_float("TRACKER_MOVE_FAILURE_BACKOFF_BASE", 0.25)),
            move_failure_backoff_max=max(0.10, _env_float("TRACKER_MOVE_FAILURE_BACKOFF_MAX", 1.0)),
            move_failure_stop_after=max(1, _env_int("TRACKER_MOVE_FAILURE_STOP_AFTER", 5)),
            home_retry_attempts=max(1, _env_int("TRACKER_HOME_RETRY_ATTEMPTS", 3)),
            home_retry_delay=max(0.10, _env_float("TRACKER_HOME_RETRY_DELAY", 0.75)),
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
        cfg.move_eta_min_samples = max(2, min(12, cfg.move_eta_min_samples))
        cfg.move_eta_history = max(cfg.move_eta_min_samples, min(100, cfg.move_eta_history))
        cfg.move_eta_min = max(0.20, min(2.0, cfg.move_eta_min))
        cfg.move_eta_max = max(cfg.move_eta_min, min(4.0, cfg.move_eta_max))
        cfg.velocity_consistency_cosine = max(-1.0, min(1.0, cfg.velocity_consistency_cosine))
        cfg.velocity_jump_ratio = max(1.5, min(20.0, cfg.velocity_jump_ratio))
        cfg.edge_rescue_error = max(0.40, min(0.98, cfg.edge_rescue_error))
        cfg.edge_rescue_gain = max(cfg.move_gain, min(1.00, cfg.edge_rescue_gain))
        cfg.hybrid_chase_pan_sign = 1 if cfg.hybrid_chase_pan_sign >= 0 else -1
        cfg.hybrid_chase_exit_error = max(0.20, min(0.65, cfg.hybrid_chase_exit_error))
        cfg.hybrid_chase_entry_error = max(cfg.hybrid_chase_exit_error + 0.10, min(0.98, cfg.hybrid_chase_entry_error))
        cfg.hybrid_chase_motion_error = max(cfg.hybrid_chase_exit_error, min(cfg.hybrid_chase_entry_error, cfg.hybrid_chase_motion_error))
        cfg.hybrid_chase_motion_speed_norm = max(0.005, min(1.0, cfg.hybrid_chase_motion_speed_norm))
        cfg.hybrid_chase_miss_grace = max(0.0, min(0.75, cfg.hybrid_chase_miss_grace))
        cfg.hybrid_chase_min_speed = max(1, min(8, cfg.hybrid_chase_min_speed))
        cfg.hybrid_chase_max_speed = max(cfg.hybrid_chase_min_speed, min(8, cfg.hybrid_chase_max_speed))
        cfg.hybrid_chase_full_speed_error = max(cfg.hybrid_chase_entry_error + 0.01, min(1.0, cfg.hybrid_chase_full_speed_error))
        cfg.hybrid_chase_command_interval = max(0.10, min(0.75, cfg.hybrid_chase_command_interval))
        cfg.hybrid_chase_keepalive = max(cfg.hybrid_chase_command_interval, min(0.90, cfg.hybrid_chase_keepalive))
        cfg.hybrid_chase_camera_timeout = max(1, min(5, cfg.hybrid_chase_camera_timeout))
        cfg.hybrid_chase_max_seconds = max(0.75, min(5.0, cfg.hybrid_chase_max_seconds))
        cfg.hybrid_chase_cooldown = max(0.0, min(2.0, cfg.hybrid_chase_cooldown))
        cfg.hybrid_chase_settle_frames = max(1, min(8, cfg.hybrid_chase_settle_frames))
        cfg.ptz_status_poll_interval = max(0.05, min(1.0, cfg.ptz_status_poll_interval))
        cfg.ptz_operation_timeout = max(0.75, min(15.0, cfg.ptz_operation_timeout))
        cfg.post_move_frames = max(1, min(20, cfg.post_move_frames))

        cfg.zoom_wide_position = max(0.01, cfg.zoom_wide_position)
        cfg.zoom_min_factor = max(1.0, cfg.zoom_min_factor)
        cfg.zoom_max_factor = max(cfg.zoom_min_factor, min(25.0, cfg.zoom_max_factor))
        cfg.zoom_target_min = max(0.03, min(0.80, cfg.zoom_target_min))
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
            "rtsp_open_timeout": self.rtsp_open_timeout,
            "rtsp_read_timeout": self.rtsp_read_timeout,
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
            "adaptive_lead": self.adaptive_lead,
            "move_eta_min_samples": self.move_eta_min_samples,
            "move_eta_history": self.move_eta_history,
            "move_eta_min": self.move_eta_min,
            "move_eta_max": self.move_eta_max,
            "velocity_consistency_cosine": self.velocity_consistency_cosine,
            "velocity_jump_ratio": self.velocity_jump_ratio,
            "edge_rescue_enabled": self.edge_rescue_enabled,
            "edge_rescue_error": self.edge_rescue_error,
            "edge_rescue_gain": self.edge_rescue_gain,
            "hybrid_chase_enabled": self.hybrid_chase_enabled,
            "hybrid_chase_pan_sign": self.hybrid_chase_pan_sign,
            "hybrid_chase_entry_error": self.hybrid_chase_entry_error,
            "hybrid_chase_exit_error": self.hybrid_chase_exit_error,
            "hybrid_chase_motion_error": self.hybrid_chase_motion_error,
            "hybrid_chase_motion_speed_norm": self.hybrid_chase_motion_speed_norm,
            "hybrid_chase_miss_grace": self.hybrid_chase_miss_grace,
            "hybrid_chase_min_speed": self.hybrid_chase_min_speed,
            "hybrid_chase_max_speed": self.hybrid_chase_max_speed,
            "hybrid_chase_full_speed_error": self.hybrid_chase_full_speed_error,
            "hybrid_chase_command_interval": self.hybrid_chase_command_interval,
            "hybrid_chase_keepalive": self.hybrid_chase_keepalive,
            "hybrid_chase_camera_timeout": self.hybrid_chase_camera_timeout,
            "hybrid_chase_max_seconds": self.hybrid_chase_max_seconds,
            "hybrid_chase_cooldown": self.hybrid_chase_cooldown,
            "hybrid_chase_settle_frames": self.hybrid_chase_settle_frames,
            "hybrid_divergence_frames": self.hybrid_divergence_frames,
            "hybrid_divergence_growth": self.hybrid_divergence_growth,
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
            "zoom_in_min_conf": self.zoom_in_min_conf,
            "zoom_in_confirm_frames": self.zoom_in_confirm_frames,
            "zoom_in_max_error": self.zoom_in_max_error,
            "zoom_in_step_ms": self.zoom_in_step_ms,
            "zoom_out_step_ms": self.zoom_out_step_ms,
            "zoom_cooldown": self.zoom_cooldown,
            "zoom_out_cooldown": self.zoom_out_cooldown,
            "zoom_noop_backoff": self.zoom_noop_backoff,
            "zoom_noop_epsilon": self.zoom_noop_epsilon,
            "coast_time": self.coast_time,
            "reacquire_time": self.reacquire_time,
            "home_timeout": self.home_timeout,
            "home_preset": self.home_preset,
            "goto_home_on_start": self.goto_home_on_start,
            "return_home_on_lost": self.return_home_on_lost,
            "ptz_http_timeout": self.ptz_http_timeout,
            "move_failure_backoff_base": self.move_failure_backoff_base,
            "move_failure_backoff_max": self.move_failure_backoff_max,
            "move_failure_stop_after": self.move_failure_stop_after,
            "home_retry_attempts": self.home_retry_attempts,
            "home_retry_delay": self.home_retry_delay,
            "frame_stale_timeout": self.frame_stale_timeout,
            "history_size": self.history_size,
        }


class LatestFrameCapture:
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
            if text and text.lstrip().lower().startswith("error"):
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
    velocity_estimate_time: Optional[float] = None

    def rebase_velocity(self, now: float) -> None:
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

    def update(
        self,
        det: Detection,
        now: float,
        *,
        update_velocity: bool = True,
        min_velocity_sample_s: float = 0.10,
        preserve_velocity: bool = False,
        allow_class_mismatch: bool = False,
    ) -> None:
        if det.class_id != self.class_id and not allow_class_mismatch:
            raise ValueError("TargetTrack.update received a different object class")

        new_center = det.center
        if update_velocity:
            if self.velocity_reference_center is None or self.velocity_reference_time is None:
                self.velocity_reference_center = self.center
                self.velocity_reference_time = self.last_update

            sample_s = max(0.0, now - self.velocity_reference_time)
            # Keep the last mature estimate usable while the next sample window
            # accumulates. Rev 3 accidentally rewrote velocity_sample_ms to
            # 50-70 ms on every intervening frame, making motion state alternate
            # between mature/immature at camera FPS.
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
                self.velocity_estimate_time = now
                self.velocity_reference_center = new_center
                self.velocity_reference_time = now
                self.velocity_sample_ms = int(sample_s * 1000)
        elif not preserve_velocity:
            # PTZ motion invalidates image-space target velocity. A rejected
            # detector-geometry sample while the camera is stationary is
            # different: the last trusted estimate may be preserved briefly.
            self.clear_velocity()

        self.bbox = det.bbox
        self.confidence = det.confidence
        self.center = new_center
        self.last_update = now
        self.last_seen = now
        self.acquire_hits += 1

    def predicted_center(self, now: float) -> Tuple[float, float]:
        if (
            not self.velocity_valid
            or self.velocity_estimate_time is None
            or (now - self.velocity_estimate_time) > 0.75
        ):
            return self.center
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


def _servo_axis_decision(
    *,
    error: float,
    error_rate: float,
    feedforward_rate: float,
    exit_error: float,
    kp: float,
    kd: float,
    feedforward_gain: float,
    brake_horizon_s: float,
) -> Dict[str, float | bool | str]:
    """Rev 6 proportional+damped image-space servo decision for one axis.

    error_rate is measured from successive tracked bbox centers while the
    camera is moving. That captures both subject motion and the camera's real
    mechanical response, so braking follows what the image is actually doing.
    """
    error = float(error)
    error_rate = float(error_rate)
    exit_error = max(0.01, abs(float(exit_error)))
    predicted_error = error + error_rate * max(0.0, float(brake_horizon_s))
    # Positive means the tracked bbox is moving toward center. Rev 5 used
    # copysign(error_rate, error), which discarded the sign of error_rate and
    # could report an outward-moving negative error as "closing".
    closing_rate = (
        0.0
        if abs(error) < 1e-6
        else -error_rate * math.copysign(1.0, error)
    )
    p_term = float(kp) * error
    d_term = float(kd) * error_rate
    ff_term = float(feedforward_gain) * float(feedforward_rate)
    raw_rate = p_term + d_term + ff_term

    phase = "track"
    brake = False
    if abs(error) <= exit_error:
        brake = True
        phase = "deadband"
    elif error * predicted_error <= 0.0 or abs(predicted_error) <= exit_error * 0.90:
        brake = True
        phase = "predicted_stop"
    elif error * raw_rate <= 0.0:
        brake = True
        phase = "closing_fast"
    elif closing_rate > 0.0 and abs(predicted_error) < abs(error) * 0.55:
        phase = "brake"

    desired_rate = 0.0 if brake else raw_rate
    return {
        "desired_rate": desired_rate,
        "raw_rate": raw_rate,
        "predicted_error": predicted_error,
        "closing_rate": closing_rate,
        "p_term": p_term,
        "d_term": d_term,
        "ff_term": ff_term,
        "brake": brake,
        "phase": phase,
    }


def _motion_control_decision(
    *,
    err_x,
    err_y,
    vx,
    vy,
    velocity_valid,
    velocity_sample_ms,
    frame_shape,
    move_eta_s,
    hybrid_enabled,
    hybrid_disabled,
    cooldown_ready,
    edge_clipped,
    exit_error,
    entry_error,
    motion_error,
    motion_speed_norm,
    min_sample_ms,
    deadline_travel_norm,
    stationary_speed_norm,
    moving_error,
):
    """Choose continuous chase vs slow/stationary precision positioning.

    Rev 4 treats moveDirectly as a precision controller, not a predictor. A
    moving target either receives continuous PTZ (when sufficiently displaced)
    or is intentionally held until its motion settles. This avoids committing
    the camera to a 1.3-1.8 second move based on a target that will be somewhere
    else when the move completes.
    """
    h, w = frame_shape[:2]
    half_w = max(1.0, float(w) / 2.0)
    half_h = max(1.0, float(h) / 2.0)
    frame_diag = max(1.0, math.hypot(w, h))
    dominant_error = max(abs(err_x), abs(err_y))
    target_speed_norm = math.hypot(vx, vy) / frame_diag if velocity_valid else 0.0
    velocity_mature = bool(velocity_valid and velocity_sample_ms >= min_sample_ms)
    moving_target = bool(
        velocity_mature and target_speed_norm > stationary_speed_norm
    )

    if abs(err_x) >= abs(err_y):
        dominant_axis_error = err_x
        dominant_axis_velocity = vx
    else:
        dominant_axis_error = err_y
        dominant_axis_velocity = vy

    moving_inward_dominant = velocity_mature and (
        abs(dominant_axis_error) >= moving_error
        and dominant_axis_error * dominant_axis_velocity < 0.0
        and abs(dominant_axis_velocity) / frame_diag >= stationary_speed_norm
    )
    moving_outward = velocity_mature and (
        (abs(err_x) >= moving_error and err_x * vx > 0.0)
        or (abs(err_y) >= moving_error and err_y * vy > 0.0)
    )

    projected_travel_norm = 0.0
    if velocity_mature:
        projected_travel_norm = max(
            abs(vx) * max(0.0, move_eta_s) / half_w,
            abs(vy) * max(0.0, move_eta_s) / half_h,
        )

    motion_escape = (
        moving_target
        and dominant_error >= moving_error
        and moving_outward
        and target_speed_norm >= min(motion_speed_norm, stationary_speed_norm * 1.25)
    )
    deadline_escape = (
        moving_target
        and dominant_error >= moving_error
        and projected_travel_norm >= deadline_travel_norm
        and not moving_inward_dominant
    )
    hard_escape_error = max(0.90, entry_error + 0.08)
    # A clipped target deserves continuous recovery sooner than the historical
    # emergency-only entry threshold.
    edge_escape = (
        edge_clipped
        and dominant_error >= max(moving_error, min(exit_error, entry_error))
        and not moving_inward_dominant
    )
    hard_escape = dominant_error >= hard_escape_error and not moving_inward_dominant

    use_continuous = bool(
        hybrid_enabled
        and not hybrid_disabled
        and cooldown_ready
        and (edge_escape or hard_escape or motion_escape or deadline_escape)
    )
    reason = None
    if use_continuous:
        if motion_escape:
            reason = "motion_escape"
        elif deadline_escape:
            reason = "deadline_motion"
        elif edge_escape:
            reason = "edge_clipped"
        else:
            reason = "hard_escape"

    # Rev 6 never falls back to a 1.3-1.8 s positional move just because the
    # continuous path is cooling down or circuit-broken. moveDirectly remains
    # a stationary-target precision tool in every controller state.
    precision_move_allowed = True
    precision_hold_reason = None
    if not velocity_mature:
        precision_move_allowed = False
        precision_hold_reason = "velocity_sample"
    elif moving_target:
        precision_move_allowed = False
        precision_hold_reason = "moving_target"

    return {
        "use_continuous": use_continuous,
        "reason": reason,
        "velocity_mature": velocity_mature,
        "moving_target": moving_target,
        "target_speed_norm": target_speed_norm,
        "projected_travel_norm": projected_travel_norm,
        "moving_outward": moving_outward,
        "moving_inward_dominant": moving_inward_dominant,
        "motion_escape": motion_escape,
        "deadline_escape": deadline_escape,
        "precision_move_allowed": precision_move_allowed,
        "precision_hold_reason": precision_hold_reason,
    }

class DogTracker:
    """Multi-class YOLO tracker using Dahua 3D moveDirectly + bounded autozoom."""

    def __init__(self, cfg: TrackerConfig, inference_cb: InferenceCallback, logger):
        self.cfg = cfg
        self.inference_cb = inference_cb
        self.logger = logger
        self.capture = LatestFrameCapture(
            cfg.resolved_rtsp_url(),
            open_timeout_s=cfg.rtsp_open_timeout,
            read_timeout_s=cfg.rtsp_read_timeout,
        )
        self.ptz = AmcrestPTZ(cfg)
        self._smart_history = TargetHistory(1.5)
        self._smart_motion = CameraMotionEstimator(120, 12)
        self._acquisition_zones = AcquisitionZonePolicy(
            os.getenv("TRACKER_ACQUIRE_ZONES_JSON", ""),
            os.getenv("TRACKER_IGNORE_ZONES_JSON", ""),
        )
        self._motion_masks = MotionMaskPolicy(os.getenv("TRACKER_MOTION_MASKS_JSON", ""))
        self._bbox_motion = BBoxMotionValidator()
        self._scene_stability = SceneStabilityGate(
            stable_frames=max(1, _env_int("TRACKER_SCENE_STABLE_FRAMES", 2)),
            max_wait_s=_env_float("TRACKER_SCENE_STABLE_MAX_WAIT", 0.60),
            flow_threshold_norm=_env_float("TRACKER_SCENE_STABLE_FLOW", 0.012),
        )
        self._scene_stable_ready = True
        self._last_velocity_geometry = None
        self._camera_motion = None
        self._frame_sharpness: Optional[float] = None
        self._sharpness_baseline: Optional[float] = None
        self._frame_sharpness_ok = True
        self._last_ptz_stopped_at = 0.0
        self._sharpness_wait_logged = False
        self._last_effective_deadzone = (cfg.move_deadzone_x, cfg.move_deadzone_y)
        self._calibration = CalibrationStore(
            os.getenv("TRACKER_CALIBRATION_PATH", "/app/models/tracker_calibration.json"),
            cfg.camera_ip,
            True,
        )
        self._onvif_zoom = OnvifAbsoluteZoom(
            cfg.camera_ip,
            _env_int("TRACKER_ONVIF_PORT", 80),
            cfg.camera_user,
            cfg.camera_password,
            logger,
            os.getenv("TRACKER_ZOOM_CONTROL_MODE", "auto"),
        )
        self._camera_max_optical_zoom = max(
            cfg.zoom_max_factor, _env_float("TRACKER_CAMERA_MAX_OPTICAL_ZOOM", 25.0)
        )
        self._onvif_retry = OnvifRetryState(
            _env_float("TRACKER_ONVIF_RETRY_BASE", 15.0),
            _env_float("TRACKER_ONVIF_RETRY_MAX", 300.0),
        )
        self._onvif_recovery_task: Optional[asyncio.Task] = None
        self._zoom_map = ZoomCalibrationMap(
            os.getenv("TRACKER_ZOOM_CALIBRATION_PATH", "/app/models/tracker_zoom_calibration.json"),
            cfg.camera_ip,
        )
        self._active_calibration = ActivePtzCalibrationStore(
            os.getenv("TRACKER_ACTIVE_CALIBRATION_PATH", "/app/models/tracker_ptz_active_calibration.json"),
            cfg.camera_ip,
        )
        self._onvif_motion = OnvifPanTiltProbe(
            cfg.camera_ip,
            _env_int("TRACKER_ONVIF_PORT", 80),
            cfg.camera_user,
            cfg.camera_password,
        )
        self._servo_onvif_available = False
        self._servo_onvif_capabilities: Dict[str, object] = {}
        self._startup_calibration_policy = os.getenv("TRACKER_CALIBRATE_ON_START", "if_missing").strip().lower()
        if self._startup_calibration_policy not in ("off", "if_missing", "if_stale", "always"):
            self._startup_calibration_policy = "if_missing"
        self._calibration_scope = os.getenv("TRACKER_CALIBRATION_SCOPE", "all").strip().lower()
        if self._calibration_scope not in ("zoom", "movedirectly", "continuous", "motion", "onvif", "all"):
            self._calibration_scope = "all"
        self._calibration_max_age_days = max(0.0, _env_float("TRACKER_CALIBRATION_MAX_AGE_DAYS", 30.0))
        self._calibration_zoom_levels = _env_float_list(
            "TRACKER_CALIBRATION_ZOOM_LEVELS", [1.0, 1.75, 2.25, 3.0], 1.0, max(1.0, cfg.zoom_max_factor)
        )
        self._calibration_offsets = _env_float_list(
            "TRACKER_CALIBRATION_OFFSETS", [0.18, 0.35], 0.08, 0.60
        )
        self._calibration_continuous_speeds = _env_int_list(
            "TRACKER_CALIBRATION_CONTINUOUS_SPEEDS", [1, 3, 6], 1, 8
        )
        self._calibration_continuous_duration = max(0.10, min(0.50, _env_float("TRACKER_CALIBRATION_CONTINUOUS_DURATION", 0.22)))
        self._calibration_onvif_benchmark = _env_bool("TRACKER_CALIBRATION_ONVIF_BENCHMARK", True)
        self._startup_calibration_task: Optional[asyncio.Task] = None
        self._startup_calibration_state = "idle"
        self._startup_calibration_last_result: Optional[dict] = None
        self._zoom_operation_target_normalized: Optional[float] = None
        self._zoom_operation_target_factor: Optional[float] = None
        self._calibrating = False
        self._zoom_control_active = "cgi_timed_fallback"
        self._pending_move_quality: Optional[dict] = None
        self._quality_improvements: Deque[float] = deque(maxlen=50)
        self._quality_overshoots = 0
        self._quality_undershoots = 0
        self._motion_control_min_sample_ms = max(
            200, min(1000, _env_int("TRACKER_MOTION_CONTROL_MIN_SAMPLE_MS", 250))
        )
        self._motion_control_deadline_travel = max(
            0.05, min(0.75, _env_float("TRACKER_MOTION_CONTROL_DEADLINE_TRAVEL", 0.18))
        )
        self._move_direct_lead_horizon_max = max(
            0.25, min(1.50, _env_float("TRACKER_MOVE_DIRECT_LEAD_HORIZON_MAX", 0.80))
        )
        self._move_direct_lead_max_fraction = max(
            0.05, min(0.25, _env_float("TRACKER_MOVE_DIRECT_LEAD_MAX_FRACTION", 0.12))
        )
        self._motion_control_stationary_speed_norm = max(
            0.003, min(0.05, _env_float("TRACKER_MOTION_CONTROL_STATIONARY_SPEED_NORM", 0.012))
        )
        self._motion_control_moving_error = max(
            0.18, min(0.60, _env_float("TRACKER_MOTION_CONTROL_MOVING_ERROR", 0.35))
        )
        self._motion_control_continuous_exit_error = max(
            0.12, min(0.45, _env_float("TRACKER_MOTION_CONTROL_CONTINUOUS_EXIT_ERROR", 0.22))
        )
        self._post_chase_precision_holdoff = max(
            0.0, min(1.5, _env_float("TRACKER_POST_CHASE_PRECISION_HOLDOFF", 0.35))
        )
        self._hybrid_confidence_grace = max(
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

        # Rev 5 closes the loop on observed image error instead of choosing
        # speed from position error alone. Acceleration is gradual; braking is immediate.
        self._servo_kp = max(0.10, min(2.00, _env_float("TRACKER_SERVO_KP", 0.70)))
        self._servo_kd = max(0.0, min(1.50, _env_float("TRACKER_SERVO_KD", 0.24)))
        self._servo_feedforward_gain = max(
            0.0, min(1.50, _env_float("TRACKER_SERVO_FEEDFORWARD_GAIN", 0.45))
        )
        self._servo_feedforward_decay_s = max(
            0.20, min(3.0, _env_float("TRACKER_SERVO_FEEDFORWARD_DECAY", 0.80))
        )
        self._servo_brake_horizon = max(
            0.05, min(0.75, _env_float("TRACKER_SERVO_BRAKE_HORIZON", 0.24))
        )
        self._servo_derivative_alpha = max(
            0.05, min(1.0, _env_float("TRACKER_SERVO_DERIVATIVE_ALPHA", 0.40))
        )
        self._servo_start_speed_max = max(
            1, min(self.cfg.hybrid_chase_max_speed, _env_int("TRACKER_SERVO_START_SPEED_MAX", 1))
        )
        self._servo_accel_step = max(1, min(3, _env_int("TRACKER_SERVO_ACCEL_STEP", 1)))
        self._servo_divergence_grace = max(
            0.20, min(2.0, _env_float("TRACKER_SERVO_DIVERGENCE_GRACE", 0.80))
        )
        self._servo_post_stop_settle_s = max(
            0.15, min(1.50, _env_float("TRACKER_SERVO_POST_STOP_SETTLE", 0.35))
        )
        self._servo_divergence_trip_limit = max(
            2, min(8, _env_int("TRACKER_SERVO_DIVERGENCE_TRIP_LIMIT", 3))
        )
        self._servo_divergence_window_s = max(
            5.0, min(120.0, _env_float("TRACKER_SERVO_DIVERGENCE_WINDOW", 30.0))
        )
        self._servo_divergence_cooldown_s = max(
            0.25, min(5.0, _env_float("TRACKER_SERVO_DIVERGENCE_COOLDOWN", 0.75))
        )
        self._servo_telemetry_interval = max(
            0.10, min(1.0, _env_float("TRACKER_SERVO_TELEMETRY_INTERVAL", 0.20))
        )
        self._servo_actuator_mode = os.getenv("TRACKER_SERVO_ACTUATOR", "auto").strip().lower()
        if self._servo_actuator_mode not in ("auto", "onvif", "native"):
            self._servo_actuator_mode = "auto"
        self._servo_onvif_max_velocity = max(
            0.05, min(1.0, _env_float("TRACKER_ONVIF_SERVO_MAX_VELOCITY", 0.35))
        )
        self._servo_onvif_calibration_velocities = _env_float_list(
            "TRACKER_ONVIF_SERVO_CALIBRATION_VELOCITIES",
            [0.04, 0.08, 0.16],
            0.02,
            0.50,
        )
        self._servo_onvif_calibration_duration = max(
            0.12,
            min(0.45, _env_float("TRACKER_ONVIF_SERVO_CALIBRATION_DURATION", 0.22)),
        )

        continuity_default = "dog,cat,bird"
        self._class_continuity_labels = {
            token.strip().lower()
            for token in os.getenv("TRACKER_CLASS_CONTINUITY_LABELS", continuity_default).split(",")
            if token.strip().lower() in self.cfg.target_classes
        }
        self._class_vote_window = max(
            0.40, min(4.0, _env_float("TRACKER_CLASS_VOTE_WINDOW", 1.50))
        )
        self._class_switch_ratio = max(
            1.05, min(3.0, _env_float("TRACKER_CLASS_SWITCH_RATIO", 1.35))
        )
        self._class_switch_min_hits = max(
            2, min(8, _env_int("TRACKER_CLASS_SWITCH_MIN_HITS", 2))
        )

        self.active = False
        self.state = "OFF"
        self.target: Optional[TargetTrack] = None
        self._task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._last_processed_seq = -1
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        self._last_debug_detections: List[Detection] = []

        self._home_sent = False
        self._last_move_point: Optional[Tuple[int, int]] = None
        self._last_error_x: Optional[float] = None
        self._last_error_y: Optional[float] = None
        self._last_target_span: Optional[float] = None
        self._last_zoom_position: Optional[float] = None
        self._last_zoom_command_at = 0.0
        self._zoom_in_candidate_frames = 0
        self._zoom_in_candidate_last_at = 0.0
        self._zoom_suppressed_until = 0.0
        self._zoom_operation_start_position: Optional[float] = None

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

        # Learned moveDirectly timing model. Samples persist across target sessions
        # for the life of the process because they describe the camera, not a target.
        self._move_timing_samples: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.move_eta_history)
        self._move_eta_intercept = max(self.cfg.move_eta_min, min(self.cfg.move_eta_max, self.cfg.lead_time))
        self._move_eta_slope = 0.0
        self._move_eta_model_ready = False
        self._last_predicted_move_eta = self._move_eta_intercept
        self._pending_move_distance: Optional[float] = None
        saved_timing = self._calibration.timing()
        if isinstance(saved_timing, dict):
            for sample in saved_timing.get("samples", []):
                try:
                    self._move_timing_samples.append((float(sample[0]), float(sample[1])))
                except (TypeError, ValueError, IndexError):
                    pass
            if len(self._move_timing_samples) >= self.cfg.move_eta_min_samples:
                try:
                    self._move_eta_intercept = max(0.10, min(self.cfg.move_eta_max, float(saved_timing["intercept"])))
                    self._move_eta_slope = max(0.0, min(2.0, float(saved_timing["slope"])))
                    self._move_eta_model_ready = True
                    self._last_predicted_move_eta = self._move_eta_intercept
                except (KeyError, TypeError, ValueError):
                    pass

        # Lightweight camera-motion compensation for association. moveDirectly
        # tells us the approximate image shift it intends to create, so detections
        # during/just after a slew are matched against that motion corridor instead
        # of only the stale pre-move bbox center.
        self._association_motion_start_center: Optional[Tuple[float, float]] = None
        self._association_motion_end_center: Optional[Tuple[float, float]] = None

        # Last clean stationary-camera subject velocity. Used only as a sanity
        # reference; a sudden reversal/jump suppresses predictive lead for one move.
        self._trusted_velocity: Optional[Tuple[float, float]] = None

        # Fast continuous motion is deliberately an escape-only mode. It pulls a
        # target away from an edge, stops early (well before the normal deadzone),
        # then hands control back to moveDirectly. This keeps normal tracking smooth
        # and avoids the center-crossing oscillation of the earlier all-continuous build.
        self._hybrid_chase_active = False
        self._hybrid_pan_speed = 0
        self._hybrid_tilt_speed = 0
        self._hybrid_actuator = "none"
        self._hybrid_pan_velocity = 0.0
        self._hybrid_tilt_velocity = 0.0
        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0
        self._hybrid_last_stopped_at = 0.0

        self._last_detection_count = 0
        self._last_inference_ms = 0
        self._last_inference_outcome = "never"
        self._inference_times: Deque[float] = deque(maxlen=128)
        self._history: Deque[dict] = deque(maxlen=self.cfg.history_size)
        self._session_started_wall: Optional[str] = None
        self._session_started_mono: Optional[float] = None
        self._stop_reason: Optional[str] = None
        self._last_task_error: Optional[str] = None
        self._session_generation = 0
        self._move_failure_count = 0
        self._move_retry_after = 0.0
        self._hybrid_disabled_for_session = False
        self._hybrid_last_error: Optional[float] = None
        self._hybrid_divergence_count = 0
        self._hybrid_divergence_strikes: Deque[float] = deque(maxlen=16)
        self._hybrid_recover_after = 0.0
        self._continuous_settle_until = 0.0
        self._hybrid_low_confidence_since: Optional[float] = None
        self._precision_hold_until = 0.0
        self._precision_slow_since: Optional[float] = None
        self._precision_defer_reason: Optional[str] = None
        self._precision_defer_last_event_at = 0.0
        self._hybrid_pan_reverse_until = 0.0
        self._hybrid_tilt_reverse_until = 0.0
        self._servo_axis_state: Dict[str, dict] = {
            "pan": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
            "tilt": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
        }
        self._servo_feedforward = {"pan": 0.0, "tilt": 0.0}
        self._servo_feedforward_started_at = 0.0
        self._servo_last_telemetry_at = 0.0
        self._servo_last_decision: Dict[str, object] = {}
        self._class_evidence: Deque[Tuple[float, str, int, float]] = deque(maxlen=32)

        self.total_inferences = 0
        self.frames_skipped_gpu_busy = 0
        self.frames_skipped_duplicate = 0
        self.frames_stale = 0
        self.ptz_commands = 0
        self.move_direct_commands = 0
        self.zoom_commands = 0
        self.hybrid_chase_entries = 0
        self.hybrid_chase_commands = 0
        self.hybrid_chase_stops = 0
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

    async def _ensure_onvif_motion_runtime(self) -> bool:
        profile_ready = self._active_calibration.has_onvif_fractional_continuous()
        if self._servo_actuator_mode == "native":
            self._servo_onvif_available = False
            self._servo_onvif_capabilities = {
                "available": False,
                "reason": "native actuator forced",
                "calibrated_fractional_profile": profile_ready,
            }
            return False
        if not profile_ready:
            self._servo_onvif_available = False
            self._servo_onvif_capabilities = {
                "available": False,
                "reason": "fractional ONVIF calibration required",
                "calibrated_fractional_profile": False,
            }
            return False
        if self._servo_onvif_available:
            return True
        try:
            await self._onvif_motion.close()
            capabilities = await asyncio.wait_for(self._onvif_motion.initialize(), timeout=5.0)
        except Exception as exc:
            capabilities = {"available": False, "continuous_supported": False, "error": str(exc)}
        self._servo_onvif_capabilities = dict(capabilities)
        self._servo_onvif_capabilities["calibrated_fractional_profile"] = profile_ready
        self._servo_onvif_available = bool(capabilities.get("continuous_supported"))
        if self._servo_onvif_available:
            self.logger.info("PTZ tracker servo: calibrated fractional ONVIF ContinuousMove enabled")
        else:
            self.logger.warning(
                "Fractional ONVIF servo unavailable; native PTZ remains edge-rescue fallback: %s",
                capabilities.get("error"),
            )
        return self._servo_onvif_available

    async def initialize(self) -> None:
        if not self.cfg.camera_ip:
            raise RuntimeError("TRACKER_CAMERA_IP is required when TRACKER_ENABLED=true")
        if not self.cfg.camera_password:
            self.logger.warning("PTZ tracker camera password is empty.")
        self.capture.start()
        if self.cfg.autozoom and os.getenv("TRACKER_ZOOM_CONTROL_MODE", "auto").strip().lower() != "cgi":
            try:
                onvif_ok = await asyncio.wait_for(self._onvif_zoom.initialize(), timeout=5.0)
            except Exception as exc:
                onvif_ok = False
                self._onvif_zoom.last_error = str(exc)
            self._zoom_control_active = "onvif_absolute" if onvif_ok else "cgi_timed_fallback"
            if onvif_ok:
                self._onvif_retry.success()
                self.logger.info("PTZ tracker zoom: ONVIF AbsoluteMove enabled")
            else:
                delay = self._onvif_retry.failure(time.monotonic(), self._onvif_zoom.last_error)
                self.logger.warning(
                    "ONVIF absolute zoom unavailable; CGI fallback remains active and re-probe is scheduled in %.1fs: %s",
                    delay, self._onvif_zoom.last_error,
                )
        await self._ensure_onvif_motion_runtime()
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
            if self._startup_calibration_policy == "off":
                await self.start()
            else:
                self._startup_calibration_task = asyncio.create_task(
                    self._startup_calibration_sequence(), name="ptz-startup-calibration"
                )

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._startup_calibration_task is not None and not self._startup_calibration_task.done():
            self._startup_calibration_task.cancel()
            try:
                await self._startup_calibration_task
            except asyncio.CancelledError:
                pass
            self._startup_calibration_task = None
        await asyncio.to_thread(self.ptz.continuous_stop)
        self.active = False
        self.state = "SHUTDOWN"
        await self._stop_hybrid_chase("shutdown", force=True)
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._onvif_recovery_task is not None and not self._onvif_recovery_task.done():
            self._onvif_recovery_task.cancel()
            try:
                await self._onvif_recovery_task
            except asyncio.CancelledError:
                pass
            self._onvif_recovery_task = None
        try:
            await self._onvif_zoom.close()
        except Exception:
            pass
        try:
            await self._onvif_motion.close()
        except Exception:
            pass
        self._calibration.flush(force=True)
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
        self._zoom_in_candidate_frames = 0
        self._zoom_in_candidate_last_at = 0.0
        self._zoom_suppressed_until = 0.0
        self._zoom_operation_start_position = None
        self._zoom_operation_target_normalized = None
        self._zoom_operation_target_factor = None
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
        self._pending_move_distance = None
        self._association_motion_start_center = None
        self._association_motion_end_center = None
        self._trusted_velocity = None
        self._hybrid_chase_active = False
        self._hybrid_pan_speed = 0
        self._hybrid_tilt_speed = 0
        self._hybrid_actuator = "none"
        self._hybrid_pan_velocity = 0.0
        self._hybrid_tilt_velocity = 0.0
        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0
        self._hybrid_last_error = None
        self._hybrid_divergence_count = 0
        self._hybrid_divergence_strikes.clear()
        self._hybrid_recover_after = 0.0
        self._continuous_settle_until = 0.0
        self._hybrid_low_confidence_since = None
        self._precision_hold_until = 0.0
        self._precision_slow_since = None
        self._precision_defer_reason = None
        self._precision_defer_last_event_at = 0.0
        self._hybrid_pan_reverse_until = 0.0
        self._hybrid_tilt_reverse_until = 0.0
        self._servo_axis_state = {
            "pan": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
            "tilt": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
        }
        self._servo_feedforward = {"pan": 0.0, "tilt": 0.0}
        self._servo_feedforward_started_at = 0.0
        self._servo_last_telemetry_at = 0.0
        self._servo_last_decision = {}
        self._class_evidence.clear()
        self._move_failure_count = 0
        self._move_retry_after = 0.0
        self._smart_history.clear()
        self._smart_motion.reset()
        self._bbox_motion.reset()
        self._scene_stability.clear()
        self._scene_stable_ready = True
        self._last_velocity_geometry = None
        self._camera_motion = None
        self._frame_sharpness = None
        self._frame_sharpness_ok = True
        self._last_ptz_stopped_at = 0.0
        self._sharpness_wait_logged = False
        self._last_effective_deadzone = (self.cfg.move_deadzone_x, self.cfg.move_deadzone_y)
        self._pending_move_quality = None

    def _session_valid(self, generation: int) -> bool:
        return self.active and generation == self._session_generation and not self._shutdown

    def _onvif_optional_enabled(self) -> bool:
        return self.cfg.autozoom and os.getenv("TRACKER_ZOOM_CONTROL_MODE", "auto").strip().lower() != "cgi"

    def _schedule_onvif_retry(self, error: Optional[str]) -> None:
        if not self._onvif_optional_enabled():
            return
        # Quarantine the optional ONVIF path immediately. Native CGI remains
        # available while the scheduled capability re-probe runs in background.
        self._onvif_zoom.available = False
        delay = self._onvif_retry.failure(time.monotonic(), error)
        self._zoom_control_active = "cgi_timed_fallback"
        self._record_event("onvif_zoom_retry_scheduled", retry_in_s=round(delay, 1), error=error)

    async def _recover_onvif_once(self) -> bool:
        if not self._onvif_optional_enabled():
            return False
        try:
            await self._onvif_zoom.close()
            ok = await asyncio.wait_for(self._onvif_zoom.initialize(), timeout=5.0)
        except Exception as exc:
            ok = False
            self._onvif_zoom.last_error = str(exc)
        if ok:
            self._onvif_retry.success()
            self._zoom_control_active = "onvif_absolute"
            self._record_event("onvif_zoom_recovered")
            self.logger.info("ONVIF absolute zoom recovered; exact zoom control restored")
            return True
        self._schedule_onvif_retry(self._onvif_zoom.last_error)
        return False

    async def _onvif_recovery_worker(self) -> None:
        try:
            await self._recover_onvif_once()
        finally:
            self._onvif_recovery_task = None

    def _maybe_start_onvif_recovery(self, now: float) -> None:
        if not self._onvif_optional_enabled() or self._onvif_zoom.available:
            return
        if self._onvif_recovery_task is not None and not self._onvif_recovery_task.done():
            return
        if self._onvif_retry.due(now):
            self._onvif_recovery_task = asyncio.create_task(
                self._onvif_recovery_worker(), name="ptz-onvif-recovery"
            )


    def _startup_calibration_requirements(self) -> Tuple[bool, bool]:
        policy = self._startup_calibration_policy
        if policy == "off":
            return False, False
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
                onvif_result = await self._calibration_onvif_benchmark_run()

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

    async def start(self) -> dict:
        if self._calibrating:
            return {"success": False, "error": "Calibration is in progress.", "status": self.status()}
        self._session_generation += 1
        self._history.clear()
        self._session_started_wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._session_started_mono = time.monotonic()
        self._stop_reason = None
        self._last_task_error = None
        self._hybrid_disabled_for_session = False
        self._shutdown = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="direct-3d-ptz-tracker")
        await self._ensure_onvif_motion_runtime()
        self.active = True
        self.state = "SEARCHING"
        self._home_sent = False
        self._reset_tracking_state()
        if self.cfg.goto_home_on_start:
            await self.home()
        self._record_event(
            "tracker_started",
            goto_home_on_start=self.cfg.goto_home_on_start,
            move_directly=self.cfg.move_directly_enabled,
            autozoom=self.cfg.autozoom,
        )
        self.logger.info("PTZ tracker STARTED (Rev6 fractional servo + settled velocity rebase)")
        return self.status()

    async def stop(self) -> dict:
        self._session_generation += 1
        self._record_event("tracker_stopping")
        self._stop_reason = None
        self.active = False
        self.state = "OFF"
        await self._stop_hybrid_chase("tracker_stop", force=True)
        self._reset_tracking_state()
        self._record_event("tracker_stopped")
        self.logger.info("PTZ tracker STOPPED")
        return self.status()

    async def _goto_home_with_retry(self, reason: str, generation: int) -> bool:
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
                "velocity_age_ms": (
                    None
                    if self.target.velocity_estimate_time is None
                    else int(max(0.0, now - self.target.velocity_estimate_time) * 1000)
                ),
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
        op_timeout_remaining_ms = 0
        if self._ptz_operation is not None:
            op_elapsed_ms = int(max(0.0, now - self._ptz_operation_started_at) * 1000)
            op_timeout_remaining_ms = max(0, int((self._ptz_operation_deadline - now) * 1000))

        post_frames_remaining = max(0, self._post_motion_release_seq - seq)
        return {
            "enabled": self.cfg.enabled,
            "active": self.active,
            "session_started": self._session_started_wall,
            "stop_reason": self._stop_reason,
            "tracker_task_running": bool(self._task is not None and not self._task.done()),
            "tracker_task_error": self._last_task_error,
            "session_generation": self._session_generation,
            "hybrid_chase_disabled_for_session": self._hybrid_disabled_for_session,
            "move_failure_count": self._move_failure_count,
            "history_events": len(self._history),
            "state": self.state,
            "control_mode": "adaptive_hybrid",
            "primary_control_mode": "continuous_when_moving",
            "hybrid_chase_active": self._hybrid_chase_active,
            "hybrid_chase_speed": [self._hybrid_pan_speed, self._hybrid_tilt_speed],
            "hybrid_chase_actuator": self._hybrid_actuator,
            "hybrid_chase_velocity": [
                round(self._hybrid_pan_velocity, 4),
                round(self._hybrid_tilt_velocity, 4),
            ],
            "hybrid_chase_elapsed_ms": (
                None
                if not self._hybrid_chase_active or self._hybrid_started_at <= 0
                else int(max(0.0, now - self._hybrid_started_at) * 1000)
            ),
            "move_timing_model": {
                "samples": len(self._move_timing_samples),
                "ready": self._move_eta_model_ready,
                "intercept_s": round(self._move_eta_intercept, 3),
                "slope_s_per_norm": round(self._move_eta_slope, 3),
                "last_predicted_eta_s": round(self._last_predicted_move_eta, 3),
            },
            "ptz_operation": self._ptz_operation,
            "ptz_operation_elapsed_ms": op_elapsed_ms,
            "ptz_operation_timeout_remaining_ms": op_timeout_remaining_ms,
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
            "smart_ptz": {
                "zoom_control": self._onvif_zoom.public_dict(),
                "frame_sharpness": None if self._frame_sharpness is None else round(self._frame_sharpness, 1),
                "sharpness_baseline": None if self._sharpness_baseline is None else round(self._sharpness_baseline, 1),
                "sharpness_ok": self._frame_sharpness_ok,
                "camera_motion": None if self._camera_motion is None else self._camera_motion.public_dict(),
                "camera_motion_failures": self._smart_motion.failures,
                "effective_deadzone": [round(self._last_effective_deadzone[0], 3), round(self._last_effective_deadzone[1], 3)],
                "target_history": self._smart_history.metrics(now, 1.0),
                "move_quality_samples": len(self._quality_improvements),
                "mean_error_improvement": (None if not self._quality_improvements else round(sum(self._quality_improvements) / len(self._quality_improvements), 3)),
                "motion_control": {
                    "controller_revision": 5,
                    "strategy": "damped_feedback_servo_semantic_continuity",
                    "min_velocity_sample_ms": self._motion_control_min_sample_ms,
                    "deadline_travel_norm": round(self._motion_control_deadline_travel, 3),
                    "stationary_speed_norm": round(self._motion_control_stationary_speed_norm, 4),
                    "moving_entry_error": round(self._motion_control_moving_error, 3),
                    "continuous_exit_error": round(self._motion_control_continuous_exit_error, 3),
                    "post_chase_precision_holdoff_s": round(self._post_chase_precision_holdoff, 3),
                    "confidence_grace_s": round(self._hybrid_confidence_grace, 3),
                    "missing_grace_s": round(self._hybrid_missing_grace, 3),
                    "retention_detection_conf": round(self._retention_detection_conf, 3),
                    "chase_detection_conf": round(self._chase_detection_conf, 3),
                    "target_retention_grace_s": round(self._target_retention_grace, 3),
                    "velocity_ttl_s": round(self._motion_velocity_ttl, 3),
                    "precision_min_error": round(self._precision_min_error, 3),
                    "precision_settle_s": round(self._precision_settle_s, 3),
                    "axis_reverse_holdoff_s": round(self._hybrid_axis_reverse_holdoff, 3),
                    "servo": {
                        "kp": round(self._servo_kp, 3),
                        "kd": round(self._servo_kd, 3),
                        "feedforward_gain": round(self._servo_feedforward_gain, 3),
                        "feedforward_decay_s": round(self._servo_feedforward_decay_s, 3),
                        "brake_horizon_s": round(self._servo_brake_horizon, 3),
                        "start_speed_max": self._servo_start_speed_max,
                        "accel_step": self._servo_accel_step,
                        "divergence_grace_s": round(self._servo_divergence_grace, 3),
                        "post_stop_settle_s": round(self._servo_post_stop_settle_s, 3),
                        "post_stop_settle_remaining_ms": max(
                            0, int((self._continuous_settle_until - now) * 1000)
                        ),
                        "divergence_strikes": len(self._hybrid_divergence_strikes),
                        "divergence_trip_limit": self._servo_divergence_trip_limit,
                        "divergence_recovery_remaining_ms": max(
                            0, int((self._hybrid_recover_after - now) * 1000)
                        ),
                        "actuator_mode": self._servo_actuator_mode,
                        "active_actuator": self._hybrid_actuator,
                        "fractional_onvif_available": self._servo_onvif_available,
                        "fractional_onvif_profile_ready": self._active_calibration.has_onvif_fractional_continuous(),
                        "onvif_max_velocity": round(self._servo_onvif_max_velocity, 3),
                        "onvif_capabilities": self._servo_onvif_capabilities or None,
                        "last_decision": self._servo_last_decision or None,
                    },
                    "semantic_continuity": {
                        "labels": sorted(self._class_continuity_labels),
                        "vote_window_s": round(self._class_vote_window, 3),
                        "switch_ratio": round(self._class_switch_ratio, 3),
                        "switch_min_hits": self._class_switch_min_hits,
                    },
                    "precision_hold_until_ms": max(0, int((self._precision_hold_until - now) * 1000)),
                    "move_direct_predictive_lead": False,
                    "live_spatial_learning": False,
                },
                "overshoots": self._quality_overshoots,
                "undershoots": self._quality_undershoots,
                "calibration": self._calibration.public_dict(),
                "zoom_calibration": self._zoom_map.public_dict(),
                "active_calibration": self._active_calibration.public_dict(),
                "startup_calibration": {
                    "policy": self._startup_calibration_policy,
                    "scope": self._calibration_scope,
                    "state": self._startup_calibration_state,
                    "task_running": bool(self._startup_calibration_task is not None and not self._startup_calibration_task.done()),
                    "last_result": self._startup_calibration_last_result,
                },
                "onvif_retry": self._onvif_retry.public_dict(now),
                "acquisition_zones": self._acquisition_zones.public_dict(),
                "motion_masks": self._motion_masks.public_dict(),
                "scene_stability": self._scene_stability.public_dict(),
                "velocity_geometry": self._last_velocity_geometry,
                "calibrating": self._calibrating,
            },
            "counters": {
                "total_inferences": self.total_inferences,
                "frames_skipped_gpu_busy": self.frames_skipped_gpu_busy,
                "frames_skipped_duplicate": self.frames_skipped_duplicate,
                "frames_stale": self.frames_stale,
                "ptz_commands": self.ptz_commands,
                "move_direct_commands": self.move_direct_commands,
                "zoom_commands": self.zoom_commands,
                "hybrid_chase_entries": self.hybrid_chase_entries,
                "hybrid_chase_commands": self.hybrid_chase_commands,
                "hybrid_chase_stops": self.hybrid_chase_stops,
                "targets_acquired": self.targets_acquired,
                "home_returns": self.home_returns,
            },
            "config": {
                **self.cfg.public_dict(),
                "calibrate_on_start": self._startup_calibration_policy,
                "calibration_scope": self._calibration_scope,
                "calibration_max_age_days": self._calibration_max_age_days,
                "calibration_zoom_levels": self._calibration_zoom_levels,
                "calibration_offsets": self._calibration_offsets,
                "calibration_continuous_speeds": self._calibration_continuous_speeds,
                "calibration_continuous_duration": self._calibration_continuous_duration,
                "calibration_onvif_benchmark": self._calibration_onvif_benchmark,
            },
        }

    async def camera_status(self) -> dict:
        return await asyncio.to_thread(self.ptz.get_status)

    @staticmethod
    def _move_distance_from_center(point: Tuple[float, float], frame_shape: Tuple[int, ...]) -> float:
        h, w = frame_shape[:2]
        half_w = max(1.0, w / 2.0)
        half_h = max(1.0, h / 2.0)
        return min(2.0, abs(point[0] - half_w) / half_w + abs(point[1] - half_h) / half_h)

    def _predict_move_eta(self, move_distance: float) -> float:
        fallback = max(self.cfg.move_eta_min, min(self.cfg.move_eta_max, self.cfg.lead_time))
        if not self.cfg.adaptive_lead or not self._move_eta_model_ready:
            eta = fallback
        else:
            eta = self._move_eta_intercept + self._move_eta_slope * max(0.0, move_distance)
            eta = max(self.cfg.move_eta_min, min(self.cfg.move_eta_max, eta))
        self._last_predicted_move_eta = eta
        return eta

    def _update_move_timing_model(self, move_distance: float, elapsed_s: float) -> None:
        if not (math.isfinite(move_distance) and math.isfinite(elapsed_s)):
            return
        if move_distance < 0.0 or elapsed_s < 0.10 or elapsed_s > self.cfg.ptz_operation_timeout:
            return

        self._move_timing_samples.append((float(move_distance), float(elapsed_s)))
        if len(self._move_timing_samples) < self.cfg.move_eta_min_samples:
            return

        x = np.array([sample[0] for sample in self._move_timing_samples], dtype=float)
        y = np.array([sample[1] for sample in self._move_timing_samples], dtype=float)

        def fit(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float]:
            if len(x_values) < 2 or float(np.ptp(x_values)) < 0.03:
                return float(np.median(y_values)), 0.0
            design = np.column_stack((np.ones(x_values.shape[0]), x_values))
            intercept, slope = np.linalg.lstsq(design, y_values, rcond=None)[0]
            if not (math.isfinite(float(intercept)) and math.isfinite(float(slope))):
                return float(np.median(y_values)), 0.0
            if slope < 0.0:
                return float(np.median(y_values)), 0.0
            return float(intercept), float(slope)

        intercept, slope = fit(x, y)

        # One robust refit keeps a delayed HTTP/status outlier from poisoning ETA.
        if len(x) >= 5:
            predicted = intercept + slope * x
            residuals = y - predicted
            median_residual = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median_residual)))
            residual_limit = max(0.20, 3.0 * 1.4826 * mad)
            mask = np.abs(residuals - median_residual) <= residual_limit
            if int(np.count_nonzero(mask)) >= self.cfg.move_eta_min_samples:
                intercept, slope = fit(x[mask], y[mask])

        self._move_eta_intercept = max(0.10, min(self.cfg.move_eta_max, intercept))
        self._move_eta_slope = max(0.0, min(2.0, slope))
        self._move_eta_model_ready = True
        self._calibration.set_timing(self._move_eta_intercept, self._move_eta_slope, list(self._move_timing_samples))

    @staticmethod
    def _point_segment_distance(
        point: Tuple[float, float],
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        px, py = point
        ax, ay = start
        bx, by = end
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-6:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))
        cx = ax + t * dx
        cy = ay + t * dy
        return math.hypot(px - cx, py - cy)

    def _validate_velocity_for_lead(self, frame_shape: Tuple[int, ...]) -> Tuple[bool, Optional[str]]:
        if self.target is None or not self.target.velocity_valid:
            return False, "velocity_sample"
        if self.target.velocity_sample_ms < self._motion_control_min_sample_ms:
            return False, "velocity_sample"

        h, w = frame_shape[:2]
        vx = self.target.vx
        vy = self.target.vy
        speed = math.hypot(vx, vy)

        # A center moving more than ~1.5 frame diagonals/second is much more likely
        # to be association/bbox noise than a useful prediction sample.
        if speed > max(100.0, math.hypot(w, h) * 1.5):
            return False, "velocity_magnitude"

        previous = self._trusted_velocity
        if previous is None:
            self._trusted_velocity = (vx, vy)
            return True, None

        prev_speed = math.hypot(previous[0], previous[1])
        if speed < 20.0 or prev_speed < 20.0:
            self._trusted_velocity = (vx, vy)
            return True, None

        cosine = (vx * previous[0] + vy * previous[1]) / max(1e-6, speed * prev_speed)
        if cosine < self.cfg.velocity_consistency_cosine:
            # One-sample quarantine: suppress this lead, but do not keep comparing
            # a legitimate new direction against a stale pre-turn reference forever.
            self._trusted_velocity = None
            return False, "velocity_direction_change"

        ratio = max(speed, prev_speed) / max(1.0, min(speed, prev_speed))
        if ratio > self.cfg.velocity_jump_ratio:
            self._trusted_velocity = None
            return False, "velocity_jump"

        # Invalid samples must never become the reference used to validate the
        # next sample. Promote the candidate only after every sanity check passes.
        self._trusted_velocity = (vx, vy)
        return True, None

    def _current_zoom_factor(self) -> float:
        if self._last_zoom_position is None or self.cfg.zoom_wide_position <= 0:
            return 1.0
        return max(1.0, min(self.cfg.zoom_max_factor, self._last_zoom_position / self.cfg.zoom_wide_position))

    def _effective_deadzone(self, target_span: float, frame_shape: Tuple[int, ...]) -> Tuple[float, float]:
        h, w = frame_shape[:2]
        zoom_scale = math.sqrt(self._current_zoom_factor())
        size_scale = max(0.90, min(1.45, math.sqrt(0.12 / max(0.03, target_span))))
        speed_norm = 0.0
        if self.target is not None and self.target.velocity_valid:
            speed_norm = math.hypot(self.target.vx, self.target.vy) / max(1.0, math.hypot(w, h))
        motion_scale = (
            1.35 if speed_norm > self._motion_control_stationary_speed_norm else 1.0
        )
        # Rev 4 never makes the precision deadzone smaller than the configured
        # base. A 1+ second positional move for ~0.11 normalized error was a
        # measurable regression in Rev 2.
        scale = max(1.0, min(1.75, zoom_scale * size_scale * motion_scale))
        result = (
            max(self.cfg.move_deadzone_x, min(0.45, self.cfg.move_deadzone_x * scale)),
            max(self.cfg.move_deadzone_y, min(0.50, self.cfg.move_deadzone_y * scale)),
        )
        self._last_effective_deadzone = result
        return result

    def _update_frame_quality(self, frame: np.ndarray, now: float) -> bool:
        score = frame_sharpness(frame)
        self._frame_sharpness = score
        in_guard = self._last_ptz_stopped_at > 0 and (now - self._last_ptz_stopped_at) <= 0.60
        if not in_guard and self._ptz_operation is None and score > 0:
            self._sharpness_baseline = score if self._sharpness_baseline is None else 0.95 * self._sharpness_baseline + 0.05 * score
        threshold = max(35.0, 0.45 * self._sharpness_baseline) if self._sharpness_baseline is not None else 35.0
        self._frame_sharpness_ok = not in_guard or score >= threshold
        return self._frame_sharpness_ok

    def _complete_move_quality(self, frame_shape: Tuple[int, ...], now: float) -> None:
        pending = self._pending_move_quality
        if pending is None or self.target is None:
            return
        if now - float(pending.get("started_at", now)) > 4.0:
            self._pending_move_quality = None
            return
        h, w = frame_shape[:2]
        post_x = (self.target.center[0] - w / 2.0) / (w / 2.0)
        post_y = (self.target.center[1] - h / 2.0) / (h / 2.0)
        pre_x, pre_y = float(pending["pre_x"]), float(pending["pre_y"])
        before, after = math.hypot(pre_x, pre_y), math.hypot(post_x, post_y)
        improvement = 0.0 if before < 1e-6 else max(-2.0, min(1.0, (before - after) / before))
        self._quality_improvements.append(improvement)
        overshoot = (pre_x * post_x < 0 and abs(pre_x) > 0.10 and abs(post_x) > 0.04) or (pre_y * post_y < 0 and abs(pre_y) > 0.10 and abs(post_y) > 0.04)
        undershoot = after > before * 0.55 and not overshoot
        self._quality_overshoots += int(overshoot)
        self._quality_undershoots += int(undershoot)
        bucket = str(pending["bucket"])
        pan_scale, tilt_scale = self._calibration.spatial_scales(bucket)
        # Live-target telemetry is observational only. Active calibration
        # uses a static scene and is authoritative for spatial camera response;
        # a walking target must never rewrite those camera-specific scales.
        learned = False
        self._record_event("move_quality", pre_error=[round(pre_x, 3), round(pre_y, 3)], post_error=[round(post_x, 3), round(post_y, 3)], improvement=round(improvement, 3), overshoot=overshoot, undershoot=undershoot, learned=learned, zoom_bucket=bucket, pan_scale=round(pan_scale, 3), tilt_scale=round(tilt_scale, 3))
        self._pending_move_quality = None

    def _reset_servo_feedback(self) -> None:
        self._servo_axis_state = {
            "pan": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
            "tilt": {"last_error": None, "last_time": 0.0, "filtered_rate": 0.0},
        }
        self._servo_feedforward = {"pan": 0.0, "tilt": 0.0}
        self._servo_feedforward_started_at = 0.0
        self._servo_last_telemetry_at = 0.0
        self._servo_last_decision = {}

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
        decision = _servo_axis_decision(
            error=error,
            error_rate=filtered_rate,
            feedforward_rate=ff,
            exit_error=axis_exit_error,
            kp=self._servo_kp,
            kd=self._servo_kd,
            feedforward_gain=self._servo_feedforward_gain,
            brake_horizon_s=self._servo_brake_horizon,
        )
        desired_rate = abs(float(decision["desired_rate"]))
        requested_speed = 0
        selected_calibrated_rate = None
        command = 0
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
            command = command_mag if float(decision["desired_rate"]) > 0.0 else -command_mag

        meta = {
            "axis": axis,
            "error": round(float(error), 4),
            "error_rate": round(filtered_rate, 4),
            "feedforward_rate": round(ff, 4),
            "predicted_error": round(float(decision["predicted_error"]), 4),
            "closing_rate": round(float(decision["closing_rate"]), 4),
            "desired_rate": round(float(decision["desired_rate"]), 4),
            "requested_speed": requested_speed,
            "command_speed": command,
            "calibrated_rate": (
                None if selected_calibrated_rate is None
                else round(float(selected_calibrated_rate), 4)
            ),
            "phase": str(decision["phase"]),
        }
        return command, meta

    def _record_servo_telemetry(
        self,
        now: float,
        pan_meta: dict,
        tilt_meta: dict,
        pan_command: int,
        tilt_command: int,
        *,
        force: bool = False,
    ) -> None:
        payload = {
            "pan": pan_meta,
            "tilt": tilt_meta,
            "camera_command": [int(pan_command), int(tilt_command)],
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
        )
        if not was_active and not force:
            return True
        t0 = time.monotonic()
        ok = await asyncio.to_thread(self.ptz.continuous_stop)
        t1 = time.monotonic()
        self._hybrid_chase_active = False
        self._hybrid_pan_speed = 0
        self._hybrid_tilt_speed = 0
        self._hybrid_started_at = 0.0
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
        self._hybrid_low_confidence_since = None
        self._hybrid_pan_reverse_until = 0.0
        self._hybrid_tilt_reverse_until = 0.0
        self._precision_slow_since = None
        self._reset_servo_feedback()

        if was_active:
            # Continuous PTZ is not represented by _ptz_operation, so Rev 5
            # could rebase velocity only a few milliseconds after Stop. Force
            # the same optical-flow settle discipline used after positional PTZ.
            self._scene_stability.reset(t1)
            self._scene_stable_ready = False
            if ok:
                self.ptz_commands += 1
                self.hybrid_chase_stops += 1
                self._record_event(
                    "hybrid_chase_stop",
                    reason=reason,
                    http_ms=int((t1 - t0) * 1000),
                    settle_ms=int(self._servo_post_stop_settle_s * 1000),
                )
            else:
                self._record_event(
                    "hybrid_chase_stop_failed",
                    reason=reason,
                    error=self.ptz.last_error,
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
        return ok

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
        pan_speed = max(-self.cfg.hybrid_chase_max_speed, min(self.cfg.hybrid_chase_max_speed, int(pan_speed)))
        tilt_speed = max(-self.cfg.hybrid_chase_max_speed, min(self.cfg.hybrid_chase_max_speed, int(tilt_speed)))
        if pan_speed == 0 and tilt_speed == 0:
            return await self._stop_hybrid_chase("safe_inner_region", seq=seq)

        raw_desired = (pan_speed, tilt_speed)
        current = (self._hybrid_pan_speed, self._hybrid_tilt_speed)

        # Rev 5 preserves Rev 4's independent per-axis center-crossing handling. Motor inertia on
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
        same_speed = self._hybrid_chase_active and desired == current
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
        command_pan, command_tilt = desired
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
            self._hybrid_last_error = max(abs(error_x), abs(error_y))
            self._hybrid_divergence_count = 0
            self.hybrid_chase_entries += 1
        self._hybrid_chase_active = True
        self._hybrid_pan_speed = command_pan
        self._hybrid_tilt_speed = command_tilt
        self._hybrid_last_command_at = t1
        self.ptz_commands += 1
        self.hybrid_chase_commands += 1
        self.state = "ESCAPE_CHASE"
        if self.target is not None:
            self.target.clear_velocity()
            self._velocity_rebase_required = True
        self._association_motion_start_center = None
        self._association_motion_end_center = None
        if entering or desired != current:
            self._record_event(
                "hybrid_chase_move",
                entering=entering,
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
        if self._hybrid_last_error is not None:
            if (
                (now - self._hybrid_started_at) >= self._servo_divergence_grace
                and dominant_error > (self._hybrid_last_error + self.cfg.hybrid_divergence_growth)
            ):
                self._hybrid_divergence_count += 1
            elif dominant_error < self._hybrid_last_error:
                self._hybrid_divergence_count = 0
            self._hybrid_last_error = dominant_error
            if self._hybrid_divergence_count >= self.cfg.hybrid_divergence_frames:
                count = self._hybrid_divergence_count
                cutoff = now - self._servo_divergence_window_s
                while self._hybrid_divergence_strikes and self._hybrid_divergence_strikes[0] < cutoff:
                    self._hybrid_divergence_strikes.popleft()
                self._hybrid_divergence_strikes.append(now)
                strikes = len(self._hybrid_divergence_strikes)
                trip = strikes >= self._servo_divergence_trip_limit
                self._hybrid_disabled_for_session = trip
                self._hybrid_recover_after = max(
                    self._hybrid_recover_after,
                    now + self._servo_divergence_cooldown_s,
                )
                self._record_event(
                    "hybrid_chase_diverging",
                    error=round(dominant_error, 3),
                    consecutive_growth_frames=count,
                    strikes_in_window=strikes,
                    trip_limit=self._servo_divergence_trip_limit,
                    session_disabled=trip,
                    pan_sign=self.cfg.hybrid_chase_pan_sign,
                )
                await self._stop_hybrid_chase(
                    "repeated_divergence" if trip else "diverging",
                    seq=seq,
                    force=True,
                )
                return
        else:
            self._hybrid_last_error = dominant_error

        if (now - self._hybrid_started_at) >= self.cfg.hybrid_chase_max_seconds:
            await self._stop_hybrid_chase("max_duration", seq=seq)
            return
        if max(abs(err_x), abs(err_y)) <= self._motion_control_continuous_exit_error:
            await self._stop_hybrid_chase("safe_inner_region", seq=seq)
            return
        if self.target is not None and self.target.confidence < self.cfg.reacquire_conf:
            if self._hybrid_low_confidence_since is None:
                self._hybrid_low_confidence_since = now
                self._record_event(
                    "hybrid_chase_confidence_grace",
                    confidence=round(self.target.confidence, 3),
                    grace_ms=int(self._hybrid_confidence_grace * 1000),
                )

                # Never coast through a detector-confidence gap at a high native
                # motor speed. Preserve direction only, immediately dropping each
                # active axis to minimum speed while association gets its grace
                # window. This is braking, not new steering from a weak bbox.
                current_pan = self._hybrid_pan_speed
                current_tilt = self._hybrid_tilt_speed
                safe_pan = (
                    int(math.copysign(self.cfg.hybrid_chase_min_speed, current_pan))
                    if current_pan != 0 else 0
                )
                safe_tilt = (
                    int(math.copysign(self.cfg.hybrid_chase_min_speed, current_tilt))
                    if current_tilt != 0 else 0
                )
                if (
                    (safe_pan, safe_tilt) != (current_pan, current_tilt)
                    and (safe_pan != 0 or safe_tilt != 0)
                ):
                    self._record_event(
                        "hybrid_confidence_decelerate",
                        confidence=round(self.target.confidence, 3),
                        from_speed=[current_pan, current_tilt],
                        to_speed=[safe_pan, safe_tilt],
                    )
                    await self._set_hybrid_chase_speed(
                        safe_pan,
                        safe_tilt,
                        seq=seq,
                        now=now,
                        error_x=err_x,
                        error_y=err_y,
                        target_span=target_span,
                        force_command=True,
                    )
            if (now - self._hybrid_low_confidence_since) <= self._hybrid_confidence_grace:
                self.state = "ESCAPE_CHASE"
                return
            await self._stop_hybrid_chase("low_confidence", seq=seq)
            return
        self._hybrid_low_confidence_since = None
        pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)
        tilt_sign = self._active_calibration.continuous_sign("tilt", -1)
        pan_control, pan_meta = self._servo_axis_command(err_x, "pan", now)
        tilt_control, tilt_meta = self._servo_axis_command(err_y, "tilt", now)
        pan_speed = pan_sign * pan_control
        tilt_speed = tilt_sign * tilt_control
        self._record_servo_telemetry(now, pan_meta, tilt_meta, pan_speed, tilt_speed)
        if pan_speed == 0 and tilt_speed == 0:
            await self._stop_hybrid_chase("servo_brake", seq=seq)
            return
        await self._set_hybrid_chase_speed(
            pan_speed,
            tilt_speed,
            seq=seq,
            now=now,
            error_x=err_x,
            error_y=err_y,
            target_span=target_span,
        )

    def _begin_ptz_operation(self, kind: str, seq: int, now: float) -> None:
        self._ptz_operation = kind
        self._ptz_operation_started_at = now
        self._ptz_operation_deadline = now + self.cfg.ptz_operation_timeout
        self._ptz_next_status_poll_at = now + self.cfg.ptz_status_poll_interval
        self._ptz_idle_polls = 0
        self._ptz_seen_motion = False
        self._ptz_last_poll_position = self._last_camera_position
        self._post_motion_release_seq = max(self._post_motion_release_seq, seq + 1)
        self._target_seen_during_ptz_operation = False
        self._loss_pause_logged = False
        # Any physical camera action breaks the run of stationary observations
        # required before another zoom-in is allowed.
        self._zoom_in_candidate_frames = 0
        self._zoom_in_candidate_last_at = 0.0
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
        if kind == "move" and not timed_out and move_distance is not None:
            self._update_move_timing_model(move_distance, elapsed_ms / 1000.0)

        event_fields = {
            "operation": kind,
            "elapsed_ms": elapsed_ms,
            "seen_motion": self._ptz_seen_motion,
            "idle_polls": self._ptz_idle_polls,
            "move_status": status.get("status.MoveStatus") if status else None,
            "pan_tilt_status": status.get("status.PanTiltStatus") if status else None,
            "zoom_status": status.get("status.ZoomStatus") if status else None,
            "position": None if position is None else [round(v, 3) for v in position],
        }
        if kind == "move":
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
            if (
                not zoom_noop
                and zoom_after is not None
                and self._zoom_operation_target_normalized is not None
                and self._zoom_control_active == "onvif_absolute"
            ):
                actual_factor = max(1.0, zoom_after / max(0.001, self.cfg.zoom_wide_position))
                self._zoom_map.record(actual_factor, self._zoom_operation_target_normalized)
                event_fields.update(
                    zoom_calibrated_factor=round(actual_factor, 3),
                    onvif_normalized=round(self._zoom_operation_target_normalized, 6),
                )
            if zoom_noop:
                self._record_event(
                    "zoom_step_noop",
                    zoom_before=round(zoom_before, 3),
                    zoom_after=round(zoom_after, 3),
                    backoff_ms=int(self.cfg.zoom_noop_backoff * 1000),
                )
        self._record_event(
            "ptz_operation_timeout" if timed_out else "ptz_operation_complete",
            **event_fields,
        )
        self._last_ptz_stopped_at = now
        self._sharpness_wait_logged = False
        self._scene_stability.reset(now)
        self._scene_stable_ready = False

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
        self._pending_move_distance = None
        self._zoom_operation_start_position = None
        self._zoom_operation_target_normalized = None
        self._zoom_operation_target_factor = None

    async def _halt_tracking_for_ptz_failure(
        self,
        reason: str,
        *,
        operation: Optional[str],
        status: Optional[Dict[str, str]] = None,
    ) -> None:
        """Fail closed when PTZ state can no longer be established safely."""
        try:
            await self._stop_hybrid_chase("ptz_failure", force=True)
        except Exception:
            pass
        self.active = False
        self.state = "PTZ_ERROR"
        self._stop_reason = reason
        self._record_event(
            "tracking_stopped_ptz_error",
            reason=reason,
            operation=operation,
            status_available=bool(status),
            move_status=status.get("status.MoveStatus") if status else None,
            pan_tilt_status=status.get("status.PanTiltStatus") if status else None,
            zoom_status=status.get("status.ZoomStatus") if status else None,
        )
        self.logger.error("PTZ tracking stopped: %s", reason)
        self._reset_tracking_state()
        self.state = "PTZ_ERROR"
        self._stop_reason = reason

    async def _poll_ptz_operation(self, seq: int, now: float, generation: int) -> None:
        if not self._session_valid(generation):
            return
        kind = self._ptz_operation
        if kind is None:
            return

        if now >= self._ptz_operation_deadline:
            # Final status validation: never release a timed-out operation blindly.
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not self._session_valid(generation):
                return
            if not status:
                await self._halt_tracking_for_ptz_failure(
                    f"PTZ status remained unavailable when '{kind}' exceeded its {self.cfg.ptz_operation_timeout:.2f}s timeout.",
                    operation=kind,
                )
                return

            self._last_camera_status = status
            self._last_camera_status_at = checked_at
            position = self.ptz.position_from_status(status)
            if position is not None:
                self._last_camera_position = position
                self._last_zoom_position = position[2]

            stable = self._position_stable(self._ptz_last_poll_position, position, kind)
            reported_idle = (
                self.ptz.zoom_reported_idle(status)
                if kind == "zoom"
                else self.ptz.pan_tilt_reported_idle(status)
            )
            status_ok = reported_idle is True or (reported_idle is None and stable is True)
            stable_ok = stable is not False
            if status_ok and stable_ok:
                self._finish_ptz_operation(seq, checked_at, timed_out=True, status=status)
                return

            await self._halt_tracking_for_ptz_failure(
                f"PTZ operation '{kind}' exceeded its {self.cfg.ptz_operation_timeout:.2f}s timeout and the camera did not report a safe idle state.",
                operation=kind,
                status=status,
            )
            return

        if now < self._ptz_next_status_poll_at:
            return

        status = await asyncio.to_thread(self.ptz.get_status)
        polled_at = time.monotonic()
        if not self._session_valid(generation):
            return
        self._ptz_next_status_poll_at = polled_at + self.cfg.ptz_status_poll_interval
        if not status:
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
        return (
            self._ptz_operation is None
            and seq >= self._post_motion_release_seq
            and self._scene_stable_ready
            and time.monotonic() >= self._continuous_settle_until
        )

    async def _run(self) -> None:
        period = 1.0 / self.cfg.fps
        next_tick = time.monotonic()
        try:
            while not self._shutdown:
                now = time.monotonic()
                if now < next_tick:
                    await asyncio.sleep(next_tick - now)
                next_tick = max(next_tick + period, time.monotonic())

                self._maybe_start_onvif_recovery(time.monotonic())
                if not self.active:
                    await asyncio.sleep(0.05)
                    continue

                generation = self._session_generation
                frame, seq, frame_time = self.capture.latest()
                now = time.monotonic()

                # PTZ completion is independent of RTSP/GPU health. Keep polling
                # the camera while an operation is active even if the next video
                # frame is duplicate/stale or tracker inference gets skipped.
                await self._poll_ptz_operation(seq, now, generation)
                now = time.monotonic()

                # A fail-closed PTZ recovery may disable tracking during the poll.
                # Do not let the remainder of this iteration overwrite PTZ_ERROR or
                # acquire/process another target after tracking has been stopped.
                if not self._session_valid(generation):
                    continue

                if frame is None or frame_time <= 0:
                    if self._hybrid_chase_active:
                        await self._stop_hybrid_chase("frame_unavailable", seq=seq, force=True)
                    self.state = "WAITING_FRAME"
                    continue
                if (now - frame_time) > self.cfg.frame_stale_timeout:
                    if self._hybrid_chase_active:
                        await self._stop_hybrid_chase("stale_frame", seq=seq, force=True)
                    self.frames_stale += 1
                    self.state = "STALE_FRAME"
                    continue
                if seq == self._last_processed_seq:
                    self.frames_skipped_duplicate += 1
                    continue

                self._last_processed_seq = seq
                self._last_frame_shape = frame.shape[:2]
                inference_conf = self.cfg.hold_conf
                if self.target is not None and (
                    self._hybrid_chase_active
                    or self.state in ("COAST", "REACQUIRE", "PTZ_MOVING", "PTZ_SETTLING")
                ):
                    inference_conf = min(inference_conf, self._retention_detection_conf)
                if self._hybrid_chase_active:
                    inference_conf = min(inference_conf, self._chase_detection_conf)
                outcome, raw_detections, infer_ms = await self.inference_cb(
                    frame,
                    inference_conf,
                    self.cfg.model_name,
                    self.cfg.target_class_ids,
                )
                if not self._session_valid(generation):
                    continue
                self._last_inference_outcome = outcome
                self._last_inference_ms = infer_ms

                if outcome == "busy":
                    self.frames_skipped_gpu_busy += 1
                    continue
                if outcome != "ok":
                    if self._hybrid_chase_active:
                        await self._stop_hybrid_chase("inference_error", seq=seq, force=True)
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
                    if float(d.get("confidence", 0.0)) >= inference_conf
                ]
                self._last_detection_count = len(detections)
                self._last_debug_detections = detections
                await self._process_observation(frame, seq, detections, time.monotonic(), generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.active = False
            self.state = "TRACKER_ERROR"
            self._last_task_error = str(exc)
            self._stop_reason = f"Tracker task crashed: {exc}"
            try:
                await self._stop_hybrid_chase("tracker_error", force=True)
            except Exception:
                pass
            self.logger.exception("PTZ tracker loop crashed: %s", exc)

    async def _process_observation(
        self,
        frame: np.ndarray,
        seq: int,
        detections: List[Detection],
        now: float,
        generation: int,
    ) -> None:
        if not self._session_valid(generation):
            return
        # Native 3D positioning is asynchronous inside the camera. The run loop
        # polls PTZ status before inference; tracking continues while the camera
        # moves, but no second PTZ command is allowed until it is truly idle.
        now = time.monotonic()
        motion_active = (
            self._ptz_operation is not None
            or self._hybrid_chase_active
            or seq < self._post_motion_release_seq
            or not self._scene_stable_ready
        )
        motion_boxes = [d.bbox for d in detections] + self._motion_masks.boxes(frame.shape)
        self._camera_motion = self._smart_motion.update(
            frame, motion_boxes, active=motion_active, use_homography=self._ptz_operation == "zoom"
        )
        self._update_frame_quality(frame, now)
        if self._ptz_operation is None and not self._scene_stable_ready:
            observed_ready = self._scene_stability.observe(
                now, self._camera_motion, self._frame_sharpness_ok, frame.shape
            )
            # A continuous stop needs both optical-flow stability and a minimum
            # physical quiet interval. If flow settles early, restart the gate so
            # we still require fresh stable frames after the inertia window.
            if observed_ready and now < self._continuous_settle_until:
                self._scene_stability.reset(now)
                observed_ready = False
            self._scene_stable_ready = observed_ready
            if self._scene_stable_ready:
                self._record_event(
                    "post_move_scene_stable",
                    reason=self._scene_stability.last_reason,
                    flow_norm=self._scene_stability.last_flow_norm,
                    timed_out=self._scene_stability.timed_out,
                    post_stop_quiet_ms=max(
                        0, int((now - self._last_ptz_stopped_at) * 1000)
                    ),
                )

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
            candidates = [
                d for d in detections
                if d.confidence >= self.cfg.acquire_conf
                and self._acquisition_zones.allows(d.center, frame.shape)
            ]
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
            self._trusted_velocity = None
            self._association_motion_start_center = None
            self._association_motion_end_center = None
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
            self._bbox_motion.reset(chosen.bbox, now)
            self._last_velocity_geometry = None
            self._smart_history.clear()
            self._smart_history.add(now, chosen.bbox, chosen.confidence, frame.shape)
            self._class_evidence.clear()
            self._class_evidence.append((now, chosen.label, chosen.class_id, chosen.confidence))
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
        if matched is not None and self.state in ("COAST", "REACQUIRE", "LOST"):
            missing_for_match = max(0.0, now - self.target.last_seen)
            required_conf = (
                self._retention_detection_conf
                if missing_for_match <= self._target_retention_grace
                else self.cfg.reacquire_conf
            )
            if matched.confidence < required_conf:
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
                and not self._hybrid_chase_active
                and ptz_ready
                and self._velocity_rebase_required
                and self._frame_sharpness_ok
            )
            geometry_valid = True
            if (
                self._ptz_operation is None
                and not self._hybrid_chase_active
                and ptz_ready
                and not self._velocity_rebase_required
                and self._frame_sharpness_ok
            ):
                geometry = self._bbox_motion.validate(matched.bbox, now, frame.shape)
                self._last_velocity_geometry = geometry.public_dict()
                geometry_valid = geometry.valid
                if not geometry_valid:
                    self._record_event(
                        "velocity_geometry_rejected",
                        reason=geometry.reason,
                        geometry=self._last_velocity_geometry,
                    )
            else:
                self._bbox_motion.reset(matched.bbox, now)
                self._last_velocity_geometry = None
            stationary_observation = (
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
            self._observe_target_class(matched, now)
            self.target.update(
                matched,
                now,
                update_velocity=velocity_learning_allowed,
                min_velocity_sample_s=max(
                    self.cfg.velocity_min_sample_ms,
                    self._motion_control_min_sample_ms,
                ) / 1000.0,
                preserve_velocity=preserve_velocity,
                allow_class_mismatch=True,
            )
            self._smart_history.add(now, matched.bbox, matched.confidence, frame.shape)
            if self._ptz_operation is None and ptz_ready and not self._frame_sharpness_ok:
                self.state = "PTZ_SETTLING"
                if not self._sharpness_wait_logged:
                    self._record_event("post_move_frame_blurry", sharpness=round(self._frame_sharpness or 0.0, 1))
                    self._sharpness_wait_logged = True
                return

            # Consume one post-move match as the stationary image-space reference.
            # Later bbox updates do not move this reference until the sample window
            # is mature, eliminating the noisy 30-60 ms velocity estimates.
            if rebasing_velocity:
                self._complete_move_quality(frame.shape, now)
                self.target.rebase_velocity(now)
                self._velocity_rebase_required = False
                self._association_motion_start_center = None
                self._association_motion_end_center = None

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

            self.state = "ESCAPE_CHASE" if self._hybrid_chase_active else (
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

            # Low-confidence detections are allowed to preserve association during
            # blur/recovery, but may never initiate a new camera movement. Active
            # chase has its own bounded confidence grace in _drive_hybrid_chase.
            if (
                not self._hybrid_chase_active
                and self.target.confidence < self.cfg.hold_conf
            ):
                self.state = "TARGET_RETENTION"
                return

            # Wait only for one robust stationary-camera sample after a PTZ move.
            # At 15 FPS the 100 ms default is about two frames, negligible beside
            # the camera's ~1.5 s mechanical slew.
            if (
                self._ptz_operation is None
                and ptz_ready
                and self.target.velocity_reference_time is not None
                and not self.target.velocity_valid
                and (now - self.target.velocity_reference_time)
                    < (
                        max(self.cfg.velocity_min_sample_ms, self._motion_control_min_sample_ms)
                        / 1000.0
                    )
            ):
                self.state = "VELOCITY_SAMPLE"
                return

            await self._drive_to_target(frame.shape, seq, now, generation)
            return

        if self.target.acquire_hits < self.cfg.acquire_frames:
            if self._hybrid_chase_active:
                await self._stop_hybrid_chase("acquire_dropped", seq=seq, force=True)
            self._record_event("acquire_dropped", label=self.target.label, hits=self.target.acquire_hits)
            self.target = None
            self.state = "SEARCHING"
            return

        if self._hybrid_chase_active:
            # Keep continuous steering alive through a short burst of motion blur.
            # The lower retention threshold is association-only; no new target can
            # be acquired at this confidence. The camera still stops at a bounded
            # grace interval if the detector does not recover.
            hybrid_missing_for = max(0.0, now - self.target.last_seen)
            if hybrid_missing_for <= self._hybrid_missing_grace:
                self.state = "ESCAPE_CHASE"
                return
            await self._stop_hybrid_chase("target_missing", seq=seq)

        # A temporary detector miss while the camera is moving is not evidence that
        # the subject is gone. Pause the loss state machine until PTZ idle + fresh
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
            ok = await self._goto_home_with_retry("target_lost", generation)
            now2 = time.monotonic()
            if not self._session_valid(generation):
                return
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
            self._hybrid_chase_active
            or self._ptz_operation is not None
            or self._last_processed_seq < self._post_motion_release_seq
        )
        motion_start: Optional[Tuple[float, float]] = None
        motion_end: Optional[Tuple[float, float]] = None
        motion_point = self._camera_motion.apply_point(self.target.center) if camera_recently_moved and self._camera_motion is not None else None
        if self._hybrid_chase_active:
            # During continuous rescue the whole frame is translating. Frigate
            # handles this with camera-motion estimation; our lightweight version
            # simply follows the latest matched center and uses the wider moving gate.
            px, py = self.target.center
        elif camera_recently_moved:
            # Approximate the camera-induced image shift from the moveDirectly
            # command itself. This is a lightweight analogue of Frigate's camera
            # motion compensation: associate against the whole expected image-motion
            # corridor, not only the stale pre-move target center.
            motion_start = self._association_motion_start_center or self.target.center
            motion_end = self._association_motion_end_center or self.target.center
            px, py = self.target.center
        else:
            px, py = self.target.predicted_center(now)
        old_area = max(1.0, (self.target.bbox[2] - self.target.bbox[0]) * (self.target.bbox[3] - self.target.bbox[1]))

        max_dist = self.cfg.association_moving_distance if camera_recently_moved else self.cfg.association_idle_distance
        best: Optional[Detection] = None
        best_score = -1.0
        for det in detections:
            cx, cy = det.center
            if (
                camera_recently_moved
                and not self._hybrid_chase_active
                and motion_start is not None
                and motion_end is not None
            ):
                dist_norm = self._point_segment_distance((cx, cy), motion_start, motion_end) / diag
                if motion_point is not None:
                    dist_norm = min(dist_norm, math.hypot(cx - motion_point[0], cy - motion_point[1]) / diag)
            elif camera_recently_moved and motion_point is not None:
                dist_norm = math.hypot(cx - motion_point[0], cy - motion_point[1]) / diag
            else:
                dist_norm = math.hypot(cx - px, cy - py) / diag
            proximity_span = 0.75 if camera_recently_moved else 0.45
            proximity = max(0.0, 1.0 - (dist_norm / proximity_span))
            overlap = _iou(self.target.bbox, det.bbox)
            if dist_norm > max_dist and overlap < 0.30:
                continue
            size_similarity = min(old_area, det.area) / max(old_area, det.area)
            if not self._class_mismatch_compatible(
                det,
                dist_norm=dist_norm,
                overlap=overlap,
                size_similarity=size_similarity,
            ):
                continue
            class_penalty = 0.0 if det.class_id == self.target.class_id else 0.12
            score = (
                0.35 * overlap
                + 0.45 * proximity
                + 0.10 * size_similarity
                + 0.10 * det.confidence
                - class_penalty
            )
            if score > best_score:
                best_score = score
                best = det

        if best is not None and best_score >= 0.20:
            return best
        return None

    async def _drive_to_target(self, frame_shape: Tuple[int, ...], seq: int, now: float, generation: int) -> None:
        if self.target is None or not self._session_valid(generation):
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

        if self._hybrid_chase_active:
            await self._drive_hybrid_chase(
                frame_shape, seq, now, err_x, err_y, target_span
            )
            return

        if not self._ptz_action_ready(seq):
            return

        deadzone_x, deadzone_y = self._effective_deadzone(target_span, frame_shape)
        outside_deadzone = abs(err_x) > deadzone_x or abs(err_y) > deadzone_y

        if outside_deadzone:
            if not self.cfg.move_directly_enabled:
                return
            if now < self._move_retry_after:
                self.state = "PTZ_BACKOFF"
                return

            x1, y1, x2, y2 = self.target.bbox
            margin_x = w * self.cfg.lead_edge_margin
            margin_y = h * self.cfg.lead_edge_margin
            edge_clipped_now = (
                x1 <= margin_x
                or y1 <= margin_y
                or x2 >= (w - margin_x)
                or y2 >= (h - margin_y)
            )
            dominant_error = max(abs(err_x), abs(err_y))
            hard_escape_error = max(0.90, self.cfg.hybrid_chase_entry_error + 0.08)

            # Dynamic handoff: keep moveDirectly for slow/stable motion, but a
            # fast target that is already well outside center and still moving
            # outward should not be committed to another ~1.5s positional move.
            frame_diag = max(1.0, math.hypot(w, h))
            target_speed_norm = (
                math.hypot(self.target.vx, self.target.vy) / frame_diag
                if self.target.velocity_valid
                else 0.0
            )
            moving_outward = self.target.velocity_valid and (
                (abs(err_x) >= self.cfg.hybrid_chase_motion_error and err_x * self.target.vx > 0.0)
                or (abs(err_y) >= self.cfg.hybrid_chase_motion_error and err_y * self.target.vy > 0.0)
            )
            if abs(err_x) >= abs(err_y):
                dominant_axis_error = err_x
                dominant_axis_velocity = self.target.vx
            else:
                dominant_axis_error = err_y
                dominant_axis_velocity = self.target.vy
            moving_inward_dominant = self.target.velocity_valid and (
                abs(dominant_axis_error) >= self.cfg.hybrid_chase_exit_error
                and dominant_axis_error * dominant_axis_velocity < 0.0
                and abs(dominant_axis_velocity) / frame_diag >= self.cfg.hybrid_chase_motion_speed_norm
            )
            rough_move_distance = min(
                2.0,
                (abs(err_x) + abs(err_y)) * self.cfg.move_gain,
            )
            handoff_eta_s = self._predict_move_eta(rough_move_distance)
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
                frame_shape=frame_shape,
                move_eta_s=handoff_eta_s,
                hybrid_enabled=self.cfg.hybrid_chase_enabled,
                hybrid_disabled=self._hybrid_disabled_for_session,
                cooldown_ready=(
                    (now - self._hybrid_last_stopped_at) >= self.cfg.hybrid_chase_cooldown
                    and now >= self._hybrid_recover_after
                    and now >= self._continuous_settle_until
                ),
                edge_clipped=edge_clipped_now,
                exit_error=self.cfg.hybrid_chase_exit_error,
                entry_error=self.cfg.hybrid_chase_entry_error,
                motion_error=self.cfg.hybrid_chase_motion_error,
                motion_speed_norm=self.cfg.hybrid_chase_motion_speed_norm,
                min_sample_ms=self._motion_control_min_sample_ms,
                deadline_travel_norm=self._motion_control_deadline_travel,
                stationary_speed_norm=self._motion_control_stationary_speed_norm,
                moving_error=self._motion_control_moving_error,
            )
            target_speed_norm = float(control_decision["target_speed_norm"])
            moving_outward = bool(control_decision["moving_outward"])
            moving_inward_dominant = bool(control_decision["moving_inward_dominant"])
            motion_escape = bool(control_decision["motion_escape"])
            deadline_escape = bool(control_decision["deadline_escape"])
            hybrid_entry = bool(control_decision["use_continuous"])
            if hybrid_entry:
                self._precision_slow_since = None
                self._seed_servo_feedback(frame_shape, now, err_x, err_y)
                pan_sign = self._active_calibration.continuous_sign("pan", self.cfg.hybrid_chase_pan_sign)
                tilt_sign = self._active_calibration.continuous_sign("tilt", -1)
                pan_control, pan_meta = self._servo_axis_command(err_x, "pan", now)
                tilt_control, tilt_meta = self._servo_axis_command(err_y, "tilt", now)
                pan_speed = pan_sign * pan_control
                tilt_speed = tilt_sign * tilt_control
                self._record_servo_telemetry(
                    now, pan_meta, tilt_meta, pan_speed, tilt_speed, force=True
                )
                self._record_event(
                    "hybrid_chase_enter",
                    label=self.target.label,
                    entry_reason=control_decision["reason"],
                    error_x=round(err_x, 3),
                    error_y=round(err_y, 3),
                    target_speed_norm=round(target_speed_norm, 4),
                    velocity_mature=bool(control_decision["velocity_mature"]),
                    projected_travel_norm=round(float(control_decision["projected_travel_norm"]), 4),
                    deadline_escape=deadline_escape,
                    moving_outward=bool(moving_outward),
                    moving_inward_dominant=bool(moving_inward_dominant),
                    edge_clipped=edge_clipped_now,
                    pan_speed=pan_speed,
                    tilt_speed=tilt_speed,
                    confidence=round(self.target.confidence, 3),
                )
                await self._set_hybrid_chase_speed(
                    pan_speed,
                    tilt_speed,
                    seq=seq,
                    now=now,
                    error_x=err_x,
                    error_y=err_y,
                    target_span=target_span,
                )
                return

            dominant_error = max(abs(err_x), abs(err_y))
            precision_hold_reason = None
            if now < self._continuous_settle_until:
                self._precision_slow_since = None
                precision_hold_reason = "camera_settling"
            elif now < self._precision_hold_until:
                self._precision_slow_since = None
                precision_hold_reason = "post_chase_holdoff"
            elif not bool(control_decision["precision_move_allowed"]):
                self._precision_slow_since = None
                precision_hold_reason = str(control_decision["precision_hold_reason"] or "moving_target")
            elif dominant_error < self._precision_min_error:
                self._precision_slow_since = None
                precision_hold_reason = "precision_deadband"
            else:
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
                        "camera_settling",
                        "post_chase_holdoff",
                        "precision_deadband",
                        "stationary_settle",
                    )
                    else "MOTION_HOLD"
                )
                if (
                    precision_hold_reason != self._precision_defer_reason
                    or (now - self._precision_defer_last_event_at) >= 1.0
                ):
                    self._record_event(
                        "precision_move_deferred",
                        reason=precision_hold_reason,
                        error_x=round(err_x, 3),
                        error_y=round(err_y, 3),
                        target_speed_norm=round(target_speed_norm, 4),
                        velocity_mature=bool(control_decision["velocity_mature"]),
                    )
                    self._precision_defer_reason = precision_hold_reason
                    self._precision_defer_last_event_at = now
                self.state = next_state
                return

            self._precision_defer_reason = None

            # A precision move consumes the stationary dwell. A second positional
            # correction must earn a new stable observation window after settling.
            self._precision_slow_since = None

            # moveDirectly centers the CURRENT measured point. In adaptive hybrid
            # move takes long enough that the instantaneous YOLO point is stale by
            # arrival. Use a deliberately simple correction:
            #
            #   1. Project the target a short distance using velocity learned only
            #      from stationary-camera observations.
            #   2. Bound that lead to 20% of the frame per axis so bbox jitter or a
            #      bad sample can never fling the camera toward an edge.
            #   3. Move only move_gain of the way from frame center to that projected
            #      point. This avoids the full-error overshoot seen in V6.
            target_cx, target_cy = self.target.center
            velocity_x = self.target.vx
            velocity_y = self.target.vy
            velocity_sample_ms = self.target.velocity_sample_ms

            margin_x = w * self.cfg.lead_edge_margin
            margin_y = h * self.cfg.lead_edge_margin
            edge_clipped = (
                x1 <= margin_x
                or y1 <= margin_y
                or x2 >= (w - margin_x)
                or y2 >= (h - margin_y)
            )

            # When the bbox is already clipped or the centroid is near escape,
            # a normal partial correction is too timid for a 1.3-1.6 s move.
            # Strengthen only this one camera-managed move; do not overlap moves.
            edge_rescue_requested = self.cfg.edge_rescue_enabled and (
                edge_clipped
                or abs(err_x) >= self.cfg.edge_rescue_error
                or abs(err_y) >= self.cfg.edge_rescue_error
            )
            edge_rescue_active = (
                edge_rescue_requested
                and not moving_inward_dominant
                and (not self.cfg.hybrid_chase_enabled or self._hybrid_disabled_for_session)
            )
            edge_rescue_reason: Optional[str] = None
            if edge_rescue_active:
                edge_rescue_reason = "edge_clipped" if edge_clipped else "extreme_error"
            elif edge_rescue_requested and moving_inward_dominant:
                edge_rescue_reason = "inward_motion_suppressed"
            active_move_gain = (
                max(self.cfg.move_gain, self.cfg.edge_rescue_gain)
                if edge_rescue_active else self.cfg.move_gain
            )
            current_zoom_factor = self._current_zoom_factor()
            current_bucket = zoom_bucket(current_zoom_factor)
            pan_scale, tilt_scale = self._calibration.spatial_scales(current_bucket)
            zoom_gain = 1.0 / math.sqrt(current_zoom_factor)
            gain_x = max(0.20, min(0.95, active_move_gain * zoom_gain * pan_scale))
            gain_y = max(0.20, min(0.95, active_move_gain * zoom_gain * tilt_scale))

            frame_cx = w / 2.0
            frame_cy = h / 2.0

            # Estimate the duration of the move we are about to ask for using the
            # camera's own completed-move history. Before enough samples exist,
            # TRACKER_LEAD_TIME remains the conservative known-good fallback.
            base_command_center = (
                frame_cx + (target_cx - frame_cx) * gain_x,
                frame_cy + (target_cy - frame_cy) * gain_y,
            )
            base_move_distance = self._move_distance_from_center(base_command_center, frame_shape)
            predicted_move_eta_s = self._predict_move_eta(base_move_distance)
            lead_horizon_s = (
                0.0
                if adaptive_hybrid_available
                else min(predicted_move_eta_s, self._move_direct_lead_horizon_max)
            )

            lead_suppressed_reason: Optional[str] = (
                "adaptive_hybrid_current_center" if adaptive_hybrid_available else None
            )
            if lead_suppressed_reason is None:
                if not self.target.velocity_valid:
                    lead_suppressed_reason = "velocity_sample"
                elif self.target.confidence < self.cfg.lead_min_conf:
                    lead_suppressed_reason = "low_confidence"
                elif target_span < self.cfg.lead_min_span:
                    lead_suppressed_reason = "small_target"
                elif edge_clipped:
                    lead_suppressed_reason = "edge_clipped"
                else:
                    velocity_ok, velocity_reason = self._validate_velocity_for_lead(frame_shape)
                    if not velocity_ok:
                        lead_suppressed_reason = velocity_reason

            lead_valid = lead_suppressed_reason is None
            if lead_valid:
                lead_dx = velocity_x * lead_horizon_s
                lead_dy = velocity_y * lead_horizon_s
                max_lead_x = w * self._move_direct_lead_max_fraction
                max_lead_y = h * self._move_direct_lead_max_fraction
                lead_dx = max(-max_lead_x, min(max_lead_x, lead_dx))
                lead_dy = max(-max_lead_y, min(max_lead_y, lead_dy))
            else:
                lead_dx = 0.0
                lead_dy = 0.0

            predicted_cx = max(w * 0.05, min(w * 0.95, target_cx + lead_dx))
            predicted_cy = max(h * 0.05, min(h * 0.95, target_cy + lead_dy))

            command_center = (
                frame_cx + (predicted_cx - frame_cx) * gain_x,
                frame_cy + (predicted_cy - frame_cy) * gain_y,
            )
            move_distance = self._move_distance_from_center(command_center, frame_shape)

            # Approximate where the predicted subject should land after the camera
            # shift. Association uses the segment from the current center to this
            # expected center while the PTZ is moving/settling.
            expected_post_move_center = (
                max(0.0, min(float(w), predicted_cx + frame_cx - command_center[0])),
                max(0.0, min(float(h), predicted_cy + frame_cy - command_center[1])),
            )
            scaled = self.ptz._scale_point(command_center, frame_shape)

            generation = self._session_generation
            t0 = time.monotonic()
            ok = await asyncio.to_thread(self.ptz.move_directly_point, command_center, frame_shape)
            t1 = time.monotonic()
            if not self._session_valid(generation):
                return
            if ok:
                self._move_failure_count = 0
                self._move_retry_after = 0.0
                self.ptz_commands += 1
                self.move_direct_commands += 1
                self._last_move_point = scaled
                self._pending_move_distance = move_distance
                self._association_motion_start_center = (target_cx, target_cy)
                self._association_motion_end_center = expected_post_move_center
                self._pending_move_quality = {"started_at": t1, "pre_x": err_x, "pre_y": err_y, "speed_norm": target_speed_norm, "bucket": current_bucket}
                self._begin_ptz_operation("move", seq, t1)
                self.state = "PTZ_MOVING"
                self._record_event(
                    "move_directly",
                    label=self.target.label,
                    point_8192=[scaled[0], scaled[1]],
                    center_px=[round(target_cx, 1), round(target_cy, 1)],
                    predicted_center_px=[round(predicted_cx, 1), round(predicted_cy, 1)],
                    command_center_px=[round(command_center[0], 1), round(command_center[1], 1)],
                    velocity_px_s=[round(velocity_x, 1), round(velocity_y, 1)],
                    velocity_sample_ms=velocity_sample_ms,
                    lead_valid=lead_valid,
                    lead_suppressed_reason=lead_suppressed_reason,
                    edge_clipped=edge_clipped,
                    lead_px=[round(lead_dx, 1), round(lead_dy, 1)],
                    move_gain=round(active_move_gain, 3),
                    gain_x=round(gain_x, 3), gain_y=round(gain_y, 3),
                    zoom_factor=round(current_zoom_factor, 2), zoom_bucket=current_bucket,
                    deadzone=[round(deadzone_x, 3), round(deadzone_y, 3)],
                    base_move_gain=round(self.cfg.move_gain, 3),
                    edge_rescue_active=edge_rescue_active,
                    edge_rescue_reason=edge_rescue_reason,
                    lead_time=round(self.cfg.lead_time, 3),
                    lead_horizon_s=round(lead_horizon_s, 3),
                    predicted_move_eta_ms=int(predicted_move_eta_s * 1000),
                    move_distance=round(move_distance, 3),
                    move_timing_samples=len(self._move_timing_samples),
                    move_eta_model_ready=self._move_eta_model_ready,
                    expected_post_move_center_px=[
                        round(expected_post_move_center[0], 1),
                        round(expected_post_move_center[1], 1),
                    ],
                    error_x=round(err_x, 3),
                    error_y=round(err_y, 3),
                    target_span=round(target_span, 3),
                    confidence=round(self.target.confidence, 3),
                    http_ms=int((t1 - t0) * 1000),
                    operation_timeout_ms=int(self.cfg.ptz_operation_timeout * 1000),
                )
            else:
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

        # Zoom is subordinate to tracking. Zoom-in is evidence-gated so the lens
        # only moves when the current framing is clearly wasting useful pixels.
        # Zoom-out stays intentionally easier so the camera can regain context.
        if not self.cfg.autozoom:
            return

        zoom_metrics = self._smart_history.metrics(now, 1.0)
        smoothed_span = float(zoom_metrics.get("weighted_span") or target_span)
        predicted_span = float(zoom_metrics.get("predicted_span") or smoothed_span)
        speed_norm = math.hypot(self.target.vx, self.target.vy) / max(1.0, math.hypot(w, h)) if self.target.velocity_valid else 0.0
        zoom_in_eligible = (
            int(zoom_metrics.get("samples", 0)) >= 5
            and max(smoothed_span, predicted_span) < self.cfg.zoom_target_min
            and self.target.confidence >= self.cfg.zoom_in_min_conf
            and abs(err_x) <= self.cfg.zoom_in_max_error and abs(err_y) <= self.cfg.zoom_in_max_error
            and int(zoom_metrics.get("edge_touches", 0)) == 0
            and speed_norm <= 0.025
            and self._frame_sharpness_ok
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
            and max(smoothed_span, predicted_span) < self.cfg.zoom_target_min
            and since_zoom >= self.cfg.zoom_cooldown
            and now >= self._zoom_suppressed_until
            and self._last_zoom_position < (self.cfg.zoom_max_position - zoom_in_guard)
        ):
            direction = "in"
            duration_ms = self.cfg.zoom_in_step_ms
        elif (
            (max(smoothed_span, predicted_span) > self.cfg.zoom_target_max or int(zoom_metrics.get("edge_touches", 0)) > 0 or speed_norm >= 0.08)
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
        current_factor = max(1.0, before / self.cfg.zoom_wide_position)
        desired_factor = min(self.cfg.zoom_max_factor, current_factor + 0.25) if direction == "in" else max(self.cfg.zoom_min_factor, current_factor - 0.50)
        t0 = time.monotonic()
        zoom_control = "cgi_timed_fallback"
        ok = False
        onvif_normalized = None
        if self._onvif_zoom.available:
            onvif_normalized = self._zoom_map.estimate_normalized(
                desired_factor, self._camera_max_optical_zoom
            )
            try:
                ok = await asyncio.wait_for(
                    self._onvif_zoom.set_normalized(onvif_normalized), timeout=1.5
                )
            except asyncio.TimeoutError:
                self._schedule_onvif_retry("ONVIF absolute zoom command timed out")
                self._record_event("zoom_step_deferred", direction=direction, reason="onvif_timeout")
                return
            if ok:
                zoom_control = "onvif_absolute"
                self._onvif_retry.success()
                self._zoom_operation_target_normalized = onvif_normalized
                self._zoom_operation_target_factor = desired_factor
            else:
                self._schedule_onvif_retry(self._onvif_zoom.last_error)
        if not ok:
            self._zoom_operation_target_normalized = None
            self._zoom_operation_target_factor = None
            ok = await asyncio.to_thread(self.ptz.zoom_step, direction, duration_ms)
        self._zoom_control_active = zoom_control
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
                control=zoom_control, desired_factor=round(desired_factor, 2),
                onvif_normalized=None if onvif_normalized is None else round(onvif_normalized, 6),
                smoothed_span=round(smoothed_span, 3), predicted_span=round(predicted_span, 3),
                target_speed_norm=round(speed_norm, 4),
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

    def _render_debug_frame(
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
            dzx, dzy = self._last_effective_deadzone
            left = int(cx - dzx * (w / 2.0))
            right = int(cx + dzx * (w / 2.0))
            top = int(cy - dzy * (h / 2.0))
            bottom = int(cy + dzy * (h / 2.0))
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
