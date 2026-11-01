from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import math
import os
from typing import Dict, List, Mapping, Tuple
from urllib.parse import quote


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
    # Class names are resolved against the LOADED model's own label map at
    # startup (see TrackerConfig.resolve_class_ids), so any label the model
    # knows is valid (e.g. Frigate+ "raccoon", "fox"). A bare number is kept as
    # a raw class id for that model; it carries no label, so it cannot take
    # part in label-based priority. Prefer names.
    for token in (raw or default):
        if token not in result:
            result.append(token)
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


def _event_timezone(name: str):
    requested = (name or "").strip() or "America/New_York"
    try:
        return requested, ZoneInfo(requested)
    except ZoneInfoNotFoundError:
        return "UTC", timezone.utc


def _acquisition_hotspot_key(
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, ...],
    quantum: float = 0.025,
) -> Tuple[int, int, int, int]:
    """Quantize a box by normalized center and size for recurring-scene fingerprints."""
    h, w = frame_shape[:2]
    width = max(1.0, float(w))
    height = max(1.0, float(h))
    x1, y1, x2, y2 = (float(v) for v in bbox)
    values = (
        ((x1 + x2) * 0.5) / width,
        ((y1 + y2) * 0.5) / height,
        max(0.0, x2 - x1) / width,
        max(0.0, y2 - y1) / height,
    )
    q = max(0.005, float(quantum))
    return tuple(int(round(v / q)) for v in values)



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
    lead_edge_margin: float = 0.02

    # Frigate-inspired predictive control: learn this camera's actual moveDirectly
    # duration from completed moves and predict where the subject will be when the
    # camera arrives, rather than assuming one fixed movement duration forever.
    adaptive_lead: bool = True
    move_eta_min_samples: int = 3
    move_eta_history: int = 24
    move_eta_min: float = 0.50
    move_eta_max: float = 2.00


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
            lead_edge_margin=_env_float("TRACKER_LEAD_EDGE_MARGIN", 0.02),
            adaptive_lead=_env_bool("TRACKER_ADAPTIVE_LEAD", True),
            move_eta_min_samples=_env_int("TRACKER_MOVE_ETA_MIN_SAMPLES", 3),
            move_eta_history=_env_int("TRACKER_MOVE_ETA_HISTORY", 24),
            move_eta_min=_env_float("TRACKER_MOVE_ETA_MIN", 0.50),
            move_eta_max=_env_float("TRACKER_MOVE_ETA_MAX", 2.00),
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
        cfg.lead_edge_margin = max(0.0, min(0.10, cfg.lead_edge_margin))
        cfg.move_eta_min_samples = max(2, min(12, cfg.move_eta_min_samples))
        cfg.move_eta_history = max(cfg.move_eta_min_samples, min(100, cfg.move_eta_history))
        cfg.move_eta_min = max(0.20, min(2.0, cfg.move_eta_min))
        cfg.move_eta_max = max(cfg.move_eta_min, min(4.0, cfg.move_eta_max))
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

    def resolve_class_ids(self, model_names: Mapping[int, str]) -> Tuple[List[int], List[str]]:
        """Map configured class names to the class ids of the loaded model.

        Returns (ids, unresolved_tokens). Different models number classes
        differently (COCO dog=16, Frigate+ dog=3, Frigate+ 16=license_plate),
        so ids must come from the model, never from a hardcoded table.
        """
        by_label: Dict[str, int] = {}
        valid_ids = set()
        for cid, label in model_names.items():
            valid_ids.add(int(cid))
            by_label.setdefault(str(label).strip().lower(), int(cid))
        ids: List[int] = []
        unresolved: List[str] = []
        for token in self.target_classes:
            if token.isdigit():
                cid = int(token)
                if cid not in valid_ids:
                    unresolved.append(token)
                    continue
            elif token in by_label:
                cid = by_label[token]
            else:
                unresolved.append(token)
                continue
            if cid not in ids:
                ids.append(cid)
        return ids, unresolved

    @property
    def target_class_ids(self) -> List[int]:
        """LEGACY COCO-only fallback. Used only when no model label map is available."""
        ids: List[int] = []
        for name in self.target_classes:
            if name.isdigit():
                ids.append(int(name))
            elif name in COCO_TRACKER_CLASSES:
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
            "lead_edge_margin": self.lead_edge_margin,
            "adaptive_lead": self.adaptive_lead,
            "move_eta_min_samples": self.move_eta_min_samples,
            "move_eta_history": self.move_eta_history,
            "move_eta_min": self.move_eta_min,
            "move_eta_max": self.move_eta_max,
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


