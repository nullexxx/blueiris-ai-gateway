
    
  
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

    # Native PTZ operation tracking. Instead of guessing how long a 3D move takes,
    # poll getStatus until the camera reports idle and its reported position is stable.
    ptz_status_poll_interval: float = 0.12
    ptz_operation_timeout: float = 4.0
    post_move_frames: int = 2

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

