from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

from tracker_config import TrackerConfig


class LatestFrameCapture:
    """Continuously drains RTSP packets via grab() and decodes on demand via retrieve()."""

    def __init__(self, url: str, open_timeout_s: float = 5.0, read_timeout_s: float = 5.0):
        self.url = url
        self.open_timeout_s = max(1.0, float(open_timeout_s))
        self.read_timeout_s = max(1.0, float(read_timeout_s))
        self._cap = None
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._decoded_seq = 0
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
            if (
                self._cap is not None
                and self._decoded_seq != self._seq
                and hasattr(self._cap, "retrieve")
            ):
                ok, frame = self._cap.retrieve()
                if ok and frame is not None:
                    self._frame = frame
                    self._decoded_seq = self._seq
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
                use_grab = hasattr(cap, "grab") and hasattr(cap, "retrieve")
                with self._lock:
                    self._cap = cap
                while not self._stop.is_set():
                    if use_grab:
                        with self._lock:
                            ok = cap.grab()
                            if not ok:
                                raise RuntimeError("RTSP read failed")
                            self._seq += 1
                            self._frame_time = time.monotonic()
                    else:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            raise RuntimeError("RTSP read failed")
                        now = time.monotonic()
                        with self._lock:
                            self._frame = frame
                            self._seq += 1
                            self._decoded_seq = self._seq
                            self._frame_time = now
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
                self.reconnects += 1
                if not self._stop.wait(1.0):
                    continue
            finally:
                with self._lock:
                    self._cap = None
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


