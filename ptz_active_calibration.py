from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from onvif import ONVIFCamera
except Exception:  # Optional for unit tests; runtime image installs onvif-zeep-async.
    ONVIFCamera = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _zoom_key(factor: float) -> str:
    return f"{max(1.0, float(factor)):.2f}"


@dataclass
class StaticFrameShift:
    dx: float
    dy: float
    points: int
    inlier_ratio: float
    method: str

    def normalized(self, frame_shape: Tuple[int, ...]) -> Tuple[float, float]:
        h, w = frame_shape[:2]
        return (
            float(self.dx) / max(1.0, float(w) / 2.0),
            float(self.dy) / max(1.0, float(h) / 2.0),
        )

    def public_dict(self) -> dict:
        return {
            "dx": round(self.dx, 2),
            "dy": round(self.dy, 2),
            "points": self.points,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "method": self.method,
        }


def estimate_static_frame_shift(before: np.ndarray, after: np.ndarray) -> Optional[StaticFrameShift]:
    """Estimate scene translation between two settled frames.

    Active calibration intentionally compares settled before/after frames rather than
    tracking every blurred frame during PTZ motion. ORB + RANSAC tolerates the larger
    displacement of moveDirectly better than frame-to-frame Lucas-Kanade flow. A
    phase-correlation fallback keeps small, texture-poor calibration moves useful.
    """
    if before is None or after is None or before.size == 0 or after.size == 0:
        return None
    if before.shape[:2] != after.shape[:2]:
        return None
    try:
        gray0 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if before.ndim == 3 else before
        gray1 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if after.ndim == 3 else after
        orb = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
        kp0, des0 = orb.detectAndCompute(gray0, None)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        if des0 is not None and des1 is not None and len(kp0) >= 12 and len(kp1) >= 12:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(des0, des1), key=lambda m: m.distance)
            if len(matches) >= 12:
                keep_count = max(12, min(160, int(len(matches) * 0.65)))
                matches = matches[:keep_count]
                p0 = np.float32([kp0[m.queryIdx].pt for m in matches])
                p1 = np.float32([kp1[m.trainIdx].pt for m in matches])
                matrix, inliers = cv2.estimateAffinePartial2D(
                    p0,
                    p1,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=4.0,
                    maxIters=2500,
                    confidence=0.995,
                    refineIters=10,
                )
                if matrix is not None and np.all(np.isfinite(matrix)):
                    h, w = gray0.shape[:2]
                    center = np.array([w / 2.0, h / 2.0, 1.0], dtype=float)
                    moved = matrix @ center
                    dx = float(moved[0] - center[0])
                    dy = float(moved[1] - center[1])
                    ratio = 1.0 if inliers is None else float(np.mean(inliers.reshape(-1).astype(bool)))
                    # Reject implausible scale/rotation transforms. Calibration wants
                    # the dominant scene translation, not a coincidental feature fit.
                    a, b = float(matrix[0, 0]), float(matrix[0, 1])
                    scale = math.hypot(a, b)
                    if 0.70 <= scale <= 1.35 and ratio >= 0.30:
                        return StaticFrameShift(dx, dy, len(matches), ratio, "orb_affine")
    except Exception:
        pass

    try:
        g0 = np.float32(gray0)
        g1 = np.float32(gray1)
        h, w = g0.shape[:2]
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(g0, g1, window)
        if math.isfinite(dx) and math.isfinite(dy) and math.isfinite(response) and response >= 0.05:
            return StaticFrameShift(float(dx), float(dy), 0, float(response), "phase_correlation")
    except Exception:
        pass
    return None


class ActivePtzCalibrationStore:
    """Persistent active calibration for moveDirectly and continuous PTZ response."""

    def __init__(self, path: str, camera_key: str) -> None:
        self.path = Path(path)
        self.camera_key = str(camera_key or "default")
        self.data: Dict[str, object] = {"version": 1, "cameras": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            value = json.loads(self.path.read_text())
            if isinstance(value, dict) and isinstance(value.get("cameras"), dict):
                self.data = value
        except Exception:
            self.data = {"version": 1, "cameras": {}}

    def _camera(self) -> dict:
        return self.data.setdefault("cameras", {}).setdefault(self.camera_key, {})

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
            temp.replace(self.path)
        except Exception:
            return

    def has_motion_calibration(self) -> bool:
        camera = self._camera()
        move = camera.get("move_directly")
        continuous = camera.get("continuous")
        return isinstance(move, dict) and bool(move.get("samples")) and isinstance(continuous, dict) and bool(continuous.get("samples"))

    def move_directly(self) -> dict:
        value = self._camera().get("move_directly", {})
        return dict(value) if isinstance(value, dict) else {}

    def continuous(self) -> dict:
        value = self._camera().get("continuous", {})
        return dict(value) if isinstance(value, dict) else {}

    def onvif_benchmark(self) -> Optional[dict]:
        value = self._camera().get("onvif_benchmark")
        return dict(value) if isinstance(value, dict) else None

    def age_days(self) -> Optional[float]:
        stamp = self._camera().get("calibrated_at_epoch")
        try:
            return max(0.0, (time.time() - float(stamp)) / 86400.0)
        except (TypeError, ValueError):
            return None

    def is_fresh(self, max_age_days: float) -> bool:
        if not self.has_motion_calibration():
            return False
        age = self.age_days()
        return age is not None and age <= max(0.0, float(max_age_days))

    def replace_motion_calibration(
        self,
        *,
        move_directly: dict,
        continuous: dict,
        onvif_benchmark: Optional[dict] = None,
    ) -> None:
        camera = self._camera()
        camera["calibrated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        camera["calibrated_at_epoch"] = time.time()
        camera["move_directly"] = move_directly
        camera["continuous"] = continuous
        if onvif_benchmark is not None:
            camera["onvif_benchmark"] = onvif_benchmark
        self._flush()

    def continuous_sign(self, axis: str, fallback: int) -> int:
        continuous = self._camera().get("continuous", {})
        signs = continuous.get("signs", {}) if isinstance(continuous, dict) else {}
        try:
            value = int(signs.get(axis, fallback))
            return 1 if value >= 0 else -1
        except Exception:
            return 1 if fallback >= 0 else -1

    def _nearest_zoom_rates(self, axis: str, factor: float) -> Dict[int, float]:
        continuous = self._camera().get("continuous", {})
        by_zoom = continuous.get("by_zoom", {}) if isinstance(continuous, dict) else {}
        if not isinstance(by_zoom, dict) or not by_zoom:
            return {}
        candidates = []
        for key, row in by_zoom.items():
            try:
                z = float(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            axis_row = row.get(axis, {})
            if isinstance(axis_row, dict) and axis_row:
                candidates.append((abs(z - factor), axis_row))
        if not candidates:
            return {}
        axis_row = min(candidates, key=lambda item: item[0])[1]
        result: Dict[int, float] = {}
        for key, value in axis_row.items():
            try:
                speed = abs(int(key))
                rate = abs(float(value))
            except (TypeError, ValueError):
                continue
            if speed > 0 and math.isfinite(rate) and rate > 0:
                result[speed] = rate
        return result

    def choose_continuous_speed(
        self,
        axis: str,
        error_magnitude: float,
        zoom_factor: float,
        *,
        min_speed: int,
        max_speed: int,
        exit_error: float,
        full_speed_error: float,
    ) -> Optional[int]:
        rates = self._nearest_zoom_rates(axis, zoom_factor)
        rates = {s: r for s, r in rates.items() if min_speed <= s <= max_speed}
        if not rates:
            return None
        span = max(0.01, float(full_speed_error) - float(exit_error))
        ratio = _clamp((abs(float(error_magnitude)) - float(exit_error)) / span, 0.0, 1.0)
        ordered = sorted(rates.items())
        low_rate = ordered[0][1]
        high_rate = ordered[-1][1]
        desired = low_rate + ratio * max(0.0, high_rate - low_rate)
        return min(ordered, key=lambda item: (abs(item[1] - desired), item[0]))[0]

    def public_dict(self) -> dict:
        camera = self._camera()
        move = camera.get("move_directly", {}) if isinstance(camera, dict) else {}
        continuous = camera.get("continuous", {}) if isinstance(camera, dict) else {}
        return {
            "path": str(self.path),
            "available": self.has_motion_calibration(),
            "calibrated_at": camera.get("calibrated_at") if isinstance(camera, dict) else None,
            "age_days": None if self.age_days() is None else round(self.age_days(), 3),
            "move_directly_samples": len(move.get("samples", [])) if isinstance(move, dict) else 0,
            "continuous_samples": len(continuous.get("samples", [])) if isinstance(continuous, dict) else 0,
            "continuous_signs": continuous.get("signs", {}) if isinstance(continuous, dict) else {},
            "onvif_benchmark": camera.get("onvif_benchmark") if isinstance(camera, dict) else None,
        }


class OnvifPanTiltProbe:
    """Probe ONVIF pan/tilt spaces and optionally perform one tiny FOV-relative move."""

    def __init__(self, host: str, port: int, user: str, password: str) -> None:
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.camera = None
        self.ptz = None
        self.profile_token: Optional[str] = None
        self.relative_uri: Optional[str] = None
        self.relative_x: Optional[Tuple[float, float]] = None
        self.relative_y: Optional[Tuple[float, float]] = None
        self.continuous_uri: Optional[str] = None
        self.last_error: Optional[str] = None

    @staticmethod
    def _get(obj, key: str, default=None):
        if obj is None:
            return default
        value = getattr(obj, key, None)
        if value is not None:
            return value
        try:
            return obj[key]
        except Exception:
            return default

    @classmethod
    def _range(cls, obj) -> Optional[Tuple[float, float]]:
        try:
            return float(cls._get(obj, "Min")), float(cls._get(obj, "Max"))
        except Exception:
            return None

    async def initialize(self) -> dict:
        result = {
            "available": False,
            "relative_fov_supported": False,
            "continuous_supported": False,
            "relative_uri": None,
            "continuous_uri": None,
            "error": None,
        }
        if ONVIFCamera is None:
            result["error"] = "onvif-zeep-async unavailable"
            return result
        try:
            self.camera = ONVIFCamera(self.host, self.port, self.user, self.password)
            await self.camera.update_xaddrs()
            media = await self.camera.create_media_service()
            profiles = await media.GetProfiles()
            profile = next((p for p in profiles if self._get(p, "PTZConfiguration") is not None), None)
            if profile is None:
                raise RuntimeError("camera exposes no PTZ-capable ONVIF media profile")
            self.profile_token = str(self._get(profile, "token"))
            config = self._get(profile, "PTZConfiguration")
            config_token = self._get(config, "token")
            self.ptz = await self.camera.create_ptz_service()
            request = self.ptz.create_type("GetConfigurationOptions")
            request.ConfigurationToken = config_token
            options = await self.ptz.GetConfigurationOptions(request)
            spaces = self._get(options, "Spaces")

            relatives = self._get(spaces, "RelativePanTiltTranslationSpace", []) or []
            preferred = None
            for space in relatives:
                uri = str(self._get(space, "URI", ""))
                if "TranslationSpaceFov" in uri or "TranslationSpaceFOV" in uri:
                    preferred = space
                    break
            if preferred is not None:
                self.relative_uri = str(self._get(preferred, "URI", ""))
                self.relative_x = self._range(self._get(preferred, "XRange"))
                self.relative_y = self._range(self._get(preferred, "YRange"))

            continuous = self._get(spaces, "ContinuousPanTiltVelocitySpace", []) or []
            if continuous:
                self.continuous_uri = str(self._get(continuous[0], "URI", ""))

            result.update(
                available=True,
                relative_fov_supported=bool(self.relative_uri and self.relative_x and self.relative_y),
                continuous_supported=bool(self.continuous_uri),
                relative_uri=self.relative_uri,
                continuous_uri=self.continuous_uri,
            )
            return result
        except Exception as exc:
            self.last_error = str(exc)
            result["error"] = self.last_error
            return result

    @staticmethod
    def _map_norm(value: float, bounds: Tuple[float, float]) -> float:
        low, high = bounds
        value = _clamp(value, -1.0, 1.0)
        center = (low + high) / 2.0
        half = (high - low) / 2.0
        return _clamp(center + value * half, low, high)

    async def relative_move(self, pan: float, tilt: float) -> bool:
        if self.ptz is None or self.profile_token is None or not self.relative_uri or not self.relative_x or not self.relative_y:
            return False
        try:
            request = self.ptz.create_type("RelativeMove")
            request.ProfileToken = self.profile_token
            request.Translation = {
                "PanTilt": {
                    "x": self._map_norm(pan, self.relative_x),
                    "y": self._map_norm(tilt, self.relative_y),
                    "space": self.relative_uri,
                }
            }
            await self.ptz.RelativeMove(request)
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    async def close(self) -> None:
        camera = self.camera
        self.camera = None
        self.ptz = None
        if camera is None:
            return
        close = getattr(camera, "close", None)
        if close is None:
            return
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            return
