from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from onvif import ONVIFCamera
except Exception:  # Optional at import/test time; Docker installs it for runtime.
    ONVIFCamera = None


BBox = Tuple[float, float, float, float]


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def frame_sharpness(frame: np.ndarray) -> float:
    """Cheap focus/motion-blur score; larger values are sharper."""
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def zoom_bucket(factor: float) -> str:
    factor = max(1.0, float(factor))
    if factor < 1.5:
        return "1.0-1.5"
    if factor < 2.0:
        return "1.5-2.0"
    if factor < 2.5:
        return "2.0-2.5"
    return "2.5+"


@dataclass
class TargetObservation:
    when: float
    bbox: BBox
    confidence: float
    span: float
    edge_touches: int


class TargetHistory:
    """Short, robust history used for Frigate-style size/zoom decisions."""

    def __init__(self, window_s: float = 1.5, maxlen: int = 128) -> None:
        self.window_s = max(0.5, float(window_s))
        self.items: Deque[TargetObservation] = deque(maxlen=max(16, int(maxlen)))

    def clear(self) -> None:
        self.items.clear()

    @staticmethod
    def _edge_touches(bbox: BBox, frame_shape: Tuple[int, ...], margin: float = 0.02) -> int:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        mx = max(1.0, w * margin)
        my = max(1.0, h * margin)
        return int(x1 <= mx) + int(x2 >= w - mx) + int(y1 <= my) + int(y2 >= h - my)

    def add(self, when: float, bbox: BBox, confidence: float, frame_shape: Tuple[int, ...]) -> None:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        span = max(
            max(0.0, x2 - x1) / max(1.0, float(w)),
            max(0.0, y2 - y1) / max(1.0, float(h)),
        )
        self.items.append(
            TargetObservation(
                when=float(when),
                bbox=tuple(float(v) for v in bbox),
                confidence=float(confidence),
                span=float(span),
                edge_touches=self._edge_touches(bbox, frame_shape),
            )
        )
        self._trim(when)

    def _trim(self, now: float) -> None:
        cutoff = float(now) - self.window_s
        while self.items and self.items[0].when < cutoff:
            self.items.popleft()

    def metrics(self, now: float, prediction_horizon_s: float = 1.0) -> dict:
        self._trim(now)
        if not self.items:
            return {
                "samples": 0,
                "weighted_span": None,
                "predicted_span": None,
                "span_slope_per_s": 0.0,
                "edge_touches": 0,
            }

        rows = list(self.items)
        spans = np.array([row.span for row in rows], dtype=float)
        if len(spans) >= 4:
            q1, q3 = np.percentile(spans, [25, 75])
            iqr = max(1e-6, float(q3 - q1))
            low = float(q1 - 1.5 * iqr)
            high = float(q3 + 1.5 * iqr)
            filtered = [row for row in rows if low <= row.span <= high]
            if len(filtered) >= 2:
                rows = filtered
                spans = np.array([row.span for row in rows], dtype=float)

        weights = np.arange(1, len(rows) + 1, dtype=float)
        weighted_span = float(np.average(spans, weights=weights))

        # Edge-clipped boxes have distorted dimensions. Keep them for the current
        # safety signal, but exclude them from size-trend regression when possible.
        trend_rows = [row for row in rows if row.edge_touches == 0]
        if len(trend_rows) < 3:
            trend_rows = rows

        slope = 0.0
        predicted = weighted_span
        if len(trend_rows) >= 3:
            t0 = trend_rows[0].when
            x = np.array([row.when - t0 for row in trend_rows], dtype=float)
            y = np.array([row.span for row in trend_rows], dtype=float)
            if float(np.ptp(x)) >= 0.15:
                design = np.column_stack((np.ones(len(x)), x))
                intercept, slope_raw = np.linalg.lstsq(design, y, rcond=None)[0]
                if math.isfinite(float(intercept)) and math.isfinite(float(slope_raw)):
                    slope = float(slope_raw)
                    future_t = max(0.0, float(now) - t0) + max(0.0, float(prediction_horizon_s))
                    predicted = float(intercept + slope * future_t)

        return {
            "samples": len(rows),
            "weighted_span": _bounded(weighted_span, 0.0, 1.5),
            "predicted_span": _bounded(predicted, 0.0, 1.5),
            "span_slope_per_s": float(slope),
            "edge_touches": int(self.items[-1].edge_touches),
        }


@dataclass
class CameraMotionEstimate:
    matrix: np.ndarray
    mode: str
    points: int
    inlier_ratio: float
    dx: float
    dy: float

    def apply_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        p = np.array([float(point[0]), float(point[1]), 1.0], dtype=float)
        out = self.matrix @ p
        if not np.all(np.isfinite(out)) or abs(float(out[2])) < 1e-8:
            return point
        return (float(out[0] / out[2]), float(out[1] / out[2]))

    def public_dict(self) -> dict:
        return {
            "mode": self.mode,
            "points": self.points,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "dx": round(self.dx, 2),
            "dy": round(self.dy, 2),
        }


class CameraMotionEstimator:
    """OpenCV-only global motion estimator inspired by Frigate/Norfair."""

    def __init__(self, max_points: int = 120, min_points: int = 12) -> None:
        self.max_points = max(24, int(max_points))
        self.min_points = max(6, int(min_points))
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_boxes: List[BBox] = []
        self.last_estimate: Optional[CameraMotionEstimate] = None
        self.failures = 0

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_boxes = []
        self.last_estimate = None

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

    @staticmethod
    def _finite_matrix(matrix: np.ndarray) -> bool:
        if matrix is None or matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            return False
        if abs(float(matrix[2, 2])) < 1e-8:
            return False
        det = float(np.linalg.det(matrix[:2, :2]))
        return math.isfinite(det) and 0.35 <= abs(det) <= 3.0

    def _feature_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        h, w = shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        pad = max(6, int(min(h, w) * 0.02))
        for x1, y1, x2, y2 in self.prev_boxes:
            xa = max(0, int(x1) - pad)
            ya = max(0, int(y1) - pad)
            xb = min(w, int(x2) + pad)
            yb = min(h, int(y2) + pad)
            if xb > xa and yb > ya:
                mask[ya:yb, xa:xb] = 0
        return mask

    def update(
        self,
        frame: np.ndarray,
        boxes: Sequence[BBox],
        *,
        active: bool,
        use_homography: bool = False,
    ) -> Optional[CameraMotionEstimate]:
        gray = self._gray(frame)
        current_boxes = [tuple(float(v) for v in box) for box in boxes]

        if self.prev_gray is None or self.prev_gray.shape != gray.shape or not active:
            self.prev_gray = gray
            self.prev_boxes = current_boxes
            self.last_estimate = None
            return None

        try:
            points0 = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.max_points,
                qualityLevel=0.01,
                minDistance=8,
                mask=self._feature_mask(self.prev_gray.shape),
                blockSize=7,
            )
            if points0 is None or len(points0) < self.min_points:
                raise ValueError("insufficient background features")

            points1, status, errors = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                points0,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
            )
            if points1 is None or status is None:
                raise ValueError("optical flow failed")

            good = status.reshape(-1).astype(bool)
            if errors is not None:
                good &= np.isfinite(errors.reshape(-1)) & (errors.reshape(-1) < 40.0)
            p0 = points0.reshape(-1, 2)[good]
            p1 = points1.reshape(-1, 2)[good]
            finite = np.all(np.isfinite(p0), axis=1) & np.all(np.isfinite(p1), axis=1)
            p0 = p0[finite]
            p1 = p1[finite]
            if len(p0) < self.min_points:
                raise ValueError("insufficient valid optical-flow points")

            matrix: Optional[np.ndarray] = None
            mode = "translation"
            inlier_ratio = 1.0

            if use_homography and len(p0) >= max(self.min_points, 16):
                homography, inliers = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
                if homography is not None and inliers is not None:
                    ratio = float(np.mean(inliers.reshape(-1).astype(bool)))
                    if ratio >= 0.40 and self._finite_matrix(homography):
                        matrix = homography.astype(float)
                        mode = "homography"
                        inlier_ratio = ratio

            if matrix is None:
                delta = p1 - p0
                med = np.median(delta, axis=0)
                residual = np.linalg.norm(delta - med, axis=1)
                med_res = float(np.median(residual))
                mad = float(np.median(np.abs(residual - med_res)))
                limit = max(1.5, med_res + 3.0 * 1.4826 * mad)
                keep = residual <= limit
                if int(np.count_nonzero(keep)) >= self.min_points:
                    p0 = p0[keep]
                    p1 = p1[keep]
                    delta = p1 - p0
                dx, dy = np.median(delta, axis=0)
                matrix = np.array(
                    [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
                    dtype=float,
                )
                inlier_ratio = float(len(delta)) / max(1.0, float(np.count_nonzero(good)))

            if not self._finite_matrix(matrix):
                raise ValueError("non-finite/degenerate camera transform")

            h, w = gray.shape[:2]
            center = np.array([w / 2.0, h / 2.0, 1.0], dtype=float)
            moved = matrix @ center
            moved = moved / moved[2]
            estimate = CameraMotionEstimate(
                matrix=matrix,
                mode=mode,
                points=int(len(p0)),
                inlier_ratio=float(inlier_ratio),
                dx=float(moved[0] - center[0]),
                dy=float(moved[1] - center[1]),
            )
            self.last_estimate = estimate
            self.failures = 0
            return estimate
        except Exception:
            self.failures += 1
            self.last_estimate = None
            return None
        finally:
            # Every estimate is frame-to-frame. A failed transform cannot poison
            # subsequent estimates because the current frame becomes the new base.
            self.prev_gray = gray
            self.prev_boxes = current_boxes


class CalibrationStore:
    """Small atomic JSON store for camera-specific timing/spatial calibration."""

    def __init__(self, path: str, camera_key: str, enabled: bool = True) -> None:
        self.path = Path(path)
        self.camera_key = str(camera_key or "default")
        self.enabled = bool(enabled)
        self.data: Dict[str, object] = {"version": 1, "cameras": {}}
        self._dirty = False
        self._last_write = 0.0
        self._load()

    def _load(self) -> None:
        if not self.enabled or not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, dict) and isinstance(loaded.get("cameras", {}), dict):
                self.data = loaded
        except Exception:
            # Corrupt calibration must never prevent tracking from starting.
            self.data = {"version": 1, "cameras": {}}

    def _camera(self) -> dict:
        cameras = self.data.setdefault("cameras", {})
        camera = cameras.setdefault(self.camera_key, {})
        return camera

    def timing(self) -> dict:
        value = self._camera().get("move_timing", {})
        return value if isinstance(value, dict) else {}

    def set_timing(self, intercept: float, slope: float, samples: Iterable[Tuple[float, float]]) -> None:
        rows = [[float(x), float(y)] for x, y in samples]
        self._camera()["move_timing"] = {
            "intercept": float(intercept),
            "slope": float(slope),
            "samples": rows[-100:],
        }
        self._dirty = True
        self.flush()

    def spatial_scales(self, bucket: str) -> Tuple[float, float]:
        spatial = self._camera().setdefault("spatial_response", {})
        row = spatial.get(bucket, {}) if isinstance(spatial, dict) else {}
        if not isinstance(row, dict):
            return 1.0, 1.0
        return (
            _bounded(float(row.get("pan_scale", 1.0)), 0.75, 1.25),
            _bounded(float(row.get("tilt_scale", 1.0)), 0.75, 1.25),
        )

    def set_spatial_scales(self, bucket: str, pan_scale: float, tilt_scale: float) -> None:
        spatial = self._camera().setdefault("spatial_response", {})
        spatial[bucket] = {
            "pan_scale": _bounded(pan_scale, 0.75, 1.25),
            "tilt_scale": _bounded(tilt_scale, 0.75, 1.25),
        }
        self._dirty = True
        self.flush()

    def public_dict(self) -> dict:
        camera = self._camera()
        timing = camera.get("move_timing", {}) if isinstance(camera, dict) else {}
        if not isinstance(timing, dict): timing = {}
        samples = timing.get("samples", [])
        return {"path": str(self.path), "loaded": bool(camera), "move_timing": {"intercept": timing.get("intercept"), "slope": timing.get("slope"), "samples": len(samples) if isinstance(samples, list) else 0}, "spatial_response": camera.get("spatial_response", {})}

    def flush(self, force: bool = False) -> None:
        if not self.enabled or not self._dirty:
            return
        now = time.monotonic()
        if not force and (now - self._last_write) < 2.0:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
            temp.replace(self.path)
            self._dirty = False
            self._last_write = now
        except Exception:
            # Persistence is an optimization; runtime tracking stays authoritative.
            return


class OnvifAbsoluteZoom:
    """Optional exact zoom controller; safe CGI fallback remains in tracker.py."""

    def __init__(self, host: str, port: int, user: str, password: str, logger, mode: str = "auto") -> None:
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.logger = logger
        self.mode = (mode or "auto").strip().lower()
        self.camera = None
        self.ptz = None
        self.profile_token: Optional[str] = None
        self.zoom_min: Optional[float] = None
        self.zoom_max: Optional[float] = None
        self.available = False
        self.last_error: Optional[str] = None
        self.failures = 0

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

    async def initialize(self) -> bool:
        if self.mode == "cgi" or ONVIFCamera is None:
            self.available = False
            if ONVIFCamera is None and self.mode != "cgi":
                self.last_error = "onvif-zeep-async is unavailable"
            return False
        try:
            self.camera = ONVIFCamera(self.host, self.port, self.user, self.password)
            await self.camera.update_xaddrs()
            media = await self.camera.create_media_service()
            profiles = await media.GetProfiles()
            profile = next((p for p in profiles if self._get(p, "PTZConfiguration") is not None), None)
            if profile is None:
                raise RuntimeError("camera exposes no PTZ-capable ONVIF media profile")

            self.ptz = await self.camera.create_ptz_service()
            ptz_config = self._get(profile, "PTZConfiguration")
            request = self.ptz.create_type("GetConfigurationOptions")
            request.ConfigurationToken = self._get(ptz_config, "token")
            options = await self.ptz.GetConfigurationOptions(request)
            spaces = self._get(options, "Spaces")
            ranges = self._get(spaces, "AbsoluteZoomPositionSpace", []) or []
            if not ranges:
                raise RuntimeError("ONVIF AbsoluteZoomPositionSpace is not supported")
            x_range = self._get(ranges[0], "XRange")
            zmin = float(self._get(x_range, "Min"))
            zmax = float(self._get(x_range, "Max"))
            if not (math.isfinite(zmin) and math.isfinite(zmax) and zmax > zmin):
                raise RuntimeError("invalid ONVIF absolute zoom range")

            self.profile_token = str(self._get(profile, "token"))
            self.zoom_min = zmin
            self.zoom_max = zmax
            self.available = True
            self.failures = 0
            self.last_error = None
            return True
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)
            try:
                await self.close()
            except Exception:
                pass
            return False

    async def set_normalized(self, normalized: float, speed: float = 1.0) -> bool:
        """Move to an exact normalized point in the camera's advertised absolute zoom range."""
        if not self.available or self.ptz is None or self.profile_token is None:
            return False
        if self.zoom_min is None or self.zoom_max is None:
            return False
        try:
            normalized = _bounded(float(normalized), 0.0, 1.0)
            target = self.zoom_min + normalized * (self.zoom_max - self.zoom_min)
            request = self.ptz.create_type("AbsoluteMove")
            request.ProfileToken = self.profile_token
            request.Speed = {"Zoom": float(speed)}
            request.Position = {"Zoom": float(target)}
            await self.ptz.AbsoluteMove(request)
            self.last_error = None
            self.failures = 0
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.failures += 1
            if self.failures >= 2:
                self.available = False
            return False

    async def set_factor(self, factor: float, max_optical_zoom: float, speed: float = 1.0) -> bool:
        max_optical_zoom = max(1.01, float(max_optical_zoom))
        factor = _bounded(float(factor), 1.0, max_optical_zoom)
        normalized = (factor - 1.0) / (max_optical_zoom - 1.0)
        return await self.set_normalized(normalized, speed=speed)

    async def close(self) -> None:
        camera = self.camera
        self.camera = None
        self.ptz = None
        self.available = False
        if camera is not None:
            try:
                await camera.close()
            except Exception:
                pass

    def public_dict(self) -> dict:
        return {
            "requested_mode": self.mode,
            "available": self.available,
            "control": "onvif_absolute" if self.available else "cgi_timed_fallback",
            "onvif_port": self.port,
            "range": None if self.zoom_min is None or self.zoom_max is None else [self.zoom_min, self.zoom_max],
            "failures": self.failures,
            "last_error": self.last_error,
        }
