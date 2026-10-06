from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import List, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Polygon = List[Point]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _parse_polygons(raw: str) -> List[Polygon]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except Exception:
        return []
    polygons: List[Polygon] = []
    if not isinstance(value, list):
        return polygons
    for candidate in value:
        if not isinstance(candidate, list) or len(candidate) < 3:
            continue
        poly: Polygon = []
        valid = True
        for point in candidate:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                valid = False
                break
            try:
                x = _clamp(float(point[0]), 0.0, 1.0)
                y = _clamp(float(point[1]), 0.0, 1.0)
            except (TypeError, ValueError):
                valid = False
                break
            poly.append((x, y))
        if valid and len(poly) >= 3:
            polygons.append(poly)
    return polygons


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        denominator = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / denominator + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


class AcquisitionZonePolicy:
    """Optional normalized required/ignore polygons used only for initial acquisition."""

    def __init__(self, required_raw: str = "", ignore_raw: str = "") -> None:
        self.required = _parse_polygons(required_raw)
        self.ignore = _parse_polygons(ignore_raw)
        self.rejected_required = 0
        self.rejected_ignore = 0

    def allows(self, center_px: Point, frame_shape: Tuple[int, ...]) -> bool:
        h, w = frame_shape[:2]
        point = (
            _clamp(center_px[0] / max(1.0, float(w)), 0.0, 1.0),
            _clamp(center_px[1] / max(1.0, float(h)), 0.0, 1.0),
        )
        if any(_point_in_polygon(point, poly) for poly in self.ignore):
            self.rejected_ignore += 1
            return False
        if self.required and not any(_point_in_polygon(point, poly) for poly in self.required):
            self.rejected_required += 1
            return False
        return True

    def public_dict(self) -> dict:
        return {
            "required_zone_count": len(self.required),
            "ignore_zone_count": len(self.ignore),
            "rejected_required": self.rejected_required,
            "rejected_ignore": self.rejected_ignore,
        }


class MotionMaskPolicy:
    """Static normalized polygons converted to conservative bounding boxes for optical-flow masking."""

    def __init__(self, raw: str = "") -> None:
        self.polygons = _parse_polygons(raw)

    def boxes(self, frame_shape: Tuple[int, ...]) -> List[BBox]:
        h, w = frame_shape[:2]
        result: List[BBox] = []
        for poly in self.polygons:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            result.append((min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h))
        return result

    def public_dict(self) -> dict:
        return {"mask_count": len(self.polygons)}


@dataclass
class BBoxMotionResult:
    valid: bool
    reason: Optional[str]
    edge_velocity_px_s: Tuple[float, float, float, float]
    scale_rate_per_s: Tuple[float, float]

    def public_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "edge_velocity_px_s": [round(v, 1) for v in self.edge_velocity_px_s],
            "scale_rate_per_s": [round(v, 3) for v in self.scale_rate_per_s],
        }


class BBoxMotionValidator:
    """Reject center-velocity samples that are likely caused by detector-box deformation."""

    def __init__(self) -> None:
        self.previous_bbox: Optional[BBox] = None
        self.previous_time: Optional[float] = None
        self.last_result: Optional[BBoxMotionResult] = None

    def reset(self, bbox: Optional[BBox] = None, when: Optional[float] = None) -> None:
        self.previous_bbox = bbox
        self.previous_time = when
        self.last_result = None

    def validate(self, bbox: BBox, when: float, frame_shape: Tuple[int, ...]) -> BBoxMotionResult:
        previous = self.previous_bbox
        previous_time = self.previous_time
        self.previous_bbox = bbox
        self.previous_time = float(when)
        if previous is None or previous_time is None:
            result = BBoxMotionResult(True, None, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0))
            self.last_result = result
            return result
        dt = float(when) - float(previous_time)
        if dt < 0.06 or dt > 0.75:
            result = BBoxMotionResult(True, "sample_interval", (0.0, 0.0, 0.0, 0.0), (0.0, 0.0))
            self.last_result = result
            return result

        x1, y1, x2, y2 = [float(v) for v in bbox]
        px1, py1, px2, py2 = [float(v) for v in previous]
        vx1, vy1, vx2, vy2 = (x1 - px1) / dt, (y1 - py1) / dt, (x2 - px2) / dt, (y2 - py2) / dt
        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))
        width0 = max(2.0, px2 - px1)
        height0 = max(2.0, py2 - py1)
        width_rate = ((x2 - x1) - width0) / width0 / dt
        height_rate = ((y2 - y1) - height0) / height0 / dt

        corner_a = (vx1, vy1)
        corner_b = (vx2, vy2)
        speed_a = math.hypot(*corner_a)
        speed_b = math.hypot(*corner_b)
        delta = math.hypot(vx2 - vx1, vy2 - vy1)
        reason: Optional[str] = None
        if max(speed_a, speed_b) > diag * 2.5:
            reason = "edge_velocity_magnitude"
        elif delta > diag * 0.80:
            reason = "edge_velocity_disagreement"
        elif abs(width_rate) > 2.5 or abs(height_rate) > 2.5:
            # Rapid scale change by itself is not bad velocity evidence. A person
            # walking toward/away from the camera legitimately changes bbox size
            # several times per second. Reject only when edge deformation clearly
            # dominates coherent translation (Frigate-style edge consensus).
            center_vx = 0.5 * (vx1 + vx2)
            center_vy = 0.5 * (vy1 + vy2)
            deform_x = 0.5 * abs(vx2 - vx1)
            deform_y = 0.5 * abs(vy2 - vy1)
            width_bad = (
                abs(width_rate) > 2.5
                and deform_x > max(120.0, abs(center_vx) * 1.75)
            )
            height_bad = (
                abs(height_rate) > 2.5
                and deform_y > max(120.0, abs(center_vy) * 1.75)
            )
            extreme_scale = (
                (abs(width_rate) > 6.0 and deform_x > 80.0)
                or (abs(height_rate) > 6.0 and deform_y > 80.0)
            )
            if width_bad or height_bad or extreme_scale:
                reason = "bbox_scale_jump"
        elif speed_a > 40.0 and speed_b > 40.0:
            cosine = (vx1 * vx2 + vy1 * vy2) / max(1e-6, speed_a * speed_b)
            if cosine < -0.20 and delta > diag * 0.30:
                reason = "edge_direction_disagreement"

        result = BBoxMotionResult(
            reason is None,
            reason,
            (vx1, vy1, vx2, vy2),
            (width_rate, height_rate),
        )
        self.last_result = result
        return result


class SceneStabilityGate:
    """Release post-PTZ tracking only after the image itself is stable, with a bounded timeout."""

    def __init__(self, stable_frames: int = 2, max_wait_s: float = 0.60, flow_threshold_norm: float = 0.012) -> None:
        self.required_frames = max(1, int(stable_frames))
        self.max_wait_s = _clamp(max_wait_s, 0.15, 2.0)
        self.flow_threshold_norm = _clamp(flow_threshold_norm, 0.001, 0.10)
        self.active = False
        self.started_at = 0.0
        self.stable_frames = 0
        self.last_flow_norm: Optional[float] = None
        self.last_reason = "idle"
        self.timed_out = False

    def reset(self, now: float) -> None:
        self.active = True
        self.started_at = float(now)
        self.stable_frames = 0
        self.last_flow_norm = None
        self.last_reason = "waiting"
        self.timed_out = False

    def clear(self) -> None:
        self.active = False
        self.stable_frames = self.required_frames
        self.last_reason = "ready"
        self.timed_out = False

    def observe(self, now: float, motion, sharp_ok: bool, frame_shape: Tuple[int, ...]) -> bool:
        if not self.active:
            return True
        elapsed = max(0.0, float(now) - self.started_at)
        if elapsed >= self.max_wait_s:
            self.active = False
            self.timed_out = True
            self.last_reason = "bounded_timeout"
            return True
        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))
        if motion is None:
            flow_norm = 0.0
        else:
            flow_norm = math.hypot(float(motion.dx), float(motion.dy)) / diag
        self.last_flow_norm = flow_norm
        if sharp_ok and flow_norm <= self.flow_threshold_norm:
            self.stable_frames += 1
            self.last_reason = "stable_candidate"
        else:
            self.stable_frames = 0
            self.last_reason = "blurry" if not sharp_ok else "camera_motion"
        if self.stable_frames >= self.required_frames:
            self.active = False
            self.last_reason = "stable"
            return True
        return False

    def public_dict(self) -> dict:
        return {
            "active": self.active,
            "required_frames": self.required_frames,
            "stable_frames": self.stable_frames,
            "max_wait_s": self.max_wait_s,
            "flow_threshold_norm": self.flow_threshold_norm,
            "last_flow_norm": None if self.last_flow_norm is None else round(self.last_flow_norm, 5),
            "last_reason": self.last_reason,
            "timed_out": self.timed_out,
        }


class OnvifRetryState:
    """Exponential backoff for an optional ONVIF enhancement; native CGI remains authoritative."""

    def __init__(self, base_s: float = 15.0, max_s: float = 300.0) -> None:
        self.base_s = _clamp(base_s, 2.0, 300.0)
        self.max_s = max(self.base_s, _clamp(max_s, self.base_s, 1800.0))
        self.failures = 0
        self.next_retry_at = 0.0
        self.last_error: Optional[str] = None

    def success(self) -> None:
        self.failures = 0
        self.next_retry_at = 0.0
        self.last_error = None

    def failure(self, now: float, error: Optional[str]) -> float:
        self.failures += 1
        delay = min(self.max_s, self.base_s * (2 ** min(8, self.failures - 1)))
        self.next_retry_at = float(now) + delay
        self.last_error = error
        return delay

    def due(self, now: float) -> bool:
        return self.next_retry_at > 0.0 and float(now) >= self.next_retry_at

    def public_dict(self, now: Optional[float] = None) -> dict:
        now = time.monotonic() if now is None else float(now)
        return {
            "failures": self.failures,
            "retry_in_s": None if self.next_retry_at <= 0 else round(max(0.0, self.next_retry_at - now), 1),
            "last_error": self.last_error,
        }


class ZoomCalibrationMap:
    """Persistent mapping from observed optical factor to ONVIF normalized absolute position."""

    def __init__(self, path: str, camera_key: str) -> None:
        self.path = Path(path)
        self.camera_key = str(camera_key or "default")
        self.data = {"version": 1, "cameras": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, dict) and isinstance(loaded.get("cameras"), dict):
                self.data = loaded
        except Exception:
            self.data = {"version": 1, "cameras": {}}

    def _camera(self) -> dict:
        return self.data.setdefault("cameras", {}).setdefault(self.camera_key, {})

    def points(self) -> List[Tuple[float, float]]:
        rows = self._camera().get("zoom_mapping", [])
        result: List[Tuple[float, float]] = []
        if isinstance(rows, list):
            for row in rows:
                try:
                    factor = float(row[0])
                    normalized = float(row[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(factor) and math.isfinite(normalized) and factor >= 1.0:
                    result.append((factor, _clamp(normalized, 0.0, 1.0)))
        return sorted(result)

    def estimate_normalized(self, factor: float, fallback_max_optical_zoom: float) -> float:
        factor = max(1.0, float(factor))
        rows = self.points()
        if len(rows) < 2:
            maximum = max(1.01, float(fallback_max_optical_zoom))
            return _clamp((factor - 1.0) / (maximum - 1.0), 0.0, 1.0)
        if factor <= rows[0][0]:
            return rows[0][1]
        if factor >= rows[-1][0]:
            a, b = rows[-2], rows[-1]
        else:
            a = rows[0]
            b = rows[-1]
            for left, right in zip(rows, rows[1:]):
                if left[0] <= factor <= right[0]:
                    a, b = left, right
                    break
        if abs(b[0] - a[0]) < 1e-6:
            return _clamp(b[1], 0.0, 1.0)
        ratio = (factor - a[0]) / (b[0] - a[0])
        return _clamp(a[1] + ratio * (b[1] - a[1]), 0.0, 1.0)

    def replace_points(self, rows: Sequence[Tuple[float, float]]) -> None:
        """Atomically replace the learned map with a compact monotonic calibration.

        Cameras often expose plateaus where several ONVIF positions report the same
        physical zoom factor. Collapse those plateaus to one representative point.
        Wide angle is special: if a measured 1x plateau includes normalized 0.0, keep
        that true optical minimum instead of allowing later 1x observations to walk
        the anchor upward.
        """
        cleaned: List[Tuple[float, float]] = []
        for factor, normalized in rows:
            try:
                factor = max(1.0, float(factor))
                normalized = _clamp(float(normalized), 0.0, 1.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(factor) and math.isfinite(normalized):
                cleaned.append((factor, normalized))
        cleaned.sort(key=lambda row: (row[0], row[1]))

        groups: List[List[Tuple[float, float]]] = []
        for row in cleaned:
            if not groups or abs(row[0] - groups[-1][-1][0]) > 0.04:
                groups.append([row])
            else:
                groups[-1].append(row)

        compact: List[Tuple[float, float]] = []
        for group in groups:
            factors = sorted(row[0] for row in group)
            positions = sorted(row[1] for row in group)
            midpoint = len(group) // 2
            factor = factors[midpoint]
            normalized = positions[midpoint]
            if min(factors) <= 1.04 and min(positions) <= 0.005:
                factor = 1.0
                normalized = 0.0
            if compact and normalized + 0.003 < compact[-1][1]:
                continue
            compact.append((factor, normalized))

        self._camera()["zoom_mapping"] = [
            [round(f, 5), round(n, 7)] for f, n in compact[-48:]
        ]
        self._flush()

    def record(self, actual_factor: float, normalized: float) -> None:
        actual_factor = max(1.0, float(actual_factor))
        normalized = _clamp(normalized, 0.0, 1.0)
        rows = self.points()

        # Never let the physical 1x plateau move the wide-angle anchor away from
        # the ONVIF minimum. This is exactly what a Dahua camera with a zoom
        # deadband can otherwise do during repeated observations.
        if actual_factor <= 1.04:
            rows.append((1.0, normalized))
            if normalized <= 0.005 or any(f <= 1.04 and n <= 0.005 for f, n in rows):
                rows.append((1.0, 0.0))
        else:
            rows.append((actual_factor, normalized))
        self.replace_points(rows)

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
            temp.replace(self.path)
        except Exception:
            return

    def public_dict(self) -> dict:
        rows = self.points()
        return {
            "path": str(self.path),
            "samples": len(rows),
            "points": [[round(f, 3), round(n, 5)] for f, n in rows],
        }
