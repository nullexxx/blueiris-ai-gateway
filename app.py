import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Mapping, Optional, Tuple
import urllib.request
from facenet_pytorch import MTCNN, InceptionResnetV1
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
import numpy as np
from PIL import Image
import tensorrt
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics import settings as yolo_settings
from tracker import DogTracker, TrackerConfig

# Disable Ultralytics' anonymous analytics/crash reporting. Doesn't persist
# via settings.json since that path isn't a mounted volume, so it's set
# here in code to survive container rebuilds.
yolo_settings.update({"sync": False})

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bi-gateway")

# Input tensor shapes are fixed (YOLO @ imgsz=640, FaceNet crops @ 160x160),
# so let cuDNN autotune and cache the fastest conv algorithms for them.
torch.backends.cudnn.benchmark = True

# --- Configuration ---
MODELS_DIR = Path("/app/models")
FACES_DB_PATH = MODELS_DIR / "faces_db.pt"
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "yolo11m")
PRELOAD_MODELS_ENV = os.getenv("PRELOAD_MODELS", "").strip()
ALLOW_LAZY_LOAD = os.getenv("ALLOW_LAZY_LOAD", "false").lower() in ("true", "1")
HALF_PRECISION = os.getenv("HALF_PRECISION", "true").lower() in ("true", "1")
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "8.0"))
MODEL_LOAD_TIMEOUT = float(os.getenv("MODEL_LOAD_TIMEOUT", "180.0"))
RECOVERY_GRACE_PERIOD = float(os.getenv("RECOVERY_GRACE_PERIOD", "4.0"))
MAX_FACE_EMBEDDINGS_PER_USER = max(1, min(100, int(os.getenv("MAX_FACE_EMBEDDINGS_PER_USER", "20"))))
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()


def _env_decode_side(name: str) -> int:
    try:
        return max(0, int(os.getenv(name, "0")))
    except ValueError:
        return 0


# Optional reduced-size JPEG decoding (0 = off). Large JPEGs are decoded via
# libjpeg DCT scaling to the smallest 1/2, 1/4 or 1/8 size whose long side is
# still >= this many pixels; returned boxes are scaled back to the original
# image, so callers see full-resolution coordinates either way.
DETECTION_DECODE_MAX_SIDE = _env_decode_side("DETECTION_DECODE_MAX_SIDE")
FACE_DECODE_MAX_SIDE = _env_decode_side("FACE_DECODE_MAX_SIDE")

MODEL_EXT_PRIORITY = [".engine", ".onnx", ".pt"]


@dataclass
class ModelEntry:
    model: YOLO
    is_pt: bool
    format_name: str


# Global YOLO State
loaded_models: Dict[str, ModelEntry] = {}
failed_models: Dict[str, str] = {}
cache_lock = asyncio.Lock()

# Global Face Recognition State
face_detector: Optional[MTCNN] = None
face_recognizer: Optional[InceptionResnetV1] = None
registered_faces: Dict[str, List[torch.Tensor]] = {}
face_state_lock = asyncio.Lock()

# Flattened (all users' embeddings stacked into one matrix) cache used for
# fast batched matching, so we don't rebuild + re-loop per detected face.
_face_matrix_cache: Optional[torch.Tensor] = None
_face_labels_cache: List[str] = []
_face_cache_dirty: bool = True

# Global GPU Execution Gate & Watchdog State
gpu_lock = asyncio.Lock()
gpu_tainted: bool = False
gpu_taint_generation: int = 0
recovery_timer_handle: Optional[asyncio.TimerHandle] = None

# Telemetry State for Concurrency/Queue Tracking
queue_depth: int = 0
active_inferences: int = 0

GATEWAY_REVISION = "6.7.4"

is_healthy: bool = True
health_failure_reason: Optional[str] = None

_FATAL_CUDA_ERROR_MARKERS = (
    "cuda_error_launch_timeout",
    "cudaerrorlaunchtimeout",
    "launch timed out and was terminated",
    "sticky error detected",
    "cuda_error_illegal_address",
    "illegal memory access",
    "device-side assert triggered",
    "unspecified launch failure",
)


def _is_fatal_cuda_exception(exc: BaseException) -> bool:
    """Return True when an exception indicates a poisoned CUDA context.

    CUDA launch timeout/illegal-address/device-assert failures are sticky at the
    process/context level. Retrying inference in the same process only repeats
    the failure, so the safe recovery is a fresh container/CUDA context.
    """
    seen = set()
    current: Optional[BaseException] = exc
    messages = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}".lower())
        current = current.__cause__ or current.__context__
    combined = " | ".join(messages)
    return any(marker in combined for marker in _FATAL_CUDA_ERROR_MARKERS)


# Optional multi-class PTZ tracker service (configured by TRACKER_* environment variables)
tracker_service: Optional[DogTracker] = None


def discover_models() -> Dict[str, Path]:
    """Map model stem -> best available file on disk (Engine > ONNX > PyTorch)."""
    all_files = list(MODELS_DIR.glob("*.*"))
    stems = {f.stem for f in all_files if f.suffix.lower() in MODEL_EXT_PRIORITY}

    found: Dict[str, Path] = {}
    for stem in stems:
        for ext in MODEL_EXT_PRIORITY:
            candidate = MODELS_DIR / f"{stem}{ext}"
            if candidate.exists():
                found[stem] = candidate
                break
    return found


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_model_path(stem: str) -> Optional[Path]:
    """Return the preferred source weights (.pt or .onnx) for building a TensorRT engine."""
    pt_path = MODELS_DIR / f"{stem}.pt"
    if pt_path.exists():
        return pt_path
    onnx_path = MODELS_DIR / f"{stem}.onnx"
    if onnx_path.exists():
        return onnx_path
    return None


def _tensorrt_manifest(source_path: Path) -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "source_file": source_path.name,
        "source_sha256": _sha256_file(source_path),
        "source_size": source_path.stat().st_size,
        "tensorrt": str(tensorrt.__version__),
        "cuda": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "imgsz": 640,
        "precision": "fp16" if HALF_PRECISION else "fp32",
    }


def ensure_tensorrt_engine(stem: str, force: bool = False) -> bool:
    """Build a TensorRT engine from .pt when its source/runtime manifest is stale.

    Note: Custom standalone .onnx models (like Frigate+) run natively via ONNX Runtime
    CUDAExecutionProvider, whereas .pt models auto-export to TensorRT .engine.
    """
    engine_path = MODELS_DIR / f"{stem}.engine"
    pt_path = MODELS_DIR / f"{stem}.pt"
    manifest_path = MODELS_DIR / f"{stem}.trt_manifest.json"
    legacy_stamp_path = MODELS_DIR / f"{stem}.trt_version"

    if not pt_path.exists():
        return engine_path.exists()

    expected = _tensorrt_manifest(pt_path)
    current = None
    if manifest_path.exists():
        try:
            current = json.loads(manifest_path.read_text())
        except Exception as exc:
            logger.warning("Invalid TensorRT manifest for '%s': %s", stem, exc)

    needs_rebuild = force or not engine_path.exists() or current != expected
    if not needs_rebuild:
        return True

    if engine_path.exists():
        logger.warning("Rebuilding stale TensorRT engine for '%s'.", stem)
        try:
            engine_path.unlink()
        except OSError as exc:
            logger.error("Could not remove stale engine %s: %s", engine_path, exc)
            return False
    manifest_path.unlink(missing_ok=True)

    logger.info(
        "Compiling TensorRT engine for '%s' on %s (%s)...",
        stem,
        expected["gpu_name"],
        expected["precision"],
    )
    try:
        model = YOLO(str(pt_path), task="detect")
        model.export(
            format="engine",
            device=0,
            quantize=(16 if HALF_PRECISION else 32),
            imgsz=640,
            dynamic=False,
        )
        if not engine_path.exists():
            raise RuntimeError(f"Ultralytics export did not create {engine_path}")

        temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        temp_manifest.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
        os.replace(temp_manifest, manifest_path)
        legacy_stamp_path.unlink(missing_ok=True)
        logger.info("Successfully compiled and cached: %s", engine_path.name)
        return True
    except Exception as exc:
        logger.error("Failed to auto-compile engine for '%s': %s", stem, exc, exc_info=True)
        engine_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        return False


def decode_image(image_bytes: bytes, max_side: int = 0) -> Tuple[Image.Image, float, float]:
    """Decode upload bytes to RGB, optionally using JPEG DCT scaling.

    Returns ``(image, scale_x, scale_y)`` where the scales map decoded-image
    pixel coordinates back to the original image.
    """
    img = Image.open(io.BytesIO(image_bytes))
    orig_w, orig_h = img.size
    long_side = max(orig_w, orig_h)
    if max_side > 0 and img.format == "JPEG" and long_side >= 2 * max_side:
        ratio = max_side / long_side
        img.draft("RGB", (max(1, math.ceil(orig_w * ratio)), max(1, math.ceil(orig_h * ratio))))
    img = img.convert("RGB")
    width, height = img.size
    return img, orig_w / width, orig_h / height


def _result_boxes(results) -> List[Tuple[float, float, float, float, float, int]]:
    """Flatten Ultralytics results into (x1, y1, x2, y2, conf, class_id) rows.

    Copies each result's box tensor to the host in a single transfer instead of
    one device sync per coordinate. Must run inside the GPU worker so the copy
    is covered by run_gpu_job's timeout and taint handling.
    """
    rows: List[Tuple[float, float, float, float, float, int]] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        data = boxes.data.cpu().numpy()  # columns: x1, y1, x2, y2, [track_id,] conf, cls
        for row in data:
            rows.append((
                float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                float(row[-2]), int(row[-1]),
            ))
    return rows


def sync_predict(
    model: YOLO,
    img: Image.Image,
    min_conf: float,
    is_pt: bool,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[dict]:
    """Run detection and return plain-Python predictions (executed inside the GPU worker).

    ``scale_x``/``scale_y`` map coordinates on a reduced-size decode back to the
    original image (see decode_image).
    """
    kwargs = {
        "conf": min_conf,
        "device": 0,
        "verbose": False,
    }
    if is_pt and HALF_PRECISION:
        kwargs["quantize"] = 16

    results = model.predict(img, **kwargs)
    names = model.names
    return [
        {
            "confidence": round(conf, 2),
            "label": names.get(class_id, f"class{class_id}"),
            "x_min": int(x1 * scale_x), "y_min": int(y1 * scale_y),
            "x_max": int(x2 * scale_x), "y_max": int(y2 * scale_y),
        }
        for x1, y1, x2, y2, conf, class_id in _result_boxes(results)
    ]


def sync_tracker_predict(entry: ModelEntry, frame: np.ndarray, min_conf: float, target_class_ids: List[int]) -> List[dict]:
    """Run tracker prediction for the configured classes and return plain Python data."""
    kwargs = {
        "conf": min_conf,
        "classes": target_class_ids,
        "imgsz": 640,
        "device": 0,
        "verbose": False,
    }
    if entry.is_pt and HALF_PRECISION:
        kwargs["quantize"] = 16

    results = entry.model.predict(frame, **kwargs)
    names = entry.model.names
    return [
        {
            "class_id": class_id,
            "label": str(names.get(class_id, f"class{class_id}")),
            "confidence": conf,
            "x_min": int(x1), "y_min": int(y1),
            "x_max": int(x2), "y_max": int(y2),
        }
        for x1, y1, x2, y2, conf, class_id in _result_boxes(results)
    ]


def load_and_warmup_sync(path: Path) -> Tuple[YOLO, bool, str]:
    """Loads weights and forces CUDA/TensorRT/ONNX Runtime buffer allocation."""
    ext = path.suffix.lower()
    is_pt = (ext == ".pt")
    model = YOLO(str(path)) if is_pt else YOLO(str(path), task="detect")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # Warm up with the same predictor-setup arguments real requests use (device, quantize):
    # Ultralytics rebuilds its predictor when those change, which would discard the warm-up
    # and any class-name override applied below.
    warmup_kwargs = {"device": 0, "verbose": False}
    if is_pt and HALF_PRECISION:
        warmup_kwargs["quantize"] = 16
    model.predict(dummy_frame, **warmup_kwargs)

    label_map = load_label_map(path)
    if label_map:
        _apply_label_map(model, label_map, path.with_suffix(".json").name)

    format_name = "TensorRT" if ext == ".engine" else ("ONNX" if ext == ".onnx" else "PyTorch")
    if ext == ".onnx":
        provider = _onnx_execution_provider(model)
        if provider is not None and provider != "CUDAExecutionProvider":
            logger.error(
                "ONNX model '%s' is running on %s, not the GPU. ONNX Runtime could not load its CUDA "
                "provider; check the image's CUDA libraries.",
                path.name,
                provider,
            )
            format_name = "ONNX (CPU)"
    return model, is_pt, format_name


def load_label_map(model_path: Path) -> Optional[Dict[int, str]]:
    """Class names from a ``<stem>.json`` file beside the model, if present.

    Accepts the model JSON Frigate+ ships with each model (its ``labelMap``), a
    ``{"names": {...}}`` object, or a plain ``{"0": "person", ...}`` map. Third-party
    exports (e.g. Frigate+ ONNX) usually carry no class names, in which case
    Ultralytics would report generic ``class0``/``class1`` labels.
    """
    sidecar = model_path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text())
        if isinstance(data, dict):
            mapping = data.get("labelMap", data.get("names", data))
        else:
            mapping = dict(enumerate(data)) if isinstance(data, list) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("no labelMap/names mapping found")
        names = {int(k): str(v) for k, v in mapping.items()}
    except Exception as exc:
        logger.warning("Ignoring label file %s: %s", sidecar.name, exc)
        return None
    return dict(sorted(names.items()))


def _apply_label_map(model: YOLO, names: Dict[int, str], source: str) -> None:
    """Override a loaded model's class names (after warm-up, when the predictor exists)."""
    auto_backend = model.predictor.model
    target = getattr(auto_backend, "backend", None) or auto_backend
    current = dict(getattr(target, "names", None) or {})
    generic = all(str(v) == f"class{k}" for k, v in current.items())
    if current and not generic and current != names:
        logger.warning(
            "Model's embedded class names differ from %s; using %s (%d labels).",
            source, source, len(names),
        )
    target.names = dict(names)
    logger.info("Using %d class labels from %s.", len(names), source)


def _onnx_execution_provider(model: YOLO) -> Optional[str]:
    """Execution provider ONNX Runtime actually selected for a loaded ONNX model, if discoverable."""
    try:
        backend = model.predictor.model  # Ultralytics AutoBackend (forwards attributes to its backend)
        session = getattr(backend, "session", None)
        if session is None:
            session = getattr(getattr(backend, "backend", None), "session", None)
        return session.get_providers()[0] if session is not None else None
    except Exception:
        return None


def load_and_warmup_with_recovery_sync(path: Path) -> Tuple[YOLO, bool, str]:
    """Load a model, rebuilding a broken preferred TensorRT engine once or falling back to .onnx/.pt."""
    try:
        return load_and_warmup_sync(path)
    except Exception as first_exc:
        if path.suffix.lower() != ".engine":
            raise

        stem = path.stem
        pt_path = MODELS_DIR / f"{stem}.pt"
        onnx_path = MODELS_DIR / f"{stem}.onnx"
        if not pt_path.exists():
            if onnx_path.exists():
                logger.warning(
                    "TensorRT engine '%s' failed to load (%s); falling back to %s.",
                    path.name,
                    first_exc,
                    onnx_path.name,
                )
                path.unlink(missing_ok=True)
                return load_and_warmup_sync(onnx_path)
            raise

        logger.warning(
            "TensorRT engine '%s' failed to load (%s); rebuilding once from %s.",
            path.name,
            first_exc,
            pt_path.name,
        )
        path.unlink(missing_ok=True)
        (MODELS_DIR / f"{stem}.trt_manifest.json").unlink(missing_ok=True)
        rebuilt = ensure_tensorrt_engine(stem, force=True)
        if rebuilt and path.exists():
            try:
                return load_and_warmup_sync(path)
            except Exception as second_exc:
                logger.error(
                    "Rebuilt TensorRT engine '%s' still failed (%s); falling back to PyTorch.",
                    path.name,
                    second_exc,
                )
        else:
            logger.error("TensorRT rebuild failed for '%s'; falling back to PyTorch.", stem)
        return load_and_warmup_sync(pt_path)


def save_faces_db(snapshot: Optional[Dict[str, List[torch.Tensor]]] = None):
    """Persist a stable CPU snapshot of enrolled face embeddings atomically."""
    data = registered_faces if snapshot is None else snapshot
    try:
        temp_path = FACES_DB_PATH.with_suffix(".tmp")
        torch.save(data, temp_path)
        os.replace(temp_path, FACES_DB_PATH)  # Atomic on POSIX/Linux
        logger.info("Faces database saved atomically (%d people enrolled).", len(data))
    except Exception as e:
        logger.error(f"Failed to persist face database: {e}")


def load_faces_db():
    """Loads enrolled face embeddings from disk on boot."""
    global registered_faces, _face_cache_dirty
    if FACES_DB_PATH.exists():
        try:
            registered_faces = torch.load(FACES_DB_PATH, map_location="cpu", weights_only=True)
            logger.info(f"Loaded {len(registered_faces)} enrolled people from {FACES_DB_PATH}")
        except Exception as e:
            logger.error(f"Error loading {FACES_DB_PATH}: {e}")
            registered_faces = {}
    _face_cache_dirty = True


def _rebuild_face_matrix():
    """Stacks every enrolled embedding into one (M, D) CPU matrix + parallel label list.

    Rebuilt lazily (only when dirty) so recognize() can do a single batched
    matmul against all enrolled faces instead of looping + torch.cat-ing
    per user, per detected face, on every request. The matrix deliberately
    stays on the CPU: it is tiny, and moving it to the GPU happens inside the
    recognition job so all CUDA work stays behind run_gpu_job.
    """
    global _face_matrix_cache, _face_labels_cache, _face_cache_dirty
    labels: List[str] = []
    mats: List[torch.Tensor] = []
    for user, embs in registered_faces.items():
        for e in embs:
            mats.append(e.detach().cpu())
            labels.append(user)

    _face_matrix_cache = torch.cat(mats, dim=0) if mats else None
    _face_labels_cache = labels
    _face_cache_dirty = False


async def _get_face_enrollment_snapshot() -> Tuple[Optional[torch.Tensor], List[str]]:
    """Return an immutable CPU snapshot of the enrolled face matrix and labels."""
    async with face_state_lock:
        if _face_cache_dirty:
            _rebuild_face_matrix()  # CPU-only and small; no CUDA work here.
        return _face_matrix_cache, list(_face_labels_cache)


def sync_face_register_embedding(img: Image.Image) -> Tuple[Optional[torch.Tensor], int]:
    """Extract exactly one normalized face embedding without repeating MTCNN detection."""
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    face_count = 0 if boxes is None else len(boxes)
    if face_count != 1:
        return None, face_count

    faces = face_detector.extract(img, boxes, None)
    if faces is None or len(faces) != 1:
        return None, face_count

    face_tensor = faces[0].unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        emb = face_recognizer(face_tensor)
        return F.normalize(emb, p=2, dim=1).cpu(), face_count


def sync_face_recognize(
    img: Image.Image,
    min_conf: float,
    face_matrix: Optional[torch.Tensor],
    face_labels: List[str],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[dict]:
    """Detect faces once, extract crops, and compare embeddings against enrollment data.

    ``face_matrix`` is the CPU snapshot from _get_face_enrollment_snapshot(); it is
    moved to the GPU here, inside the GPU worker. With nobody enrolled, the
    embedding model is skipped and every detected face is reported as unknown.
    """
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    best_sims = best_idx = None
    if face_matrix is not None and face_labels:
        faces = face_detector.extract(img, boxes, None)
        if faces is None or len(faces) == 0:
            return []

        with torch.no_grad():
            embeddings = face_recognizer(faces.to("cuda:0"))
            embeddings = F.normalize(embeddings, p=2, dim=1)
            sims = embeddings @ face_matrix.to(embeddings.device, embeddings.dtype).T
            best_sims, best_idx = sims.max(dim=1)
        best_sims = best_sims.cpu().tolist()
        best_idx = best_idx.cpu().tolist()

    predictions = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        if best_sims is not None:
            sim = float(best_sims[i])
            matched_id = face_labels[int(best_idx[i])] if sim >= min_conf else "unknown"
        else:
            sim = 0.0
            matched_id = "unknown"

        predictions.append({
            "confidence": round(sim, 2),
            "userid": matched_id,
            "label": matched_id,
            "x_min": int(max(0, x1) * scale_x),
            "y_min": int(max(0, y1) * scale_y),
            "x_max": int(x2 * scale_x),
            "y_max": int(y2 * scale_y),
        })

    return predictions


def send_crash_alert(reason: str):
    if not ALERT_WEBHOOK_URL:
        return
    try:
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=f"Blue Iris Gateway Fatal: {reason}".encode("utf-8"),
            headers={"Title": "AI Gateway GPU Deadlock", "Priority": "urgent", "Tags": "warning,gpu"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1.5)
        logger.info("External crash alert sent.")
    except Exception as e:
        logger.error(f"Failed to dispatch alert webhook: {e}")


def trigger_self_termination(reason: str):
    global is_healthy, health_failure_reason
    if not is_healthy:
        return
    is_healthy = False
    health_failure_reason = reason
    logger.critical(f"FATAL: {reason} — terminating process for container restart in 200ms.")

    if ALERT_WEBHOOK_URL:
        import threading
        threading.Thread(target=send_crash_alert, args=(reason,), daemon=True).start()

    loop = asyncio.get_running_loop()
    loop.call_later(0.2, os._exit, 1)


def on_orphaned_worker_done(fut: asyncio.Future, generation: int):
    global gpu_tainted, gpu_taint_generation, recovery_timer_handle
    if generation != gpu_taint_generation or fut.cancelled():
        return

    exc = fut.exception()
    if exc:
        logger.critical(f"Orphaned worker (gen {generation}) failed: {exc}")
        trigger_self_termination(f"CUDA exception in worker: {exc}")
    else:
        logger.warning(f"Orphaned worker (gen {generation}) recovered within grace window.")
        gpu_tainted = False
        if recovery_timer_handle and not recovery_timer_handle.cancelled():
            recovery_timer_handle.cancel()
            recovery_timer_handle = None


def on_grace_period_expired(generation: int):
    if generation == gpu_taint_generation:
        trigger_self_termination(f"CUDA operation failed to exit within {RECOVERY_GRACE_PERIOD}s grace window.")


class GpuBusyError(RuntimeError):
    pass


class GpuUnavailableError(RuntimeError):
    pass


class GpuJobTimeoutError(TimeoutError):
    pass


def _taint_gpu_for_orphan(worker_future: asyncio.Future, reason: str) -> int:
    """Block all new CUDA work until an orphaned worker exits or the process restarts."""
    global gpu_tainted, gpu_taint_generation, recovery_timer_handle
    gpu_tainted = True
    gpu_taint_generation += 1
    generation = gpu_taint_generation
    logger.error("GPU pipeline tainted (gen %d): %s", generation, reason)

    if recovery_timer_handle and not recovery_timer_handle.cancelled():
        recovery_timer_handle.cancel()
    worker_future.add_done_callback(
        lambda fut, gen=generation: on_orphaned_worker_done(fut, gen)
    )
    loop = asyncio.get_running_loop()
    recovery_timer_handle = loop.call_later(
        RECOVERY_GRACE_PERIOD,
        lambda gen=generation: on_grace_period_expired(gen),
    )
    return generation


async def run_gpu_job(
    func,
    *args,
    label: str,
    timeout: float = INFERENCE_TIMEOUT,
    enqueue: bool = True,
    acquire_timeout: Optional[float] = None,
    yield_to_queue: bool = False,
):
    """The only runtime path allowed to launch CUDA executor work.

    Timed-out or cancelled requests leave their executor thread running. Such a
    worker taints the pipeline so releasing the asyncio lock cannot permit a
    second CUDA job to collide with the orphaned worker.
    """
    global queue_depth, active_inferences

    if not is_healthy:
        raise GpuUnavailableError(f"Gateway shutting down: {health_failure_reason}")
    if gpu_tainted:
        raise GpuUnavailableError("GPU execution is in the recovery grace window.")

    queued = False
    acquired = False
    active = False
    worker_future: Optional[asyncio.Future] = None

    if enqueue:
        queue_depth += 1
        queued = True

    try:
        try:
            if acquire_timeout is None:
                await gpu_lock.acquire()
            else:
                await asyncio.wait_for(gpu_lock.acquire(), timeout=acquire_timeout)
        except asyncio.TimeoutError as exc:
            raise GpuBusyError("GPU execution gate is busy.") from exc

        acquired = True
        if queued:
            queue_depth -= 1
            queued = False

        if yield_to_queue and queue_depth > 0:
            raise GpuBusyError("Queued gateway work has priority over tracker inference.")
        if not is_healthy:
            raise GpuUnavailableError(f"Gateway shutting down: {health_failure_reason}")
        if gpu_tainted:
            raise GpuUnavailableError("GPU execution is in the recovery grace window.")

        active_inferences += 1
        active = True
        loop = asyncio.get_running_loop()
        worker_future = loop.run_in_executor(None, func, *args)
        try:
            return await asyncio.wait_for(
                asyncio.shield(worker_future), timeout=max(0.1, float(timeout))
            )
        except asyncio.TimeoutError as exc:
            msg = f"{label} timed out after {timeout:.1f}s."
            _taint_gpu_for_orphan(worker_future, msg)
            raise GpuJobTimeoutError(msg) from exc
        except asyncio.CancelledError:
            if worker_future is not None and not worker_future.done():
                _taint_gpu_for_orphan(
                    worker_future,
                    f"{label} request was cancelled while its CUDA worker was still running.",
                )
            raise
        except Exception as exc:
            if _is_fatal_cuda_exception(exc):
                reason = f"{label} hit unrecoverable CUDA error: {exc}"
                logger.critical("%s", reason, exc_info=True)
                trigger_self_termination(reason)
            raise
    finally:
        if active:
            active_inferences -= 1
        if acquired:
            gpu_lock.release()
        if queued:
            queue_depth -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_healthy, health_failure_reason, face_detector, face_recognizer, tracker_service
    load_faces_db()

    # Initialize and warm up Face Recognition models off the event loop
    logger.info("Initializing FaceNet (MTCNN + InceptionResnetV1) on CUDA...")
    def _init_facenet_sync():
        detector = MTCNN(keep_all=True, device="cuda:0", post_process=True)
        recognizer = InceptionResnetV1(pretrained="vggface2").eval().to("cuda:0")
        dummy = Image.fromarray(np.zeros((160, 160, 3), dtype=np.uint8))
        detector.detect(dummy)
        return detector, recognizer

    try:
        face_detector, face_recognizer = await asyncio.to_thread(_init_facenet_sync)
        logger.info("FaceNet pipeline initialized and warmed on GPU.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceNet: {e}")
        if _is_fatal_cuda_exception(e):
            trigger_self_termination(f"FaceNet initialization hit unrecoverable CUDA error: {e}")

    # Determine target stems to preload/compile
    target_stems = set()
    if PRELOAD_MODELS_ENV.lower() == "all":
        discovered = discover_models()
        target_stems = set(discovered.keys())
    elif PRELOAD_MODELS_ENV:
        target_stems = {Path(s.strip()).stem for s in PRELOAD_MODELS_ENV.split(",") if s.strip()}
    else:
        target_stems = {Path(DEFAULT_MODEL_NAME).stem}

    # Ensure tracker model is also in target_stems if tracker is enabled
    tracker_model_env = os.getenv("TRACKER_MODEL", "").strip()
    if tracker_model_env:
        target_stems.add(Path(tracker_model_env).stem)

    logger.info("Checking TensorRT engine compilation status...")
    for stem in target_stems:
        await asyncio.to_thread(ensure_tensorrt_engine, stem)

    # Discover and preload models
    discovered = discover_models()
    default_stem = Path(DEFAULT_MODEL_NAME).stem

    logger.info("Starting model preloading sequence...")
    for stem in target_stems:
        if stem in discovered:
            model_path = discovered[stem]
        elif (MODELS_DIR / f"{stem}.onnx").exists():
            model_path = MODELS_DIR / f"{stem}.onnx"
        else:
            model_path = MODELS_DIR / f"{stem}.pt"
        try:
            model, is_pt, fmt = await asyncio.to_thread(load_and_warmup_with_recovery_sync, model_path)
            loaded_models[stem] = ModelEntry(model=model, is_pt=is_pt, format_name=fmt)
        except Exception as e:
            failed_models[stem] = str(e)
            if stem == default_stem:
                logger.critical(f"Default model '{stem}' failed to load: {e}")
                send_crash_alert(f"Startup crash: default model '{stem}' failed to load.")
                os._exit(1)

    logger.info("==================================================")
    logger.info("            AI GATEWAY STARTUP SUMMARY            ")
    logger.info("==================================================")
    for name, m in loaded_models.items():
        fp_mode = str(HALF_PRECISION).lower() if m.is_pt else "Native"
        logger.info(f"  [+] {name:<24} Format: {m.format_name:<8} FP16: {fp_mode}")
    logger.info(f"  [+] Face Recognition Enrolled: {len(registered_faces)} person(s)")
    for name, err in failed_models.items():
        logger.warning(f"  [!] {name:<24} Error: {err}")
    logger.info("==================================================")

    tracker_cfg = TrackerConfig.from_env()
    if tracker_cfg.enabled:
        try:
            tracker_service = DogTracker(
                tracker_cfg, run_tracker_inference, logger, model_names_cb=tracker_model_names
            )
            await tracker_service.initialize()
        except Exception as e:
            tracker_service = None
            logger.error(f"Failed to initialize dog PTZ tracker: {e}", exc_info=True)

    yield

    if tracker_service is not None:
        try:
            await tracker_service.shutdown()
        except Exception as e:
            logger.error(f"Error while shutting down dog PTZ tracker: {e}", exc_info=True)
        tracker_service = None

    loaded_models.clear()
    failed_models.clear()


app = FastAPI(title="Blue Iris YOLO & Face Gateway", lifespan=lifespan)


async def get_model_entry(requested_name: str, allow_lazy: bool = True) -> Optional[ModelEntry]:
    if gpu_tainted:
        return None

    stem = Path(requested_name).stem
    if stem in loaded_models:
        return loaded_models[stem]

    if not ALLOW_LAZY_LOAD or not allow_lazy:
        return None

    async with cache_lock:
        if gpu_tainted:
            return None
        if stem in loaded_models:
            return loaded_models[stem]

        discovered = discover_models()
        if stem not in discovered:
            return None

        target_path = discovered[stem]
        logger.info("Lazy loading model: %s", target_path.name)
        try:
            model, is_pt, fmt = await run_gpu_job(
                load_and_warmup_with_recovery_sync,
                target_path,
                label=f"Lazy model load '{stem}'",
                timeout=MODEL_LOAD_TIMEOUT,
                enqueue=True,
            )
            entry = ModelEntry(model=model, is_pt=is_pt, format_name=fmt)
            loaded_models[stem] = entry
            failed_models.pop(stem, None)
            return entry
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Failed to lazy load %s: %s", target_path.name, exc)
            failed_models[stem] = str(exc)
            return None


def tracker_model_names(model_name: str) -> Optional[Mapping[int, str]]:
    """Label map ({class_id: label}) of a loaded model, for tracker class resolution."""
    entry = loaded_models.get(Path(model_name).stem)
    if entry is None:
        return None
    names = getattr(entry.model, "names", None)
    return dict(names) if names else None


async def run_tracker_inference(frame: np.ndarray, min_conf: float, model_name: str, target_class_ids: List[int]):
    """Opportunistic tracker inference that always yields to gateway work."""
    if not is_healthy or gpu_tainted:
        return "unavailable", [], 0
    if queue_depth > 0 or gpu_lock.locked():
        return "busy", [], 0

    entry = await get_model_entry(model_name, allow_lazy=False)
    if entry is None:
        return "model_unavailable", [], 0

    t_infer_start = time.perf_counter()
    try:
        predictions = await run_gpu_job(
            sync_tracker_predict,
            entry,
            frame,
            min_conf,
            target_class_ids,
            label=f"Tracker inference on '{model_name}'",
            timeout=INFERENCE_TIMEOUT,
            enqueue=False,
            acquire_timeout=0.002,
            yield_to_queue=True,
        )
        return "ok", predictions, int((time.perf_counter() - t_infer_start) * 1000)
    except GpuBusyError:
        return "busy", [], 0
    except GpuUnavailableError:
        return "unavailable", [], 0
    except GpuJobTimeoutError as exc:
        logger.error("%s", exc)
        return "timeout", [], int((time.perf_counter() - t_infer_start) * 1000)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Tracker inference error on '%s': %s", model_name, exc, exc_info=True)
        return "error", [], int((time.perf_counter() - t_infer_start) * 1000)


def _failure(error: str, t_start: float, t_infer_start: Optional[float] = None) -> dict:
    """CodeProject.AI-style failure payload with timing fields."""
    now = time.perf_counter()
    return {
        "success": False,
        "error": error,
        "inferenceMs": int((now - t_infer_start) * 1000) if t_infer_start is not None else 0,
        "processMs": int((now - t_start) * 1000),
    }


async def run_inference(image_bytes: bytes, min_conf: float, model_name: str, t_start: float):
    if not is_healthy:
        return _failure(f"Gateway shutting down: {health_failure_reason}", t_start)
    if gpu_tainted:
        return _failure("GPU execution frozen during recovery grace window.", t_start)

    try:
        entry = await get_model_entry(model_name)
        if entry is None:
            return _failure(f"Model '{model_name}' is not loaded or available on server.", t_start)
        img, scale_x, scale_y = await asyncio.to_thread(decode_image, image_bytes, DETECTION_DECODE_MAX_SIDE)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return _failure(f"Bad request/corrupt image: {exc}", t_start)

    t_infer_start = time.perf_counter()
    try:
        predictions = await run_gpu_job(
            sync_predict,
            entry.model,
            img,
            min_conf,
            entry.is_pt,
            scale_x,
            scale_y,
            label=f"Inference on '{model_name}'",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
    except GpuJobTimeoutError as exc:
        return _failure(str(exc), t_start, t_infer_start)
    except GpuUnavailableError as exc:
        return _failure(str(exc), t_start)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Inference error on '%s': %s", model_name, exc, exc_info=True)
        return _failure(str(exc), t_start)

    t_infer_end = time.perf_counter()
    return {
        "success": True,
        "message": f"{len(predictions)} object(s) detected",
        "predictions": predictions,
        "count": len(predictions),
        "command": "detect",
        "moduleName": f"YOLO Gateway ({entry.format_name})",
        "executionProvider": "CUDA",
        "canUseGPU": True,
        "inferenceMs": int((t_infer_end - t_infer_start) * 1000),
        "processMs": int((time.perf_counter() - t_start) * 1000),
    }


# ============================================================================
# FACE RECOGNITION ENDPOINTS
# ============================================================================

def _face_db_snapshot() -> Dict[str, List[torch.Tensor]]:
    return {
        user: [embedding.detach().cpu().clone() for embedding in embeddings]
        for user, embeddings in registered_faces.items()
    }


@app.api_route("/v1/vision/face/list", methods=["GET", "POST"])
async def face_list():
    async with face_state_lock:
        faces = sorted(registered_faces.keys())
    return {"success": True, "faces": faces}


@app.post("/v1/vision/face/register")
async def face_register(
    image: UploadFile = File(...),
    userid: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
):
    global _face_cache_dirty
    target_id = (userid or name or "").strip()
    if not target_id:
        return {"success": False, "error": "Missing user ID or name."}
    if face_detector is None or face_recognizer is None:
        return {"success": False, "error": "Face recognition is unavailable."}

    contents = await image.read()
    try:
        img, _, _ = await asyncio.to_thread(decode_image, contents)
    except Exception as exc:
        return {"success": False, "error": f"Invalid image: {exc}"}

    try:
        embedding, face_count = await run_gpu_job(
            sync_face_register_embedding,
            img,
            label=f"Face registration for '{target_id}'",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
    except (GpuJobTimeoutError, GpuUnavailableError) as exc:
        return {"success": False, "error": str(exc)}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Face registration failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}

    if embedding is None:
        if face_count == 0:
            return {"success": False, "error": "No face detected in provided image."}
        return {"success": False, "error": f"Expected exactly one face for enrollment; detected {face_count}."}

    async with face_state_lock:
        embeddings = registered_faces.setdefault(target_id, [])
        embeddings.append(embedding)
        if len(embeddings) > MAX_FACE_EMBEDDINGS_PER_USER:
            del embeddings[:-MAX_FACE_EMBEDDINGS_PER_USER]
        _face_cache_dirty = True
        snapshot = _face_db_snapshot()
    await asyncio.to_thread(save_faces_db, snapshot)
    return {"success": True, "message": f"Face registered for {target_id}"}


@app.post("/v1/vision/face/delete")
async def face_delete(
    userid: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
):
    global _face_cache_dirty
    target_id = (userid or name or "").strip()
    async with face_state_lock:
        if target_id not in registered_faces:
            return {"success": False, "error": f"User '{target_id}' not found."}
        del registered_faces[target_id]
        _face_cache_dirty = True
        snapshot = _face_db_snapshot()
    await asyncio.to_thread(save_faces_db, snapshot)
    return {"success": True, "message": f"Face deleted for {target_id}"}


@app.post("/v1/vision/face/recognize")
async def face_recognize(
    image: UploadFile = File(...),
    min_confidence: float = Form(0.60),
):
    t_start = time.perf_counter()
    if face_detector is None or face_recognizer is None:
        return _failure("Face recognition is unavailable.", t_start)

    contents = await image.read()
    try:
        img, scale_x, scale_y = await asyncio.to_thread(decode_image, contents, FACE_DECODE_MAX_SIDE)
    except Exception as exc:
        return _failure(f"Invalid image: {exc}", t_start)

    t_infer_start = time.perf_counter()
    try:
        face_matrix, face_labels = await _get_face_enrollment_snapshot()
        predictions = await run_gpu_job(
            sync_face_recognize,
            img,
            min_confidence,
            face_matrix,
            face_labels,
            scale_x,
            scale_y,
            label="Face recognition",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
    except (GpuJobTimeoutError, GpuUnavailableError) as exc:
        return _failure(str(exc), t_start, t_infer_start)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Face recognition failed: %s", exc, exc_info=True)
        return _failure(str(exc), t_start)

    t_infer_end = time.perf_counter()
    return {
        "success": True,
        "message": f"{len(predictions)} face(s) detected",
        "predictions": predictions,
        "count": len(predictions),
        "command": "recognize",
        "moduleName": "Face Recognition (FaceNet)",
        "executionProvider": "CUDA",
        "canUseGPU": True,
        "inferenceMs": int((t_infer_end - t_infer_start) * 1000),
        "processMs": int((time.perf_counter() - t_start) * 1000),
    }


# ============================================================================
# DIAGNOSTIC & DETECTION ENDPOINTS
# ============================================================================

def _runtime_health_snapshot() -> dict:
    tracker = tracker_service.status() if tracker_service is not None else {"enabled": False}
    reasons = []

    if not is_healthy:
        reasons.append(health_failure_reason or "gateway marked unhealthy")
    if gpu_tainted:
        reasons.append("GPU execution pipeline is tainted")

    tracker_state = str(tracker.get("state") or "")
    tracker_outcome = str(tracker.get("last_inference_outcome") or "")
    tracker_active = bool(tracker.get("active"))
    tracker_task_running = bool(tracker.get("tracker_task_running"))

    if tracker_state == "TRACKER_ERROR":
        reasons.append(
            str(tracker.get("tracker_task_error") or tracker.get("stop_reason") or "tracker task crashed")
        )
    elif tracker_active and not tracker_task_running:
        reasons.append("tracker is active but its task is not running")
    elif (
        tracker_active
        and tracker_state == "INFERENCE_ERROR"
        and tracker_outcome in {"error", "timeout", "unavailable"}
    ):
        reasons.append(f"tracker inference is unhealthy ({tracker_outcome})")

    return {
        "healthy": not reasons,
        "reasons": reasons,
        "gateway_revision": GATEWAY_REVISION,
        "gpu_tainted": gpu_tainted,
        "tracker": tracker,
    }


@app.get("/healthz")
async def healthz(response: Response):
    snapshot = _runtime_health_snapshot()
    if not snapshot["healthy"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if snapshot["healthy"] else "degraded",
        "gateway_revision": snapshot["gateway_revision"],
        "gpu_tainted": snapshot["gpu_tainted"],
        "reasons": snapshot["reasons"],
        "tracker": {
            "enabled": snapshot["tracker"].get("enabled"),
            "active": snapshot["tracker"].get("active"),
            "state": snapshot["tracker"].get("state"),
            "tracker_task_running": snapshot["tracker"].get("tracker_task_running"),
            "last_inference_outcome": snapshot["tracker"].get("last_inference_outcome"),
        },
    }


@app.get("/")
async def root_health(response: Response):
    if not is_healthy or gpu_tainted:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "gpu_tainted": gpu_tainted, "reason": health_failure_reason}
    return {
        "status": "ok",
        "gateway_revision": GATEWAY_REVISION,
        "loaded_models": list(loaded_models.keys()),
        "registered_faces": sorted(list(registered_faces.keys())),
        "tracker": tracker_service.status() if tracker_service is not None else {"enabled": False},
    }


@app.api_route("/status", methods=["GET"])
@app.api_route("/v1/status", methods=["GET"])
async def detailed_status(response: Response):
    if not is_healthy or gpu_tainted:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "success": is_healthy and not gpu_tainted,
        "status": "ok" if (is_healthy and not gpu_tainted) else "degraded",
        "gateway_revision": GATEWAY_REVISION,
        "health_failure_reason": health_failure_reason,
        "gpu_tainted": gpu_tainted,
        "canUseGPU": True,
        "executionProvider": "CUDA",
        "queue_depth": queue_depth,
        "active_inferences": active_inferences,
        "models": {k: {"status": "ready", "format": v.format_name} for k, v in loaded_models.items()},
        "registered_faces": sorted(list(registered_faces.keys())),
        "tracker": tracker_service.status() if tracker_service is not None else {"enabled": False},
    }


@app.api_route("/v1/vision/custom/list", methods=["GET", "POST"])
async def list_custom_models():
    return {"success": True, "models": sorted(discover_models().keys())}


@app.post("/v1/vision/detection")
async def detection(
    image: UploadFile = File(...),
    min_confidence: float = Form(0.4),
    model: Optional[str] = Form(None),
):
    t_start = time.perf_counter()
    image_bytes = await image.read()
    return await run_inference(image_bytes, min_confidence, model or DEFAULT_MODEL_NAME, t_start=t_start)


@app.post("/v1/vision/custom/{model_name}")
async def custom_detection(
    model_name: str,
    image: UploadFile = File(...),
    min_confidence: float = Form(0.4),
):
    t_start = time.perf_counter()
    image_bytes = await image.read()
    return await run_inference(image_bytes, min_confidence, model_name, t_start=t_start)


# ============================================================================
# PTZ TRACKER ENDPOINTS
# ============================================================================

def _require_tracker() -> DogTracker:
    if tracker_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PTZ tracker is not enabled/initialized. Set TRACKER_ENABLED=true and check startup logs.",
        )
    return tracker_service


@app.get("/v1/tracker/status")
async def tracker_status():
    return _require_tracker().status()


@app.post("/v1/tracker/start")
async def tracker_start():
    return await _require_tracker().start()


@app.post("/v1/tracker/stop")
async def tracker_stop():
    return await _require_tracker().stop()


@app.post("/v1/tracker/home")
async def tracker_home():
    return await _require_tracker().home()


@app.get("/v1/tracker/camera-status")
async def tracker_camera_status():
    return await _require_tracker().camera_status()


@app.post("/v1/tracker/calibrate")
async def tracker_calibrate(mode: str = "zoom"):
    return await _require_tracker().calibrate(mode=mode)


@app.get("/v1/tracker/history")
async def tracker_history(limit: int = 200):
    return _require_tracker().history(limit=limit)


@app.post("/v1/tracker/history/clear")
async def tracker_history_clear():
    return _require_tracker().clear_history()


@app.get("/v1/tracker/debug.jpg")
async def tracker_debug_frame():
    jpeg = await _require_tracker().debug_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No tracker debug frame is available yet.")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
