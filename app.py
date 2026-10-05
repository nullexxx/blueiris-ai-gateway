import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
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

is_healthy: bool = True
health_failure_reason: Optional[str] = None

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


def _tensorrt_manifest(pt_path: Path) -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "source_sha256": _sha256_file(pt_path),
        "source_size": pt_path.stat().st_size,
        "tensorrt": str(tensorrt.__version__),
        "cuda": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "imgsz": 640,
        "precision": "fp16" if HALF_PRECISION else "fp32",
    }


def ensure_tensorrt_engine(stem: str, force: bool = False) -> bool:
    """Build a TensorRT engine when its source/runtime manifest is stale."""
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
        model = YOLO(str(pt_path))
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


def sync_predict(model: YOLO, img: Image.Image, min_conf: float, is_pt: bool):
    """Synchronous inference worker executed inside threadpool."""
    kwargs = {
        "conf": min_conf,
        "device": 0,
        "verbose": False,
    }
    if is_pt and HALF_PRECISION:
        kwargs["quantize"] = 16

    return model.predict(img, **kwargs)


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
    predictions: List[dict] = []
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            predictions.append({
                "class_id": class_id,
                "label": str(entry.model.names[class_id]),
                "confidence": float(box.conf[0]),
                "x_min": int(coords[0]),
                "y_min": int(coords[1]),
                "x_max": int(coords[2]),
                "y_max": int(coords[3]),
            })
    return predictions


def load_and_warmup_sync(path: Path) -> Tuple[YOLO, bool, str]:
    """Loads weights and forces CUDA/TensorRT buffer allocation."""
    model = YOLO(str(path))
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_frame, device=0, verbose=False)

    ext = path.suffix.lower()
    is_pt = (ext == ".pt")
    format_name = "TensorRT" if ext == ".engine" else ("ONNX" if ext == ".onnx" else "PyTorch")
    return model, is_pt, format_name


def load_and_warmup_with_recovery_sync(path: Path) -> Tuple[YOLO, bool, str]:
    """Load a model, rebuilding a broken preferred TensorRT engine once."""
    try:
        return load_and_warmup_sync(path)
    except Exception as first_exc:
        if path.suffix.lower() != ".engine":
            raise

        stem = path.stem
        pt_path = MODELS_DIR / f"{stem}.pt"
        if not pt_path.exists():
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
    """Stacks every enrolled embedding into one (M, D) matrix + parallel label list.

    Rebuilt lazily (only when dirty) so recognize() can do a single batched
    matmul against all enrolled faces instead of looping + torch.cat-ing
    per user, per detected face, on every request.
    """
    global _face_matrix_cache, _face_labels_cache, _face_cache_dirty
    labels: List[str] = []
    mats: List[torch.Tensor] = []
    for user, embs in registered_faces.items():
        for e in embs:
            mats.append(e)
            labels.append(user)

    _face_matrix_cache = torch.cat(mats, dim=0).to("cuda:0") if mats else None
    _face_labels_cache = labels
    _face_cache_dirty = False


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


def sync_face_recognize(img: Image.Image, min_conf: float) -> List[dict]:
    """Detect faces once, extract crops, and compare embeddings against enrollment data."""
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    faces = face_detector.extract(img, boxes, None)
    if faces is None or len(faces) == 0:
        return []

    faces = faces.to("cuda:0")
    with torch.no_grad():
        embeddings = face_recognizer(faces)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    if _face_cache_dirty:
        _rebuild_face_matrix()

    if _face_matrix_cache is not None:
        sims = embeddings @ _face_matrix_cache.T
        best_sims, best_idx = sims.max(dim=1)
    else:
        best_sims = best_idx = None

    predictions = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        if best_sims is not None:
            sim = float(best_sims[i].item())
            matched_id = _face_labels_cache[int(best_idx[i].item())] if sim >= min_conf else "unknown"
        else:
            sim = 0.0
            matched_id = "unknown"

        predictions.append({
            "confidence": round(sim, 2),
            "userid": matched_id,
            "label": matched_id,
            "x_min": int(max(0, x1)),
            "y_min": int(max(0, y1)),
            "x_max": int(x2),
            "y_max": int(y2),
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

    # Initialize and warm up Face Recognition models
    logger.info("Initializing FaceNet (MTCNN + InceptionResnetV1) on CUDA...")
    try:
        face_detector = MTCNN(keep_all=True, device="cuda:0", post_process=True)
        face_recognizer = InceptionResnetV1(pretrained="vggface2").eval().to("cuda:0")
        dummy = Image.fromarray(np.zeros((160, 160, 3), dtype=np.uint8))
        face_detector.detect(dummy)
        logger.info("FaceNet pipeline initialized and warmed on GPU.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceNet: {e}")

    # Determine target stems to preload/compile
    target_stems = set()
    if PRELOAD_MODELS_ENV.lower() == "all":
        discovered = discover_models()
        target_stems = set(discovered.keys())
    elif PRELOAD_MODELS_ENV:
        target_stems = {s.strip() for s in PRELOAD_MODELS_ENV.split(",") if s.strip()}
    else:
        target_stems = {Path(DEFAULT_MODEL_NAME).stem}

    logger.info("Checking TensorRT engine compilation status...")
    for stem in target_stems:
        await asyncio.to_thread(ensure_tensorrt_engine, stem)

    # Discover and preload models
    discovered = discover_models()
    default_stem = Path(DEFAULT_MODEL_NAME).stem

    logger.info("Starting model preloading sequence...")
    for stem in target_stems:
        model_path = discovered[stem] if stem in discovered else MODELS_DIR / f"{stem}.pt"
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
        logger.info(f"  [+] {name:<16} Format: {m.format_name:<8} FP16: {fp_mode}")
    logger.info(f"  [+] Face Recognition Enrolled: {len(registered_faces)} person(s)")
    for name, err in failed_models.items():
        logger.warning(f"  [!] {name:<16} Error: {err}")
    logger.info("==================================================")

    tracker_cfg = TrackerConfig.from_env()
    if tracker_cfg.enabled:
        try:
            tracker_service = DogTracker(tracker_cfg, run_tracker_inference, logger)
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


async def run_inference(image_bytes: bytes, min_conf: float, model_name: str, t_start: float):
    if not is_healthy:
        return {"success": False, "error": f"Gateway shutting down: {health_failure_reason}", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}
    if gpu_tainted:
        return {"success": False, "error": "GPU execution frozen during recovery grace window.", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    try:
        entry = await get_model_entry(model_name)
        if entry is None:
            return {"success": False, "error": f"Model '{model_name}' is not loaded or available on server.", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}
        img = await asyncio.to_thread(lambda: Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {"success": False, "error": f"Bad request/corrupt image: {exc}", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    t_infer_start = time.perf_counter()
    try:
        results = await run_gpu_job(
            sync_predict,
            entry.model,
            img,
            min_conf,
            entry.is_pt,
            label=f"Inference on '{model_name}'",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
    except GpuJobTimeoutError as exc:
        return {"success": False, "error": str(exc), "inferenceMs": int((time.perf_counter() - t_infer_start) * 1000), "processMs": int((time.perf_counter() - t_start) * 1000)}
    except GpuUnavailableError as exc:
        return {"success": False, "error": str(exc), "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Inference error on '%s': %s", model_name, exc, exc_info=True)
        return {"success": False, "error": str(exc), "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    t_infer_end = time.perf_counter()
    predictions = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            predictions.append({
                "confidence": round(float(box.conf[0]), 2),
                "label": entry.model.names[int(box.cls[0])],
                "x_min": int(coords[0]), "y_min": int(coords[1]),
                "x_max": int(coords[2]), "y_max": int(coords[3]),
            })

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
        img = await asyncio.to_thread(lambda: Image.open(io.BytesIO(contents)).convert("RGB"))
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
        return {"success": False, "error": "Face recognition is unavailable.", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    contents = await image.read()
    try:
        img = await asyncio.to_thread(lambda: Image.open(io.BytesIO(contents)).convert("RGB"))
    except Exception as exc:
        return {"success": False, "error": f"Invalid image: {exc}", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    t_infer_start = time.perf_counter()
    try:
        # Freeze face mutations while the worker may rebuild the flattened GPU matrix.
        async with face_state_lock:
            predictions = await run_gpu_job(
                sync_face_recognize,
                img,
                min_confidence,
                label="Face recognition",
                timeout=INFERENCE_TIMEOUT,
                enqueue=True,
            )
    except (GpuJobTimeoutError, GpuUnavailableError) as exc:
        return {"success": False, "error": str(exc), "inferenceMs": int((time.perf_counter() - t_infer_start) * 1000), "processMs": int((time.perf_counter() - t_start) * 1000)}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Face recognition failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc), "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

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

@app.get("/")
async def root_health(response: Response):
    if not is_healthy or gpu_tainted:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "gpu_tainted": gpu_tainted, "reason": health_failure_reason}
    return {
        "status": "ok",
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
