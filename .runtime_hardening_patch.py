from pathlib import Path
import re
import textwrap


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing expected block: {label}")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"missing start marker for {label}: {start!r}")
    j = text.find(end, i)
    if j < 0:
        raise SystemExit(f"missing end marker for {label}: {end!r}")
    return text[:i] + replacement + text[j:]


# ---------------------------------------------------------------------------
# app.py
# ---------------------------------------------------------------------------
p = Path("app.py")
app = p.read_text()

app = replace_exact(
    app,
    "from dataclasses import dataclass\nimport io\nimport logging\n",
    "from dataclasses import dataclass\nimport hashlib\nimport io\nimport json\nimport logging\n",
    "app imports",
)
app = replace_exact(
    app,
    'INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "8.0"))\nRECOVERY_GRACE_PERIOD = float(os.getenv("RECOVERY_GRACE_PERIOD", "4.0"))\n',
    'INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "8.0"))\nMODEL_LOAD_TIMEOUT = float(os.getenv("MODEL_LOAD_TIMEOUT", "180.0"))\nRECOVERY_GRACE_PERIOD = float(os.getenv("RECOVERY_GRACE_PERIOD", "4.0"))\n',
    "model load timeout config",
)
app = replace_exact(
    app,
    'registered_faces: Dict[str, List[torch.Tensor]] = {}\n\n# Flattened',
    'registered_faces: Dict[str, List[torch.Tensor]] = {}\nface_state_lock = asyncio.Lock()\n\n# Flattened',
    "face state lock",
)

trt_block = '''def _sha256_file(path: Path) -> str:
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
        temp_manifest.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\\n")
        os.replace(temp_manifest, manifest_path)
        legacy_stamp_path.unlink(missing_ok=True)
        logger.info("Successfully compiled and cached: %s", engine_path.name)
        return True
    except Exception as exc:
        logger.error("Failed to auto-compile engine for '%s': %s", stem, exc, exc_info=True)
        engine_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        return False


'''
app = replace_between(
    app,
    "def ensure_tensorrt_engine(stem: str):\n",
    "def sync_predict(",
    trt_block,
    "TensorRT manifest builder",
)

recovery_loader = '''

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
'''
app = replace_exact(
    app,
    '    return model, is_pt, format_name\n\n\ndef save_faces_db():',
    '    return model, is_pt, format_name\n' + recovery_loader + '\n\ndef save_faces_db(snapshot: Optional[Dict[str, List[torch.Tensor]]] = None):',
    "engine load recovery",
)
app = replace_exact(
    app,
    '''def save_faces_db(snapshot: Optional[Dict[str, List[torch.Tensor]]] = None):
    """Persists registered face embeddings atomically to prevent file corruption."""
    try:
        temp_path = FACES_DB_PATH.with_suffix(".tmp")
        torch.save(registered_faces, temp_path)
        os.replace(temp_path, FACES_DB_PATH)  # Atomic on POSIX/Linux
        logger.info(f"Faces database saved atomically ({len(registered_faces)} people enrolled).")
    except Exception as e:
        logger.error(f"Failed to persist face database: {e}")
''',
    '''def save_faces_db(snapshot: Optional[Dict[str, List[torch.Tensor]]] = None):
    """Persist a stable CPU snapshot of enrolled face embeddings atomically."""
    data = registered_faces if snapshot is None else snapshot
    try:
        temp_path = FACES_DB_PATH.with_suffix(".tmp")
        torch.save(data, temp_path)
        os.replace(temp_path, FACES_DB_PATH)  # Atomic on POSIX/Linux
        logger.info("Faces database saved atomically (%d people enrolled).", len(data))
    except Exception as e:
        logger.error(f"Failed to persist face database: {e}")
''',
    "face db snapshot persistence",
)
app = replace_exact(
    app,
    'registered_faces = torch.load(FACES_DB_PATH, map_location="cpu", weights_only=False)',
    'registered_faces = torch.load(FACES_DB_PATH, map_location="cpu", weights_only=True)',
    "safe face db load",
)

face_register_worker = '''def sync_face_register_embedding(img: Image.Image) -> Optional[torch.Tensor]:
    """Extract and return one normalized face embedding without mutating shared state."""
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return None

    faces = face_detector(img)
    if faces is None or len(faces) == 0:
        return None

    face_tensor = faces[0].unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        emb = face_recognizer(face_tensor)
        return F.normalize(emb, p=2, dim=1).cpu()


'''
app = replace_between(
    app,
    "def sync_face_register(",
    "def sync_face_recognize(",
    face_register_worker,
    "face registration worker",
)

gpu_gate = '''

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
'''
app = replace_exact(
    app,
    'def on_grace_period_expired(generation: int):\n    if generation == gpu_taint_generation:\n        trigger_self_termination(f"CUDA operation failed to exit within {RECOVERY_GRACE_PERIOD}s grace window.")\n\n\n@asynccontextmanager',
    'def on_grace_period_expired(generation: int):\n    if generation == gpu_taint_generation:\n        trigger_self_termination(f"CUDA operation failed to exit within {RECOVERY_GRACE_PERIOD}s grace window.")\n' + gpu_gate + '\n\n@asynccontextmanager',
    "unified GPU execution gate",
)
app = replace_exact(
    app,
    'model, is_pt, fmt = await asyncio.to_thread(load_and_warmup_sync, model_path)',
    'model, is_pt, fmt = await asyncio.to_thread(load_and_warmup_with_recovery_sync, model_path)',
    "startup model recovery loader",
)

runtime_block = '''async def get_model_entry(requested_name: str, allow_lazy: bool = True) -> Optional[ModelEntry]:
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


'''
app = replace_between(
    app,
    "async def get_model_entry(",
    "# ============================================================================\n# FACE RECOGNITION ENDPOINTS",
    runtime_block,
    "runtime GPU paths",
)

face_endpoints = '''# ============================================================================
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
        embedding = await run_gpu_job(
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
        return {"success": False, "error": "No face detected in provided image."}

    async with face_state_lock:
        registered_faces.setdefault(target_id, []).append(embedding)
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


'''
app = replace_between(
    app,
    "# ============================================================================\n# FACE RECOGNITION ENDPOINTS",
    "# ============================================================================\n# DIAGNOSTIC & DETECTION ENDPOINTS",
    face_endpoints,
    "face endpoints",
)
p.write_text(app)


# ---------------------------------------------------------------------------
# tracker.py
# ---------------------------------------------------------------------------
p = Path("tracker.py")
tracker = p.read_text()
tracker = replace_exact(
    tracker,
    '        self._session_started_wall: Optional[str] = None\n        self._session_started_mono: Optional[float] = None\n',
    '        self._session_started_wall: Optional[str] = None\n        self._session_started_mono: Optional[float] = None\n        self._stop_reason: Optional[str] = None\n        self._last_task_error: Optional[str] = None\n',
    "tracker task/error fields",
)
tracker = replace_exact(
    tracker,
    '            "active": self.active,\n            "session_started": self._session_started_wall,\n',
    '            "active": self.active,\n            "session_started": self._session_started_wall,\n            "stop_reason": self._stop_reason,\n            "tracker_task_running": bool(self._task is not None and not self._task.done()),\n            "tracker_task_error": self._last_task_error,\n',
    "tracker status task health",
)

new_start = '''    async def start(self) -> dict:
        self._history.clear()
        self._session_started_wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._session_started_mono = time.monotonic()
        self._stop_reason = None
        self._last_task_error = None
        self._shutdown = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="direct-3d-ptz-tracker")
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
        self.logger.info("PTZ tracker STARTED (hybrid moveDirectly + escape chase)")
        return self.status()

'''
tracker = replace_between(
    tracker,
    "    async def start(self) -> dict:\n",
    "    async def stop(self) -> dict:\n",
    new_start,
    "restartable tracker start",
)
tracker = replace_exact(
    tracker,
    '    async def stop(self) -> dict:\n        self._record_event("tracker_stopping")\n        self.active = False\n',
    '    async def stop(self) -> dict:\n        self._record_event("tracker_stopping")\n        self._stop_reason = None\n        self.active = False\n',
    "manual stop clears error reason",
)

ptz_recovery = '''    async def _halt_tracking_for_ptz_failure(
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

    async def _poll_ptz_operation(self, seq: int, now: float) -> None:
        kind = self._ptz_operation
        if kind is None:
            return

        if now >= self._ptz_operation_deadline:
            # Final status validation: never release a timed-out operation blindly.
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
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

'''
tracker = replace_between(
    tracker,
    "    async def _poll_ptz_operation(self, seq: int, now: float) -> None:\n",
    "    def _ptz_action_ready(self, seq: int) -> bool:\n",
    ptz_recovery,
    "PTZ timeout fail-closed recovery",
)

crash_pattern = re.compile(
    r'(?ms)^(\s*)except asyncio\.CancelledError:\n'
    r'\1    raise\n'
    r'\1except Exception as exc:\n'
    r'\1    self\.state = "TRACKER_ERROR"\n'
    r'\1    try:\n'
    r'\1        await self\._stop_hybrid_chase\("tracker_error", force=True\)\n'
    r'\1    except Exception:\n'
    r'\1        pass\n'
    r'\1    self\.logger\.exception\("PTZ tracker loop crashed: %s", exc\)\n'
)
crash_match = crash_pattern.search(tracker)
if not crash_match:
    raise SystemExit("missing expected tracker crash handler")
indent = crash_match.group(1)
crash_replacement = (
    f'{indent}except asyncio.CancelledError:\n'
    f'{indent}    raise\n'
    f'{indent}except Exception as exc:\n'
    f'{indent}    self.active = False\n'
    f'{indent}    self.state = "TRACKER_ERROR"\n'
    f'{indent}    self._last_task_error = str(exc)\n'
    f'{indent}    self._stop_reason = f"Tracker task crashed: {{exc}}"\n'
    f'{indent}    try:\n'
    f'{indent}        await self._stop_hybrid_chase("tracker_error", force=True)\n'
    f'{indent}    except Exception:\n'
    f'{indent}        pass\n'
    f'{indent}    self.logger.exception("PTZ tracker loop crashed: %s", exc)\n'
)
tracker = crash_pattern.sub(lambda _m: crash_replacement, tracker, count=1)
p.write_text(tracker)


# ---------------------------------------------------------------------------
# docker-compose.example.yml
# ---------------------------------------------------------------------------
p = Path("docker-compose.example.yml")
compose = p.read_text()
compose = replace_exact(
    compose,
    '      INFERENCE_TIMEOUT: "8.0"\n      RECOVERY_GRACE_PERIOD: "4.0"\n',
    '      INFERENCE_TIMEOUT: "8.0"\n      MODEL_LOAD_TIMEOUT: "180.0"\n      RECOVERY_GRACE_PERIOD: "4.0"\n',
    "compose model load timeout",
)
current_tracker_knobs = '''      # Adaptive positional lead and one-shot edge rescue.
      TRACKER_ADAPTIVE_LEAD: "true"
      TRACKER_MOVE_ETA_MIN_SAMPLES: "3"
      TRACKER_MOVE_ETA_HISTORY: "24"
      TRACKER_MOVE_ETA_MIN: "0.50"
      TRACKER_MOVE_ETA_MAX: "2.00"
      TRACKER_VELOCITY_CONSISTENCY_COSINE: "0.25"
      TRACKER_VELOCITY_JUMP_RATIO: "4.0"
      TRACKER_EDGE_RESCUE_ENABLED: "true"
      TRACKER_EDGE_RESCUE_ERROR: "0.75"
      TRACKER_EDGE_RESCUE_GAIN: "0.85"

      # Escape-only continuous chase. Normal tracking remains moveDirectly.
      TRACKER_HYBRID_CHASE_ENABLED: "true"
      TRACKER_HYBRID_CHASE_PAN_SIGN: "-1"
      TRACKER_HYBRID_CHASE_ENTRY_ERROR: "0.82"
      TRACKER_HYBRID_CHASE_EXIT_ERROR: "0.50"
      TRACKER_HYBRID_CHASE_MOTION_ERROR: "0.55"
      TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM: "0.03"
      TRACKER_HYBRID_CHASE_MISS_GRACE: "0.15"
      TRACKER_HYBRID_CHASE_MIN_SPEED: "1"
      TRACKER_HYBRID_CHASE_MAX_SPEED: "6"
      TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR: "0.90"
      TRACKER_HYBRID_CHASE_COMMAND_INTERVAL: "0.18"
      TRACKER_HYBRID_CHASE_KEEPALIVE: "0.45"
      TRACKER_HYBRID_CHASE_CAMERA_TIMEOUT: "1"
      TRACKER_HYBRID_CHASE_MAX_SECONDS: "2.50"
      TRACKER_HYBRID_CHASE_COOLDOWN: "0.35"
      TRACKER_HYBRID_CHASE_SETTLE_FRAMES: "2"

'''
compose = replace_between(
    compose,
    "      # Bounded destination updates while a native moveDirectly is still slewing.\n",
    "      # Camera status drives completion; no fixed movement sleep is used.\n",
    current_tracker_knobs,
    "compose tracker knobs",
)
p.write_text(compose)


# ---------------------------------------------------------------------------
# README + tracker doc
# ---------------------------------------------------------------------------
p = Path("README.md")
readme = p.read_text()
readme = readme.replace(
    "- **Bounded in-flight retargeting** so a slow native `moveDirectly` slew can have its destination replaced when the target reverses or is escaping toward a frame edge.\n- **Escape-zone gain** for stronger corrections near frame edges while retaining the normal conservative gain near center.\n",
    "- **Adaptive `moveDirectly` lead** learned from completed camera moves, with bounded one-shot edge rescue.\n- **Hybrid escape chase** that uses continuous PTZ only when a fast target is in genuine danger of leaving the frame.\n",
)
readme = readme.replace(
    "If a `.pt` model exists without a matching TensorRT engine, the gateway can build the `.engine` automatically for the current TensorRT/CUDA environment.\n",
    "If a `.pt` model exists without a matching TensorRT engine, the gateway can build the `.engine` automatically. A per-model manifest records the source-weight SHA-256, TensorRT/CUDA versions, GPU identity/compute capability, image size, and precision. A source/runtime change invalidates the cached engine, and an engine load failure triggers one rebuild attempt before falling back to the `.pt` model.\n",
)
readme = readme.replace(
    "| `INFERENCE_TIMEOUT` | `8.0` | GPU inference timeout in seconds. |\n| `RECOVERY_GRACE_PERIOD` | `4.0` | Grace period for an orphaned CUDA worker before the container self-terminates. |",
    "| `INFERENCE_TIMEOUT` | `8.0` | Runtime CUDA job timeout in seconds. Applies to YOLO and FaceNet. |\n| `MODEL_LOAD_TIMEOUT` | `180.0` | Timeout for an optional lazy model load/warmup. |\n| `RECOVERY_GRACE_PERIOD` | `4.0` | Grace period for an orphaned CUDA worker before the container self-terminates. |",
)

design = '''## PTZ Tracker Design

The tracker continuously drains the camera RTSP substream and retains only the newest decoded frame. Tracker inference is opportunistic: regular Blue Iris and FaceNet work has priority, so the tracker drops a frame instead of waiting behind queued gateway inference.

Normal pan/tilt tracking uses Dahua/Amcrest `moveDirectly` 3D positioning. The tracker polls camera status and position until the camera reports idle and stable rather than sleeping for a guessed movement duration. Completed moves feed a small timing model that estimates future `moveDirectly` duration for bounded predictive lead.

Predictive target velocity is learned only while the camera is stationary. It is suppressed for immature/noisy velocity samples, low-confidence or tiny detections, and edge-clipped detections. A one-shot **edge rescue** can use a stronger positional gain when the target is already near escape.

For fast outward motion, the tracker can enter an **escape-only hybrid chase**. Continuous PTZ is used only until the target returns to a safe inner region, then it stops and hands control back to `moveDirectly`. Continuous chase is bounded by speed, keepalive, maximum duration, cooldown, and camera-side timeout controls.

PTZ completion is fail-closed. At the operation deadline the tracker performs a final status read. If status remains unavailable, or the camera still cannot be established as safely idle, tracking stops with `state=PTZ_ERROR` rather than issuing another movement command. A later `/v1/tracker/start` can resume tracking; if the tracker task itself crashed, `start` recreates it.

Target-loss timing is paused while the camera is moving so global image motion is not mistaken for subject loss. After movement completes, the velocity estimator is rebased from a fresh stationary-camera frame.

The tracker has been developed against a Dahua/Amcrest-style PTZ CGI camera, including an Amcrest IP2M-863EW-AI. Other cameras may require changes to the PTZ transport or coordinate behavior.

'''
readme = replace_between(
    readme,
    "## PTZ Tracker Design\n",
    "## Main Environment Variables\n",
    design,
    "README PTZ design",
)

pan_sections = '''### Pan / Tilt Prediction

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_MOVE_DIRECTLY_ENABLED` | `true` | Use Dahua/Amcrest 3D `moveDirectly` for normal tracking. |
| `TRACKER_MOVE_DEADZONE_X` | `0.14` | Horizontal normalized deadzone. |
| `TRACKER_MOVE_DEADZONE_Y` | `0.18` | Vertical normalized deadzone. |
| `TRACKER_MOVE_GAIN` | `0.60` | Fraction of projected frame error applied to a normal positional move. |
| `TRACKER_LEAD_TIME` | `0.75` | Fallback lead horizon before enough move timing samples exist. |
| `TRACKER_VELOCITY_MIN_SAMPLE_MS` | `100` | Minimum stationary-camera sample window before velocity is trusted. |
| `TRACKER_LEAD_MIN_CONF` | `0.50` | Suppress predictive lead below this confidence. |
| `TRACKER_LEAD_MIN_SPAN` | `0.08` | Suppress predictive lead for very small targets. |
| `TRACKER_LEAD_EDGE_MARGIN` | `0.02` | Suppress predictive lead when the bbox touches this frame margin. |
| `TRACKER_ADAPTIVE_LEAD` | `true` | Learn move duration from completed `moveDirectly` operations. |
| `TRACKER_MOVE_ETA_MIN_SAMPLES` | `3` | Completed moves required before the learned timing model is trusted. |
| `TRACKER_MOVE_ETA_HISTORY` | `24` | Maximum completed move samples retained. |
| `TRACKER_MOVE_ETA_MIN` | `0.50` | Minimum learned/fallback lead horizon. |
| `TRACKER_MOVE_ETA_MAX` | `2.00` | Maximum learned lead horizon. |
| `TRACKER_VELOCITY_CONSISTENCY_COSINE` | `0.25` | Reject abrupt direction changes from predictive lead. |
| `TRACKER_VELOCITY_JUMP_RATIO` | `4.0` | Reject implausible velocity magnitude jumps. |

Predictive lead is hard-clamped to 20% of frame width/height per axis in code.

### Edge Rescue / Hybrid Escape Chase

| Variable | Default | Description |
| --- | --- | --- |
| `TRACKER_EDGE_RESCUE_ENABLED` | `true` | Permit a stronger one-shot positional correction near an edge. |
| `TRACKER_EDGE_RESCUE_ERROR` | `0.75` | Normalized error that can request edge rescue. |
| `TRACKER_EDGE_RESCUE_GAIN` | `0.85` | Positional gain used by edge rescue. |
| `TRACKER_HYBRID_CHASE_ENABLED` | `true` | Allow continuous PTZ only for genuine escape conditions. |
| `TRACKER_HYBRID_CHASE_PAN_SIGN` | `-1` | Camera-specific horizontal continuous-move direction sign. |
| `TRACKER_HYBRID_CHASE_ENTRY_ERROR` | `0.82` | Hard normalized error threshold for chase entry. |
| `TRACKER_HYBRID_CHASE_EXIT_ERROR` | `0.50` | Stop chase after the target returns inside this error. |
| `TRACKER_HYBRID_CHASE_MOTION_ERROR` | `0.55` | Lower error threshold used for fast outward-motion entry. |
| `TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM` | `0.03` | Minimum normalized target speed for motion-based entry. |
| `TRACKER_HYBRID_CHASE_MISS_GRACE` | `0.15` | Brief detector-miss grace during motion blur. |
| `TRACKER_HYBRID_CHASE_MIN_SPEED` | `1` | Minimum continuous PTZ speed. |
| `TRACKER_HYBRID_CHASE_MAX_SPEED` | `6` | Maximum continuous PTZ speed. |
| `TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR` | `0.90` | Error at which maximum continuous speed is reached. |
| `TRACKER_HYBRID_CHASE_COMMAND_INTERVAL` | `0.18` | Minimum interval between changed chase commands. |
| `TRACKER_HYBRID_CHASE_KEEPALIVE` | `0.45` | Keepalive interval when chase speed is unchanged. |
| `TRACKER_HYBRID_CHASE_CAMERA_TIMEOUT` | `1` | Camera-side continuous movement timeout. |
| `TRACKER_HYBRID_CHASE_MAX_SECONDS` | `2.50` | Maximum duration of one escape chase. |
| `TRACKER_HYBRID_CHASE_COOLDOWN` | `0.35` | Cooldown before re-entering chase. |
| `TRACKER_HYBRID_CHASE_SETTLE_FRAMES` | `2` | Fresh frames required after chase stops. |

'''
readme = replace_between(
    readme,
    "### Pan / Tilt Prediction\n",
    "### PTZ Operation Settling\n",
    pan_sections,
    "README current tracker settings",
)

tuning = '''## Recommended Tracker Tuning

The included example Compose reflects the current hybrid controller:

```text
TRACKER_FPS=15
TRACKER_MOVE_GAIN=0.60
TRACKER_LEAD_TIME=0.75
TRACKER_ADAPTIVE_LEAD=true
TRACKER_EDGE_RESCUE_ENABLED=true
TRACKER_EDGE_RESCUE_ERROR=0.75
TRACKER_EDGE_RESCUE_GAIN=0.85
TRACKER_HYBRID_CHASE_ENABLED=true
TRACKER_HYBRID_CHASE_PAN_SIGN=-1
TRACKER_HYBRID_CHASE_ENTRY_ERROR=0.82
TRACKER_HYBRID_CHASE_EXIT_ERROR=0.50
TRACKER_HYBRID_CHASE_MOTION_ERROR=0.55
TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM=0.03
TRACKER_HYBRID_CHASE_MIN_SPEED=1
TRACKER_HYBRID_CHASE_MAX_SPEED=6
TRACKER_PTZ_STATUS_POLL_INTERVAL=0.06
TRACKER_PTZ_OPERATION_TIMEOUT=4.0
TRACKER_POST_MOVE_FRAMES=1
TRACKER_ZOOM_MAX_FACTOR=6.0
```

When tuning, use `/v1/tracker/history`. `move_directly` events include target/predicted/command centers, velocity quality, lead suppression, effective gain, move distance, predicted ETA, confidence, and CGI latency. Hybrid chase history records entry reason, signed speed, frame error, outward-motion state, stops, and failures. `ptz_operation_complete`, `ptz_operation_timeout`, and `tracking_stopped_ptz_error` show whether camera motion settled safely.

`/v1/tracker/status` also exposes `stop_reason`, `tracker_task_running`, and `tracker_task_error`. If the tracker loop itself crashes, a subsequent `/v1/tracker/start` recreates the task instead of requiring a container restart.

'''
readme = replace_between(
    readme,
    "## Recommended Tracker Tuning\n",
    "## Updating\n",
    tuning,
    "README tuning/history",
)
p.write_text(readme)

Path("docs/continuous-chase.md").write_text('''# Hybrid PTZ Escape-Chase Controller

Normal tracking uses Dahua/Amcrest `moveDirectly` positional moves. Continuous pan/tilt is reserved for a bounded escape mode when a fast target is genuinely in danger of leaving the frame.

Current controls:

```env
TRACKER_HYBRID_CHASE_ENABLED=true
TRACKER_HYBRID_CHASE_PAN_SIGN=-1
TRACKER_HYBRID_CHASE_ENTRY_ERROR=0.82
TRACKER_HYBRID_CHASE_EXIT_ERROR=0.50
TRACKER_HYBRID_CHASE_MOTION_ERROR=0.55
TRACKER_HYBRID_CHASE_MOTION_SPEED_NORM=0.03
TRACKER_HYBRID_CHASE_MISS_GRACE=0.15
TRACKER_HYBRID_CHASE_MIN_SPEED=1
TRACKER_HYBRID_CHASE_MAX_SPEED=6
TRACKER_HYBRID_CHASE_FULL_SPEED_ERROR=0.90
TRACKER_HYBRID_CHASE_COMMAND_INTERVAL=0.18
TRACKER_HYBRID_CHASE_KEEPALIVE=0.45
TRACKER_HYBRID_CHASE_CAMERA_TIMEOUT=1
TRACKER_HYBRID_CHASE_MAX_SECONDS=2.50
TRACKER_HYBRID_CHASE_COOLDOWN=0.35
TRACKER_HYBRID_CHASE_SETTLE_FRAMES=2
```

Edge rescue remains a positional `moveDirectly` correction and is configured separately with `TRACKER_EDGE_RESCUE_ENABLED`, `TRACKER_EDGE_RESCUE_ERROR`, and `TRACKER_EDGE_RESCUE_GAIN`.

Chase can start from a hard edge/error condition or from sufficiently fast outward target motion. It stops before center, on direction reversal, low confidence, target loss beyond the blur grace period, stale/unavailable frames, inference errors, maximum chase duration, tracker stop/shutdown, or home commands. Camera-side timeout is an additional fail-safe.

Useful diagnostics are `control_mode=hybrid`, `hybrid_chase_active`, `hybrid_chase_speed`, and the `hybrid_chase_*` history events.
''')


# ---------------------------------------------------------------------------
# Static regression tests. Keep the existing latest Ultralytics base by design.
# ---------------------------------------------------------------------------
Path("tests").mkdir(exist_ok=True)
Path("tests/test_static_contracts.py").write_text('''import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StaticContracts(unittest.TestCase):
    def test_python_sources_compile_to_ast(self):
        for name in ("app.py", "tracker.py"):
            ast.parse((ROOT / name).read_text(), filename=name)

    def test_every_compose_tracker_variable_is_consumed(self):
        compose = (ROOT / "docker-compose.example.yml").read_text()
        tracker = (ROOT / "tracker.py").read_text()
        names = set(re.findall(r"^\\s+(TRACKER_[A-Z0-9_]+):", compose, flags=re.MULTILINE))
        self.assertTrue(names)
        missing = sorted(name for name in names if f'"{name}"' not in tracker)
        self.assertEqual(missing, [], f"Compose tracker settings not consumed by tracker.py: {missing}")

    def test_obsolete_tracker_variables_are_gone_from_docs_and_compose(self):
        text = "\\n".join(
            (ROOT / name).read_text()
            for name in ("README.md", "docker-compose.example.yml", "docs/continuous-chase.md")
        )
        for stale in ("TRACKER_INFLIGHT_", "TRACKER_ESCAPE_", "TRACKER_CONTINUOUS_"):
            self.assertNotIn(stale, text)

    def test_runtime_cuda_executor_is_centralized(self):
        app = (ROOT / "app.py").read_text()
        self.assertIn("async def run_gpu_job(", app)
        self.assertEqual(app.count("run_in_executor("), 1)
        self.assertIn("request was cancelled while its CUDA worker was still running", app)

    def test_tensorrt_cache_tracks_source_and_gpu(self):
        app = (ROOT / "app.py").read_text()
        self.assertIn(".trt_manifest.json", app)
        self.assertIn('"source_sha256"', app)
        self.assertIn('"compute_capability"', app)

    def test_ptz_timeout_stops_when_state_is_unsafe(self):
        tracker = (ROOT / "tracker.py").read_text()
        self.assertIn("async def _halt_tracking_for_ptz_failure", tracker)
        self.assertIn("PTZ status remained unavailable", tracker)
        self.assertIn('self.state = "PTZ_ERROR"', tracker)

    def test_latest_ultralytics_base_is_intentional(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("FROM ultralytics/ultralytics:latest", dockerfile)


if __name__ == "__main__":
    unittest.main()
''')
