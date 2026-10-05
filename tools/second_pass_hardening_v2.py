from pathlib import Path

# Adjust the reviewed patch driver in memory so harmless source formatting does
# not break literal replacements. Then apply a small semantic post-pass for the
# two places where a generic replacement would be too broad.
driver_path = Path("tools/second_pass_hardening.py")
driver = driver_path.read_text()

replacements = [
    (
        """'''                embedding = await run_gpu_job(
                    sync_face_register_embedding,
                    img,
                    label=f\"Face registration for '{target_id}'\",
                    timeout=INFERENCE_TIMEOUT,
                    enqueue=True,
                )
'''""",
        """'''        embedding = await run_gpu_job(
            sync_face_register_embedding,
            img,
            label=f\"Face registration for '{target_id}'\",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
'''""",
    ),
    (
        """'''                embedding, face_count = await run_gpu_job(
                    sync_face_register_embedding,
                    img,
                    label=f\"Face registration for '{target_id}'\",
                    timeout=INFERENCE_TIMEOUT,
                    enqueue=True,
                )
'''""",
        """'''        embedding, face_count = await run_gpu_job(
            sync_face_register_embedding,
            img,
            label=f\"Face registration for '{target_id}'\",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
'''""",
    ),
    (
        """'''            if embedding is None:
                return {\"success\": False, \"error\": \"No face detected in provided image.\"}

            async with face_state_lock:
                registered_faces.setdefault(target_id, []).append(embedding)
                _face_cache_dirty = True
                await asyncio.to_thread(save_faces_db, _face_db_snapshot())
'''""",
        """'''    if embedding is None:
        return {\"success\": False, \"error\": \"No face detected in provided image.\"}

    async with face_state_lock:
        registered_faces.setdefault(target_id, []).append(embedding)
        _face_cache_dirty = True
        snapshot = _face_db_snapshot()
    await asyncio.to_thread(save_faces_db, snapshot)
'''""",
    ),
    (
        """'''            if embedding is None:
                if face_count == 0:
                    return {\"success\": False, \"error\": \"No face detected in provided image.\"}
                return {\"success\": False, \"error\": f\"Expected exactly one face for enrollment; detected {face_count}.\"}

            async with face_state_lock:
                embeddings = registered_faces.setdefault(target_id, [])
                embeddings.append(embedding)
                if len(embeddings) > MAX_FACE_EMBEDDINGS_PER_USER:
                    del embeddings[:-MAX_FACE_EMBEDDINGS_PER_USER]
                _face_cache_dirty = True
                await asyncio.to_thread(save_faces_db, _face_db_snapshot())
'''""",
        """'''    if embedding is None:
        if face_count == 0:
            return {\"success\": False, \"error\": \"No face detected in provided image.\"}
        return {\"success\": False, \"error\": f\"Expected exactly one face for enrollment; detected {face_count}.\"}

    async with face_state_lock:
        embeddings = registered_faces.setdefault(target_id, [])
        embeddings.append(embedding)
        if len(embeddings) > MAX_FACE_EMBEDDINGS_PER_USER:
            del embeddings[:-MAX_FACE_EMBEDDINGS_PER_USER]
        _face_cache_dirty = True
        snapshot = _face_db_snapshot()
    await asyncio.to_thread(save_faces_db, snapshot)
'''""",
    ),
]
for old, new in replacements:
    if old not in driver:
        raise SystemExit("expected face-registration patch literal not found")
    driver = driver.replace(old, new, 1)

# Neutralize the generic get_status generation edit. The correct guard is added
# specifically to the zoom path after the main patch has run.
old_status_patch = """    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not status:\\n',
    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not self._session_valid(generation):\\n                return\\n            if not status:\\n',
    \"zoom status stale-result guard\",
"""
new_status_patch = """    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not status:\\n',
    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not status:\\n',
    \"zoom status stale-result guard\",
"""
if old_status_patch not in driver:
    raise SystemExit("could not neutralize generic zoom status patch")
driver = driver.replace(old_status_patch, new_status_patch, 1)

exec(compile(driver, str(driver_path), "exec"), {"__name__": "__main__", "__file__": str(driver_path)})

tracker_path = Path("tracker.py")
tracker = tracker_path.read_text()

# Reset transient retry/divergence state whenever a track session is reset.
reset_marker = """        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0

    def _session_valid(self, generation: int) -> bool:
"""
reset_repl = """        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0
        self._hybrid_last_error = None
        self._hybrid_divergence_count = 0
        self._move_failure_count = 0
        self._move_retry_after = 0.0

    def _session_valid(self, generation: int) -> bool:
"""
if reset_marker not in tracker:
    raise SystemExit("missing reset-state marker after main patch")
tracker = tracker.replace(reset_marker, reset_repl, 1)

# Discard a camera-status result if /stop or /home happened while it was awaited.
zoom_marker = """        if self._last_zoom_position is None:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not status:
                return
"""
zoom_repl = """        if self._last_zoom_position is None:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not self._session_valid(generation):
                return
            if not status:
                return
"""
if zoom_marker not in tracker:
    raise SystemExit("missing zoom status block after main patch")
tracker = tracker.replace(zoom_marker, zoom_repl, 1)

poll_start = tracker.index("    async def _poll_ptz_operation(")
poll_end = tracker.index("    def _ptz_action_ready", poll_start)
if "_session_valid(generation)" in tracker[poll_start:poll_end]:
    raise SystemExit("undefined generation guard leaked into _poll_ptz_operation")

tracker_path.write_text(tracker)
print("second-pass hardening v2 applied")
