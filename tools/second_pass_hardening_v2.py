from pathlib import Path

# Reuse the reviewed patch driver, but normalize the two endpoint replacements
# that were accidentally written with source-code indentation instead of the
# actual function-body indentation. Also prevent its generic get_status edit
# from touching _poll_ptz_operation; the correct zoom-only guard is applied in
# the post-pass below.
driver_path = Path("tools/second_pass_hardening.py")
driver = driver_path.read_text()

old_register = ''''''                embedding = await run_gpu_job(
                    sync_face_register_embedding,
                    img,
                    label=f"Face registration for '{target_id}'",
                    timeout=INFERENCE_TIMEOUT,
                    enqueue=True,
                )
''''''
new_register = ''''''        embedding = await run_gpu_job(
            sync_face_register_embedding,
            img,
            label=f"Face registration for '{target_id}'",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
''''''
driver = driver.replace(old_register, new_register, 1)

old_register_repl = ''''''                embedding, face_count = await run_gpu_job(
                    sync_face_register_embedding,
                    img,
                    label=f"Face registration for '{target_id}'",
                    timeout=INFERENCE_TIMEOUT,
                    enqueue=True,
                )
''''''
new_register_repl = ''''''        embedding, face_count = await run_gpu_job(
            sync_face_register_embedding,
            img,
            label=f"Face registration for '{target_id}'",
            timeout=INFERENCE_TIMEOUT,
            enqueue=True,
        )
''''''
driver = driver.replace(old_register_repl, new_register_repl, 1)

old_enroll = ''''''            if embedding is None:
                return {"success": False, "error": "No face detected in provided image."}

            async with face_state_lock:
                registered_faces.setdefault(target_id, []).append(embedding)
                _face_cache_dirty = True
                await asyncio.to_thread(save_faces_db, _face_db_snapshot())
''''''
new_enroll = ''''''    if embedding is None:
        return {"success": False, "error": "No face detected in provided image."}

    async with face_state_lock:
        registered_faces.setdefault(target_id, []).append(embedding)
        _face_cache_dirty = True
        snapshot = _face_db_snapshot()
    await asyncio.to_thread(save_faces_db, snapshot)
''''''
driver = driver.replace(old_enroll, new_enroll, 1)

old_enroll_repl = ''''''            if embedding is None:
                if face_count == 0:
                    return {"success": False, "error": "No face detected in provided image."}
                return {"success": False, "error": f"Expected exactly one face for enrollment; detected {face_count}."}

            async with face_state_lock:
                embeddings = registered_faces.setdefault(target_id, [])
                embeddings.append(embedding)
                if len(embeddings) > MAX_FACE_EMBEDDINGS_PER_USER:
                    del embeddings[:-MAX_FACE_EMBEDDINGS_PER_USER]
                _face_cache_dirty = True
                await asyncio.to_thread(save_faces_db, _face_db_snapshot())
''''''
new_enroll_repl = ''''''    if embedding is None:
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
''''''
driver = driver.replace(old_enroll_repl, new_enroll_repl, 1)

# Make the generic status replacement a no-op. The original patch driver would
# otherwise match _poll_ptz_operation first, where no generation local exists.
needle = '''    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not status:\\n',
    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not self._session_valid(generation):\\n                return\\n            if not status:\\n',
    "zoom status stale-result guard",
'''
replacement = '''    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not status:\\n',
    '            status = await asyncio.to_thread(self.ptz.get_status)\\n            checked_at = time.monotonic()\\n            if not status:\\n',
    "zoom status stale-result guard",
'''
if needle not in driver:
    raise SystemExit("could not neutralize generic zoom status patch")
driver = driver.replace(needle, replacement, 1)

# Execute the adjusted reviewed driver as a script.
exec(compile(driver, str(driver_path), "exec"), {"__name__": "__main__", "__file__": str(driver_path)})

# ---------------------------------------------------------------------------
# Semantic post-pass: reset transient failure/divergence state on tracker reset,
# and place the stale-result guard specifically in the zoom status lookup.
# ---------------------------------------------------------------------------
tracker_path = Path("tracker.py")
tracker = tracker_path.read_text()

reset_marker = '''        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0

    def _session_valid(self, generation: int) -> bool:
'''
reset_repl = '''        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0
        self._hybrid_last_error = None
        self._hybrid_divergence_count = 0
        self._move_failure_count = 0
        self._move_retry_after = 0.0

    def _session_valid(self, generation: int) -> bool:
'''
if reset_marker not in tracker:
    raise SystemExit("missing reset-state marker after patch")
tracker = tracker.replace(reset_marker, reset_repl, 1)

zoom_marker = '''        if self._last_zoom_position is None:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not status:
                return
'''
zoom_repl = '''        if self._last_zoom_position is None:
            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not self._session_valid(generation):
                return
            if not status:
                return
'''
if zoom_marker not in tracker:
    raise SystemExit("missing zoom status block after patch")
tracker = tracker.replace(zoom_marker, zoom_repl, 1)

# Assert that _poll_ptz_operation did not receive an undefined generation check.
poll_start = tracker.index("    async def _poll_ptz_operation(")
poll_end = tracker.index("    def _ptz_action_ready", poll_start)
if "_session_valid(generation)" in tracker[poll_start:poll_end]:
    raise SystemExit("undefined generation guard leaked into _poll_ptz_operation")

tracker_path.write_text(tracker)
print("second-pass hardening v2 applied")
