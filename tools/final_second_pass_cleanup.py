from pathlib import Path

p = Path("tracker.py")
text = p.read_text()

old = '''        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0
        self._hybrid_last_error = None
        self._hybrid_divergence_count = 0
        self._move_failure_count = 0
        self._move_retry_after = 0.0
        self._hybrid_last_stopped_at = 0.0
'''
new = '''        self._hybrid_started_at = 0.0
        self._hybrid_last_command_at = 0.0
        self._hybrid_last_stopped_at = 0.0
'''
if old not in text:
    raise SystemExit("duplicate initializer block not found")
text = text.replace(old, new, 1)

old = '    async def _poll_ptz_operation(self, seq: int, now: float) -> None:\n'
new = '    async def _poll_ptz_operation(self, seq: int, now: float, generation: int) -> None:\n'
if old not in text:
    raise SystemExit("poll signature not found")
text = text.replace(old, new, 1)

old = '''        kind = self._ptz_operation
        if kind is None:
            return
'''
new = '''        if not self._session_valid(generation):
            return
        kind = self._ptz_operation
        if kind is None:
            return
'''
poll_pos = text.find('    async def _poll_ptz_operation(')
block_pos = text.find(old, poll_pos)
if block_pos < 0:
    raise SystemExit("poll opening block not found")
text = text[:block_pos] + text[block_pos:].replace(old, new, 1)

# Both camera-status awaits inside _poll_ptz_operation must discard a stale
# result before touching tracker state or entering fail-closed PTZ recovery.
poll_start = text.index('    async def _poll_ptz_operation(')
poll_end = text.index('    def _ptz_action_ready', poll_start)
poll = text[poll_start:poll_end]
needle1 = '''            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not status:
'''
repl1 = '''            status = await asyncio.to_thread(self.ptz.get_status)
            checked_at = time.monotonic()
            if not self._session_valid(generation):
                return
            if not status:
'''
if needle1 not in poll:
    raise SystemExit("timeout status await not found")
poll = poll.replace(needle1, repl1, 1)

needle2 = '''        status = await asyncio.to_thread(self.ptz.get_status)
        polled_at = time.monotonic()
        self._ptz_next_status_poll_at = polled_at + self.cfg.ptz_status_poll_interval
'''
repl2 = '''        status = await asyncio.to_thread(self.ptz.get_status)
        polled_at = time.monotonic()
        if not self._session_valid(generation):
            return
        self._ptz_next_status_poll_at = polled_at + self.cfg.ptz_status_poll_interval
'''
if needle2 not in poll:
    raise SystemExit("normal status await not found")
poll = poll.replace(needle2, repl2, 1)
text = text[:poll_start] + poll + text[poll_end:]

old = '                await self._poll_ptz_operation(seq, now)\n'
new = '                await self._poll_ptz_operation(seq, now, generation)\n'
if old not in text:
    raise SystemExit("poll call not found")
text = text.replace(old, new, 1)

p.write_text(text)

# Extend the static contracts so this exact stop-during-status race cannot regress.
tp = Path("tests/test_static_contracts.py")
tests = tp.read_text()
needle = '        self.assertIn("CAP_PROP_READ_TIMEOUT_MSEC", tracker)\n'
addition = '''        self.assertIn("CAP_PROP_READ_TIMEOUT_MSEC", tracker)
        self.assertIn("async def _poll_ptz_operation(self, seq: int, now: float, generation: int)", tracker)
        poll = tracker[tracker.index("async def _poll_ptz_operation"):tracker.index("def _ptz_action_ready")]
        self.assertGreaterEqual(poll.count("if not self._session_valid(generation):"), 3)
'''
if needle not in tests:
    raise SystemExit("test insertion point not found")
tests = tests.replace(needle, addition, 1)
tp.write_text(tests)

print("final second-pass cleanup applied")
