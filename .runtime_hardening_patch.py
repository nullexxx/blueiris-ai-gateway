from pathlib import Path

p = Path("tracker.py")
text = p.read_text()
old = '''                await self._poll_ptz_operation(seq, now)
                now = time.monotonic()

                if frame is None or frame_time <= 0:
'''
new = '''                await self._poll_ptz_operation(seq, now)
                now = time.monotonic()

                # A fail-closed PTZ recovery may disable tracking during the poll.
                # Do not let the remainder of this iteration overwrite PTZ_ERROR or
                # acquire/process another target after tracking has been stopped.
                if not self.active:
                    continue

                if frame is None or frame_time <= 0:
'''
if old not in text:
    raise SystemExit("expected tracker poll block not found")
p.write_text(text.replace(old, new, 1))

# Strengthen the static contract for fail-closed behavior.
t = Path("tests/test_static_contracts.py")
tests = t.read_text()
needle = '''        self.assertIn('self.state = "PTZ_ERROR"', tracker)\n'''
replacement = needle + '''        self.assertIn("if not self.active:\\n                    continue", tracker)\n'''
if needle not in tests:
    raise SystemExit("expected PTZ test block not found")
t.write_text(tests.replace(needle, replacement, 1))
