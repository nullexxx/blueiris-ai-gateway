from pathlib import Path

app_path = Path("app.py")
app = app_path.read_text()
old = "    jpeg = _require_tracker().debug_jpeg()\n"
new = "    jpeg = await _require_tracker().debug_jpeg()\n"
if old not in app:
    raise SystemExit("missing tracker debug endpoint call")
app_path.write_text(app.replace(old, new, 1))

tests_path = Path("tests/test_static_contracts.py")
tests = tests_path.read_text()
tests = tests.replace(
    'self.assertIn("if not self.active:\\n                    continue", tracker)',
    'self.assertIn("if not self._session_valid(generation):\\n                    continue", tracker)',
)
tests = tests.replace(
    'self.assertIn("await tracker_service.debug_jpeg()", app)',
    'self.assertIn("await _require_tracker().debug_jpeg()", app)',
)
tests_path.write_text(tests)
print("second-pass validation fixups applied")
