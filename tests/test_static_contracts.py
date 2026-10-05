import ast
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
        names = set(re.findall(r"^\s+(TRACKER_[A-Z0-9_]+):", compose, flags=re.MULTILINE))
        self.assertTrue(names)
        missing = sorted(name for name in names if f'"{name}"' not in tracker)
        self.assertEqual(missing, [], f"Compose tracker settings not consumed by tracker.py: {missing}")

    def test_obsolete_tracker_variables_are_gone_from_docs_and_compose(self):
        text = "\n".join(
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
        self.assertIn("if not self.active:\n                    continue", tracker)

    def test_latest_ultralytics_base_is_intentional(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("FROM ultralytics/ultralytics:latest", dockerfile)


if __name__ == "__main__":
    unittest.main()
