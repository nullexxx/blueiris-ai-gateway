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
        self.assertIn("if not self._session_valid(generation):\n                    continue", tracker)

    def test_latest_ultralytics_base_is_intentional(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("FROM ultralytics/ultralytics:latest", dockerfile)

    def test_second_pass_tracker_safety_contracts(self):
        tracker = (ROOT / "tracker.py").read_text()
        compose = (ROOT / "docker-compose.example.yml").read_text()
        self.assertIn("if dist_norm > max_dist and overlap < 0.30", tracker)
        self.assertNotIn("best_score >= 0.20 or best_dist_norm <= max_dist", tracker)
        self.assertIn("if not self._session_valid(generation):", tracker)
        self.assertIn("hybrid_chase_diverging", tracker)
        self.assertIn("moveDirectly failed", tracker)
        self.assertIn("CAP_PROP_READ_TIMEOUT_MSEC", tracker)
        self.assertIn("async def _poll_ptz_operation(self, seq: int, now: float, generation: int)", tracker)
        poll = tracker[tracker.index("async def _poll_ptz_operation"):tracker.index("def _ptz_action_ready")]
        self.assertGreaterEqual(poll.count("if not self._session_valid(generation):"), 3)
        self.assertIn('TRACKER_PTZ_STATUS_POLL_INTERVAL: "0.12"', compose)

    def test_face_and_debug_optimizations(self):
        app = (ROOT / "app.py").read_text()
        tracker = (ROOT / "tracker.py").read_text()
        self.assertIn("face_detector.extract(img, boxes, None)", app)
        self.assertNotIn("faces = face_detector(img)", app)
        self.assertIn('kwargs["quantize"] = 16', app)
        self.assertNotIn('kwargs["half"] = True', app)
        self.assertIn("MAX_FACE_EMBEDDINGS_PER_USER", app)
        self.assertNotIn("self._make_debug_frame(frame, detections)", tracker)
        self.assertIn("async def debug_jpeg", tracker)
        self.assertIn("await _require_tracker().debug_jpeg()", app)


if __name__ == "__main__":
    unittest.main()
