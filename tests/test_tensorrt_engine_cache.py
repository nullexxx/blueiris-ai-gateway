"""Behavior tests for TensorRT engine caching and model discovery in app.py.

app.py needs torch/ultralytics/tensorrt, which CI does not install, so the functions
under test are isolated with ``ast`` and run against a temporary models directory with
the GPU-dependent pieces (manifest, engine builders) stubbed.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP_TREE = ast.parse((ROOT / "app.py").read_text(), filename="app.py")

FUNCTIONS = (
    "discover_models", "_source_model_path", "_trt_paths", "_write_json_atomic",
    "_mark_tensorrt_failed", "ensure_tensorrt_engine",
)


class Log:
    def __init__(self):
        self.records = []

    def __getattr__(self, level):
        return lambda msg, *args, **kwargs: self.records.append((level, msg % args if args else msg))


class EngineCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name)
        self.builds: List[str] = []
        self.fail_build = False
        self.source_version = "v1"
        self.log = Log()

        def fake_manifest(source: Path) -> dict:
            return {
                "source_file": source.name, "source_sha256": self.source_version, "tensorrt": "11.4",
                "gpu_name": "NVIDIA GeForce RTX 5060 Ti", "precision": "fp16",
            }

        def fake_onnx_build(onnx_path: Path, engine_path: Path) -> None:
            self.builds.append(onnx_path.name)
            (self.models / f"{onnx_path.stem}.fp16.onnx").write_text("intermediate")
            if self.fail_build:
                raise RuntimeError("simulated TensorRT failure")
            engine_path.write_text("engine")

        class FakeYOLO:
            def __init__(inner, path, task=None):
                inner.path = Path(path)

            def export(inner, **kwargs):
                self.builds.append(inner.path.name)
                inner.path.with_suffix(".engine").write_text("engine")

        nodes = [n for n in APP_TREE.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
        self.ns = {
            "Path": Path, "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple,
            "json": json, "os": os, "logger": self.log, "MODELS_DIR": self.models,
            "MODEL_EXT_PRIORITY": [".engine", ".onnx", ".pt"], "HALF_PRECISION": True,
            "TENSORRT_FROM_ONNX": True, "YOLO": FakeYOLO,
            "_tensorrt_manifest": fake_manifest, "_build_engine_from_onnx": fake_onnx_build,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), self.ns)
        self.ensure = self.ns["ensure_tensorrt_engine"]

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, name: str) -> Path:
        path = self.models / name
        path.write_text("x")
        return path

    def test_onnx_only_model_gets_a_cached_engine(self):
        self.touch("frigate.onnx")
        self.assertTrue(self.ensure("frigate"))
        self.assertEqual(self.builds, ["frigate.onnx"])
        self.assertTrue((self.models / "frigate.engine").exists())
        manifest = json.loads((self.models / "frigate.trt_manifest.json").read_text())
        self.assertEqual(manifest["source_file"], "frigate.onnx")
        self.assertFalse((self.models / "frigate.fp16.onnx").exists(), "ModelOpt intermediate must be removed")
        self.assertEqual(self.ns["discover_models"]()["frigate"].name, "frigate.engine")

        self.assertTrue(self.ensure("frigate"))
        self.assertEqual(len(self.builds), 1, "an up-to-date engine must not be rebuilt")

    def test_changed_source_rebuilds(self):
        self.touch("frigate.onnx")
        self.ensure("frigate")
        self.source_version = "v2"
        self.assertTrue(self.ensure("frigate"))
        self.assertEqual(len(self.builds), 2)

    def test_pt_weights_take_priority_over_onnx(self):
        self.touch("yolo11l.pt")
        self.touch("yolo11l.onnx")
        self.assertTrue(self.ensure("yolo11l"))
        self.assertEqual(self.builds, ["yolo11l.pt"])

    def test_failed_build_is_not_retried_until_something_changes(self):
        self.touch("frigate.onnx")
        self.fail_build = True
        self.assertFalse(self.ensure("frigate"))
        self.assertFalse((self.models / "frigate.engine").exists())
        self.assertFalse((self.models / "frigate.fp16.onnx").exists())
        self.assertTrue((self.models / "frigate.trt_failed.json").exists())

        self.assertFalse(self.ensure("frigate"))
        self.assertEqual(len(self.builds), 1, "a known-bad build must not rerun on every restart")
        self.assertEqual(self.ns["discover_models"]()["frigate"].name, "frigate.onnx")

        self.fail_build = False
        self.source_version = "v2"  # e.g. a new TensorRT or a new model file
        self.assertTrue(self.ensure("frigate"))
        self.assertEqual(len(self.builds), 2)
        self.assertFalse((self.models / "frigate.trt_failed.json").exists())

    def test_force_retries_even_after_a_failure(self):
        self.touch("frigate.onnx")
        self.fail_build = True
        self.ensure("frigate")
        self.fail_build = False
        self.assertTrue(self.ensure("frigate", force=True))
        self.assertEqual(len(self.builds), 2)

    def test_onnx_engines_can_be_disabled(self):
        self.touch("frigate.onnx")
        self.ns["TENSORRT_FROM_ONNX"] = False
        self.assertFalse(self.ensure("frigate"))
        self.assertEqual(self.builds, [])

    def test_discovery_hides_fp16_intermediates_but_keeps_standalone_models(self):
        self.touch("yolo11l.engine")
        self.touch("yolo11l.fp16.onnx")
        self.touch("custom.fp16.onnx")  # no "custom" base model: a real model that happens to be named .fp16
        found = self.ns["discover_models"]()
        self.assertEqual(sorted(found), ["custom.fp16", "yolo11l"])


if __name__ == "__main__":
    unittest.main()
