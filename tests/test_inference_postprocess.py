"""Behavior tests for detection post-processing and GPU-boundary contracts in app.py.

app.py imports torch/ultralytics/tensorrt, which CI does not install, so the
functions under test are isolated from the source with ``ast`` and exercised
against small stand-ins for Ultralytics result objects.
"""
import ast
import io
import json
import math
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple
import unittest

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is installed in CI
    Image = None

ROOT = Path(__file__).resolve().parents[1]
APP_TEXT = (ROOT / "app.py").read_text()
APP_TREE = ast.parse(APP_TEXT, filename="app.py")


def function_source(name: str) -> str:
    for node in APP_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(APP_TEXT, node) or ""
    raise AssertionError(f"{name} not found in app.py")


def load_functions(*names, **namespace):
    nodes = [
        node for node in APP_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    ns = {"List": List, "Optional": Optional, "Tuple": Tuple, "Dict": Dict, "np": np, **namespace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), ns)
    return [ns[name] for name in names]


class FakeTensor:
    """Mimics a CUDA tensor: data is only reachable through .cpu().numpy()."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    def __init__(self, rows):
        array = np.asarray(rows, dtype=np.float32)
        self.data = FakeTensor(array if rows else array.reshape(0, 6))

    def __len__(self):
        return len(self.data.numpy())


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    names = {0: "person", 16: "dog"}

    def __init__(self, results):
        self._results = results
        self.calls = []

    def predict(self, img, **kwargs):
        self.calls.append(kwargs)
        return self._results


class Placeholder:
    Image = object


class DetectionPostprocessTests(unittest.TestCase):
    def test_result_boxes_reads_each_result_with_one_host_transfer(self):
        result_boxes, = load_functions("_result_boxes")
        first = FakeResult([[1.6, 2.2, 30.9, 40.1, 0.876, 16], [5, 6, 7, 8, 0.5, 0]])
        empty = FakeResult([])
        rows = result_boxes([first, empty])
        self.assertEqual(first.boxes.data.cpu_calls, 1)
        self.assertEqual(empty.boxes.data.cpu_calls, 0)
        self.assertEqual(len(rows), 2)
        x1, y1, x2, y2, conf, cls = rows[0]
        self.assertAlmostEqual(x1, 1.6, places=4)
        self.assertAlmostEqual(conf, 0.876, places=4)
        self.assertEqual(cls, 16)
        self.assertIsInstance(cls, int)

    def test_result_boxes_handles_tracking_id_column(self):
        result_boxes, = load_functions("_result_boxes")
        rows = result_boxes([FakeResult([[1, 2, 3, 4, 99, 0.7, 16]])])
        self.assertEqual(rows[0][4:], (np.float32(0.7).item(), 16))

    def test_sync_predict_returns_codeproject_shaped_predictions(self):
        _, sync_predict = load_functions(
            "_result_boxes", "sync_predict",
            YOLO=object, Image=Placeholder, HALF_PRECISION=True,
        )
        model = FakeModel([FakeResult([[1.6, 2.2, 30.9, 40.1, 0.876, 16]])])
        predictions = sync_predict(model, object(), 0.4, True)
        self.assertEqual(
            predictions,
            [{"confidence": 0.88, "label": "dog", "x_min": 1, "y_min": 2, "x_max": 30, "y_max": 40}],
        )
        self.assertEqual(model.calls[0]["quantize"], 16)
        self.assertEqual(model.calls[0]["conf"], 0.4)

    def test_classes_missing_from_names_fall_back_to_generic_label(self):
        _, sync_predict = load_functions(
            "_result_boxes", "sync_predict",
            YOLO=object, Image=Placeholder, HALF_PRECISION=False,
        )
        model = FakeModel([FakeResult([[1, 2, 3, 4, 0.9, 45]])])
        self.assertEqual(sync_predict(model, object(), 0.4, False)[0]["label"], "class45")

    def test_sync_predict_only_quantizes_pytorch_weights(self):
        _, sync_predict = load_functions(
            "_result_boxes", "sync_predict",
            YOLO=object, Image=Placeholder, HALF_PRECISION=True,
        )
        model = FakeModel([])
        self.assertEqual(sync_predict(model, object(), 0.4, False), [])
        self.assertNotIn("quantize", model.calls[0])


def encode(width: int, height: int, fmt: str = "JPEG") -> bytes:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt)
    return buf.getvalue()


@unittest.skipIf(Image is None, "Pillow is not installed")
class DecodeImageTests(unittest.TestCase):
    def setUp(self):
        self.decode, = load_functions("decode_image", Image=Image, io=io, math=math)

    def test_disabled_decodes_full_resolution(self):
        img, sx, sy = self.decode(encode(3840, 2160), 0)
        self.assertEqual(img.size, (3840, 2160))
        self.assertEqual((sx, sy), (1.0, 1.0))
        self.assertEqual(img.mode, "RGB")

    def test_large_jpeg_keeps_long_side_at_or_above_target(self):
        img, sx, sy = self.decode(encode(3840, 2160), 1280)
        self.assertEqual(img.size, (1920, 1080))
        self.assertEqual((sx, sy), (2.0, 2.0))
        img, sx, sy = self.decode(encode(3840, 2160), 640)
        self.assertEqual(img.size, (960, 540))
        self.assertEqual((sx, sy), (4.0, 4.0))

    def test_portrait_and_odd_sizes_map_back_to_original(self):
        img, sx, sy = self.decode(encode(1521, 2689), 640)
        self.assertGreaterEqual(max(img.size), 640)
        self.assertAlmostEqual(img.size[0] * sx, 1521, places=6)
        self.assertAlmostEqual(img.size[1] * sy, 2689, places=6)

    def test_small_jpeg_and_non_jpeg_are_untouched(self):
        img, sx, _ = self.decode(encode(1000, 600), 640)  # cannot halve without going below 640
        self.assertEqual((img.size, sx), ((1000, 600), 1.0))
        img, sx, _ = self.decode(encode(2560, 1440, "PNG"), 640)
        self.assertEqual((img.size, sx), ((2560, 1440), 1.0))

    def test_predictions_are_scaled_back_to_original_pixels(self):
        _, sync_predict = load_functions(
            "_result_boxes", "sync_predict",
            YOLO=object, Image=Placeholder, HALF_PRECISION=False,
        )
        model = FakeModel([FakeResult([[100, 50, 200, 150, 0.9, 0]])])
        predictions = sync_predict(model, object(), 0.4, False, 4.0, 4.0)
        self.assertEqual(
            (predictions[0]["x_min"], predictions[0]["y_min"], predictions[0]["x_max"], predictions[0]["y_max"]),
            (400, 200, 800, 600),
        )


class OnnxProviderDetectionTests(unittest.TestCase):
    def setUp(self):
        self.provider, = load_functions("_onnx_execution_provider", YOLO=object)

    @staticmethod
    def model_with(backend):
        predictor = type("Predictor", (), {"model": backend})()
        return type("Model", (), {"predictor": predictor})()

    @staticmethod
    def session(*providers):
        return type("Session", (), {"get_providers": lambda self: list(providers)})()

    def test_reads_provider_forwarded_by_autobackend(self):
        backend = type("AutoBackend", (), {"session": self.session("CPUExecutionProvider")})()
        self.assertEqual(self.provider(self.model_with(backend)), "CPUExecutionProvider")

    def test_reads_provider_from_nested_backend(self):
        inner = type("ONNXBackend", (), {"session": self.session("CUDAExecutionProvider", "CPUExecutionProvider")})()
        backend = type("AutoBackend", (), {"backend": inner})()
        self.assertEqual(self.provider(self.model_with(backend)), "CUDAExecutionProvider")

    def test_unknown_layout_returns_none(self):
        self.assertIsNone(self.provider(object()))
        self.assertIsNone(self.provider(self.model_with(object())))


class SilentLogger:
    def __init__(self):
        self.messages = []

    def _log(self, level, msg, *args):
        self.messages.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._log("info", msg, *args)

    def warning(self, msg, *args):
        self._log("warning", msg, *args)


FRIGATE_PLUS_JSON = {
    "name": "yolov9c",
    "type": "yolo-generic",
    "width": 640,
    "labelMap": {"0": "person", "1": "dhl", "16": "license_plate", "29": "car"},
}


class LabelMapTests(unittest.TestCase):
    def setUp(self):
        self.logger = SilentLogger()
        self.load, self.apply = load_functions(
            "load_label_map", "_apply_label_map", json=json, logger=self.logger, Path=Path, YOLO=object,
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, payload, name="frigate_yolov9c_640.json"):
        (self.dir / name).write_text(payload if isinstance(payload, str) else json.dumps(payload))

    def test_reads_frigate_plus_label_map_for_any_model_format(self):
        self.write(FRIGATE_PLUS_JSON)
        for ext in (".onnx", ".engine", ".pt"):
            names = self.load(self.dir / f"frigate_yolov9c_640{ext}")
            self.assertEqual(names, {0: "person", 1: "dhl", 16: "license_plate", 29: "car"})
            self.assertEqual(list(names), [0, 1, 16, 29])

    def test_accepts_names_object_list_and_plain_map(self):
        self.write({"names": {"0": "a", "1": "b"}}, "m1.json")
        self.write(["a", "b"], "m2.json")
        self.write({"0": "a", "1": "b"}, "m3.json")
        for stem in ("m1", "m2", "m3"):
            self.assertEqual(self.load(self.dir / f"{stem}.onnx"), {0: "a", 1: "b"}, stem)

    def test_missing_or_bad_files_are_ignored_with_a_warning(self):
        self.assertIsNone(self.load(self.dir / "absent.onnx"))
        self.write("{not json", "broken.json")
        self.write({"labelMap": {"zero": "person"}}, "badkeys.json")
        self.write({"width": 640}, "nolabels.json")
        for stem in ("broken", "badkeys", "nolabels"):
            self.assertIsNone(self.load(self.dir / f"{stem}.onnx"), stem)
        self.assertEqual(sum(1 for level, _ in self.logger.messages if level == "warning"), 3)

    def test_trt_manifest_is_not_mistaken_for_a_label_file(self):
        self.write({"source_sha256": "x"}, "yolo11l.trt_manifest.json")
        self.assertIsNone(self.load(self.dir / "yolo11l.engine"))

    @staticmethod
    def model_with_backend(names):
        inner = type("ONNXBackend", (), {})()
        inner.names = names
        auto = type("AutoBackend", (), {})()
        auto.backend = inner
        predictor = type("Predictor", (), {})()
        predictor.model = auto
        model = type("Model", (), {})()
        model.predictor = predictor
        return model, inner

    def test_apply_replaces_generic_names_on_the_inner_backend(self):
        model, inner = self.model_with_backend({i: f"class{i}" for i in range(999)})
        self.apply(model, {0: "person", 29: "car"}, "frigate.json")
        self.assertEqual(inner.names, {0: "person", 29: "car"})
        self.assertFalse(any(level == "warning" for level, _ in self.logger.messages))

    def test_apply_warns_when_overriding_real_embedded_names(self):
        model, inner = self.model_with_backend({0: "person", 1: "bicycle"})
        self.apply(model, {0: "person", 1: "dhl"}, "frigate.json")
        self.assertEqual(inner.names[1], "dhl")
        self.assertTrue(any(level == "warning" for level, _ in self.logger.messages))


class GpuBoundaryContracts(unittest.TestCase):
    def test_request_handlers_do_not_touch_result_tensors(self):
        # Tensor reads on the event loop block it on the GPU with no timeout.
        for name in ("run_inference", "face_recognize"):
            source = function_source(name)
            self.assertNotIn(".boxes", source, name)
            self.assertNotIn(".tolist()", source, name)
            self.assertNotIn(".item()", source, name)

    def test_face_matrix_is_built_on_cpu_outside_executor(self):
        rebuild = function_source("_rebuild_face_matrix")
        self.assertNotIn("cuda", rebuild)
        snapshot = function_source("_get_face_enrollment_snapshot")
        self.assertNotIn("to_thread", snapshot)
        recognize = function_source("sync_face_recognize")
        self.assertIn("face_matrix.to(embeddings.device", recognize)

    def test_no_cuda_work_is_dispatched_with_to_thread_at_runtime(self):
        # Startup (lifespan) runs before requests are served; everything else
        # must go through run_gpu_job.
        for node in APP_TREE.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name == "lifespan":
                continue
            source = ast.get_source_segment(APP_TEXT, node) or ""
            for gpu_fn in ("_rebuild_face_matrix", "sync_predict", "sync_face_recognize",
                           "sync_face_register_embedding", "load_and_warmup_with_recovery_sync"):
                self.assertNotIn(f"to_thread({gpu_fn}", source, node.name)


if __name__ == "__main__":
    unittest.main()
