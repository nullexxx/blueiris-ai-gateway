"""Behavior tests for detection post-processing and GPU-boundary contracts in app.py.

app.py imports torch/ultralytics/tensorrt, which CI does not install, so the
functions under test are isolated from the source with ``ast`` and exercised
against small stand-ins for Ultralytics result objects.
"""
import ast
from pathlib import Path
from typing import List, Optional, Tuple
import unittest

import numpy as np

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
    ns = {"List": List, "Optional": Optional, "Tuple": Tuple, "np": np, **namespace}
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

    def test_sync_predict_only_quantizes_pytorch_weights(self):
        _, sync_predict = load_functions(
            "_result_boxes", "sync_predict",
            YOLO=object, Image=Placeholder, HALF_PRECISION=True,
        )
        model = FakeModel([])
        self.assertEqual(sync_predict(model, object(), 0.4, False), [])
        self.assertNotIn("quantize", model.calls[0])


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
