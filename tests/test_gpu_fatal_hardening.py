import ast
from pathlib import Path
from typing import Optional
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP_TEXT = (ROOT / "app.py").read_text()


def load_fatal_classifier():
    tree = ast.parse(APP_TEXT, filename="app.py")
    wanted = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_FATAL_CUDA_ERROR_MARKERS"
                for target in node.targets
            )
        ):
            wanted.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_is_fatal_cuda_exception":
            wanted.append(node)
    if len(wanted) != 2:
        raise AssertionError("Could not isolate CUDA fatal classifier from app.py")
    ns = {"Optional": Optional}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "app.py", "exec"), ns)
    return ns["_is_fatal_cuda_exception"]


class GpuFatalHardeningTests(unittest.TestCase):
    def test_known_sticky_cuda_errors_are_fatal(self):
        fatal = load_fatal_classifier()
        messages = (
            "CUDA error: the launch timed out and was terminated",
            "Returning 702 (CUDA_ERROR_LAUNCH_TIMEOUT) from cuCtxSynchronize_v2",
            "Sticky error detected",
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "CUDA error: an illegal memory access was encountered",
            "device-side assert triggered",
            "unspecified launch failure",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertTrue(fatal(RuntimeError(message)))

    def test_nested_cuda_cause_is_detected(self):
        fatal = load_fatal_classifier()
        inner = RuntimeError("CUDA_ERROR_LAUNCH_TIMEOUT")
        outer = RuntimeError("Ultralytics predictor failed")
        outer.__cause__ = inner
        self.assertTrue(fatal(outer))

    def test_nonsticky_runtime_errors_do_not_force_restart(self):
        fatal = load_fatal_classifier()
        for exc in (
            RuntimeError("CUDA out of memory"),
            RuntimeError("GPU execution gate is busy"),
            TimeoutError("ordinary HTTP request timed out"),
            ValueError("bad tensor shape"),
        ):
            with self.subTest(exc=str(exc)):
                self.assertFalse(fatal(exc))

    def test_run_gpu_job_terminates_only_on_fatal_worker_exception(self):
        self.assertIn("except Exception as exc:", APP_TEXT)
        self.assertIn("if _is_fatal_cuda_exception(exc):", APP_TEXT)
        self.assertIn("trigger_self_termination(reason)", APP_TEXT)
        self.assertIn("raise\n    finally:", APP_TEXT)

    def test_healthz_covers_gateway_and_tracker_failure_states(self):
        for token in (
            '@app.get("/healthz")',
            'tracker_state == "TRACKER_ERROR"',
            'tracker_state == "INFERENCE_ERROR"',
            '"timeout", "unavailable"',
            'response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE',
            'GATEWAY_REVISION = "6.7.3"',
        ):
            self.assertIn(token, APP_TEXT)


if __name__ == "__main__":
    unittest.main()
