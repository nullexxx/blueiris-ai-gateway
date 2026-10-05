import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()
HELPER = (ROOT / "ptz_enhancements.py").read_text()
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "docker-compose.example.yml").read_text()


class SmartPtzIntegrationContracts(unittest.TestCase):
    def test_python_sources_parse(self):
        ast.parse(TRACKER)
        ast.parse(HELPER)

    def test_camera_motion_compensation_is_background_based_and_guarded(self):
        self.assertIn("cv2.goodFeaturesToTrack", HELPER)
        self.assertIn("cv2.calcOpticalFlowPyrLK", HELPER)
        self.assertIn("cv2.findHomography", HELPER)
        self.assertIn("mask[ya:yb, xa:xb] = 0", HELPER)
        self.assertIn("_finite_matrix", HELPER)
        self.assertIn("motion_point = self._camera_motion.apply_point", TRACKER)

    def test_zoom_prefers_exact_onvif_but_keeps_cgi_fallback(self):
        self.assertIn("class OnvifAbsoluteZoom", HELPER)
        self.assertIn('create_type("AbsoluteMove")', HELPER)
        self.assertIn("await self.ptz.AbsoluteMove(request)", HELPER)
        self.assertIn("if self._onvif_zoom.available", TRACKER)
        self.assertIn("await self._onvif_zoom.set_factor", TRACKER)
        self.assertIn("await asyncio.to_thread(self.ptz.zoom_step", TRACKER)

    def test_zoom_uses_history_prediction_and_conservative_gates(self):
        self.assertIn("class TargetHistory", HELPER)
        self.assertIn("np.percentile", HELPER)
        self.assertIn("weighted_span", HELPER)
        self.assertIn("predicted_span", HELPER)
        self.assertIn("self._smart_history.metrics(now, 1.0)", TRACKER)
        self.assertIn("speed_norm <= 0.025", TRACKER)
        self.assertIn('int(zoom_metrics.get("edge_touches", 0)) == 0', TRACKER)

    def test_blur_gate_and_closed_loop_quality_learning_are_present(self):
        self.assertIn("frame_sharpness", TRACKER)
        self.assertIn('self._record_event("post_move_frame_blurry"', TRACKER)
        self.assertIn("def _complete_move_quality", TRACKER)
        self.assertIn('self._record_event("move_quality"', TRACKER)
        self.assertIn("self._calibration.spatial_scales", TRACKER)
        self.assertIn("self._calibration.set_spatial_scales", TRACKER)

    def test_calibration_persists_and_is_zoom_bucketed(self):
        self.assertIn("class CalibrationStore", HELPER)
        self.assertIn("temp.replace(self.path)", HELPER)
        self.assertIn("def zoom_bucket", HELPER)
        self.assertIn("self._calibration.set_timing", TRACKER)
        self.assertIn('os.getenv("TRACKER_CALIBRATION_PATH"', TRACKER)

    def test_dynamic_deadzone_and_zoom_aware_control_are_present(self):
        self.assertIn("def _effective_deadzone", TRACKER)
        self.assertIn("math.sqrt(self._current_zoom_factor())", TRACKER)
        self.assertIn("zoom_gain = 1.0 / math.sqrt(current_zoom_factor)", TRACKER)
        self.assertIn("gain_x", TRACKER)
        self.assertIn("gain_y", TRACKER)

    def test_container_and_example_config_include_smart_ptz_runtime(self):
        self.assertIn("FROM ultralytics/ultralytics:latest", DOCKERFILE)
        self.assertIn("onvif-zeep-async", DOCKERFILE)
        self.assertIn("COPY ptz_enhancements.py /app/ptz_enhancements.py", DOCKERFILE)
        for name in (
            "TRACKER_ZOOM_CONTROL_MODE",
            "TRACKER_ONVIF_PORT",
            "TRACKER_CAMERA_MAX_OPTICAL_ZOOM",
            "TRACKER_CALIBRATION_PATH",
        ):
            self.assertIn(name, COMPOSE)
            self.assertIn(name, TRACKER)


if __name__ == "__main__":
    unittest.main()
