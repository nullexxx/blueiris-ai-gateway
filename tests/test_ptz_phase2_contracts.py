import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()
HELPER = (ROOT / "ptz_enhancements.py").read_text()
PHASE2 = (ROOT / "ptz_phase2.py").read_text()
APP = (ROOT / "app.py").read_text()
COMPOSE = (ROOT / "docker-compose.example.yml").read_text()
DOCKER = (ROOT / "Dockerfile").read_text()
BUILD = (ROOT / ".github/workflows/build.yml").read_text()

class PtzPhase2Contracts(unittest.TestCase):
    def test_sources_parse(self):
        for source in (TRACKER, HELPER, PHASE2, APP):
            ast.parse(source)

    def test_onvif_is_optional_and_self_healing(self):
        self.assertIn("class OnvifRetryState", PHASE2)
        self.assertIn("_maybe_start_onvif_recovery", TRACKER)
        self.assertIn("cgi_timed_fallback", TRACKER)
        self.assertIn("self._onvif_zoom.available = False", TRACKER)
        self.assertIn("self._onvif_recovery_task.cancel()", TRACKER)
        self.assertIn("async def set_normalized", HELPER)

    def test_zoom_mapping_is_persisted_and_used(self):
        self.assertIn("class ZoomCalibrationMap", PHASE2)
        self.assertIn("estimate_normalized", TRACKER)
        self.assertIn("self._zoom_map.record", TRACKER)
        self.assertIn("TRACKER_ZOOM_CALIBRATION_PATH", COMPOSE)

    def test_bbox_geometry_and_scene_stability_guard_control(self):
        self.assertIn("class BBoxMotionValidator", PHASE2)
        self.assertIn("velocity_geometry_rejected", TRACKER)
        self.assertIn("class SceneStabilityGate", PHASE2)
        self.assertIn("post_move_scene_stable", TRACKER)
        self.assertIn("and self._scene_stable_ready", TRACKER)

    def test_zones_and_motion_masks_are_optional(self):
        self.assertIn("class AcquisitionZonePolicy", PHASE2)
        self.assertIn("class MotionMaskPolicy", PHASE2)
        self.assertIn("self._acquisition_zones.allows", TRACKER)
        self.assertIn("self._motion_masks.boxes", TRACKER)
        for key in ("TRACKER_ACQUIRE_ZONES_JSON", "TRACKER_IGNORE_ZONES_JSON", "TRACKER_MOTION_MASKS_JSON"):
            self.assertIn(key, COMPOSE)

    def test_manual_calibration_is_bounded_and_not_automatic(self):
        self.assertIn("async def calibrate", TRACKER)
        self.assertIn('/v1/tracker/calibrate', APP)
        self.assertIn("Stop tracking before calibration", TRACKER)
        self.assertNotIn("await self.calibrate()", TRACKER)

    def test_container_and_ci_track_phase2_module(self):
        self.assertIn("COPY ptz_phase2.py /app/ptz_phase2.py", DOCKER)
        self.assertIn('      - "ptz_phase2.py"', BUILD)

if __name__ == "__main__":
    unittest.main()
