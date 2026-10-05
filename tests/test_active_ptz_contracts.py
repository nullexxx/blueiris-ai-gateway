from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()
APP = (ROOT / "app.py").read_text()
DOCKER = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "docker-compose.example.yml").read_text()


class ActiveCalibrationContracts(unittest.TestCase):
    def test_active_calibration_runtime_is_wired(self):
        self.assertIn("ActivePtzCalibrationStore", TRACKER)
        self.assertIn("estimate_static_frame_shift", TRACKER)
        self.assertIn("_calibration_move_directly_sample", TRACKER)
        self.assertIn("_calibration_continuous_sample", TRACKER)
        self.assertIn("OnvifPanTiltProbe", TRACKER)
        self.assertIn("choose_continuous_speed", TRACKER)
        self.assertIn("continuous_sign", TRACKER)

    def test_startup_policy_is_nonblocking_and_guarded(self):
        self.assertIn('TRACKER_CALIBRATE_ON_START', TRACKER)
        self.assertIn('"if_missing"', TRACKER)
        self.assertIn('asyncio.create_task(\n                    self._startup_calibration_sequence()', TRACKER)
        self.assertIn('if self._calibrating:', TRACKER)
        self.assertIn('Calibration is in progress.', TRACKER)

    def test_api_and_container_include_active_calibration(self):
        self.assertIn('async def tracker_calibrate(mode: str = "zoom")', APP)
        self.assertIn('COPY ptz_active_calibration.py /app/ptz_active_calibration.py', DOCKER)
        for name in (
            "TRACKER_CALIBRATE_ON_START",
            "TRACKER_CALIBRATION_SCOPE",
            "TRACKER_ACTIVE_CALIBRATION_PATH",
            "TRACKER_CALIBRATION_MAX_AGE_DAYS",
            "TRACKER_CALIBRATION_ZOOM_LEVELS",
            "TRACKER_CALIBRATION_OFFSETS",
            "TRACKER_CALIBRATION_CONTINUOUS_SPEEDS",
            "TRACKER_CALIBRATION_CONTINUOUS_DURATION",
            "TRACKER_CALIBRATION_ONVIF_BENCHMARK",
        ):
            self.assertIn(name, TRACKER)
            self.assertIn(name, COMPOSE)


if __name__ == "__main__":
    unittest.main()
