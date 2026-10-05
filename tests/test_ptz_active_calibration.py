import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from ptz_active_calibration import ActivePtzCalibrationStore, estimate_static_frame_shift


class ActivePtzCalibrationTests(unittest.TestCase):
    def test_store_persists_motion_calibration_and_freshness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active.json"
            store = ActivePtzCalibrationStore(str(path), "camera-a")
            self.assertFalse(store.has_motion_calibration())
            store.replace_motion_calibration(
                move_directly={"samples": [{"zoom_factor": 1.0}]},
                continuous={
                    "samples": [{"axis": "pan", "speed": 1}],
                    "signs": {"pan": -1, "tilt": -1},
                    "by_zoom": {"1.00": {"pan": {"1": 0.2}, "tilt": {"1": 0.15}}},
                },
                onvif_benchmark={"available": True},
            )
            self.assertTrue(store.has_motion_calibration())
            self.assertTrue(store.is_fresh(1.0))
            reloaded = ActivePtzCalibrationStore(str(path), "camera-a")
            self.assertTrue(reloaded.has_motion_calibration())
            self.assertEqual(reloaded.continuous_sign("pan", 1), -1)
            self.assertTrue(reloaded.public_dict()["onvif_benchmark"]["available"])

    def test_continuous_speed_selection_uses_measured_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ActivePtzCalibrationStore(str(Path(tmp) / "active.json"), "camera-a")
            store.replace_motion_calibration(
                move_directly={"samples": [{"ok": True}]},
                continuous={
                    "samples": [{"ok": True}],
                    "signs": {"pan": -1, "tilt": -1},
                    "by_zoom": {
                        "1.00": {
                            "pan": {"1": 0.10, "3": 0.32, "6": 0.70},
                            "tilt": {"1": 0.08, "3": 0.25, "6": 0.55},
                        }
                    },
                },
            )
            low = store.choose_continuous_speed(
                "pan", 0.53, 1.0,
                min_speed=1, max_speed=6, exit_error=0.50, full_speed_error=0.90,
            )
            high = store.choose_continuous_speed(
                "pan", 0.90, 1.0,
                min_speed=1, max_speed=6, exit_error=0.50, full_speed_error=0.90,
            )
            self.assertEqual(low, 1)
            self.assertEqual(high, 6)

    def test_static_frame_shift_finds_synthetic_translation(self):
        rng = np.random.default_rng(123)
        before = np.zeros((360, 480, 3), dtype=np.uint8)
        for _ in range(80):
            x = int(rng.integers(20, 460))
            y = int(rng.integers(20, 340))
            radius = int(rng.integers(2, 7))
            value = int(rng.integers(100, 255))
            cv2.circle(before, (x, y), radius, (value, value, value), -1)
        matrix = np.float32([[1, 0, 24], [0, 1, -15]])
        after = cv2.warpAffine(before, matrix, (480, 360))
        shift = estimate_static_frame_shift(before, after)
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(shift.dx, 24.0, delta=4.0)
        self.assertAlmostEqual(shift.dy, -15.0, delta=4.0)
        nx, ny = shift.normalized(before.shape)
        self.assertAlmostEqual(nx, 24.0 / 240.0, delta=0.03)
        self.assertAlmostEqual(ny, -15.0 / 180.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
