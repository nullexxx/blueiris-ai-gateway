import tempfile
from pathlib import Path
import unittest

from ptz_phase2 import (
    AcquisitionZonePolicy,
    BBoxMotionValidator,
    OnvifRetryState,
    SceneStabilityGate,
    ZoomCalibrationMap,
)


class DummyMotion:
    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy


class PtzPhase2Behavior(unittest.TestCase):
    def test_acquisition_required_and_ignore_zones(self):
        policy = AcquisitionZonePolicy(
            '[[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]]',
            '[[[0.4,0.4],[0.6,0.4],[0.6,0.6],[0.4,0.6]]]',
        )
        shape = (100, 100, 3)
        self.assertTrue(policy.allows((20, 20), shape))
        self.assertFalse(policy.allows((50, 50), shape))
        self.assertFalse(policy.allows((5, 5), shape))
        status = policy.public_dict()
        self.assertEqual(status["rejected_ignore"], 1)
        self.assertEqual(status["rejected_required"], 1)

    def test_scene_stability_requires_consecutive_good_frames(self):
        gate = SceneStabilityGate(stable_frames=2, max_wait_s=0.6, flow_threshold_norm=0.01)
        gate.reset(10.0)
        shape = (100, 100, 3)
        self.assertFalse(gate.observe(10.1, DummyMotion(5.0, 0.0), True, shape))
        self.assertFalse(gate.observe(10.2, DummyMotion(0.2, 0.2), True, shape))
        self.assertTrue(gate.observe(10.3, DummyMotion(0.1, 0.1), True, shape))
        self.assertEqual(gate.last_reason, "stable")
        self.assertFalse(gate.timed_out)

    def test_scene_stability_frame_at_timeout_boundary_can_complete_stability(self):
        gate = SceneStabilityGate(stable_frames=2, max_wait_s=0.2, flow_threshold_norm=0.01)
        gate.reset(1.0)
        shape = (100, 100, 3)
        self.assertFalse(gate.observe(1.1, DummyMotion(0.1, 0.1), True, shape))
        self.assertTrue(gate.observe(1.21, DummyMotion(0.1, 0.1), True, shape))
        self.assertEqual(gate.last_reason, "stable")
        self.assertFalse(gate.timed_out)

    def test_scene_stability_has_bounded_timeout(self):
        gate = SceneStabilityGate(stable_frames=3, max_wait_s=0.2, flow_threshold_norm=0.001)
        gate.reset(1.0)
        self.assertTrue(gate.observe(1.25, DummyMotion(50.0, 50.0), False, (100, 100, 3)))
        self.assertTrue(gate.timed_out)
        self.assertEqual(gate.last_reason, "bounded_timeout")

    def test_bbox_geometry_rejects_detector_box_deformation(self):
        validator = BBoxMotionValidator()
        shape = (480, 640, 3)
        validator.reset((200, 150, 300, 250), 1.0)
        result = validator.validate((20, 150, 610, 250), 1.1, shape)
        self.assertFalse(result.valid)
        self.assertIn(
            result.reason,
            {"edge_velocity_magnitude", "edge_velocity_disagreement", "bbox_scale_jump"},
        )

    def test_bbox_geometry_accepts_coherent_translation(self):
        validator = BBoxMotionValidator()
        shape = (480, 640, 3)
        validator.reset((200, 150, 300, 250), 1.0)
        result = validator.validate((205, 153, 305, 253), 1.1, shape)
        self.assertTrue(result.valid)
        self.assertIsNone(result.reason)

    def test_onvif_retry_backoff_and_reset(self):
        state = OnvifRetryState(base_s=10.0, max_s=40.0)
        self.assertEqual(state.failure(100.0, "one"), 10.0)
        self.assertFalse(state.due(109.9))
        self.assertTrue(state.due(110.0))
        self.assertEqual(state.failure(110.0, "two"), 20.0)
        self.assertEqual(state.failure(130.0, "three"), 40.0)
        self.assertEqual(state.failure(170.0, "four"), 40.0)
        state.success()
        self.assertEqual(state.failures, 0)
        self.assertFalse(state.due(1000.0))

    def test_zoom_calibration_interpolates_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zoom.json"
            mapping = ZoomCalibrationMap(str(path), "camera-a")
            mapping.record(1.0, 0.0)
            mapping.record(2.0, 0.10)
            mapping.record(3.0, 0.30)
            self.assertAlmostEqual(mapping.estimate_normalized(1.5, 25.0), 0.05, places=6)
            self.assertAlmostEqual(mapping.estimate_normalized(2.5, 25.0), 0.20, places=6)

            reloaded = ZoomCalibrationMap(str(path), "camera-a")
            self.assertEqual(len(reloaded.points()), 3)
            self.assertAlmostEqual(reloaded.estimate_normalized(2.5, 25.0), 0.20, places=6)

    def test_zoom_calibration_wide_anchor_survives_deadband_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zoom.json"
            mapping = ZoomCalibrationMap(str(path), "camera-a")
            mapping.record(1.0, 0.0)
            mapping.record(1.0, 0.010417)
            mapping.record(1.0, 0.020833)
            mapping.record(1.0, 0.041667)
            points = mapping.points()
            self.assertEqual(points[0], (1.0, 0.0))
            self.assertAlmostEqual(mapping.estimate_normalized(1.0, 25.0), 0.0, places=6)

    def test_zoom_calibration_replace_collapses_plateaus_monotonically(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = ZoomCalibrationMap(str(Path(tmp) / "zoom.json"), "camera-a")
            mapping.replace_points([
                (1.0, 0.0),
                (1.0, 0.02),
                (1.0, 0.04),
                (1.5, 0.0625),
                (2.0, 0.085),
                (2.5, 0.105),
                (3.1, 0.125),
            ])
            points = mapping.points()
            self.assertEqual(points[0], (1.0, 0.0))
            self.assertGreaterEqual(len(points), 5)
            self.assertTrue(all(a[1] <= b[1] for a, b in zip(points, points[1:])))
            self.assertGreater(mapping.estimate_normalized(2.0, 25.0), 0.06)
            self.assertLess(mapping.estimate_normalized(2.0, 25.0), 0.11)

    def test_zoom_calibration_fallback_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = ZoomCalibrationMap(str(Path(tmp) / "zoom.json"), "camera-a")
            self.assertAlmostEqual(mapping.estimate_normalized(1.0, 25.0), 0.0)
            self.assertAlmostEqual(mapping.estimate_normalized(25.0, 25.0), 1.0)
            self.assertEqual(mapping.estimate_normalized(100.0, 25.0), 1.0)


if __name__ == "__main__":
    unittest.main()
