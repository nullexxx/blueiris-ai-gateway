import ast
from pathlib import Path
import unittest

from ptz_phase2 import BBoxMotionValidator


ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()


class Rev4ControllerTests(unittest.TestCase):
    def test_velocity_maturity_is_not_rewritten_on_every_subwindow_frame(self):
        tree = ast.parse(TRACKER)
        target = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TargetTrack"
        )
        update = next(
            node for node in target.body
            if isinstance(node, ast.FunctionDef) and node.name == "update"
        )
        source = ast.get_source_segment(TRACKER, update) or ""
        self.assertIn("velocity_estimate_time", source)
        self.assertIn("preserve_velocity", source)
        # There should be one assignment after a mature sample, not the Rev 3
        # pre-threshold assignment that toggled maturity every frame.
        self.assertEqual(source.count("self.velocity_sample_ms = int(sample_s * 1000)"), 1)

    def test_rev4_has_stationary_dwell_and_meaningful_precision_threshold(self):
        self.assertIn("TRACKER_PRECISION_MIN_ERROR", TRACKER)
        self.assertIn("TRACKER_PRECISION_SETTLE_TIME", TRACKER)
        self.assertIn('"precision_min_error"', TRACKER)
        self.assertIn('"precision_settle_s"', TRACKER)
        self.assertIn('"stationary_settle"', TRACKER)
        self.assertIn('"precision_deadband"', TRACKER)

    def test_rev4_uses_low_confidence_only_for_target_retention(self):
        self.assertIn("TRACKER_RETENTION_DETECTION_CONF", TRACKER)
        self.assertIn("TRACKER_CHASE_DETECTION_CONF", TRACKER)
        self.assertIn("inference_conf = self.cfg.hold_conf", TRACKER)
        self.assertIn('self.state in ("COAST", "REACQUIRE", "PTZ_MOVING", "PTZ_SETTLING")', TRACKER)
        self.assertIn('self.state = "TARGET_RETENTION"', TRACKER)
        self.assertIn("self.target.confidence < self.cfg.hold_conf", TRACKER)
        # New acquisition remains independently gated at acquire_conf.
        self.assertIn("d.confidence >= self.cfg.acquire_conf", TRACKER)

    def test_axis_reversal_no_longer_ends_whole_chase(self):
        start = TRACKER.index("    async def _set_hybrid_chase_speed")
        end = TRACKER.index("    async def _drive_hybrid_chase", start)
        block = TRACKER[start:end]
        self.assertIn("hybrid_axis_reversal_suppressed", block)
        self.assertIn("hybrid_actuator_pause", TRACKER)
        self.assertIn("command_pan, command_tilt = desired", block)
        self.assertIn("self.ptz.continuous_move,\n            command_pan,\n            command_tilt,", block)
        self.assertNotIn('_stop_hybrid_chase("direction_reversal"', block)

    def test_bbox_scale_change_with_coherent_translation_is_accepted(self):
        validator = BBoxMotionValidator()
        shape = (480, 640, 3)
        validator.reset((100, 100, 200, 300), 1.0)
        # Width grows 30% in 100 ms, but both x edges translate strongly in
        # the same direction. This is a plausible walking/perspective sample.
        result = validator.validate((120, 100, 250, 300), 1.1, shape)
        self.assertTrue(result.valid, result.reason)

    def test_extreme_bbox_deformation_is_still_rejected(self):
        validator = BBoxMotionValidator()
        shape = (480, 640, 3)
        validator.reset((200, 150, 300, 250), 1.0)
        result = validator.validate((20, 150, 610, 250), 1.1, shape)
        self.assertFalse(result.valid)

    def test_rev6_status_contract(self):
        self.assertIn('"controller_revision": 6', TRACKER)
        self.assertIn('"latency_aware_fast_handoff_continuity_frozen_loss_clock"', TRACKER)
        self.assertIn('"missing_grace_s"', TRACKER)
        self.assertIn('"velocity_ttl_s"', TRACKER)
        self.assertIn('"axis_reverse_holdoff_s"', TRACKER)
        self.assertIn('"servo": {', TRACKER)
        self.assertIn('"semantic_continuity": {', TRACKER)


if __name__ == "__main__":
    unittest.main()
