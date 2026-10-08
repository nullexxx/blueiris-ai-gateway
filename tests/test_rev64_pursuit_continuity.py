import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()

def load_functions(*names):
    tree = ast.parse(TRACKER)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, "tracker.py", "exec"), ns)
    return [ns[name] for name in names]

def block(start, end):
    a=TRACKER.index(start); b=TRACKER.index(end,a+len(start)); return TRACKER[a:b]

class Rev64PursuitContinuityTests(unittest.TestCase):
    def test_outward_catchup_only_when_error_grows(self):
        large, catchup = load_functions("_servo_large_error_gain", "_servo_outward_catchup_gain")
        kw=dict(start_error=0.60, full_error=0.85, max_gain=1.30)
        self.assertEqual(catchup(0.80,-0.20,**kw),1.0)
        self.assertEqual(catchup(-0.80,0.20,**kw),1.0)
        self.assertGreater(catchup(0.80,0.20,**kw),1.0)
        self.assertAlmostEqual(catchup(-0.85,-0.20,**kw),1.30,places=3)

    def test_coast_scale_decays_without_overshooting_bounds(self):
        coast, = load_functions("_servo_coast_scale")
        kw=dict(duration_s=0.40,start_scale=0.70,end_scale=0.20)
        self.assertAlmostEqual(coast(0.0,**kw),0.70,places=3)
        self.assertAlmostEqual(coast(0.20,**kw),0.45,places=3)
        self.assertAlmostEqual(coast(0.40,**kw),0.20,places=3)
        self.assertAlmostEqual(coast(1.0,**kw),0.20,places=3)

    def test_confidence_grace_coasts_and_floor_stops(self):
        chase=block("    async def _drive_hybrid_chase(","    def _begin_ptz_operation")
        self.assertIn("self.target.confidence < self._chase_detection_conf",chase)
        self.assertIn('_stop_hybrid_chase("confidence_floor"',chase)
        self.assertIn("self._hybrid_confidence_coast_velocity",chase)
        self.assertIn("_servo_coast_scale(",chase)
        self.assertNotIn('_pause_hybrid_actuator("confidence_grace")',chase)

    def test_detector_miss_decays_snapshot_velocity(self):
        process=block("    async def _process_observation(","    def _associate(")
        self.assertIn('"hybrid_missing_coast"',process)
        self.assertIn("self._hybrid_missing_coast_velocity",process)
        self.assertIn("_servo_coast_scale(",process)

    def test_servo_zero_command_enters_hold(self):
        chase=block("    async def _drive_hybrid_chase(","    def _begin_ptz_operation")
        self.assertIn('"servo_hold_enter"',chase)
        self.assertIn('"servo_hold_resume"',chase)
        self.assertIn('_stop_hybrid_chase("servo_hold_settled"',chase)
        self.assertNotIn('_stop_hybrid_chase("servo_brake"',chase)

    def test_status_identifies_rev64(self):
        self.assertIn('"controller_patch": "6.7.3"',TRACKER)
        for token in ['"outward_catchup_gain"','"hold_seconds"','"hold_active"','"confidence_coast_start_scale"']:
            self.assertIn(token,TRACKER)

if __name__ == "__main__":
    unittest.main()
