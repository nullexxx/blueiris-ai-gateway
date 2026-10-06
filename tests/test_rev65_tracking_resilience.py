import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRACKER = (ROOT / "tracker.py").read_text()

def load_functions(*names):
    tree=ast.parse(TRACKER)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=nodes,type_ignores=[])
    ast.fix_missing_locations(module)
    ns={}
    exec(compile(module,"tracker.py","exec"),ns)
    return [ns[name] for name in names]

def block(start,end):
    a=TRACKER.index(start); b=TRACKER.index(end,a+len(start)); return TRACKER[a:b]

class Rev65TrackingResilienceTests(unittest.TestCase):
    def test_axis_divergence_resets_on_crossing_and_requires_correction(self):
        fn,=load_functions("_servo_axis_divergence_count")
        self.assertEqual(fn(0.40,-0.45,-0.30,2,0.05),0)
        self.assertEqual(fn(0.40,0.52,0.0,2,0.05),0)
        self.assertEqual(fn(0.40,0.52,0.30,2,0.05),3)
        self.assertEqual(fn(-0.40,-0.52,-0.30,1,0.05),2)

    def test_servo_hold_settles_only_inside_exit_region(self):
        fn,=load_functions("_servo_hold_action")
        kw=dict(exit_error=0.22,resume_growth=0.06,elapsed_s=0.31,hold_seconds=0.30,frames=3,min_frames=3)
        self.assertEqual(fn(0.39,0.05,-0.36,-0.05,**kw),"resume_center_cross")
        self.assertEqual(fn(0.30,0.10,0.28,0.10,**kw),"resume_outside_exit")
        self.assertEqual(fn(0.20,0.10,0.18,0.10,**kw),"settled")

    def test_native_rescue_matches_fast_dog_case_but_not_closing_target(self):
        requested,speed=load_functions("_native_edge_rescue_axis_requested","_native_edge_rescue_speed")
        kw=dict(max_velocity=0.35,start_error=0.75,saturation=0.80,full_error=0.92)
        self.assertTrue(requested(-0.759,-1.0,-0.90,-0.312,**kw))
        self.assertFalse(requested(-0.759,0.50,-0.60,-0.312,**kw))
        self.assertFalse(requested(-0.60,-1.0,-0.90,-0.312,**kw))
        self.assertEqual(speed(-0.759,start_error=0.65,full_error=0.92,min_speed=3,max_speed=6),-3)
        self.assertEqual(speed(0.94,start_error=0.65,full_error=0.92,min_speed=3,max_speed=6),6)

    def test_drive_has_bounded_native_rescue_and_per_axis_divergence(self):
        chase=block("    async def _drive_hybrid_chase(","    def _begin_ptz_operation")
        self.assertIn("_servo_axis_divergence_count(",chase)
        self.assertIn('"axis_counts"',chase)
        self.assertIn('"native_edge_rescue_enter"',chase)
        self.assertIn('"native_edge_rescue_exit"',chase)
        self.assertIn("self._native_edge_rescue_cooldown_until",chase)

    def test_association_diagnostics_expose_nonmatching_candidates(self):
        associate=block("    def _associate(","    async def _drive_to_target")
        for token in ['"nearest_any"','"rejected_candidates"','"class_compatible"','"same_class"']:
            self.assertIn(token,associate)

    def test_status_identifies_rev65(self):
        self.assertIn('"controller_patch": "6.5"',TRACKER)
        self.assertIn('"axis_divergence_counts"',TRACKER)
        self.assertIn('"native_edge_rescue"',TRACKER)

if __name__=="__main__":
    unittest.main()
