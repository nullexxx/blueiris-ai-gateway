import ast
from pathlib import Path
import unittest


class HybridChaseContractTests(unittest.TestCase):
    def test_axis_specific_speed_calls_are_explicit(self):
        source = Path("tracker.py").read_text()
        tree = ast.parse(source)
        legacy_bad = []
        servo_axes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_hybrid_axis_speed":
                if len(node.args) < 2:
                    legacy_bad.append(node.lineno)
            if isinstance(func, ast.Attribute) and func.attr == "_servo_axis_command":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    servo_axes.append(node.args[1].value)

        self.assertEqual(
            legacy_bad,
            [],
            f"_hybrid_axis_speed calls missing explicit axis argument at lines {legacy_bad}",
        )
        # Rev 5 may retire the legacy speed helper entirely. The active feedback
        # servo must still make both pan and tilt axis selection explicit.
        self.assertIn("pan", servo_axes)
        self.assertIn("tilt", servo_axes)


if __name__ == "__main__":
    unittest.main()
