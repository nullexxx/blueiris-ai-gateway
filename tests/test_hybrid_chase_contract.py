import ast
from pathlib import Path
import unittest


class HybridChaseContractTests(unittest.TestCase):
    def test_hybrid_axis_speed_calls_include_axis(self):
        source = Path("tracker.py").read_text()
        tree = ast.parse(source)
        bad_calls = []
        total_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_hybrid_axis_speed":
                total_calls += 1
                if len(node.args) < 2:
                    bad_calls.append(node.lineno)
        self.assertGreater(total_calls, 0, "tracker.py contains no _hybrid_axis_speed calls")
        self.assertEqual(
            bad_calls,
            [],
            f"_hybrid_axis_speed calls missing explicit axis argument at lines {bad_calls}",
        )


if __name__ == "__main__":
    unittest.main()
