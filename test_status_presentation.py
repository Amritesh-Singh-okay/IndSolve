"""
IndSolve — Unit Test for Status Presentation & UI Mapping
Validates that raw solver statuses are mapped to standard honest descriptions and semantic tokens.
"""

import unittest
from ui.components import format_solver_status


class TestStatusPresentation(unittest.TestCase):
    def test_optimal_status_mapping(self):
        """Validates that OPTIMAL with success=True maps to 'Optimal solution found' with green/optimal tokens."""
        s = format_solver_status("OPTIMAL", True)
        self.assertEqual(s["label"], "Optimal solution found")
        self.assertTrue(s["is_optimal"])
        self.assertEqual(s["icon"], "🟢")

    def test_optimal_status_with_success_false_is_not_optimal(self):
        """Validates that OPTIMAL without success is never treated as optimal."""
        s = format_solver_status("OPTIMAL", False)
        self.assertFalse(s["is_optimal"])
        self.assertNotEqual(s["label"], "Optimal solution found")

    def test_infeasible_status_mapping(self):
        """Validates that INFEASIBLE maps to 'No feasible solution'."""
        s = format_solver_status("INFEASIBLE", False)
        self.assertEqual(s["label"], "No feasible solution")
        self.assertFalse(s["is_optimal"])
        self.assertEqual(s["icon"], "🔴")

    def test_unbounded_status_mapping(self):
        """Validates that UNBOUNDED maps to 'Objective is unbounded'."""
        s = format_solver_status("UNBOUNDED", False)
        self.assertEqual(s["label"], "Objective is unbounded")
        self.assertFalse(s["is_optimal"])
        self.assertEqual(s["icon"], "🔴")

    def test_iteration_limit_status_mapping(self):
        """Validates that ITERATION_LIMIT and MAX_ITER map to 'Solve stopped before optimality was proven'."""
        s1 = format_solver_status("ITERATION_LIMIT", False)
        self.assertEqual(s1["label"], "Solve stopped before optimality was proven")
        self.assertFalse(s1["is_optimal"])
        self.assertEqual(s1["icon"], "⚠️")

        s2 = format_solver_status("MAX_ITER", False)
        self.assertEqual(s2["label"], "Solve stopped before optimality was proven")
        self.assertFalse(s2["is_optimal"])

    def test_node_limit_status_mapping(self):
        """Validates that NODE_LIMIT maps to 'Node limit reached before optimality was proven'."""
        s = format_solver_status("NODE_LIMIT_FEASIBLE", False)
        self.assertEqual(s["label"], "Node limit reached before optimality was proven")
        self.assertFalse(s["is_optimal"])


if __name__ == "__main__":
    unittest.main()
