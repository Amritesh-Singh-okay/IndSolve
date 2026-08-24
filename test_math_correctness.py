"""
IndSolve — Mathematical Correctness & Bound Transformation Regression Suite
Tests free-variable splitting, one-sided/two-sided/fixed bound transformations,
iteration limits, and honest propagation of inconclusive child LP statuses in Branch-and-Bound.
All bounded test problems are independently compared against SciPy HiGHS.
"""

import sys
import os
import unittest
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(__file__))

from solver.tableau_simplex import SimplexSolver, SimplexResult
from solver.branch_and_bound import BranchAndBoundSolver, MILPResult


class TestMathematicalCorrectnessAndBounds(unittest.TestCase):

    def setUp(self):
        self.solver = SimplexSolver(tol=1e-8, max_iter=5000)

    # -------------------------------------------------------------------------
    # 1. ITERATION LIMIT TESTS
    # -------------------------------------------------------------------------
    def test_max_iter_zero_returns_iteration_limit(self):
        """A solve with max_iter=0 must return success=False, status='ITERATION_LIMIT'."""
        solver_zero = SimplexSolver(max_iter=0)
        c = np.array([3.0, 5.0])
        A_ub = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
        b_ub = np.array([4.0, 12.0, 18.0])

        res = solver_zero.solve(c=c, A_ub=A_ub, b_ub=b_ub, maximize=True)
        self.assertFalse(res.success, "max_iter=0 must not claim success")
        self.assertEqual(res.status, "ITERATION_LIMIT")
        self.assertEqual(res.nit, 0)

    def test_branch_and_bound_lp_iteration_limit_propagation(self):
        """B&B encountering child LP with max_iter=0 must not prune as infeasible."""
        lp_limited = SimplexSolver(max_iter=0)
        milp_solver = BranchAndBoundSolver(lp_solver=lp_limited)

        c = np.array([10.0, 20.0])
        A_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([5.0])
        bounds = [(0.0, 5.0), (0.0, 5.0)]
        integrality = [1, 1]

        res = milp_solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality, maximize=True)
        self.assertFalse(res.success, "Unresolved LP limit in B&B must not succeed")
        self.assertEqual(res.status, "ITERATION_LIMIT")

    # -------------------------------------------------------------------------
    # 2. FREE VARIABLE TESTS
    # -------------------------------------------------------------------------
    def test_free_variable_unconstrained_minimization_unbounded(self):
        """minimize x, with x free and no constraints, must return UNBOUNDED, not OPTIMAL x=0."""
        c = np.array([1.0])
        bounds = [(None, None)]  # Fully free variable

        res = self.solver.solve(c=c, bounds=bounds, maximize=False)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "UNBOUNDED")

        # Independent comparison with SciPy HiGHS
        scipy_res = linprog(c=c, bounds=bounds, method="highs")
        self.assertEqual(scipy_res.status, 3, "SciPy HiGHS confirms problem is unbounded (status 3)")

    def test_free_variable_unconstrained_maximization_unbounded(self):
        """maximize x, with x free and no constraints, must return UNBOUNDED."""
        c = np.array([1.0])
        bounds = [(None, None)]

        res = self.solver.solve(c=c, bounds=bounds, maximize=True)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "UNBOUNDED")

    def test_free_variables_constrained_finite_optimum(self):
        """
        minimize 3*x1 + 2*x2
        subject to:
            x1 + x2 >= 5
            2*x1 - x2 >= 1
            x1 <= 6
            x1, x2 in (-inf, +inf) [Free Variables]
        """
        c = np.array([3.0, 2.0])
        A_ge = np.array([[1.0, 1.0], [2.0, -1.0]])
        b_ge = np.array([5.0, 1.0])
        A_ub = np.array([[1.0, 0.0]])
        b_ub = np.array([6.0])
        bounds = [(None, None), (None, None)]

        res = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge, bounds=bounds, maximize=False)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OPTIMAL")

        # SciPy HiGHS ground truth check
        scipy_A_ub = np.vstack([A_ub, -A_ge])
        scipy_b_ub = np.concatenate([b_ub, -b_ge])
        scipy_res = linprog(c=c, A_ub=scipy_A_ub, b_ub=scipy_b_ub, bounds=bounds, method="highs")

        self.assertTrue(scipy_res.success)
        self.assertAlmostEqual(res.fun, scipy_res.fun, places=5)
        np.testing.assert_allclose(res.x, scipy_res.x, atol=1e-5)

    # -------------------------------------------------------------------------
    # 3. BOUND COMBINATIONS (LOWER-ONLY, UPPER-ONLY, TWO-SIDED, FIXED)
    # -------------------------------------------------------------------------
    def test_lower_only_negative_bound(self):
        """
        minimize 3*x1 + 4*x2
        subject to:
            x1 + x2 >= 2
            x1 >= -5 (Lower-only), x2 >= 0
        """
        c = np.array([3.0, 4.0])
        A_ge = np.array([[1.0, 1.0]])
        b_ge = np.array([2.0])
        bounds = [(-5.0, None), (0.0, None)]

        res = self.solver.solve(c=c, A_ge=A_ge, b_ge=b_ge, bounds=bounds, maximize=False)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OPTIMAL")

        scipy_res = linprog(c=c, A_ub=-A_ge, b_ub=-b_ge, bounds=bounds, method="highs")
        self.assertAlmostEqual(res.fun, scipy_res.fun, places=5)
        np.testing.assert_allclose(res.x, scipy_res.x, atol=1e-5)

    def test_upper_only_bound(self):
        """
        maximize 2*x1 - x2
        subject to:
            x1 + x2 <= 6
            x1 <= 4 (Upper-only, no lower bound: x1 in (-inf, 4])
            x2 >= 0
        """
        c = np.array([2.0, -1.0])
        A_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([6.0])
        bounds = [(None, 4.0), (0.0, None)]

        res = self.solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, maximize=True)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OPTIMAL")

        scipy_res = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        self.assertAlmostEqual(res.fun, -scipy_res.fun, places=5)
        np.testing.assert_allclose(res.x, scipy_res.x, atol=1e-5)

    def test_two_sided_negative_to_positive_bounds(self):
        """
        minimize x1 + 2*x2 + 3*x3
        subject to:
            x1 + x2 + x3 >= 5
            -3 <= x1 <= 2
            -1 <= x2 <= 4
             0 <= x3 <= 10
        """
        c = np.array([1.0, 2.0, 3.0])
        A_ge = np.array([[1.0, 1.0, 1.0]])
        b_ge = np.array([5.0])
        bounds = [(-3.0, 2.0), (-1.0, 4.0), (0.0, 10.0)]

        res = self.solver.solve(c=c, A_ge=A_ge, b_ge=b_ge, bounds=bounds, maximize=False)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OPTIMAL")

        scipy_res = linprog(c=c, A_ub=-A_ge, b_ub=-b_ge, bounds=bounds, method="highs")
        self.assertAlmostEqual(res.fun, scipy_res.fun, places=5)
        np.testing.assert_allclose(res.x, scipy_res.x, atol=1e-5)

    def test_fixed_variable_bound(self):
        """
        minimize 5*x1 + 2*x2
        subject to:
            x1 + x2 >= 10
            x1 == 4 (Fixed: lb=4, ub=4)
            x2 >= 0
        """
        c = np.array([5.0, 2.0])
        A_ge = np.array([[1.0, 1.0]])
        b_ge = np.array([10.0])
        bounds = [(4.0, 4.0), (0.0, None)]

        res = self.solver.solve(c=c, A_ge=A_ge, b_ge=b_ge, bounds=bounds, maximize=False)
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OPTIMAL")
        self.assertAlmostEqual(res.x[0], 4.0, places=5)
        self.assertAlmostEqual(res.x[1], 6.0, places=5)
        self.assertAlmostEqual(res.fun, 5.0*4.0 + 2.0*6.0, places=5)

        scipy_res = linprog(c=c, A_ub=-A_ge, b_ub=-b_ge, bounds=bounds, method="highs")
        self.assertAlmostEqual(res.fun, scipy_res.fun, places=5)
        np.testing.assert_allclose(res.x, scipy_res.x, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
