"""
IndSolve — Regression Tests for External Reference Verification Logic
Guarantees that verification state is NEVER falsely claimed when SciPy reference solver
fails, errors, throws exceptions, or produces mismatches.
"""

import sys
import os
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from solver.tableau_simplex import SimplexSolver, SimplexResult
from solver.branch_and_bound import BranchAndBoundSolver, MILPResult
from solver.verify_reference import verify_with_scipy_reference


class TestReferenceVerification(unittest.TestCase):

    def setUp(self):
        self.lp_solver = SimplexSolver(tol=1e-7)
        self.milp_solver = BranchAndBoundSolver(lp_solver=self.lp_solver)

    def test_lp_exact_match_verified(self):
        """Standard feasible LP with SciPy agreement must return 'verified'."""
        c = np.array([3.0, 5.0])
        A_ub = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
        b_ub = np.array([4.0, 12.0, 18.0])
        res = self.lp_solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, maximize=True)

        v = verify_with_scipy_reference(
            ind_res=res, c=c, A_ub=A_ub, b_ub=b_ub, maximize=True
        )
        self.assertEqual(v["state"], "verified")
        self.assertIsNotNone(v["ref_obj"])
        self.assertAlmostEqual(v["ref_obj"], 36.0, places=2)
        self.assertIn("Independently Confirmed", v["verdict_label"])

    def test_milp_exact_match_verified(self):
        """Standard integer MILP with SciPy milp agreement must return 'verified'."""
        c = np.array([10.0, 15.0, 40.0])
        A_ub = np.array([[1.0, 2.0, 3.0]])
        b_ub = np.array([4.0])
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        integrality = [1, 1, 1]
        res = self.milp_solver.solve(
            c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality, maximize=True
        )

        v = verify_with_scipy_reference(
            ind_res=res, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality, maximize=True
        )
        self.assertEqual(v["state"], "verified")
        self.assertIsNotNone(v["ref_obj"])
        self.assertAlmostEqual(v["ref_obj"], 50.0, places=2)

    def test_infeasible_status_match_without_arithmetic(self):
        """Infeasible models must compare status without attempting float arithmetic."""
        c = np.array([1.0])
        A_ub = np.array([[1.0]])
        b_ub = np.array([2.0])
        A_ge = np.array([[1.0]])
        b_ge = np.array([5.0])
        res = self.lp_solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge)

        v = verify_with_scipy_reference(
            ind_res=res, c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge
        )
        self.assertEqual(v["state"], "verified")
        self.assertEqual(v["ref_status"], "INFEASIBLE")
        self.assertIsNone(v["ref_obj"])
        self.assertIn("Infeasibility Confirmed", v["verdict_label"])

    def test_unbounded_status_match(self):
        """Unbounded models must confirm unboundedness without arithmetic errors."""
        c = np.array([1.0, 0.0])
        A_ub = np.array([[1.0, -1.0]])
        b_ub = np.array([5.0])
        res = self.lp_solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, maximize=True)

        v = verify_with_scipy_reference(
            ind_res=res, c=c, A_ub=A_ub, b_ub=b_ub, maximize=True
        )
        self.assertEqual(v["state"], "verified")
        self.assertEqual(v["ref_status"], "UNBOUNDED")
        self.assertIsNone(v["ref_obj"])

    def test_objective_mismatch_detection(self):
        """If IndSolve result differs from SciPy, must flag 'mismatch'."""
        c = np.array([1.0, 1.0])
        A_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([10.0])

        # Create a fake mismatch result
        fake_res = SimplexResult(
            success=True, status="OPTIMAL", message="fake", fun=999.0,
            x=np.array([5.0, 5.0]), nit=1, solve_time=0.001, history=[]
        )

        v = verify_with_scipy_reference(
            ind_res=fake_res, c=c, A_ub=A_ub, b_ub=b_ub, maximize=False
        )
        self.assertEqual(v["state"], "mismatch")
        self.assertNotEqual(v["state"], "verified")
        self.assertIn("Mismatch", v["verdict_label"])

    @patch("solver.verify_reference.linprog")
    def test_forced_linprog_exception_never_verified(self, mock_linprog):
        """Regression test: When linprog throws an exception, state MUST BE 'reference_failed', never verified."""
        mock_linprog.side_effect = RuntimeError("Simulated C++ reference crash / memory fault")

        c = np.array([3.0, 5.0])
        A_ub = np.array([[1.0, 0.0]])
        b_ub = np.array([4.0])
        res = self.lp_solver.solve(c=c, A_ub=A_ub, b_ub=b_ub, maximize=True)

        v = verify_with_scipy_reference(
            ind_res=res, c=c, A_ub=A_ub, b_ub=b_ub, maximize=True
        )

        # STRICT ASSERTIONS:
        self.assertNotEqual(v["state"], "verified", "CRITICAL ERROR: Exception resulted in false 'verified' claim!")
        self.assertEqual(v["state"], "reference_failed")
        self.assertIsNone(v["ref_obj"])
        self.assertEqual(v["message"], "Reference verification unavailable — solver result is not independently confirmed.")

    @patch("solver.verify_reference.milp")
    def test_forced_milp_exception_never_verified(self, mock_milp):
        """Regression test: When milp throws an exception, state MUST BE 'reference_failed', never verified."""
        mock_milp.side_effect = Exception("Simulated SciPy MILP backend failure")

        c = np.array([10.0, 20.0])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        integrality = [1, 1]
        res = self.milp_solver.solve(c=c, bounds=bounds, integrality=integrality, maximize=True)

        v = verify_with_scipy_reference(
            ind_res=res, c=c, bounds=bounds, integrality=integrality, maximize=True
        )

        # STRICT ASSERTIONS:
        self.assertNotEqual(v["state"], "verified", "CRITICAL ERROR: Exception resulted in false 'verified' claim!")
        self.assertEqual(v["state"], "reference_failed")
        self.assertIsNone(v["ref_obj"])
        self.assertEqual(v["message"], "Reference verification unavailable — solver result is not independently confirmed.")


if __name__ == "__main__":
    unittest.main()
