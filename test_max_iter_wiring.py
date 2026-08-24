"""
IndSolve — Unit Test for Max Simplex Pivots Setting Wiring
Validates that max_iter properly propagates and halts optimization with ITERATION_LIMIT status.
"""

import unittest
import numpy as np

from solver.tableau_simplex import SimplexSolver
from solver.branch_and_bound import BranchAndBoundSolver
from solver.refinery_simulation import run_what_if_simulation
from solver.problems import get_preloaded_problems


class TestMaxIterWiring(unittest.TestCase):
    def test_simplex_solver_max_iter_zero(self):
        """Validates that SimplexSolver with max_iter=0 returns ITERATION_LIMIT rather than OPTIMAL."""
        c = np.array([70.0, 65.0])
        A_ub = np.array([[1.0, 0.0], [0.0, 1.0]])
        b_ub = np.array([500.0, 600.0])
        A_ge = np.array([[1.0, 1.0]])
        b_ge = np.array([800.0])

        solver_zero = SimplexSolver(tol=1e-7, max_iter=0)
        res_zero = solver_zero.solve(c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge, maximize=False)

        self.assertFalse(res_zero.success)
        self.assertIn(res_zero.status, ["ITERATION_LIMIT", "MAX_ITER"])
        self.assertEqual(res_zero.nit, 0)

        # Contrast with standard max_iter
        solver_normal = SimplexSolver(tol=1e-7, max_iter=5000)
        res_normal = solver_normal.solve(c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge, maximize=False)
        self.assertTrue(res_normal.success)
        self.assertEqual(res_normal.status, "OPTIMAL")
        self.assertGreater(res_normal.nit, 0)

    def test_run_what_if_simulation_max_iter_propagation(self):
        """Validates that run_what_if_simulation propagates max_iter setting to its internal solver."""
        sim_zero = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=100000.0,
            what_if_sulfur=1.20,
            price_adjustments={},
            avail_adjustments={},
            max_iter=0
        )
        self.assertFalse(sim_zero["res_whatif"].success)
        self.assertIn(sim_zero["res_whatif"].status, ["ITERATION_LIMIT", "MAX_ITER"])

        sim_normal = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=100000.0,
            what_if_sulfur=1.20,
            price_adjustments={},
            avail_adjustments={},
            max_iter=5000
        )
        self.assertTrue(sim_normal["res_whatif"].success)
        self.assertEqual(sim_normal["res_whatif"].status, "OPTIMAL")

    def test_branch_and_bound_max_iter_wiring(self):
        """Validates that BranchAndBound receives an LP solver configured with max_iter."""
        probs = get_preloaded_problems()
        milp_prob = probs["📦 Multi-City Warehouse Facility Location & Logistics (Binary MILP)"]

        lp_solver_zero = SimplexSolver(tol=1e-7, max_iter=0)
        bb_solver_zero = BranchAndBoundSolver(lp_solver=lp_solver_zero)

        res_bb_zero = bb_solver_zero.solve(
            c=milp_prob["c"],
            A_ub=milp_prob["A_ub"],
            b_ub=milp_prob["b_ub"],
            A_eq=milp_prob.get("A_eq"),
            b_eq=milp_prob.get("b_eq"),
            A_ge=milp_prob.get("A_ge"),
            b_ge=milp_prob.get("b_ge"),
            bounds=milp_prob.get("bounds"),
            integrality=milp_prob.get("integrality"),
            maximize=False
        )
        # With subproblem LP pivots capped at 0, root LP cannot converge
        self.assertFalse(res_bb_zero.success)

        # Standard solver
        lp_solver_norm = SimplexSolver(tol=1e-7, max_iter=5000)
        bb_solver_norm = BranchAndBoundSolver(lp_solver=lp_solver_norm)
        res_bb_norm = bb_solver_norm.solve(
            c=milp_prob["c"],
            A_ub=milp_prob["A_ub"],
            b_ub=milp_prob["b_ub"],
            A_eq=milp_prob.get("A_eq"),
            b_eq=milp_prob.get("b_eq"),
            A_ge=milp_prob.get("A_ge"),
            b_ge=milp_prob.get("b_ge"),
            bounds=milp_prob.get("bounds"),
            integrality=milp_prob.get("integrality"),
            maximize=False
        )
        self.assertTrue(res_bb_norm.success)
        self.assertEqual(res_bb_norm.status, "OPTIMAL")


if __name__ == "__main__":
    unittest.main()
