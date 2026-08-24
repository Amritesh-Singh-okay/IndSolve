"""
IndSolve — Netlib Benchmark Provenance & Regression Test Suite
Verifies authentic Netlib AFIRO benchmark properties, SHA-256 checksum, matrix density,
and exact agreement with the published rational optimum (-464.75314286).
"""

import sys
import os
import hashlib
import unittest
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(__file__))

from solver.mps_parser import parse_mps_text
from solver.tableau_simplex import SimplexSolver


class TestNetlibAFIROProvenance(unittest.TestCase):
    """
    Benchmark Provenance Specification:
    - Source: Netlib LP Test Problem Library (https://netlib.sandia.gov/lp/data/afiro)
              & HiGHS / Coin-OR Standard Benchmark Instances
    - Expected SHA-256: 9cd304f02717cbd6f85068cb777b69d28539b22a4868ae0f0fb425f514f0eea5
    - Dimensions: 27 constraint rows (8 Equality, 19 Inequality; 28 rows with COST), 32 structural variables
    - Non-Zero Count:
        * Constraint Matrix Non-Zeros (A_ub, A_ge, A_eq): 83 NNZ (Matrix Density: 9.61%)
        * Objective Row Non-Zeros (COST): 5 NNZ
        * Total COLUMNS Section Non-Zeros: 88 NNZ
    - Published Exact Rational Optimum: -464.75314285714285 (Koch 2004, Operations Research Letters)
    """

    AFIRO_EXPECTED_SHA256 = "9cd304f02717cbd6f85068cb777b69d28539b22a4868ae0f0fb425f514f0eea5"
    AFIRO_PUBLISHED_OPTIMUM = -464.75314285714285  # Koch (2004) Rational Arithmetic Optimum
    AFIRO_CONSTRAINT_ROWS = 27
    AFIRO_VARS = 32
    AFIRO_CONSTRAINT_NNZ = 83

    def setUp(self):
        self.afiro_path = os.path.join(os.path.dirname(__file__), "benchmarks", "afiro.mps")
        self.assertTrue(os.path.exists(self.afiro_path), f"Missing benchmark file at {self.afiro_path}")
        with open(self.afiro_path, "r", encoding="utf-8") as f:
            self.afiro_text = f.read()

    def test_afiro_sha256_provenance(self):
        """Authentic Netlib AFIRO benchmark file must match official SHA-256 hash."""
        computed_hash = hashlib.sha256(self.afiro_text.encode("utf-8")).hexdigest()
        self.assertEqual(
            computed_hash,
            self.AFIRO_EXPECTED_SHA256,
            f"AFIRO MPS SHA-256 mismatch! Expected {self.AFIRO_EXPECTED_SHA256}, got {computed_hash}"
        )

    def test_afiro_structural_statistics(self):
        """AFIRO must parse with 27 constraint rows, 32 structural variables, and 83 constraint nonzeros."""
        model = parse_mps_text(self.afiro_text)
        self.assertEqual(model["problem_name"], "AFIRO")
        self.assertEqual(model["num_rows"], self.AFIRO_CONSTRAINT_ROWS, f"Expected {self.AFIRO_CONSTRAINT_ROWS} constraint rows, got {model['num_rows']}")
        self.assertEqual(model["num_vars"], self.AFIRO_VARS, f"Expected {self.AFIRO_VARS} variables, got {model['num_vars']}")
        self.assertEqual(model["nnz"], self.AFIRO_CONSTRAINT_NNZ, f"Expected {self.AFIRO_CONSTRAINT_NNZ} constraint NNZ, got {model['nnz']}")

        expected_density = (self.AFIRO_CONSTRAINT_NNZ / (self.AFIRO_CONSTRAINT_ROWS * self.AFIRO_VARS)) * 100.0
        self.assertAlmostEqual(model["density"], expected_density, places=2)

    def test_afiro_exact_optimization(self):
        """IndSolve and SciPy HiGHS must both solve authentic AFIRO to the published optimum (-464.75314286)."""
        model = parse_mps_text(self.afiro_text)

        # 1. Solve with IndSolve TableauSimplex
        solver = SimplexSolver(tol=1e-7, max_iter=3000)
        res_ind = solver.solve(
            c=model["c"],
            A_ub=model["A_ub"],
            b_ub=model["b_ub"],
            A_ge=model["A_ge"],
            b_ge=model["b_ge"],
            A_eq=model["A_eq"],
            b_eq=model["b_eq"],
            bounds=model["bounds"],
            maximize=False,
        )

        self.assertTrue(res_ind.success, f"IndSolve failed to solve AFIRO: {res_ind.message}")
        self.assertEqual(res_ind.status, "OPTIMAL")
        self.assertAlmostEqual(
            res_ind.fun,
            self.AFIRO_PUBLISHED_OPTIMUM,
            places=5,
            msg=f"IndSolve objective ({res_ind.fun}) diverged from published Netlib AFIRO optimum ({self.AFIRO_PUBLISHED_OPTIMUM})"
        )

        # 2. Independent SciPy HiGHS comparison
        scipy_A_ub, scipy_b_ub = [], []
        if model["A_ub"] is not None:
            scipy_A_ub.append(model["A_ub"])
            scipy_b_ub.append(model["b_ub"])
        if model["A_ge"] is not None:
            scipy_A_ub.append(-model["A_ge"])
            scipy_b_ub.append(-model["b_ge"])
        A_ub_c = np.vstack(scipy_A_ub) if scipy_A_ub else None
        b_ub_c = np.concatenate(scipy_b_ub) if scipy_b_ub else None

        res_scipy = linprog(
            c=model["c"],
            A_ub=A_ub_c,
            b_ub=b_ub_c,
            A_eq=model["A_eq"],
            b_eq=model["b_eq"],
            bounds=model["bounds"],
            method="highs",
        )
        self.assertTrue(res_scipy.success, "SciPy HiGHS failed to solve AFIRO")
        self.assertAlmostEqual(res_scipy.fun, self.AFIRO_PUBLISHED_OPTIMUM, places=5)

        # 3. Inter-solver agreement check
        diff = abs(res_ind.fun - res_scipy.fun)
        self.assertLess(diff, 1e-6, f"Difference between IndSolve and SciPy ({diff}) exceeds 1e-6")


if __name__ == "__main__":
    unittest.main()
