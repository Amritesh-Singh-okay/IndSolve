"""
IndSolve — Validated Continuous-LP MPS Subset Safety Test Suite
Verifies strict rejection of unsupported constructs (integer markers, binary bounds,
quadratic terms, ranges, undeclared rows/vars) and correct parsing of supported LP features.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from solver.mps_parser import parse_mps_text, MPSParseError
from solver.tableau_simplex import SimplexSolver


class TestMPSSubsetSafety(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. SUPPORTED CONTINUOUS-LP CONSTRUCTS
    # -------------------------------------------------------------------------
    def test_supported_continuous_lp_with_all_bound_types(self):
        """Valid continuous LP with LO, UP, FX, and FR bounds must parse and solve."""
        mps_text = """
NAME          SAMPLE_LP
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
COLUMNS
    X1        COST      1.0       R1        2.0
    X1        R2        1.0
    X2        COST      3.0       R1        1.0
    X2        R3        1.0
    X3        COST      2.0       R2        1.0
    X3        R1        1.0
    X4        COST      5.0       R3        2.0
RHS
    RHS1      R1        10.0      R2        4.0
    RHS1      R3        6.0
BOUNDS
 LO BND       X1        1.0
 UP BND       X2        5.0
 FX BND       X4        2.0
 FR BND       X3
ENDATA
"""
        model = parse_mps_text(mps_text)
        self.assertEqual(model["problem_name"], "SAMPLE_LP")
        self.assertEqual(model["num_rows"], 3)
        self.assertEqual(model["num_vars"], 4)
        self.assertEqual(model["nnz"], 7)        # 7 entries in constraint matrix
        self.assertEqual(model["total_nnz"], 11)  # 7 constraints + 4 objective
        self.assertEqual(model["bounds"][0], (1.0, None))  # LO
        self.assertEqual(model["bounds"][1], (0.0, 5.0))   # UP
        self.assertEqual(model["bounds"][2], (None, None)) # FR
        self.assertEqual(model["bounds"][3], (2.0, 2.0))   # FX

        solver = SimplexSolver(tol=1e-7)
        res = solver.solve(
            c=model["c"], A_ub=model["A_ub"], b_ub=model["b_ub"],
            A_ge=model["A_ge"], b_ge=model["b_ge"], A_eq=model["A_eq"], b_eq=model["b_eq"],
            bounds=model["bounds"], maximize=False
        )
        self.assertTrue(res.success)
        self.assertEqual(res.status, "OPTIMAL")

    # -------------------------------------------------------------------------
    # 2. REJECTION OF INTEGER MARKERS (INTORG / INTEND)
    # -------------------------------------------------------------------------
    def test_reject_integer_marker_intorg(self):
        """MPS with INTORG marker in COLUMNS must be rejected with explicit MPSParseError."""
        mps_text = """
NAME          MIP_INTORG
ROWS
 N  COST
 L  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      1.0       R1        1.0
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS1      R1        10.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("Integer marker detected", str(ctx.exception))
        self.assertIn("continuous-LP", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 3. REJECTION OF DISCRETE / INTEGER BOUND TYPES
    # -------------------------------------------------------------------------
    def test_reject_binary_bound_type_bv(self):
        """MPS with binary bound 'BV' must be rejected."""
        mps_text = """
NAME          MIP_BV
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0       R1        1.0
RHS
    RHS1      R1        1.0
BOUNDS
 BV BND       X1
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("Unsupported discrete/integer bound type 'BV'", str(ctx.exception))

    def test_reject_integer_bound_type_ui(self):
        """MPS with integer bound 'UI' must be rejected."""
        mps_text = """
NAME          MIP_UI
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0       R1        1.0
RHS
    RHS1      R1        5.0
BOUNDS
 UI BND       X1        10.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("Unsupported discrete/integer bound type 'UI'", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. REJECTION OF ADVANCED / UNSUPPORTED SECTIONS
    # -------------------------------------------------------------------------
    def test_reject_ranges_section(self):
        """MPS with RANGES section must be rejected."""
        mps_text = """
NAME          RANGES_MODEL
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0       R1        1.0
RHS
    RHS1      R1        10.0
RANGES
    RNG1      R1        2.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("Unsupported MPS Section 'RANGES'", str(ctx.exception))

    def test_reject_sos_section(self):
        """MPS with SOS section must be rejected."""
        mps_text = """
NAME          SOS_MODEL
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0       R1        1.0
RHS
    RHS1      R1        10.0
SOS
 S1 S0001     1.0
    X1        1.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("Unsupported MPS Section 'SOS'", str(ctx.exception))

    def test_reject_quadratic_quadobj_section(self):
        """MPS with QUADOBJ section must be rejected."""
        mps_text = """
NAME          QP_MODEL
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0       R1        1.0
RHS
    RHS1      R1        10.0
QUADOBJ
    X1        X1        2.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("Unsupported MPS Section 'QUADOBJ'", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 5. STRUCTURAL INTEGRITY & UNDECLARED REFERENCES
    # -------------------------------------------------------------------------
    def test_reject_undeclared_row_in_columns(self):
        """COLUMNS record referencing a row not in ROWS must be rejected."""
        mps_text = """
NAME          BAD_ROW_REF
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0       R99       1.0
RHS
    RHS1      R1        10.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("references undefined row 'R99'", str(ctx.exception))

    def test_reject_undeclared_variable_in_bounds(self):
        """BOUNDS record referencing a variable not in COLUMNS must be rejected."""
        mps_text = """
NAME          BAD_VAR_REF
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0       R1        1.0
RHS
    RHS1      R1        10.0
BOUNDS
 UP BND       X99       5.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("references undeclared variable 'X99'", str(ctx.exception))

    def test_reject_missing_objective_row(self):
        """ROWS section with no 'N' row must be rejected."""
        mps_text = """
NAME          NO_OBJ
ROWS
 L  R1
 L  R2
COLUMNS
    X1        R1        1.0       R2        1.0
RHS
    RHS1      R1        10.0      R2        20.0
ENDATA
"""
        with self.assertRaises(MPSParseError) as ctx:
            parse_mps_text(mps_text)
        self.assertIn("No objective row (Type 'N')", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
