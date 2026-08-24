"""
IndSolve — Refinery Simulation & Zero-Trust Audit Regression Suite
Verifies exact model construction, dynamic binding constraint detection (Coker limit,
Sulfur ceiling, Supplier quotas), infeasibility handling, and independent mathematical auditing.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from solver.refinery_simulation import build_refinery_model, run_what_if_simulation, BASE_CRUDES
from solver.audit import audit_solution
from solver.tableau_simplex import SimplexSolver


class TestRefineryAuditAndSimulation(unittest.TestCase):

    def test_default_scenario_exact_numbers(self):
        """Verify baseline default 100k bpd @ 1.20% sulfur scenario exact values."""
        sim = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=100000.0,
            what_if_sulfur=1.20,
            price_adjustments={},
            avail_adjustments={},
            carbon_penalty=0.0
        )
        self.assertTrue(sim["res_whatif"].success)
        self.assertEqual(sim["res_whatif"].status, "OPTIMAL")

        # Check KPIs (100k bpd, $71.39/bbl -> $7,139,000.00/day)
        self.assertAlmostEqual(sim["cost_whatif"], 7139000.0, delta=10.0)
        self.assertAlmostEqual(sim["cost_per_bbl"], 71.39, delta=0.1)
        self.assertAlmostEqual(sim["actual_blend_sulfur_pct"], 1.20, delta=0.001)

        # Audit exact model matrices
        wm = sim["whatif_model"]
        audit_res = audit_solution(
            c=wm["c"],
            A_ub=wm["A_ub"],
            b_ub=wm["b_ub"],
            A_ge=None,
            b_ge=None,
            A_eq=wm["A_eq"],
            b_eq=wm["b_eq"],
            bounds=wm["bounds"],
            integrality=[0] * len(wm["var_names"]),
            var_names=wm["var_names"],
            x_sol=sim["res_whatif"].x,
            reported_obj=sim["res_whatif"].fun,
            maximize=False
        )
        self.assertTrue(audit_res["all_passed"], "Zero-trust audit must pass all constraints")
        self.assertTrue(audit_res["obj_verified"], "Objective recalculation must match")

    def test_coker_limit_binds(self):
        """Basra heavy crude is cheapest ($58/bbl), so it will hit the coker unit capacity."""
        sim = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=2.00,  # Relax sulfur so cheap Basra Heavy is strongly favored
            what_if_throughput=100000.0,
            what_if_sulfur=2.00,
            price_adjustments={},
            avail_adjustments={},
            carbon_penalty=0.0,
            heavy_limit_bpd=20000.0,  # Custom coker limit
        )
        self.assertTrue(sim["res_whatif"].success)
        basra_idx = sim["whatif_model"]["crude_keys"].index("Basra_Heavy")
        self.assertAlmostEqual(sim["res_whatif"].x[basra_idx], 20000.0, places=3)
        self.assertTrue(any("Basra Heavy Coker Capacity" in bc and "BINDING" in bc for bc in sim["binding_constraints"]))

    def test_sulfur_limit_binds(self):
        """Tight sulfur limit (0.80%) forces blend to exactly hit the environmental ceiling."""
        sim = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=100000.0,
            what_if_sulfur=0.80,
            price_adjustments={},
            avail_adjustments={},
            carbon_penalty=0.0
        )
        self.assertTrue(sim["res_whatif"].success)
        self.assertAlmostEqual(sim["actual_blend_sulfur_pct"], 0.80, places=3)
        self.assertTrue(any("Blended Feed Sulfur Ceiling" in bc and "BINDING" in bc for bc in sim["binding_constraints"]))

    def test_supplier_quota_binds(self):
        """Discounting Dubai Sour causes it to hit its supplier quota."""
        sim = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.80,
            what_if_throughput=100000.0,
            what_if_sulfur=1.80,
            price_adjustments={"Dubai_Sour": -10.0},
            avail_adjustments={"Dubai_Sour": 30000.0},
            carbon_penalty=0.0
        )
        self.assertTrue(sim["res_whatif"].success)
        dubai_idx = sim["whatif_model"]["crude_keys"].index("Dubai_Sour")
        self.assertAlmostEqual(sim["res_whatif"].x[dubai_idx], 30000.0, places=3)
        self.assertTrue(any("Dubai Sour Supplier Availability" in bc and "BINDING" in bc for bc in sim["binding_constraints"]))

    def test_infeasible_refinery_scenario(self):
        """Throughput target (200,000 bpd) exceeding total global crude supply must be detected as INFEASIBLE."""
        sim = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=200000.0,  # Total max available is 40k+35k+30k+45k+25k = 175k
            what_if_sulfur=1.20,
            price_adjustments={},
            avail_adjustments={},
            carbon_penalty=0.0
        )
        self.assertFalse(sim["res_whatif"].success)
        self.assertEqual(sim["res_whatif"].status, "INFEASIBLE")

    def test_audit_detects_no_hidden_violations(self):
        """Zero-trust audit on solved model verifies 100% of rows and bounds without omissions."""
        sim = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=90000.0,
            what_if_sulfur=1.10,
            price_adjustments={"Brent": 5.0, "Basra_Heavy": -3.0},
            avail_adjustments={"Arabian_Light": 35000.0},
            carbon_penalty=2.0
        )
        self.assertTrue(sim["res_whatif"].success)
        wm = sim["whatif_model"]
        audit_res = audit_solution(
            c=wm["c"],
            A_ub=wm["A_ub"],
            b_ub=wm["b_ub"],
            A_ge=None,
            b_ge=None,
            A_eq=wm["A_eq"],
            b_eq=wm["b_eq"],
            bounds=wm["bounds"],
            integrality=[0] * len(wm["var_names"]),
            var_names=wm["var_names"],
            x_sol=sim["res_whatif"].x,
            reported_obj=sim["res_whatif"].fun,
            maximize=False
        )
        self.assertTrue(audit_res["all_passed"])
        self.assertEqual(audit_res["passed_constraints"], audit_res["total_constraints"])


if __name__ == "__main__":
    unittest.main()
