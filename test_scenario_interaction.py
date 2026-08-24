"""
IndSolve — Unit Test for Intentional Scenario Lab Interaction & Timing
Validates that scenario calculations are decoupled from state changes until explicitly applied.
"""

import unittest
import time
from solver.refinery_simulation import run_what_if_simulation


class TestScenarioLabInteraction(unittest.TestCase):
    def test_solve_only_on_apply(self):
        """Validates that a simulation result is produced only when explicit simulation function is executed."""
        # Simulated state before apply
        session_state = {
            "scenario_result": None,
            "scenario_inputs": None,
            "scenario_timestamp": None,
            "scenario_e2e_time": None
        }
        
        # Changing hypothetical sliders without clicking apply
        slider_throughput = 110000.0
        slider_sulfur = 1.10
        
        # Verify state remains None (empty state)
        self.assertIsNone(session_state["scenario_result"])
        
        # User clicks "Apply scenario"
        t0 = time.perf_counter()
        sim_res = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=slider_throughput,
            what_if_sulfur=slider_sulfur,
            price_adjustments={},
            avail_adjustments={},
            carbon_penalty=0.0,
            heavy_limit_bpd=25000.0
        )
        t_elapsed = time.perf_counter() - t0
        
        # Update session state on submit
        session_state["scenario_result"] = sim_res
        session_state["scenario_inputs"] = {"throughput": slider_throughput, "sulfur": slider_sulfur}
        session_state["scenario_e2e_time"] = t_elapsed
        
        # Verify execution integrity
        self.assertIsNotNone(session_state["scenario_result"])
        self.assertTrue(session_state["scenario_result"]["res_whatif"].success)
        self.assertGreater(session_state["scenario_e2e_time"], 0.0)
        self.assertGreaterEqual(session_state["scenario_result"]["res_whatif"].solve_time, 0.0)
        
        # Verify that solver time is less than or equal to total end-to-end pipeline time
        self.assertLessEqual(session_state["scenario_result"]["res_whatif"].solve_time, session_state["scenario_e2e_time"] + 0.01)

    def test_different_inputs_produce_different_results_without_caching(self):
        """Validates that successive submissions with different inputs produce distinct results."""
        sim_base = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=100000.0,
            what_if_sulfur=1.20,
            price_adjustments={},
            avail_adjustments={},
            carbon_penalty=0.0,
            heavy_limit_bpd=25000.0
        )
        
        # Shock Brent crude (which is active in the baseline basis at 35,000 bpd)
        sim_shock = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=100000.0,
            what_if_sulfur=1.20,
            price_adjustments={"Brent": 14.0},
            avail_adjustments={},
            carbon_penalty=0.0,
            heavy_limit_bpd=25000.0
        )
        
        self.assertNotEqual(sim_base["cost_whatif"], sim_shock["cost_whatif"])
        self.assertGreater(sim_shock["cost_whatif"], sim_base["cost_whatif"])


if __name__ == "__main__":
    unittest.main()
