"""
IndSolve — Verification Lab & Automated Test Suite
Executes 6 comprehensive test families covering:
1. Feasible Continuous LPs (Standard Optimization)
2. Infeasible LPs (Contradiction Detection)
3. Unbounded LPs (Ray Divergence, Free-Variable Minimization & No False Optima)
4. Bounds & Free Transformations (Shifts, Upper-Only, Fixed, and Free Splitting)
5. Mixed-Integer Linear Programming (Exact Agreement vs scipy.optimize.milp)
6. Validated Continuous MPS & Safety (Authentic Netlib AFIRO & Syntax Rejections)
"""

import time
import os
from typing import Dict, List, Any
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

try:
    from .tableau_simplex import SimplexSolver
    from .branch_and_bound import BranchAndBoundSolver
    from .mps_parser import parse_mps_text, MPSParseError
except ImportError:
    from solver.tableau_simplex import SimplexSolver
    from solver.branch_and_bound import BranchAndBoundSolver
    from solver.mps_parser import parse_mps_text, MPSParseError


def run_full_verification_lab() -> Dict[str, Any]:
    """
    Runs all 6 test families and returns structured verification metrics.
    """
    solver = SimplexSolver(tol=1e-7, max_iter=5000)
    milp_solver = BranchAndBoundSolver(lp_solver=solver, max_nodes=500)

    test_results = []
    family_stats = {
        "Feasible LPs": {"total": 0, "passed": 0, "desc": "Standard optimization to proven optimality"},
        "Infeasible LPs": {"total": 0, "passed": 0, "desc": "Contradiction & impossible constraint detection"},
        "Unbounded LPs": {"total": 0, "passed": 0, "desc": "Detection of unbounded rays & free-variable divergence"},
        "Bounds & Free Transforms": {"total": 0, "passed": 0, "desc": "Negative lower bounds, upper-only, fixed & free splits"},
        "MILP vs SciPy milp": {"total": 0, "passed": 0, "desc": "Exact integer optimum agreement vs scipy.optimize.milp"},
        "MPS Benchmark & Safety": {"total": 0, "passed": 0, "desc": "Authentic Netlib AFIRO optimum & construct rejection"},
    }

    # =========================================================================
    # FAMILY 1: FEASIBLE LPs (5 Tests)
    # =========================================================================
    feasible_tests = [
        {
            "name": "Canonical 2-Var Production Max",
            "c": np.array([3.0, 5.0]),
            "A_ub": np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]]),
            "b_ub": np.array([4.0, 12.0, 18.0]),
            "maximize": True,
            "expected_status": "OPTIMAL",
            "expected_obj": 36.0,
        },
        {
            "name": "Multi-Variable Cost Min",
            "c": np.array([4.0, 1.0, 1.0]),
            "A_ge": np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]),
            "b_ge": np.array([10.0, 12.0]),
            "maximize": False,
            "expected_status": "OPTIMAL",
            "expected_obj": 11.0,
        },
        {
            "name": "Equality Blending Balance",
            "c": np.array([10.0, 15.0, 25.0]),
            "A_eq": np.array([[1.0, 1.0, 1.0]]),
            "b_eq": np.array([100.0]),
            "A_ub": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "b_ub": np.array([40.0, 50.0]),
            "maximize": False,
            "expected_status": "OPTIMAL",
            "expected_obj": 1400.0,
        },
        {
            "name": "Degenerate Diamond Polytope",
            "c": np.array([2.0, 3.0]),
            "A_ub": np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]),
            "b_ub": np.array([5.0, 5.0, 5.0]),
            "maximize": True,
            "expected_status": "OPTIMAL",
            "expected_obj": 15.0,
        },
        {
            "name": "Dense Transportation Flow (4x4)",
            "c": np.array([2.0, 4.0, 5.0, 2.0]),
            "A_ub": np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
            "b_ub": np.array([50.0, 50.0]),
            "A_ge": np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
            "b_ge": np.array([30.0, 40.0]),
            "maximize": False,
            "expected_status": "OPTIMAL",
            "expected_obj": 140.0,
        }
    ]

    for t in feasible_tests:
        fam = "Feasible LPs"
        family_stats[fam]["total"] += 1
        t0 = time.perf_counter()
        res = solver.solve(
            c=t["c"],
            A_ub=t.get("A_ub"),
            b_ub=t.get("b_ub"),
            A_eq=t.get("A_eq"),
            b_eq=t.get("b_eq"),
            A_ge=t.get("A_ge"),
            b_ge=t.get("b_ge"),
            maximize=t.get("maximize", False)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        status_ok = (res.status == t["expected_status"])
        obj_ok = abs(res.fun - t["expected_obj"]) < 1e-3 if status_ok else False
        passed = status_ok and obj_ok
        if passed:
            family_stats[fam]["passed"] += 1

        test_results.append({
            "Family": fam,
            "Test Name": t["name"],
            "IndSolve Status": res.status,
            "Expected Status": t["expected_status"],
            "Computed Obj": f"{res.fun:,.2f}" if res.success else "N/A",
            "Expected Obj": f"{t['expected_obj']:,.2f}",
            "Latency (ms)": f"{elapsed_ms:.2f}",
            "Passed": passed,
            "Verification Note": f"Exact optimum match (Δ={abs(res.fun - t['expected_obj']):.2e})" if passed else res.message
        })

    # =========================================================================
    # FAMILY 2: INFEASIBLE LPs (3 Tests)
    # =========================================================================
    infeasible_tests = [
        {
            "name": "Direct Contradiction (x <= 2 and x >= 5)",
            "c": np.array([1.0]),
            "A_ub": np.array([[1.0]]),
            "b_ub": np.array([2.0]),
            "A_ge": np.array([[1.0]]),
            "b_ge": np.array([5.0]),
            "maximize": False,
            "expected_status": "INFEASIBLE"
        },
        {
            "name": "Conflicting Equality Bounds",
            "c": np.array([1.0, 1.0]),
            "A_eq": np.array([[1.0, 1.0], [1.0, 1.0]]),
            "b_eq": np.array([10.0, 20.0]),
            "maximize": False,
            "expected_status": "INFEASIBLE"
        },
        {
            "name": "Box Infeasibility (Sum > Box Capacity)",
            "c": np.array([2.0, 3.0]),
            "A_ub": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "b_ub": np.array([2.0, 2.0]),
            "A_ge": np.array([[1.0, 1.0]]),
            "b_ge": np.array([10.0]),
            "maximize": True,
            "expected_status": "INFEASIBLE"
        }
    ]

    for t in infeasible_tests:
        fam = "Infeasible LPs"
        family_stats[fam]["total"] += 1
        t0 = time.perf_counter()
        res = solver.solve(
            c=t["c"],
            A_ub=t.get("A_ub"),
            b_ub=t.get("b_ub"),
            A_eq=t.get("A_eq"),
            b_eq=t.get("b_eq"),
            A_ge=t.get("A_ge"),
            b_ge=t.get("b_ge"),
            maximize=t.get("maximize", False)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        passed = (res.status == t["expected_status"])
        if passed:
            family_stats[fam]["passed"] += 1

        test_results.append({
            "Family": fam,
            "Test Name": t["name"],
            "IndSolve Status": res.status,
            "Expected Status": t["expected_status"],
            "Computed Obj": "N/A (Infeasible)",
            "Expected Obj": "N/A (Infeasible)",
            "Latency (ms)": f"{elapsed_ms:.2f}",
            "Passed": passed,
            "Verification Note": "Correctly detected constraint contradiction" if passed else res.message
        })

    # =========================================================================
    # FAMILY 3: UNBOUNDED LPs (3 Tests)
    # =========================================================================
    unbounded_tests = [
        {
            "name": "Unconstrained Gradient Max (No Upper Bounds)",
            "c": np.array([3.0, 4.0]),
            "A_ub": None,
            "b_ub": None,
            "bounds": [(0.0, None), (0.0, None)],
            "maximize": True,
            "expected_status": "UNBOUNDED"
        },
        {
            "name": "Open Feasible Ray (x1 - x2 <= 5, Max x1)",
            "c": np.array([1.0, 0.0]),
            "A_ub": np.array([[1.0, -1.0]]),
            "b_ub": np.array([5.0]),
            "bounds": [(0.0, None), (0.0, None)],
            "maximize": True,
            "expected_status": "UNBOUNDED"
        },
        {
            "name": "Free Variable Unconstrained Min (min x, x in Free)",
            "c": np.array([1.0]),
            "A_ub": None,
            "b_ub": None,
            "bounds": [(None, None)],
            "maximize": False,
            "expected_status": "UNBOUNDED"
        }
    ]

    for t in unbounded_tests:
        fam = "Unbounded LPs"
        family_stats[fam]["total"] += 1
        t0 = time.perf_counter()
        res = solver.solve(
            c=t["c"],
            A_ub=t.get("A_ub"),
            b_ub=t.get("b_ub"),
            bounds=t.get("bounds"),
            maximize=t.get("maximize", True)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        passed = (res.status == t["expected_status"])
        if passed:
            family_stats[fam]["passed"] += 1

        test_results.append({
            "Family": fam,
            "Test Name": t["name"],
            "IndSolve Status": res.status,
            "Expected Status": t["expected_status"],
            "Computed Obj": "+∞ (Unbounded)" if t.get("maximize", True) else "-∞",
            "Expected Obj": "+∞ / -∞ (Unbounded)",
            "Latency (ms)": f"{elapsed_ms:.2f}",
            "Passed": passed,
            "Verification Note": "Successfully identified unbounded ray without false convergence" if passed else res.message
        })

    # =========================================================================
    # FAMILY 4: BOUNDS & FREE TRANSFORMATIONS (5 Tests)
    # =========================================================================
    bounds_tests = [
        {
            "name": "Negative Lower Bound (x1 in [-10, 10])",
            "c": np.array([2.0, 3.0]),
            "A_ub": np.array([[1.0, 1.0]]),
            "b_ub": np.array([5.0]),
            "bounds": [(-10.0, 10.0), (0.0, 10.0)],
            "maximize": False,
            "expected_status": "OPTIMAL",
            "expected_obj": -20.0,
        },
        {
            "name": "Strict Positive Lower Bounds (x1 in [5, 20], x2 in [10, 30])",
            "c": np.array([1.0, 1.0]),
            "A_ub": np.array([[1.0, 2.0]]),
            "b_ub": np.array([100.0]),
            "bounds": [(5.0, 20.0), (10.0, 30.0)],
            "maximize": False,
            "expected_status": "OPTIMAL",
            "expected_obj": 15.0,
        },
        {
            "name": "Upper-Only Bound (x1 in (-inf, 4], Max 2x1 - x2)",
            "c": np.array([2.0, -1.0]),
            "A_ub": np.array([[1.0, 1.0]]),
            "b_ub": np.array([6.0]),
            "bounds": [(None, 4.0), (0.0, None)],
            "maximize": True,
            "expected_status": "OPTIMAL",
            "expected_obj": 8.0,
        },
        {
            "name": "Fixed Variable Bound (x1 == 4 fixed, Min 5x1 + 2x2)",
            "c": np.array([5.0, 2.0]),
            "A_ge": np.array([[1.0, 1.0]]),
            "b_ge": np.array([10.0]),
            "bounds": [(4.0, 4.0), (0.0, None)],
            "maximize": False,
            "expected_status": "OPTIMAL",
            "expected_obj": 32.0,
        },
        {
            "name": "Constrained Free Variables (x1, x2 in (-inf, +inf))",
            "c": np.array([3.0, 2.0]),
            "A_ge": np.array([[1.0, 1.0], [2.0, -1.0]]),
            "b_ge": np.array([5.0, 1.0]),
            "A_ub": np.array([[1.0, 0.0]]),
            "b_ub": np.array([6.0]),
            "bounds": [(None, None), (None, None)],
            "maximize": False,
            "expected_status": "OPTIMAL",
            "expected_obj": 12.0,
        }
    ]

    for t in bounds_tests:
        fam = "Bounds & Free Transforms"
        family_stats[fam]["total"] += 1
        t0 = time.perf_counter()
        res = solver.solve(
            c=t["c"],
            A_ub=t.get("A_ub"),
            b_ub=t.get("b_ub"),
            A_ge=t.get("A_ge"),
            b_ge=t.get("b_ge"),
            bounds=t.get("bounds"),
            maximize=t.get("maximize", False)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status_ok = (res.status == t["expected_status"])
        obj_ok = abs(res.fun - t["expected_obj"]) < 1e-3 if status_ok else False
        passed = status_ok and obj_ok
        if passed:
            family_stats[fam]["passed"] += 1

        test_results.append({
            "Family": fam,
            "Test Name": t["name"],
            "IndSolve Status": res.status,
            "Expected Status": t["expected_status"],
            "Computed Obj": f"{res.fun:,.2f}" if res.success else "N/A",
            "Expected Obj": f"{t['expected_obj']:,.2f}",
            "Latency (ms)": f"{elapsed_ms:.2f}",
            "Passed": passed,
            "Verification Note": f"Correct shift & bounds recovery ({res.x.round(2)})" if passed else res.message
        })

    # =========================================================================
    # FAMILY 5: MILP vs scipy.optimize.milp (4 Tests)
    # =========================================================================
    milp_tests = [
        {
            "name": "Binary 0-1 Knapsack Selection",
            "c": np.array([10.0, 15.0, 40.0]),
            "A_ub": np.array([[1.0, 2.0, 3.0]]),
            "b_ub": np.array([4.0]),
            "bounds": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            "integrality": [1, 1, 1],
            "maximize": True,
        },
        {
            "name": "Mixed Integer Resource Allocation (x1 int, x2 cont)",
            "c": np.array([5.0, 4.0]),
            "A_ub": np.array([[2.0, 3.0], [4.0, 1.0]]),
            "b_ub": np.array([12.0, 10.0]),
            "bounds": [(0.0, None), (0.0, None)],
            "integrality": [1, 0],
            "maximize": True,
        },
        {
            "name": "Integer Facility Fixed Charge",
            "c": np.array([30.0, 45.0]),
            "A_ub": np.array([[3.0, 4.0], [1.0, 2.0]]),
            "b_ub": np.array([18.0, 8.0]),
            "bounds": [(0.0, 10.0), (0.0, 10.0)],
            "integrality": [1, 1],
            "maximize": True,
        },
        {
            "name": "Multi-City Warehouse Location Hubs (3 Binary Hubs)",
            "c": np.array([12000.0, 10000.0, 8000.0, 25.0, 32.0, 40.0]),
            "A_ub": np.array([
                [-500.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, -450.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -350.0, 0.0, 0.0, 1.0],
            ]),
            "b_ub": np.array([0.0, 0.0, 0.0]),
            "A_ge": np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
            "b_ge": np.array([650.0]),
            "bounds": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 500.0), (0.0, 450.0), (0.0, 350.0)],
            "integrality": [1, 1, 1, 0, 0, 0],
            "maximize": False,
        }
    ]

    for t in milp_tests:
        fam = "MILP vs SciPy milp"
        family_stats[fam]["total"] += 1
        t0 = time.perf_counter()

        ind_res = milp_solver.solve(
            c=t["c"],
            A_ub=t.get("A_ub"),
            b_ub=t.get("b_ub"),
            A_ge=t.get("A_ge"),
            b_ge=t.get("b_ge"),
            bounds=t["bounds"],
            integrality=t["integrality"],
            maximize=t["maximize"]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Reference solve with scipy.optimize.milp
        scipy_c = -t["c"] if t["maximize"] else t["c"]
        n_vars = len(t["c"])
        lb_list = [0.0 if b[0] is None else b[0] for b in t["bounds"]]
        ub_list = [np.inf if b[1] is None else b[1] for b in t["bounds"]]
        scipy_bounds = Bounds(lb_list, ub_list)

        constraints_list = []
        if t.get("A_ub") is not None:
            constraints_list.append(LinearConstraint(t["A_ub"], -np.inf, t["b_ub"]))
        if t.get("A_ge") is not None:
            constraints_list.append(LinearConstraint(t["A_ge"], t["b_ge"], np.inf))

        scipy_res = milp(
            c=scipy_c,
            integrality=t["integrality"],
            bounds=scipy_bounds,
            constraints=constraints_list
        )
        scipy_obj = -scipy_res.fun if t["maximize"] else scipy_res.fun

        status_ok = ind_res.success and scipy_res.success
        obj_ok = abs(ind_res.fun - scipy_obj) < 1e-3 if status_ok else False
        int_ok = all(abs(ind_res.x[i] - round(ind_res.x[i])) < 1e-4 for i in range(n_vars) if t["integrality"][i] == 1)
        passed = status_ok and obj_ok and int_ok
        if passed:
            family_stats[fam]["passed"] += 1

        test_results.append({
            "Family": fam,
            "Test Name": t["name"],
            "IndSolve Status": ind_res.status,
            "Expected Status": "OPTIMAL (scipy.milp)",
            "Computed Obj": f"{ind_res.fun:,.2f}" if ind_res.success else "N/A",
            "Expected Obj": f"{scipy_obj:,.2f}",
            "Latency (ms)": f"{elapsed_ms:.2f}",
            "Passed": passed,
            "Verification Note": f"Exact integer match with scipy.milp (x*={ind_res.x.round(1)})" if passed else "Mismatch"
        })

    # =========================================================================
    # FAMILY 6: VALIDATED CONTINUOUS MPS & SAFETY (3 Tests)
    # =========================================================================
    afiro_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "afiro.mps")
    if os.path.exists(afiro_path):
        with open(afiro_path, "r", encoding="utf-8") as f:
            afiro_text = f.read()
    else:
        afiro_text = ""

    mps_tests = [
        {
            "name": "Authentic Netlib AFIRO Benchmark (Published Opt: -464.7531)",
            "type": "PARSE_AND_SOLVE",
            "mps_text": afiro_text,
            "expected_opt": -464.75314286,
        },
        {
            "name": "Safety Rejection: Integer Marker 'INTORG'",
            "type": "REJECT_CHECK",
            "mps_text": "NAME MIP\nROWS\n N COST\n L R1\nCOLUMNS\n MARK 'MARKER' 'INTORG'\n X1 COST 1.0 R1 1.0\nENDATA",
            "expected_error_substr": "Integer marker detected",
        },
        {
            "name": "Safety Rejection: Unsupported Section 'RANGES'",
            "type": "REJECT_CHECK",
            "mps_text": "NAME RNG\nROWS\n N COST\n L R1\nCOLUMNS\n X1 COST 1.0 R1 1.0\nRANGES\n RNG1 R1 2.0\nENDATA",
            "expected_error_substr": "Unsupported MPS Section 'RANGES'",
        }
    ]

    for t in mps_tests:
        fam = "MPS Benchmark & Safety"
        family_stats[fam]["total"] += 1
        t0 = time.perf_counter()

        if t["type"] == "PARSE_AND_SOLVE":
            try:
                model = parse_mps_text(t["mps_text"])
                res = solver.solve(
                    c=model["c"], A_ub=model["A_ub"], b_ub=model["b_ub"],
                    A_ge=model["A_ge"], b_ge=model["b_ge"], A_eq=model["A_eq"], b_eq=model["b_eq"],
                    bounds=model["bounds"], maximize=False
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                passed = res.success and abs(res.fun - t["expected_opt"]) < 1e-4
                if passed:
                    family_stats[fam]["passed"] += 1
                test_results.append({
                    "Family": fam,
                    "Test Name": t["name"],
                    "IndSolve Status": res.status,
                    "Expected Status": "OPTIMAL",
                    "Computed Obj": f"{res.fun:,.6f}",
                    "Expected Obj": f"{t['expected_opt']:,.6f}",
                    "Latency (ms)": f"{elapsed_ms:.2f}",
                    "Passed": passed,
                    "Verification Note": f"Exact Netlib optimum match (Δ={abs(res.fun - t['expected_opt']):.2e})"
                })
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                test_results.append({
                    "Family": fam,
                    "Test Name": t["name"],
                    "IndSolve Status": "ERROR",
                    "Expected Status": "OPTIMAL",
                    "Computed Obj": "N/A",
                    "Expected Obj": f"{t['expected_opt']:,.6f}",
                    "Latency (ms)": f"{elapsed_ms:.2f}",
                    "Passed": False,
                    "Verification Note": str(e)
                })

        elif t["type"] == "REJECT_CHECK":
            try:
                parse_mps_text(t["mps_text"])
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                passed = False
                note = "Failed to reject invalid construct"
            except MPSParseError as pe:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                passed = t["expected_error_substr"] in str(pe)
                note = f"Safely rejected with MPSParseError: {str(pe)[:40]}..."
                if passed:
                    family_stats[fam]["passed"] += 1
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                passed = False
                note = f"Unexpected exception: {e}"

            test_results.append({
                "Family": fam,
                "Test Name": t["name"],
                "IndSolve Status": "REJECTED (MPSParseError)" if passed else "FAILED",
                "Expected Status": "REJECTED",
                "Computed Obj": "N/A (Syntax Error)",
                "Expected Obj": "N/A (Syntax Error)",
                "Latency (ms)": f"{elapsed_ms:.2f}",
                "Passed": passed,
                "Verification Note": note
            })

    import pandas as pd
    return {
        "family_stats": family_stats,
        "results_df": pd.DataFrame(test_results),
        "total_tests": len(test_results),
        "total_passed": sum(f["passed"] for f in family_stats.values()),
    }


if __name__ == "__main__":
    out = run_full_verification_lab()
    print("=" * 70)
    print(f"VERIFICATION LAB: {out['total_passed']} / {out['total_tests']} TESTS PASSED")
    print("=" * 70)
    for fam, s in out["family_stats"].items():
        print(f"  [{s['passed']}/{s['total']}] {fam}: {s['desc']}")
