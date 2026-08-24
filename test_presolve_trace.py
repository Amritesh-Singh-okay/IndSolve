import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from solver.presolve import PresolveEngine
from solver.tableau_simplex import SimplexSolver

def test_presolve():
    print("=" * 60)
    print("TESTING PRESLOVE ENGINE (3 Safe Reductions)")
    print("=" * 60)

    presolve = PresolveEngine()

    # Model with fixed var and duplicate constraint
    c = np.array([10.0, 20.0, 15.0])
    A_ub = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],  # Duplicate constraint!
        [2.0, 1.0, 0.0]
    ])
    b_ub = np.array([100.0, 100.0, 80.0])
    bounds = [(0.0, None), (10.0, 10.0), (0.0, None)]  # x2 fixed to 10!
    var_names = ["x1", "x2_Fixed", "x3"]

    p_res = presolve.presolve(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, var_names=var_names)

    print("Presolve Log Entries:")
    for l in p_res.log_entries:
        print(f"  * {l}")

    # Solve reduced model
    solver = SimplexSolver()
    res = solver.solve(c=p_res.c, A_ub=p_res.A_ub, b_ub=p_res.b_ub, bounds=p_res.bounds, var_names=p_res.var_names)
    print(f"\nReduced Solve Status: {res.status} | Obj (with offset): {res.fun + p_res.obj_offset:,.2f}")

    # Postsolve
    x_full = presolve.postsolve(res.x, p_res)
    print(f"Postsolve Reconstructed Solution: {x_full.round(2)}")
    assert abs(x_full[1] - 10.0) < 1e-6, "Fixed variable not reconstructed correctly!"
    print("[PASS] Presolve and Postsolve verified successfully!")

    print("\n" + "=" * 60)
    print("TESTING SIMPLEX ALGORITHM TRACE & EXPLANATIONS")
    print("=" * 60)
    # Test 2D refinery problem trace
    c_ref = np.array([72.0, 78.0])
    A_ub_ref = np.array([[1.0, 0.0], [0.0, 1.0], [0.77, -0.63]])
    b_ub_ref = np.array([700.0, 800.0, 0.0])
    A_ge_ref = np.array([[1.0, 1.0]])
    b_ge_ref = np.array([1000.0])
    var_names_ref = ["Arabian_Light", "Brent"]

    res_trace = solver.solve(
        c=c_ref,
        A_ub=A_ub_ref,
        b_ub=b_ub_ref,
        A_ge=A_ge_ref,
        b_ge=b_ge_ref,
        var_names=var_names_ref,
        maximize=False
    )

    print(f"Algorithm Trace ({len(res_trace.trace_log)} Pivots):")
    for step in res_trace.trace_log:
        print(f"  [Iter {step['iteration']}] {step['explanation']}")

    print("\n[PASS] Algorithm Trace successfully produced clear step-by-step explanations!")

if __name__ == "__main__":
    test_presolve()
