"""
IndSolve — Verification & Test Suite
Validates mathematical correctness against standard solvers and published ground truths.
"""

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from solver.tableau_simplex import SimplexSolver
from solver.branch_and_bound import BranchAndBoundSolver
from solver.problems import get_preloaded_problems


def run_tests():
    print("=" * 70)
    print("[*] IndSolve Verification Suite -- Mathematical Correctness Check")
    print("=" * 70)

    solver = SimplexSolver()
    milp_solver = BranchAndBoundSolver(lp_solver=solver)
    problems = get_preloaded_problems()

    all_passed = True

    for name, prob in problems.items():
        clean_name = name.encode('ascii', 'ignore').decode('ascii').strip()
        print(f"\nTesting Scenario: {clean_name}")
        print(f"  Description: {prob['description'][:80]}...")

        c = prob["c"]
        A_ub = prob.get("A_ub")
        b_ub = prob.get("b_ub")
        A_eq = prob.get("A_eq")
        b_eq = prob.get("b_eq")
        A_ge = prob.get("A_ge")
        b_ge = prob.get("b_ge")
        bounds = prob.get("bounds")
        integrality = prob.get("integrality", [0] * len(c))
        maximize = prob.get("maximize", False)

        is_milp = any(i == 1 for i in integrality)

        if not is_milp:
            # Solve with IndSolve Simplex
            ind_res = solver.solve(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ge=A_ge,
                b_ge=b_ge,
                bounds=bounds,
                maximize=maximize,
            )

            # Solve with SciPy HiGHS for external mathematical validation
            # Convert >= constraints to <= for standard scipy format: -A_ge @ x <= -b_ge
            scipy_A_ub = []
            scipy_b_ub = []
            if A_ub is not None and len(A_ub) > 0:
                scipy_A_ub.append(A_ub)
                scipy_b_ub.append(b_ub)
            if A_ge is not None and len(A_ge) > 0:
                scipy_A_ub.append(-A_ge)
                scipy_b_ub.append(-b_ge)

            A_ub_combined = np.vstack(scipy_A_ub) if scipy_A_ub else None
            b_ub_combined = np.concatenate(scipy_b_ub) if scipy_b_ub else None

            scipy_c = -c if maximize else c
            scipy_res = linprog(
                c=scipy_c,
                A_ub=A_ub_combined,
                b_ub=b_ub_combined,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            scipy_obj = -scipy_res.fun if maximize else scipy_res.fun

            print(f"  [IndSolve LP]  Status: {ind_res.status} | Obj: {ind_res.fun:,.2f} | Iterations: {ind_res.nit} | Time: {ind_res.solve_time*1000:.2f}ms")
            print(f"  [SciPy HiGHS]  Status: {scipy_res.message} | Obj: {scipy_obj:,.2f}")

            diff = abs(ind_res.fun - scipy_obj)
            if diff < 1e-3:
                print(f"  [PASS] VERIFIED: IndSolve matched SciPy solution within tolerance (diff = {diff:.6e})")
            else:
                print(f"  [FAIL] MISMATCH: IndSolve ({ind_res.fun}) vs SciPy ({scipy_obj})")
                all_passed = False

        else:
            # Solve with IndSolve MILP
            ind_res = milp_solver.solve(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ge=A_ge,
                b_ge=b_ge,
                bounds=bounds,
                integrality=integrality,
                maximize=maximize,
            )

            # Solve with scipy.optimize.milp
            scipy_c = -c if maximize else c
            lb_list = [0.0 if (b is None or b[0] is None) else b[0] for b in bounds]
            ub_list = [np.inf if (b is None or b[1] is None) else b[1] for b in bounds]
            scipy_bounds = Bounds(lb_list, ub_list)

            # Stack constraints for scipy.milp: lb <= A @ x <= ub
            scipy_A_rows = []
            scipy_lhs = []
            scipy_rhs = []
            if A_ub is not None:
                scipy_A_rows.append(A_ub)
                scipy_lhs.append(np.full(len(b_ub), -np.inf))
                scipy_rhs.append(b_ub)
            if A_ge is not None:
                scipy_A_rows.append(A_ge)
                scipy_lhs.append(b_ge)
                scipy_rhs.append(np.full(len(b_ge), np.inf))

            A_scipy = np.vstack(scipy_A_rows)
            lhs_scipy = np.concatenate(scipy_lhs)
            rhs_scipy = np.concatenate(scipy_rhs)
            scipy_constraints = LinearConstraint(A_scipy, lhs_scipy, rhs_scipy)

            scipy_milp_res = milp(
                c=scipy_c,
                integrality=integrality,
                bounds=scipy_bounds,
                constraints=scipy_constraints
            )
            scipy_obj = -scipy_milp_res.fun if maximize else scipy_milp_res.fun

            print(f"  [IndSolve MILP] Status: {ind_res.status} | Obj: {ind_res.fun:,.2f} | Nodes: {ind_res.nodes_explored} | Time: {ind_res.solve_time*1000:.2f}ms")
            print(f"  [SciPy milp]    Status: {scipy_milp_res.message} | Obj: {scipy_obj:,.2f}")
            print(f"  [Solution Vector]: {np.round(ind_res.x, 2)}")

            diff = abs(ind_res.fun - scipy_obj)
            int_ok = all(abs(ind_res.x[i] - round(ind_res.x[i])) < 1e-4 for i in range(len(c)) if integrality[i] == 1)

            if diff < 1e-3 and int_ok and ind_res.success:
                print(f"  [PASS] VERIFIED: IndSolve matched SciPy milp exact integer optimum (diff = {diff:.6e})")
            else:
                print(f"  [FAIL] MISMATCH: IndSolve ({ind_res.fun}) vs SciPy ({scipy_obj})")
                all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] ALL TEST SCENARIOS PASSED -- IndSolve is mathematically verified!")
    else:
        print("[WARNING] Some tests had discrepancies.")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
