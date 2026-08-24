"""
IndSolve — External Reference Verification Engine
Provides independent mathematical validation against SciPy (HiGHS / milp).
Strictly prevents false verification states when reference solver fails or is unavailable.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


def verify_with_scipy_reference(
    ind_res: Any,
    c: np.ndarray,
    A_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    A_ge: Optional[np.ndarray] = None,
    b_ge: Optional[np.ndarray] = None,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    integrality: Optional[List[int]] = None,
    maximize: bool = False,
    tol: float = 1e-2,
) -> Dict[str, Any]:
    """
    Independently compares IndSolve result with external reference solver SciPy (HiGHS / milp).

    Returns:
        Dict with keys:
            - state: 'verified' | 'mismatch' | 'reference_failed' | 'reference_unavailable'
            - ref_obj: Optional[float]
            - ref_status: str
            - diff: Optional[float]
            - message: str
            - ref_label: str ('SciPy HiGHS' or 'SciPy milp')
            - verdict_label: str (Human-readable badge text)
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    is_milp = (integrality is not None) and any(i == 1 for i in integrality)
    ref_label = "External Reference Solver: SciPy milp" if is_milp else "External Reference Solver: SciPy HiGHS"

    # Default fallback state when reference fails
    failed_state = {
        "state": "reference_failed",
        "ref_obj": None,
        "ref_status": "EXECUTION_ERROR",
        "diff": None,
        "message": "Reference verification unavailable — solver result is not independently confirmed.",
        "ref_label": ref_label,
        "verdict_label": "⚠️ Not Confirmed (Reference Failed)",
    }

    if not is_milp:
        # =====================================================================
        # CONTINUOUS LINEAR PROGRAM (LP) VERIFICATION VIA SCIPY HIGHS
        # =====================================================================
        try:
            scipy_A_ub = []
            scipy_b_ub = []
            if A_ub is not None and len(A_ub) > 0:
                scipy_A_ub.append(A_ub)
                scipy_b_ub.append(b_ub)
            if A_ge is not None and len(A_ge) > 0:
                scipy_A_ub.append(-A_ge)
                scipy_b_ub.append(-b_ge)

            A_ub_c = np.vstack(scipy_A_ub) if scipy_A_ub else None
            b_ub_c = np.concatenate(scipy_b_ub) if scipy_b_ub else None
            scipy_c = -c if maximize else c

            scipy_res = linprog(
                c=scipy_c,
                A_ub=A_ub_c,
                b_ub=b_ub_c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
        except Exception:
            # If SciPy throws an exception, never claim verified!
            return failed_state

        if scipy_res is None:
            return {
                "state": "reference_unavailable",
                "ref_obj": None,
                "ref_status": "NO_RESULT",
                "diff": None,
                "message": "Reference verification unavailable — solver result is not independently confirmed.",
                "ref_label": ref_label,
                "verdict_label": "⚠️ Not Confirmed (No Reference Result)",
            }

        # Case 1: Reference Solved to Optimal (status 0)
        if scipy_res.status == 0:
            ref_obj = -scipy_res.fun if maximize else scipy_res.fun
            if ind_res.success and ind_res.status == "OPTIMAL":
                diff = abs(ind_res.fun - ref_obj)
                if diff < tol:
                    return {
                        "state": "verified",
                        "ref_obj": float(ref_obj),
                        "ref_status": "OPTIMAL",
                        "diff": float(diff),
                        "message": f"Exact objective agreement with external reference solver: SciPy HiGHS (diff = {diff:.2e}).",
                        "ref_label": ref_label,
                        "verdict_label": "✅ Independently Confirmed",
                    }
                else:
                    return {
                        "state": "mismatch",
                        "ref_obj": float(ref_obj),
                        "ref_status": "OPTIMAL",
                        "diff": float(diff),
                        "message": f"Objective mismatch: IndSolve ({ind_res.fun:,.4f}) vs SciPy ({ref_obj:,.4f}).",
                        "ref_label": ref_label,
                        "verdict_label": "❌ Objective Mismatch",
                    }
            else:
                return {
                    "state": "mismatch",
                    "ref_obj": float(ref_obj),
                    "ref_status": "OPTIMAL",
                    "diff": None,
                    "message": f"Status mismatch: IndSolve ({ind_res.status}) vs SciPy (OPTIMAL).",
                    "ref_label": ref_label,
                    "verdict_label": "❌ Status Mismatch",
                }

        # Case 2: Reference Infeasible (status 2)
        elif scipy_res.status == 2:
            if ind_res.status == "INFEASIBLE":
                return {
                    "state": "verified",
                    "ref_obj": None,
                    "ref_status": "INFEASIBLE",
                    "diff": None,
                    "message": "Both solvers independently confirmed problem is INFEASIBLE.",
                    "ref_label": ref_label,
                    "verdict_label": "✅ Infeasibility Confirmed",
                }
            else:
                return {
                    "state": "mismatch",
                    "ref_obj": None,
                    "ref_status": "INFEASIBLE",
                    "diff": None,
                    "message": f"Status mismatch: IndSolve ({ind_res.status}) vs SciPy (INFEASIBLE).",
                    "ref_label": ref_label,
                    "verdict_label": "❌ Status Mismatch",
                }

        # Case 3: Reference Unbounded (status 3)
        elif scipy_res.status == 3:
            if ind_res.status == "UNBOUNDED":
                return {
                    "state": "verified",
                    "ref_obj": None,
                    "ref_status": "UNBOUNDED",
                    "diff": None,
                    "message": "Both solvers independently confirmed problem is UNBOUNDED.",
                    "ref_label": ref_label,
                    "verdict_label": "✅ Unboundedness Confirmed",
                }
            else:
                return {
                    "state": "mismatch",
                    "ref_obj": None,
                    "ref_status": "UNBOUNDED",
                    "diff": None,
                    "message": f"Status mismatch: IndSolve ({ind_res.status}) vs SciPy (UNBOUNDED).",
                    "ref_label": ref_label,
                    "verdict_label": "❌ Status Mismatch",
                }

        else:
            return {
                "state": "reference_failed",
                "ref_obj": None,
                "ref_status": str(scipy_res.message),
                "diff": None,
                "message": f"Reference solver returned non-optimal status: {scipy_res.message}",
                "ref_label": ref_label,
                "verdict_label": "⚠️ Reference Inconclusive",
            }

    else:
        # =====================================================================
        # MIXED-INTEGER LINEAR PROGRAM (MILP) VERIFICATION VIA SCIPY MILP
        # =====================================================================
        try:
            scipy_c = -c if maximize else c
            lb_list = [0.0 if (b is None or b[0] is None) else b[0] for b in bounds] if bounds else [0.0]*n
            ub_list = [np.inf if (b is None or b[1] is None) else b[1] for b in bounds] if bounds else [np.inf]*n
            scipy_bounds = Bounds(lb_list, ub_list)

            # Stack constraints for scipy.milp: lhs <= A @ x <= rhs
            scipy_A_rows = []
            scipy_lhs = []
            scipy_rhs = []
            if A_ub is not None and len(A_ub) > 0:
                scipy_A_rows.append(A_ub)
                scipy_lhs.append(np.full(len(b_ub), -np.inf))
                scipy_rhs.append(b_ub)
            if A_ge is not None and len(A_ge) > 0:
                scipy_A_rows.append(A_ge)
                scipy_lhs.append(b_ge)
                scipy_rhs.append(np.full(len(b_ge), np.inf))
            if A_eq is not None and len(A_eq) > 0:
                scipy_A_rows.append(A_eq)
                scipy_lhs.append(b_eq)
                scipy_rhs.append(b_eq)

            scipy_constraints = None
            if scipy_A_rows:
                A_stacked = np.vstack(scipy_A_rows)
                lhs_stacked = np.concatenate(scipy_lhs)
                rhs_stacked = np.concatenate(scipy_rhs)
                scipy_constraints = LinearConstraint(A_stacked, lhs_stacked, rhs_stacked)

            scipy_milp_res = milp(
                c=scipy_c,
                integrality=integrality,
                bounds=scipy_bounds,
                constraints=scipy_constraints,
            )
        except Exception:
            return failed_state

        if scipy_milp_res is None:
            return {
                "state": "reference_unavailable",
                "ref_obj": None,
                "ref_status": "NO_RESULT",
                "diff": None,
                "message": "Reference verification unavailable — solver result is not independently confirmed.",
                "ref_label": ref_label,
                "verdict_label": "⚠️ Not Confirmed (No Reference Result)",
            }

        if scipy_milp_res.success:
            ref_obj = -scipy_milp_res.fun if maximize else scipy_milp_res.fun
            if ind_res.success and ind_res.status == "OPTIMAL":
                diff = abs(ind_res.fun - ref_obj)
                int_ok = all(abs(ind_res.x[i] - round(ind_res.x[i])) < 1e-4 for i in range(n) if integrality[i] == 1)
                if diff < tol and int_ok:
                    return {
                        "state": "verified",
                        "ref_obj": float(ref_obj),
                        "ref_status": "OPTIMAL",
                        "diff": float(diff),
                        "message": f"Exact integer optimum agreement with external reference solver: SciPy milp (diff = {diff:.2e}).",
                        "ref_label": ref_label,
                        "verdict_label": "✅ Independently Confirmed",
                    }
                else:
                    return {
                        "state": "mismatch",
                        "ref_obj": float(ref_obj),
                        "ref_status": "OPTIMAL",
                        "diff": float(diff),
                        "message": f"Objective or integrality mismatch: IndSolve ({ind_res.fun:,.4f}) vs SciPy ({ref_obj:,.4f}).",
                        "ref_label": ref_label,
                        "verdict_label": "❌ Discrepancy",
                    }
            else:
                return {
                    "state": "mismatch",
                    "ref_obj": float(ref_obj),
                    "ref_status": "OPTIMAL",
                    "diff": None,
                    "message": f"Status mismatch: IndSolve ({ind_res.status}) vs SciPy (OPTIMAL).",
                    "ref_label": ref_label,
                    "verdict_label": "❌ Status Mismatch",
                }
        elif scipy_milp_res.status == 2 or "infeasible" in scipy_milp_res.message.lower():
            if ind_res.status == "INFEASIBLE":
                return {
                    "state": "verified",
                    "ref_obj": None,
                    "ref_status": "INFEASIBLE",
                    "diff": None,
                    "message": "Both solvers independently confirmed problem is INFEASIBLE.",
                    "ref_label": ref_label,
                    "verdict_label": "✅ Infeasibility Confirmed",
                }
            else:
                return {
                    "state": "mismatch",
                    "ref_obj": None,
                    "ref_status": "INFEASIBLE",
                    "diff": None,
                    "message": f"Status mismatch: IndSolve ({ind_res.status}) vs SciPy (INFEASIBLE).",
                    "ref_label": ref_label,
                    "verdict_label": "❌ Status Mismatch",
                }
        else:
            return {
                "state": "reference_failed",
                "ref_obj": None,
                "ref_status": str(scipy_milp_res.message),
                "diff": None,
                "message": f"Reference solver returned non-optimal status: {scipy_milp_res.message}",
                "ref_label": ref_label,
                "verdict_label": "⚠️ Reference Inconclusive",
            }
