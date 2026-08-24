"""
IndSolve — Solver Proof & Zero-Trust Constraint Audit Engine
Performs independent mathematical verification directly from raw solution vectors.
Recomputes objective dot products, audits all inequality/equality constraints,
evaluates variable bounds, and verifies integrality without trusting solver trace text.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


def audit_solution(
    c: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_ge: Optional[np.ndarray],
    b_ge: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]],
    integrality: Optional[List[int]],
    var_names: List[str],
    x_sol: np.ndarray,
    reported_obj: float,
    maximize: bool = False,
    tol: float = 1e-4,
    constraint_names_ub: Optional[List[str]] = None,
    constraint_names_ge: Optional[List[str]] = None,
    constraint_names_eq: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Independently verifies every constraint, variable bound, integrality rule,
    and recalculates the exact objective dot product c @ x from scratch.
    """
    c = np.asarray(c, dtype=np.float64)
    x_sol = np.asarray(x_sol, dtype=np.float64)
    n = len(c)

    # 1. Objective Recalculation: c @ x
    computed_raw_dot = float(np.dot(c, x_sol))
    computed_obj = computed_raw_dot

    obj_diff = abs(reported_obj - computed_obj)
    obj_verified = obj_diff < tol

    # Detailed term-by-term formula string
    terms = []
    for i in range(n):
        if abs(x_sol[i]) > 1e-6 or n <= 5:
            terms.append(f"({c[i]:g} × {x_sol[i]:.3f})")
    obj_formula_str = " + ".join(terms) if terms else "0.0"
    obj_recalc_display = f"{obj_formula_str} = {computed_obj:,.4f}"

    # 2. Constraint Audits
    constraint_rows = []
    total_constraints = 0
    passed_constraints = 0

    def make_lhs_formula(A_row, vnames):
        row_terms = []
        for j, val in enumerate(A_row):
            if abs(val) > 1e-6:
                coeff = f"{val:g}" if val != 1.0 else ""
                row_terms.append(f"{coeff}{vnames[j]}")
        return " + ".join(row_terms) if row_terms else "0"

    # Auditing <= Constraints
    if A_ub is not None and b_ub is not None and len(b_ub) > 0:
        for i in range(len(b_ub)):
            total_constraints += 1
            lhs_val = float(np.dot(A_ub[i], x_sol))
            rhs_val = float(b_ub[i])
            slack = rhs_val - lhs_val
            passed = lhs_val <= rhs_val + tol
            if passed:
                passed_constraints += 1

            status_str = "✅ Pass" if passed else "❌ Violated"
            margin_str = f"+{slack:,.4f} (Slack)" if slack >= 0 else f"{slack:,.4f} (Excess)"
            formula = make_lhs_formula(A_ub[i], var_names)

            label = constraint_names_ub[i] if (constraint_names_ub and i < len(constraint_names_ub)) else f"Limit (≤) #{i+1}"

            constraint_rows.append({
                "Constraint / Rule": label,
                "Mathematical Formula": f"{formula} ≤ {rhs_val:g}",
                "Computed LHS": f"{lhs_val:,.4f}",
                "Rule": "≤",
                "Allowed RHS": f"{rhs_val:,.4f}",
                "Slack / Margin": margin_str,
                "Audit Result": status_str,
                "is_passed": passed
            })

    # Auditing >= Constraints
    if A_ge is not None and b_ge is not None and len(b_ge) > 0:
        for i in range(len(b_ge)):
            total_constraints += 1
            lhs_val = float(np.dot(A_ge[i], x_sol))
            rhs_val = float(b_ge[i])
            surplus = lhs_val - rhs_val
            passed = lhs_val >= rhs_val - tol
            if passed:
                passed_constraints += 1

            status_str = "✅ Pass" if passed else "❌ Violated"
            margin_str = f"+{surplus:,.4f} (Surplus)" if surplus >= 0 else f"{surplus:,.4f} (Deficit)"
            formula = make_lhs_formula(A_ge[i], var_names)

            label = constraint_names_ge[i] if (constraint_names_ge and i < len(constraint_names_ge)) else f"Requirement (≥) #{i+1}"

            constraint_rows.append({
                "Constraint / Rule": label,
                "Mathematical Formula": f"{formula} ≥ {rhs_val:g}",
                "Computed LHS": f"{lhs_val:,.4f}",
                "Rule": "≥",
                "Allowed RHS": f"{rhs_val:,.4f}",
                "Slack / Margin": margin_str,
                "Audit Result": status_str,
                "is_passed": passed
            })

    # Auditing == Constraints
    if A_eq is not None and b_eq is not None and len(b_eq) > 0:
        for i in range(len(b_eq)):
            total_constraints += 1
            lhs_val = float(np.dot(A_eq[i], x_sol))
            rhs_val = float(b_eq[i])
            diff = abs(lhs_val - rhs_val)
            passed = diff <= tol
            if passed:
                passed_constraints += 1

            status_str = "✅ Pass" if passed else "❌ Violated"
            margin_str = f"{diff:.6f} (Residual)"
            formula = make_lhs_formula(A_eq[i], var_names)

            label = constraint_names_eq[i] if (constraint_names_eq and i < len(constraint_names_eq)) else f"Balance (==) #{i+1}"

            constraint_rows.append({
                "Constraint / Rule": label,
                "Mathematical Formula": f"{formula} == {rhs_val:g}",
                "Computed LHS": f"{lhs_val:,.4f}",
                "Rule": "==",
                "Allowed RHS": f"{rhs_val:,.4f}",
                "Slack / Margin": margin_str,
                "Audit Result": status_str,
                "is_passed": passed
            })

    # 3. Variable Bound Audits
    if bounds is not None:
        for i, (lb, ub) in enumerate(bounds):
            total_constraints += 1
            val = float(x_sol[i])
            lb_val = -float("inf") if lb is None else float(lb)
            ub_val = float("inf") if ub is None else float(ub)

            lb_pass = val >= lb_val - tol
            ub_pass = val <= ub_val + tol
            passed = lb_pass and ub_pass
            if passed:
                passed_constraints += 1

            status_str = "✅ Pass" if passed else "❌ Violated"
            lb_display = "-∞" if lb is None else f"{lb_val:g}"
            ub_display = "+∞" if ub is None else f"{ub_val:g}"

            constraint_rows.append({
                "Constraint / Rule": f"Bound: {var_names[i]}",
                "Mathematical Formula": f"{lb_display} ≤ {var_names[i]} ≤ {ub_display}",
                "Computed LHS": f"{val:,.4f}",
                "Rule": "in range",
                "Allowed RHS": f"[{lb_display}, {ub_display}]",
                "Slack / Margin": "Within Bounds" if passed else "Out of Bounds",
                "Audit Result": status_str,
                "is_passed": passed
            })

    # 4. Integrality Audits
    if integrality is not None:
        for i, is_int in enumerate(integrality):
            if is_int == 1:
                total_constraints += 1
                val = float(x_sol[i])
                round_val = round(val)
                diff = abs(val - round_val)
                passed = diff <= tol
                if passed:
                    passed_constraints += 1

                status_str = "✅ Pass" if passed else "❌ Violated"
                constraint_rows.append({
                    "Constraint / Rule": f"Integrality: {var_names[i]}",
                    "Mathematical Formula": f"{var_names[i]} ∈ ℤ",
                    "Computed LHS": f"{val:.6f}",
                    "Rule": "∈ ℤ",
                    "Allowed RHS": f"{round_val}",
                    "Slack / Margin": f"{diff:.6e} (Fractional Part)",
                    "Audit Result": status_str,
                    "is_passed": passed
                })

    all_passed = (passed_constraints == total_constraints) and obj_verified
    constraint_df = pd.DataFrame(constraint_rows)

    return {
        "all_passed": all_passed,
        "total_constraints": total_constraints,
        "passed_constraints": passed_constraints,
        "obj_verified": obj_verified,
        "computed_obj": computed_obj,
        "reported_obj": reported_obj,
        "obj_recalc_display": obj_recalc_display,
        "constraint_df": constraint_df,
    }
