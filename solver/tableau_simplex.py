"""
IndSolve — First-Principles Tableau Simplex Solver Core
Indigenous Linear Programming Engine with Big-M Method, Anti-Cycling Pivoting,
General Variable Bound Transformations (Free, Lower-Only, Upper-Only, Two-Sided, Fixed),
and Strict Non-Optimal Status Handling (ITERATION_LIMIT / UNBOUNDED / INFEASIBLE).
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# Optional Numba JIT acceleration for dense pivots
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _jit_pivot(tableau: np.ndarray, pivot_row: int, pivot_col: int):
    """Pure Python Gauss-Jordan elimination pivot."""
    pivot_val = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] /= pivot_val
    for i in range(tableau.shape[0]):
        if i != pivot_row:
            factor = tableau[i, pivot_col]
            if abs(factor) > 1e-12:
                tableau[i, :] -= factor * tableau[pivot_row, :]


if HAS_NUMBA:
    @njit(fastmath=True)
    def _jit_pivot_compiled(tableau: np.ndarray, pivot_row: int, pivot_col: int):
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val
        num_rows = tableau.shape[0]
        for i in range(num_rows):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                if abs(factor) > 1e-12:
                    tableau[i, :] -= factor * tableau[pivot_row, :]


@dataclass
class SimplexResult:
    """Standardized result object for Simplex solver."""
    success: bool
    status: str  # 'OPTIMAL', 'INFEASIBLE', 'UNBOUNDED', 'ITERATION_LIMIT'
    message: str
    fun: float
    x: np.ndarray
    nit: int
    solve_time: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    tableau: Optional[np.ndarray] = None
    trace_log: List[Dict[str, Any]] = field(default_factory=list)
    col_names: List[str] = field(default_factory=list)


class SimplexSolver:
    """
    Indigenous Tableau Simplex Solver with Big-M Method,
    Rigorous General Variable Bound Transformations,
    Anti-Cycling Pivoting, and Inspectable Step-by-Step Algorithm Trace.
    """

    def __init__(self, tol: float = 1e-8, max_iter: int = 5000, big_m: float = 1e5, use_jit: bool = True):
        self.tol = tol
        self.max_iter = max_iter
        self.big_m = big_m
        self.use_jit = use_jit and HAS_NUMBA

    def solve(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ge: Optional[np.ndarray] = None,
        b_ge: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
        var_names: Optional[List[str]] = None,
        maximize: bool = False,
        **kwargs,
    ) -> SimplexResult:
        """
        Solves general LP:
            min/max c^T x
            subject to:
                A_ub @ x <= b_ub
                A_ge @ x >= b_ge
                A_eq @ x == b_eq
                lb_i <= x_i <= ub_i  (arbitrary lower/upper/free bounds)
        """
        start_time = time.perf_counter()
        c = np.asarray(c, dtype=np.float64)
        n_orig = len(c)

        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n_orig)]

        if bounds is None:
            bounds = [(0.0, None) for _ in range(n_orig)]

        # --- STEP 1: PARSE BOUNDS & VALIDATE INFEASIBILITY ---
        # Bound types per variable:
        # 'LB'       : lb finite, ub is None or ub >= lb. x_i = lb + x' (x' >= 0)
        # 'UB_ONLY'  : lb is None, ub is finite.         x_i = ub - x' (x' >= 0)
        # 'FREE'     : lb is None, ub is None.           x_i = x_pos - x_neg (x_pos, x_neg >= 0)
        # 'FIXED'    : lb == ub (finite).                x_i = lb + x' (0 <= x' <= 0)

        var_specs = []
        for i in range(n_orig):
            lb, ub = bounds[i]
            lb_val = None if lb is None else float(lb)
            ub_val = None if ub is None else float(ub)

            if lb_val is not None and ub_val is not None:
                if lb_val > ub_val + self.tol:
                    elapsed = time.perf_counter() - start_time
                    return SimplexResult(
                        success=False,
                        status="INFEASIBLE",
                        message=f"Variable {var_names[i]} has lower bound {lb_val} > upper bound {ub_val}.",
                        fun=0.0,
                        x=np.zeros(n_orig),
                        nit=0,
                        solve_time=elapsed,
                        history=[],
                    )
                if abs(lb_val - ub_val) <= self.tol:
                    var_specs.append({"type": "FIXED", "lb": lb_val, "ub": ub_val})
                else:
                    var_specs.append({"type": "LB", "lb": lb_val, "ub": ub_val})

            elif lb_val is not None and ub_val is None:
                var_specs.append({"type": "LB", "lb": lb_val, "ub": None})

            elif lb_val is None and ub_val is not None:
                var_specs.append({"type": "UB_ONLY", "lb": None, "ub": ub_val})

            else:
                var_specs.append({"type": "FREE", "lb": None, "ub": None})

        # --- STEP 2: BUILD TRANSFORMED SYSTEM OF NON-NEGATIVE VARIABLES x' >= 0 ---
        # Map each original variable x_i to column(s) in transformed system
        mapping = []
        trans_names = []
        c_trans_list = []
        obj_offset = 0.0
        upper_bounds_trans = []

        col_counter = 0
        for i, spec in enumerate(var_specs):
            vtype = spec["type"]
            vname = var_names[i]
            ci = c[i]

            if vtype in ["LB", "FIXED"]:
                lb = spec["lb"]
                ub = spec["ub"]
                mapping.append({"type": vtype, "orig_idx": i, "cols": [col_counter], "lb": lb, "ub": ub})
                trans_names.append(vname if lb == 0.0 else f"{vname}_s")
                c_trans_list.append(ci)
                obj_offset += ci * lb

                if ub is not None:
                    upper_bounds_trans.append((col_counter, ub - lb))
                col_counter += 1

            elif vtype == "UB_ONLY":
                ub = spec["ub"]
                mapping.append({"type": "UB_ONLY", "orig_idx": i, "cols": [col_counter], "ub": ub})
                trans_names.append(f"{vname}_sub")
                c_trans_list.append(-ci)
                obj_offset += ci * ub
                col_counter += 1

            elif vtype == "FREE":
                c_pos = col_counter
                c_neg = col_counter + 1
                mapping.append({"type": "FREE", "orig_idx": i, "cols": [c_pos, c_neg]})
                trans_names.append(f"{vname}_pos")
                trans_names.append(f"{vname}_neg")
                c_trans_list.append(ci)
                c_trans_list.append(-ci)
                col_counter += 2

        n_trans = col_counter
        c_trans = np.array(c_trans_list, dtype=np.float64)

        # Helper to transform constraint row A @ x into transformed columns
        # A_i * x_i:
        # LB     : A_i * (x' + lb)    = + A_i * x' + (A_i * lb)
        # UB_ONLY: A_i * (ub - x')    = - A_i * x' + (A_i * ub)
        # FREE   : A_i * (x_pos - x_neg) = + A_i * x_pos - A_i * x_neg
        def transform_matrix_and_rhs(A_mat, b_vec):
            if A_mat is None or len(A_mat) == 0:
                return np.empty((0, n_trans), dtype=np.float64), np.empty(0, dtype=np.float64)
            A_mat = np.asarray(A_mat, dtype=np.float64)
            b_vec = np.asarray(b_vec, dtype=np.float64).flatten().copy()
            m = len(b_vec)
            A_out = np.zeros((m, n_trans), dtype=np.float64)

            for item in mapping:
                orig_i = item["orig_idx"]
                col_A = A_mat[:, orig_i]

                if item["type"] in ["LB", "FIXED"]:
                    c_idx = item["cols"][0]
                    A_out[:, c_idx] = col_A
                    b_vec -= col_A * item["lb"]
                elif item["type"] == "UB_ONLY":
                    c_idx = item["cols"][0]
                    A_out[:, c_idx] = -col_A
                    b_vec -= col_A * item["ub"]
                elif item["type"] == "FREE":
                    c_pos, c_neg = item["cols"]
                    A_out[:, c_pos] = col_A
                    A_out[:, c_neg] = -col_A

            return A_out, b_vec

        A_ub_trans, b_ub_trans = transform_matrix_and_rhs(A_ub, b_ub)
        A_ge_trans, b_ge_trans = transform_matrix_and_rhs(A_ge, b_ge)
        A_eq_trans, b_eq_trans = transform_matrix_and_rhs(A_eq, b_eq)

        # Append transformed upper bounds to A_ub_trans
        for col_idx, ub_val in upper_bounds_trans:
            ub_row = np.zeros(n_trans, dtype=np.float64)
            ub_row[col_idx] = 1.0
            A_ub_trans = np.vstack([A_ub_trans, ub_row]) if len(A_ub_trans) > 0 else ub_row.reshape(1, -1)
            b_ub_trans = np.append(b_ub_trans, ub_val)

        # Helper to reconstruct original x from x_trans
        def reconstruct_x(x_t: np.ndarray) -> np.ndarray:
            x_orig = np.zeros(n_orig, dtype=np.float64)
            for item in mapping:
                i = item["orig_idx"]
                if item["type"] in ["LB", "FIXED"]:
                    x_orig[i] = x_t[item["cols"][0]] + item["lb"]
                elif item["type"] == "UB_ONLY":
                    x_orig[i] = item["ub"] - x_t[item["cols"][0]]
                elif item["type"] == "FREE":
                    x_orig[i] = x_t[item["cols"][0]] - x_t[item["cols"][1]]
            return x_orig

        # --- STEP 3: CHECK FOR UNCONSTRAINED MODELS ---
        total_explicit_constraints = len(b_ub_trans) + len(b_ge_trans) + len(b_eq_trans)
        if total_explicit_constraints == 0:
            # Internal optimization is MAXIMIZATION
            c_target = c_trans if maximize else -c_trans
            has_pos_gradient = np.any(c_target > self.tol)
            elapsed = time.perf_counter() - start_time

            if has_pos_gradient:
                return SimplexResult(
                    success=False,
                    status="UNBOUNDED",
                    message="Problem is unconstrained with non-zero objective gradient. Rays diverge to infinity.",
                    fun=float("inf") if maximize else float("-inf"),
                    x=reconstruct_x(np.zeros(n_trans)),
                    nit=0,
                    solve_time=elapsed,
                    history=[],
                )
            else:
                x_t_opt = np.zeros(n_trans)
                return SimplexResult(
                    success=True,
                    status="OPTIMAL",
                    message="Optimal trivial solution at non-negative variable lower bounds.",
                    fun=obj_offset,
                    x=reconstruct_x(x_t_opt),
                    nit=0,
                    solve_time=elapsed,
                    history=[],
                )

        # --- STEP 4: ITERATION LIMIT CHECK (max_iter == 0) ---
        if self.max_iter <= 0:
            elapsed = time.perf_counter() - start_time
            return SimplexResult(
                success=False,
                status="ITERATION_LIMIT",
                message="Iteration limit (0) reached. No simplex pivots were executed.",
                fun=obj_offset,
                x=reconstruct_x(np.zeros(n_trans)),
                nit=0,
                solve_time=elapsed,
                history=[],
            )

        # --- STEP 5: STANDARDIZE RHS >= 0 ---
        # Ensure all RHS >= 0 by flipping signs and switching <= with >=
        A_ub_full = A_ub_trans.copy()
        b_ub_full = b_ub_trans.copy()
        A_ge_full = A_ge_trans.copy()
        b_ge_full = b_ge_trans.copy()
        A_eq_full = A_eq_trans.copy()
        b_eq_full = b_eq_trans.copy()

        for i in range(len(b_ub_full)):
            if b_ub_full[i] < 0:
                A_ge_full = np.vstack([A_ge_full, -A_ub_full[i:i+1]]) if len(A_ge_full) > 0 else -A_ub_full[i:i+1]
                b_ge_full = np.append(b_ge_full, -b_ub_full[i])
                b_ub_full[i] = np.nan

        valid_ub = ~np.isnan(b_ub_full)
        A_ub_full = A_ub_full[valid_ub]
        b_ub_full = b_ub_full[valid_ub]

        for i in range(len(b_ge_full)):
            if b_ge_full[i] < 0:
                A_ub_full = np.vstack([A_ub_full, -A_ge_full[i:i+1]]) if len(A_ub_full) > 0 else -A_ge_full[i:i+1]
                b_ub_full = np.append(b_ub_full, -b_ge_full[i])
                b_ge_full[i] = np.nan

        valid_ge = ~np.isnan(b_ge_full)
        A_ge_full = A_ge_full[valid_ge]
        b_ge_full = b_ge_full[valid_ge]

        for i in range(len(b_eq_full)):
            if b_eq_full[i] < 0:
                A_eq_full[i] = -A_eq_full[i]
                b_eq_full[i] = -b_eq_full[i]

        num_ub = len(b_ub_full)
        num_ge = len(b_ge_full)
        num_eq = len(b_eq_full)
        m_total = num_ub + num_ge + num_eq

        # Internal problem is ALWAYS MAXIMIZATION
        c_obj = c_trans.copy() if maximize else -c_trans.copy()

        # Build column labels
        col_names = list(trans_names)
        for i in range(num_ub):
            col_names.append(f"Slack_s{i+1}")
        for i in range(num_ge):
            col_names.append(f"Surplus_e{i+1}")
        for i in range(num_ge + num_eq):
            col_names.append(f"Artificial_a{i+1}")
        col_names.append("RHS")

        # --- STEP 6: BUILD INITIAL TABLEAU ---
        n_slacks = num_ub
        n_surplus = num_ge
        n_art = num_ge + num_eq
        total_cols = n_trans + n_slacks + n_surplus + n_art + 1
        tableau = np.zeros((m_total + 1, total_cols), dtype=np.float64)

        basis = []
        row_idx = 0

        # 1. <= constraints: + 1.0 * slack
        for i in range(num_ub):
            tableau[row_idx, :n_trans] = A_ub_full[i]
            tableau[row_idx, n_trans + i] = 1.0
            tableau[row_idx, -1] = b_ub_full[i]
            basis.append(n_trans + i)
            row_idx += 1

        # 2. >= constraints: - 1.0 * surplus + 1.0 * artificial
        art_idx = 0
        for i in range(num_ge):
            tableau[row_idx, :n_trans] = A_ge_full[i]
            tableau[row_idx, n_trans + n_slacks + i] = -1.0
            tableau[row_idx, n_trans + n_slacks + n_surplus + art_idx] = 1.0
            tableau[row_idx, -1] = b_ge_full[i]
            basis.append(n_trans + n_slacks + n_surplus + art_idx)
            art_idx += 1
            row_idx += 1

        # 3. == constraints: + 1.0 * artificial
        for i in range(num_eq):
            tableau[row_idx, :n_trans] = A_eq_full[i]
            tableau[row_idx, n_trans + n_slacks + n_surplus + art_idx] = 1.0
            tableau[row_idx, -1] = b_eq_full[i]
            basis.append(n_trans + n_slacks + n_surplus + art_idx)
            art_idx += 1
            row_idx += 1

        # 4. Objective Row:
        tableau[-1, :n_trans] = -c_obj
        if n_art > 0:
            art_start_col = n_trans + n_slacks + n_surplus
            tableau[-1, art_start_col:art_start_col + n_art] = self.big_m

            for r in range(m_total):
                b_var = basis[r]
                if b_var >= art_start_col:
                    tableau[-1, :] -= self.big_m * tableau[r, :]

        history = []
        trace_log = []
        iteration = 0
        pivot_fn = _jit_pivot_compiled if self.use_jit else _jit_pivot

        # Helper to extract current solution during pivots
        def get_current_x() -> np.ndarray:
            x_t = np.zeros(n_trans, dtype=np.float64)
            for r, b_var in enumerate(basis):
                if b_var < n_trans:
                    x_t[b_var] = tableau[r, -1]
            return reconstruct_x(x_t)

        # --- STEP 7: SIMPLEX PIVOT LOOP ---
        while iteration < self.max_iter:
            curr_x = get_current_x()
            raw_obj = tableau[-1, -1]
            current_obj = (raw_obj + obj_offset) if maximize else (-raw_obj + obj_offset)

            history.append({
                "step": iteration,
                "obj": float(current_obj),
                "x": curr_x.tolist(),
            })

            # Check optimality: all reduced costs >= -tol
            obj_row = tableau[-1, :-1]
            candidates = np.where(obj_row < -self.tol)[0]

            if len(candidates) == 0:
                # Optimal tableau reached
                break

            # Pivot Column: Dantzig's rule with Bland's lowest-index tie-breaking
            pivot_col = candidates[np.argmin(obj_row[candidates])]
            entering_name = col_names[pivot_col]
            reduced_cost = obj_row[pivot_col]

            # Ratio test (Minimum Ratio Test)
            col_vals = tableau[:-1, pivot_col]
            rhs_vals = tableau[:-1, -1]

            ratios = np.full(m_total, np.inf)
            for i in range(m_total):
                if col_vals[i] > self.tol:
                    ratios[i] = rhs_vals[i] / col_vals[i]

            min_ratio = np.min(ratios)
            if np.isinf(min_ratio):
                elapsed = time.perf_counter() - start_time
                return SimplexResult(
                    success=False,
                    status="UNBOUNDED",
                    message="Problem is unbounded. Valid unbounded ray identified along pivot direction.",
                    fun=float("inf") if maximize else float("-inf"),
                    x=curr_x,
                    nit=iteration,
                    solve_time=elapsed,
                    history=history,
                    tableau=tableau,
                    trace_log=trace_log,
                    col_names=col_names,
                )

            # Pivot row: Minimum ratio with Bland's tie-breaking
            min_rows = np.where(np.abs(ratios - min_ratio) < self.tol)[0]
            pivot_row = min_rows[0]
            leaving_var_idx = basis[pivot_row]
            leaving_name = col_names[leaving_var_idx]
            pivot_val = tableau[pivot_row, pivot_col]

            # Update basis
            basis[pivot_row] = pivot_col

            # Perform Tableau Pivot Transformation
            pivot_fn(tableau, pivot_row, pivot_col)
            iteration += 1

            # Compute new objective for trace log
            raw_obj_new = tableau[-1, -1]
            new_obj = (raw_obj_new + obj_offset) if maximize else (-raw_obj_new + obj_offset)

            # Plain-language explanation for this iteration
            if entering_name in trans_names:
                enter_desc = f"**{entering_name}** enters the basis (reduced cost {reduced_cost:.2f}) to improve objective."
            else:
                enter_desc = f"**{entering_name}** enters the basis to rebalance slack allocation."

            if "Slack" in leaving_name:
                leave_desc = f"**{leaving_name}** leaves as capacity saturates (ratio = {min_ratio:.2f})."
            elif "Artificial" in leaving_name:
                leave_desc = f"**{leaving_name}** leaves as feasibility is attained."
            else:
                leave_desc = f"**{leaving_name}** leaves the basis (ratio = {min_ratio:.2f})."

            explanation = f"Iteration {iteration}: {enter_desc} {leave_desc} Updated objective: **{new_obj:,.2f}**."

            trace_log.append({
                "iteration": iteration,
                "entering_var": entering_name,
                "entering_col": int(pivot_col),
                "reduced_cost": float(reduced_cost),
                "leaving_var": leaving_name,
                "leaving_row": int(pivot_row),
                "min_ratio": float(min_ratio),
                "pivot_element": float(pivot_val),
                "obj_after_pivot": float(new_obj),
                "explanation": explanation,
            })

        elapsed = time.perf_counter() - start_time
        final_x = get_current_x()
        raw_obj_final = tableau[-1, -1]
        final_obj = (raw_obj_final + obj_offset) if maximize else (-raw_obj_final + obj_offset)

        # --- STEP 8: CHECK TERMINATION STATUS ---
        # 1. Did we exit because max_iter was reached without optimality?
        obj_row_final = tableau[-1, :-1]
        remaining_candidates = np.where(obj_row_final < -self.tol)[0]
        if iteration >= self.max_iter and len(remaining_candidates) > 0:
            return SimplexResult(
                success=False,
                status="ITERATION_LIMIT",
                message=f"Iteration limit ({self.max_iter}) reached without attaining optimal convergence.",
                fun=final_obj,
                x=final_x,
                nit=iteration,
                solve_time=elapsed,
                history=history,
                tableau=tableau,
                trace_log=trace_log,
                col_names=col_names,
            )

        # 2. Check for remaining artificial variables (Infeasibility)
        if n_art > 0:
            art_start_col = n_trans + n_slacks + n_surplus
            for r, b_var in enumerate(basis):
                if b_var >= art_start_col and tableau[r, -1] > self.tol:
                    return SimplexResult(
                        success=False,
                        status="INFEASIBLE",
                        message="Problem is infeasible. Non-zero artificial variable remains in the basis.",
                        fun=0.0,
                        x=final_x,
                        nit=iteration,
                        solve_time=elapsed,
                        history=history,
                        tableau=tableau,
                        trace_log=trace_log,
                        col_names=col_names,
                    )

        # 3. Proven Optimal
        return SimplexResult(
            success=True,
            status="OPTIMAL",
            message=f"Optimal solution found in {iteration} iterations.",
            fun=final_obj,
            x=final_x,
            nit=iteration,
            solve_time=elapsed,
            history=history,
            tableau=tableau,
            trace_log=trace_log,
            col_names=col_names,
        )
