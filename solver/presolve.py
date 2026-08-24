"""
IndSolve — Small, Transparent Presolve & Postsolve Engine
Performs 3 mathematically safe model reductions:
1. Fixed Variable Propagation (lb == ub)
2. Duplicate Constraint Elimination
3. Immediate Bound Contradiction Detection
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class PresolveResult:
    def __init__(
        self,
        is_infeasible: bool,
        infeasible_reason: Optional[str],
        c: np.ndarray,
        A_ub: Optional[np.ndarray],
        b_ub: Optional[np.ndarray],
        A_ge: Optional[np.ndarray],
        b_ge: Optional[np.ndarray],
        A_eq: Optional[np.ndarray],
        b_eq: Optional[np.ndarray],
        bounds: List[Tuple[Optional[float], Optional[float]]],
        var_names: List[str],
        fixed_vars: Dict[int, float],
        active_indices: List[int],
        obj_offset: float,
        log_entries: List[str],
        original_stats: Dict[str, int],
        reduced_stats: Dict[str, int],
    ):
        self.is_infeasible = is_infeasible
        self.infeasible_reason = infeasible_reason
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_ge = A_ge
        self.b_ge = b_ge
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
        self.var_names = var_names
        self.fixed_vars = fixed_vars
        self.active_indices = active_indices
        self.obj_offset = obj_offset
        self.log_entries = log_entries
        self.original_stats = original_stats
        self.reduced_stats = reduced_stats


class PresolveEngine:
    """
    Transparent Presolve Engine with Postsolve Solution Reconstruction.
    """

    def __init__(self, tol: float = 1e-8):
        self.tol = tol

    def presolve(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_ge: Optional[np.ndarray] = None,
        b_ge: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
        var_names: Optional[List[str]] = None,
    ) -> PresolveResult:
        c = np.asarray(c, dtype=np.float64)
        n_orig = len(c)
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n_orig)]
        if bounds is None:
            bounds = [(0.0, None) for _ in range(n_orig)]

        log_entries = []
        n_ub_orig = len(b_ub) if (b_ub is not None and len(b_ub) > 0) else 0
        n_ge_orig = len(b_ge) if (b_ge is not None and len(b_ge) > 0) else 0
        n_eq_orig = len(b_eq) if (b_eq is not None and len(b_eq) > 0) else 0
        total_constraints_orig = n_ub_orig + n_ge_orig + n_eq_orig

        orig_stats = {
            "vars": n_orig,
            "constraints": total_constraints_orig,
            "ub_rows": n_ub_orig,
            "ge_rows": n_ge_orig,
            "eq_rows": n_eq_orig,
        }
        log_entries.append(f"Original model: {n_orig} variables, {total_constraints_orig} constraints.")

        # --- 1. IMMEDIATE BOUND CONTRADICTION DETECTION ---
        for j in range(n_orig):
            lb, ub = bounds[j]
            if lb is not None and ub is not None:
                if lb > ub + self.tol:
                    reason = f"Variable '{var_names[j]}' lower bound ({lb:g}) strictly exceeds upper bound ({ub:g})."
                    log_entries.append(f"❌ Bound Contradiction: {reason}")
                    return PresolveResult(
                        is_infeasible=True,
                        infeasible_reason=reason,
                        c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, var_names=var_names, fixed_vars={}, active_indices=list(range(n_orig)),
                        obj_offset=0.0, log_entries=log_entries, original_stats=orig_stats, reduced_stats=orig_stats
                    )

        # --- 2. FIXED VARIABLE PROPAGATION (lb == ub) ---
        fixed_vars = {}
        active_indices = []
        obj_offset = 0.0

        for j in range(n_orig):
            lb, ub = bounds[j]
            if lb is not None and ub is not None and abs(lb - ub) <= self.tol:
                fixed_val = float(lb)
                fixed_vars[j] = fixed_val
                obj_offset += float(c[j] * fixed_val)
                log_entries.append(f"Fixed variable '{var_names[j]}' = {fixed_val:g} -> substituted into objective (offset: {c[j]*fixed_val:g})")
            else:
                active_indices.append(j)

        # Adjust constraints for fixed variables
        def substitute_fixed(A_mat, b_vec, row_type="<="):
            if A_mat is None or len(A_mat) == 0:
                return None, None, 0
            A_mat = np.asarray(A_mat, dtype=np.float64)
            b_vec = np.asarray(b_vec, dtype=np.float64).flatten()
            num_subbed_rows = 0

            for j, val in fixed_vars.items():
                col_vals = A_mat[:, j]
                impacted = np.where(np.abs(col_vals) > self.tol)[0]
                if len(impacted) > 0:
                    b_vec -= col_vals * val
                    num_subbed_rows += len(impacted)
                    log_entries.append(f"  -> Propagated '{var_names[j]}' = {val:g} into {len(impacted)} {row_type} constraint(s).")

            # Slice to only active variable columns
            A_reduced = A_mat[:, active_indices]
            return A_reduced, b_vec, num_subbed_rows

        A_ub_red, b_ub_red, _ = substitute_fixed(A_ub, b_ub, "<=")
        A_ge_red, b_ge_red, _ = substitute_fixed(A_ge, b_ge, ">=")
        A_eq_red, b_eq_red, _ = substitute_fixed(A_eq, b_eq, "=")

        # --- 3. DUPLICATE CONSTRAINT REMOVAL ---
        def remove_duplicates(A_mat, b_vec, name_prefix="<="):
            if A_mat is None or len(A_mat) == 0:
                return None, None, 0
            unique_rows = []
            unique_rhs = []
            duplicates_count = 0

            for i in range(len(b_vec)):
                row = A_mat[i]
                rhs = b_vec[i]
                is_dup = False
                for u_row, u_rhs in zip(unique_rows, unique_rhs):
                    if np.allclose(row, u_row, atol=self.tol) and abs(rhs - u_rhs) <= self.tol:
                        is_dup = True
                        duplicates_count += 1
                        break
                if not is_dup:
                    unique_rows.append(row)
                    unique_rhs.append(rhs)

            if duplicates_count > 0:
                log_entries.append(f"Removed {duplicates_count} redundant/duplicate {name_prefix} constraint(s).")

            A_clean = np.array(unique_rows) if unique_rows else None
            b_clean = np.array(unique_rhs) if unique_rhs else None
            return A_clean, b_clean, duplicates_count

        A_ub_clean, b_ub_clean, _ = remove_duplicates(A_ub_red, b_ub_red, "<=")
        A_ge_clean, b_ge_clean, _ = remove_duplicates(A_ge_red, b_ge_red, ">=")
        A_eq_clean, b_eq_clean, _ = remove_duplicates(A_eq_red, b_eq_red, "=")

        # Reduced structures
        c_red = c[active_indices]
        bounds_red = [bounds[j] for j in active_indices]
        var_names_red = [var_names[j] for j in active_indices]

        n_ub_red = len(b_ub_clean) if (b_ub_clean is not None and len(b_ub_clean) > 0) else 0
        n_ge_red = len(b_ge_clean) if (b_ge_clean is not None and len(b_ge_clean) > 0) else 0
        n_eq_red = len(b_eq_clean) if (b_eq_clean is not None and len(b_eq_clean) > 0) else 0
        total_constraints_red = n_ub_red + n_ge_red + n_eq_red

        reduced_stats = {
            "vars": len(active_indices),
            "constraints": total_constraints_red,
            "ub_rows": n_ub_red,
            "ge_rows": n_ge_red,
            "eq_rows": n_eq_red,
        }

        dim_reduction = (1.0 - (reduced_stats["vars"] * max(1, reduced_stats["constraints"])) / max(1, (orig_stats["vars"] * max(1, orig_stats["constraints"])))) * 100.0
        log_entries.append(f"Reduced model: {len(active_indices)} variables, {total_constraints_red} constraints (Matrix compression: {dim_reduction:.1f}%).")
        log_entries.append("Postsolve mapping ready for exact solution vector reconstruction.")

        return PresolveResult(
            is_infeasible=False,
            infeasible_reason=None,
            c=c_red,
            A_ub=A_ub_clean,
            b_ub=b_ub_clean,
            A_ge=A_ge_clean,
            b_ge=b_ge_clean,
            A_eq=A_eq_clean,
            b_eq=b_eq_clean,
            bounds=bounds_red,
            var_names=var_names_red,
            fixed_vars=fixed_vars,
            active_indices=active_indices,
            obj_offset=obj_offset,
            log_entries=log_entries,
            original_stats=orig_stats,
            reduced_stats=reduced_stats
        )

    def postsolve(self, x_reduced: np.ndarray, presolve_res: PresolveResult) -> np.ndarray:
        """
        Reconstructs the full original variable vector from the reduced solved vector.
        """
        n_orig = presolve_res.original_stats["vars"]
        x_full = np.zeros(n_orig, dtype=np.float64)

        # Place reduced solution back into active variable slots
        for red_idx, orig_idx in enumerate(presolve_res.active_indices):
            if red_idx < len(x_reduced):
                x_full[orig_idx] = x_reduced[red_idx]

        # Place fixed values back
        for orig_idx, fixed_val in presolve_res.fixed_vars.items():
            x_full[orig_idx] = fixed_val

        return x_full
