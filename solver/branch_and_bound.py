"""
IndSolve — Branch and Bound Solver for Mixed-Integer Linear Programming (MILP)
Built from mathematical foundation on top of IndSolve Simplex core.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .tableau_simplex import SimplexSolver, SimplexResult


class MILPResult:
    def __init__(
        self,
        success: bool,
        status: str,
        message: str,
        fun: float,
        x: np.ndarray,
        nodes_explored: int,
        solve_time: float,
        tree_log: List[Dict],
    ):
        self.success = success
        self.status = status          # 'OPTIMAL', 'NODE_LIMIT_FEASIBLE', 'NODE_LIMIT_INCONCLUSIVE', 'INFEASIBLE'
        self.message = message
        self.fun = fun
        self.x = x
        self.nodes_explored = nodes_explored
        self.solve_time = solve_time
        self.tree_log = tree_log


class BranchAndBoundSolver:
    """
    Branch and Bound Solver for Mixed-Integer Linear Programs (MILP).
    Distinguishes strictly between mathematically proven OPTIMAL and NODE_LIMIT_FEASIBLE.
    """

    def __init__(
        self,
        lp_solver: Optional[SimplexSolver] = None,
        int_tol: float = 1e-5,
        max_nodes: int = 1000,
    ):
        self.lp_solver = lp_solver if lp_solver is not None else SimplexSolver()
        self.int_tol = int_tol
        self.max_nodes = max_nodes

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
        integrality: Optional[List[int]] = None,
        var_names: Optional[List[str]] = None,
        maximize: bool = False,
        **kwargs,
    ) -> MILPResult:
        """
        Solves MILP:
            min/max c^T x
            subject to constraints and x_i in Z for i where integrality[i] == 1
        """
        start_time = time.perf_counter()
        c = np.asarray(c, dtype=np.float64)
        n = len(c)

        if integrality is None:
            integrality = [0] * n

        if bounds is None:
            bounds = [(0.0, None) for _ in range(n)]

        best_obj = float("-inf") if maximize else float("inf")
        best_x = None
        nodes_explored = 0
        tree_log = []

        # Stack for DFS Branch and Bound: (node_id, current_bounds, depth)
        stack = [(0, list(bounds), 0)]
        node_counter = 0
        hit_node_limit = False
        hit_lp_iter_limit = False
        unresolved_lp_node = None

        while stack:
            if nodes_explored >= self.max_nodes:
                hit_node_limit = True
                break

            node_id, curr_bounds, depth = stack.pop()
            nodes_explored += 1

            # Solve LP Relaxation with current bounds
            lp_res = self.lp_solver.solve(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ge=A_ge,
                b_ge=b_ge,
                bounds=curr_bounds,
                maximize=maximize,
            )

            node_info = {
                "node": node_id,
                "depth": depth,
                "lp_status": lp_res.status,
                "lp_obj": lp_res.fun if lp_res.success else None,
                "action": "pruned",
            }

            # 1. Handle Unresolved LP Status (Iteration Limit) — DO NOT PRUNE AS INFEASIBLE!
            if lp_res.status in ["ITERATION_LIMIT", "MAX_ITER", "ERROR"]:
                hit_lp_iter_limit = True
                unresolved_lp_node = node_id
                node_info["action"] = f"unresolved child LP ({lp_res.status})"
                tree_log.append(node_info)
                break

            # 2. Prune by True Infeasibility / Unboundedness
            if not lp_res.success or lp_res.status != "OPTIMAL":
                node_info["action"] = f"pruned ({lp_res.status.lower()})"
                tree_log.append(node_info)
                continue

            # 3. Prune by Bound
            if maximize:
                if lp_res.fun <= best_obj + 1e-9:
                    node_info["action"] = "pruned (bound worse than incumbent)"
                    tree_log.append(node_info)
                    continue
            else:
                if lp_res.fun >= best_obj - 1e-9:
                    node_info["action"] = "pruned (bound worse than incumbent)"
                    tree_log.append(node_info)
                    continue

            # 4. Check Integrality
            x_sol = lp_res.x
            fractional_candidates = []
            for i in range(n):
                if integrality[i] == 1:
                    val = x_sol[i]
                    frac = abs(val - round(val))
                    if frac > self.int_tol:
                        fractional_candidates.append((i, val, frac))

            if not fractional_candidates:
                # Integer feasible solution found (New Incumbent)
                best_obj = lp_res.fun
                best_x = x_sol.copy()
                node_info["action"] = f"new incumbent found (Obj: {best_obj:.4f})"
                tree_log.append(node_info)
                continue

            # 5. Branch on Most Fractional Variable
            fractional_candidates.sort(key=lambda item: abs(item[2] - 0.5))
            branch_var_idx, branch_val, _ = fractional_candidates[0]

            floor_val = np.floor(branch_val)
            ceil_val = np.ceil(branch_val)

            node_info["action"] = f"branch on x{branch_var_idx+1} = {branch_val:.2f}"
            tree_log.append(node_info)

            # Left child: x_i <= floor(val)
            left_bounds = list(curr_bounds)
            lb, ub = left_bounds[branch_var_idx]
            new_ub = floor_val if ub is None else min(ub, floor_val)
            if (lb is None) or (lb <= new_ub + 1e-9):
                left_bounds[branch_var_idx] = (lb, new_ub)
                node_counter += 1
                stack.append((node_counter, left_bounds, depth + 1))

            # Right child: x_i >= ceil(val)
            right_bounds = list(curr_bounds)
            lb, ub = right_bounds[branch_var_idx]
            new_lb = ceil_val if lb is None else max(lb, ceil_val)
            if (ub is None) or (new_lb <= ub + 1e-9):
                right_bounds[branch_var_idx] = (new_lb, ub)
                node_counter += 1
                stack.append((node_counter, right_bounds, depth + 1))

        elapsed = time.perf_counter() - start_time

        if hit_lp_iter_limit:
            status = "ITERATION_LIMIT"
            success = False
            msg = f"Branch-and-Bound aborted: Child LP relaxation at node {unresolved_lp_node} reached iteration limit. Solution is unresolved."
        elif hit_node_limit:
            if best_x is not None:
                status = "NODE_LIMIT_FEASIBLE"
                msg = f"Feasible integer solution found, but stopped at maximum node limit ({self.max_nodes} nodes). Optimality not mathematically proven."
                success = True
            else:
                status = "NODE_LIMIT_INCONCLUSIVE"
                msg = f"Node limit reached ({self.max_nodes} nodes) without finding any feasible integer point."
                success = False
        else:
            if best_x is not None:
                status = "OPTIMAL"
                msg = f"Optimal integer solution mathematically proven across {nodes_explored} nodes."
                success = True
            else:
                status = "INFEASIBLE"
                msg = "No feasible integer solution exists in the bounded region."
                success = False

        return MILPResult(
            success=success,
            status=status,
            message=msg,
            fun=float(best_obj) if (best_x is not None) else 0.0,
            x=best_x if (best_x is not None) else np.zeros(n),
            nodes_explored=nodes_explored,
            solve_time=elapsed,
            tree_log=tree_log,
        )
