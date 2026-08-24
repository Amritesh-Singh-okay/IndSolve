"""
IndSolve — Model Explorer View
Exploration of industrial optimization benchmarks and custom formulations.
"""

import time
import re
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from solver.tableau_simplex import SimplexSolver, SimplexResult
from solver.branch_and_bound import BranchAndBoundSolver
from solver.presolve import PresolveEngine
from solver.problems import get_preloaded_problems
from solver.audit import audit_solution
from solver.verify_reference import verify_with_scipy_reference
from ui.components import render_engineering_roadmap, format_solver_status


def _clean_var_name(name: str) -> str:
    """Removes redundant nested units like '(bbl) (bbl)'."""
    cleaned = re.sub(r'\(bbl\)\s*\(bbl\)', '(bbl)', name)
    cleaned = re.sub(r'\(MW\)\s*\(MW\)', '(MW)', cleaned)
    cleaned = re.sub(r'\(INR\)\s*\(INR\)', '(INR)', cleaned)
    return cleaned.replace('_', ' ')


def render_model_explorer_view(tolerance: float = 1e-7, max_iter: int = 5000) -> None:
    """Renders the Model Explorer view with dynamic facility logic and dynamic 2D plot range."""
    problem_dict = get_preloaded_problems()
    problem_options = list(problem_dict.keys()) + ["✍️ Custom Problem (Manual Input)"]

    st.markdown("""
    <div style="margin-bottom: 12px;">
        <span style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">Model Explorer</span><br>
        <span style="font-size: 0.86rem; color: #64748B;">
            Benchmark formulations across refining, electrical dispatch, facility logistics, and custom LP models.
        </span>
    </div>
    """, unsafe_allow_html=True)

    selected_problem_name = st.selectbox(
        "Select Benchmark Case Study or Custom Formulation:",
        problem_options,
        index=0,
        label_visibility="collapsed"
    )

    if selected_problem_name != "✍️ Custom Problem (Manual Input)":
        prob = problem_dict[selected_problem_name]
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px;">
            <b style="color: #0F172A; font-size: 0.92rem;">{prob['name']}</b> — 
            <span style="color: #475569; font-size: 0.85rem;">{prob['description']}</span><br>
            <small style="color: #64748B;"><b>Industrial Context:</b> {prob['context']}</small>
        </div>
        """, unsafe_allow_html=True)

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
        var_names = [_clean_var_name(v) for v in prob.get("var_names", [f"x{i+1}" for i in range(len(c))])]
        units = prob.get("units", "Units")
        is_2d = prob.get("is_2d", False)

    else:
        st.subheader("Custom Problem Input")
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            maximize = st.radio("Optimization Direction:", ["Minimize", "Maximize"], index=0) == "Maximize"
            c_str = st.text_input("Objective Vector c (comma-separated):", value="70, 65, 80")
            c = np.array([float(x.strip()) for x in c_str.split(",") if x.strip()])
            n_vars = len(c)
            var_names = [f"x{i+1}" for i in range(n_vars)]
            units = "Cost"
            is_2d = (n_vars == 2)
            integrality = [0] * n_vars
        with col_in2:
            a_ub_str = st.text_area("A_ub Matrix (rows separated by newline):", value="1, 0, 0\n0, 1, 0\n0, 0, 1")
            b_ub_str = st.text_input("b_ub Vector (comma-separated):", value="500, 600, 400")
            try:
                A_ub = np.array([[float(v) for v in row.split(",") if v.strip()] for row in a_ub_str.split("\n") if row.strip()])
                b_ub = np.array([float(v) for v in b_ub_str.split(",") if v.strip()])
            except Exception:
                A_ub, b_ub = None, None
            A_ge, b_ge = None, None
            A_eq, b_eq = None, None
            bounds = [(0.0, None) for _ in range(n_vars)]

    solve_btn = st.button("Solve Optimization Model", type="primary", width="stretch")

    if solve_btn:
        solver = SimplexSolver(tol=tolerance, max_iter=max_iter)
        milp_solver = BranchAndBoundSolver(lp_solver=solver)
        presolve_engine = PresolveEngine(tol=tolerance)
        is_milp = any(i == 1 for i in integrality)

        with st.spinner("Executing Mathematical Optimization Engine..."):
            if not is_milp:
                p_res = presolve_engine.presolve(
                    c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, var_names=var_names
                )

                if p_res.is_infeasible:
                    res = SimplexResult(
                        success=False, status="INFEASIBLE",
                        message=f"Presolve contradiction: {p_res.infeasible_reason}",
                        fun=0.0, x=np.zeros(len(c)), nit=0, solve_time=0.0001, history=[]
                    )
                else:
                    res = solver.solve(
                        c=p_res.c, A_ub=p_res.A_ub, b_ub=p_res.b_ub, A_eq=p_res.A_eq, b_eq=p_res.b_eq,
                        A_ge=p_res.A_ge, b_ge=p_res.b_ge, bounds=p_res.bounds, var_names=p_res.var_names,
                        maximize=maximize,
                    )
                    if res.success:
                        full_x = presolve_engine.postsolve(res.x, p_res)
                        res.x = full_x
                        res.fun = res.fun + p_res.obj_offset
            else:
                p_res = None
                res = milp_solver.solve(
                    c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, A_ge=A_ge, b_ge=b_ge,
                    bounds=bounds, integrality=integrality, maximize=maximize
                )

            # Independent External Reference Verification
            verify_info = verify_with_scipy_reference(
                ind_res=res,
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_ge=A_ge,
                b_ge=b_ge,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                integrality=integrality,
                maximize=maximize,
                tol=1e-2,
            )

        # Primary Results Banner (Objective, Status, Latency)
        status_info = format_solver_status(res.status, res.success)
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        kr1, kr2, kr3, kr4 = st.columns(4)
        with kr1:
            if status_info["is_optimal"]:
                obj_val_str = f"{res.fun:,.2f}"
                obj_sub_str = 'Min Cost' if not maximize else 'Max Value'
                border_c = "#244855"
            else:
                obj_val_str = "Unresolved"
                obj_sub_str = status_info["label"]
                border_c = "#E64833"

            st.markdown(f"""
            <div class="metric-tile" style="border-top: 3px solid {border_c};">
                <div class="metric-tile-label">Optimal Objective ({units})</div>
                <div class="metric-tile-value">{obj_val_str}</div>
                <div class="metric-tile-sub">{obj_sub_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with kr2:
            st.markdown(f"""
            <div class="metric-tile" style="border-top: 3px solid {status_info['badge_border']};">
                <div class="metric-tile-label">Convergence Status</div>
                <div class="metric-tile-value" style="font-size: 1.02rem; color: {status_info['badge_color']};">{status_info['icon']} {status_info['label']}</div>
                <div class="metric-tile-sub">{status_info['status_subtitle']}</div>
            </div>
            """, unsafe_allow_html=True)
        with kr3:
            iters = res.nit if not is_milp else res.nodes_explored
            lbl = "Simplex Pivots" if not is_milp else "B&B Nodes"
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-tile-label">{lbl}</div>
                <div class="metric-tile-value">{iters}</div>
                <div class="metric-tile-sub">Extreme Points</div>
            </div>
            """, unsafe_allow_html=True)
        with kr4:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-tile-label">Solve Latency</div>
                <div class="metric-tile-value">{res.solve_time*1000:.2f} ms</div>
                <div class="metric-tile-sub">CPU Simplex Core</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<small style='color: #874F41; display: block; margin-top: 4px; margin-bottom: 12px;'>⏱️ Solver time excludes browser rendering and one-time JIT preparation.</small>", unsafe_allow_html=True)

        # Main Decision Allocation Section
        col_v1, col_v2 = st.columns([1.2, 1])
        with col_v1:
            st.markdown("##### Optimal Allocation Vector")
            df_vars = pd.DataFrame({
                "Variable": var_names,
                "Optimal Value": np.round(res.x, 3),
                "Unit Cost": c,
                "Total Contribution": np.round(res.x * c, 2),
            })
            st.dataframe(df_vars, width="stretch", hide_index=True)
        with col_v2:
            st.markdown("##### Resource Allocation Distribution")
            fig_alloc, ax_alloc = plt.subplots(figsize=(4.5, 2.0), dpi=130)
            fig_alloc.patch.set_facecolor('#FFFFFF')
            ax_alloc.set_facecolor('#FFFFFF')
            clean_labels = [v.replace('_', ' ') for v in var_names]
            bars = ax_alloc.bar(clean_labels, res.x, color='#244855', width=0.45, alpha=0.92)
            ax_alloc.set_ylabel(units, fontsize=7.5, color='#874F41')
            ax_alloc.tick_params(axis='x', labelsize=8, colors='#244855')
            ax_alloc.tick_params(axis='y', labelsize=7.5, colors='#244855')
            ax_alloc.yaxis.grid(True, linestyle=':', alpha=0.5, color='#E5DFD5')
            ax_alloc.set_axisbelow(True)
            for spine in ['top', 'right', 'left']:
                ax_alloc.spines[spine].set_visible(False)
            ax_alloc.spines['bottom'].set_color('#90AEAD')
            plt.tight_layout()
            st.pyplot(fig_alloc)

        # Dynamic MILP Facility Location Cards (Prompt 10: Dynamic from res.x and model data)
        if is_milp and prob.get("is_facility_milp"):
            st.markdown("##### Regional Facility Hub Opening & Freight Flow")
            fac_names = prob.get("facility_names", ["Hub 1", "Hub 2", "Hub 3"])
            fixed_costs = prob.get("fixed_costs", [12000.0, 10000.0, 8000.0])
            capacities = prob.get("capacities", [500.0, 450.0, 350.0])
            shipping_rates = prob.get("shipping_rates", [25.0, 32.0, 40.0])
            n_fac = len(fac_names)

            f_cols = st.columns(n_fac)
            for idx in range(n_fac):
                y_val = int(round(res.x[idx])) if res.success else 0
                flow_val = float(res.x[idx + n_fac]) if res.success else 0.0
                f_cost = fixed_costs[idx] if y_val == 1 else 0.0
                v_cost = flow_val * shipping_rates[idx]
                tot_cost = f_cost + v_cost
                cap = capacities[idx]

                border_c = "#90AEAD" if y_val == 1 else "#874F41"
                bg_c = "#F4F8F8" if y_val == 1 else "#FDF7F4"
                status_text = "OPEN (y = 1)" if y_val == 1 else "CLOSED (y = 0)"
                status_c = "#244855" if y_val == 1 else "#874F41"

                with f_cols[idx]:
                    st.markdown(f"""
                    <div style="border: 1px solid {border_c}; border-radius: 6px; padding: 10px; background: {bg_c};">
                        <b style="color: #244855; font-size: 0.88rem;">🏢 {fac_names[idx]}</b><br>
                        <span style="color: {status_c}; font-weight: 700; font-size: 0.82rem;">STATUS: {status_text}</span><br>
                        <small style="color: #5C6B73;">
                            • Fixed: ₹{f_cost:,.0f} | Flow: {flow_val:,.0f} / {cap:,.0f} t<br>
                            • Total Cost: <b>₹{tot_cost:,.0f}</b>
                        </small>
                    </div>
                    """, unsafe_allow_html=True)

        # 2D Geometry View (Dynamic limits & plain language objective equation)
        if is_2d and hasattr(res, 'history') and len(res.history) > 0:
            with st.expander("📐 **2D Feasible Polytope Geometry & Simplex Path**", expanded=True):
                st.markdown(f"**Objective Function**: `Minimize z = {c[0]:.2f}·x₁ + {c[1]:.2f}·x₂`  |  **Optimal Solution** (x*): `x₁ = {res.x[0]:,.2f}, x₂ = {res.x[1]:,.2f}` (Optimal {units}: `{res.fun:,.2f}`)")
                
                # Derive Dynamic Bounding Limits from Constraints and Solution
                x_cand = [res.x[0] * 1.3, 10.0]
                y_cand = [res.x[1] * 1.3, 10.0]
                if bounds:
                    if bounds[0][1] is not None:
                        x_cand.append(bounds[0][1] * 1.15)
                    if bounds[1][1] is not None:
                        y_cand.append(bounds[1][1] * 1.15)
                if b_ub is not None and A_ub is not None:
                    for i in range(len(b_ub)):
                        if A_ub[i, 0] > 1e-4:
                            x_cand.append((b_ub[i] / A_ub[i, 0]) * 1.15)
                        if A_ub[i, 1] > 1e-4:
                            y_cand.append((b_ub[i] / A_ub[i, 1]) * 1.15)

                x_max_plot = float(min(2000.0, max(x_cand)))
                y_max_plot = float(min(2000.0, max(y_cand)))

                fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FAFAFA')

                x_grid = np.linspace(0, x_max_plot, 250)
                y_grid = np.linspace(0, y_max_plot, 250)
                X, Y = np.meshgrid(x_grid, y_grid)

                feasible = np.ones_like(X, dtype=bool)
                if A_ub is not None:
                    for i in range(len(b_ub)):
                        feasible &= (A_ub[i, 0] * X + A_ub[i, 1] * Y <= b_ub[i] + 1e-5)
                if A_ge is not None:
                    for i in range(len(b_ge)):
                        feasible &= (A_ge[i, 0] * X + A_ge[i, 1] * Y >= b_ge[i] - 1e-5)

                # Translucent Feasible Fill in Soft Sage / Cream
                ax.contourf(X, Y, feasible.astype(int), levels=[0.5, 1.5], colors=['#E1ECEB'], alpha=0.7)

                line_colors = ['#244855', '#E64833', '#874F41', '#90AEAD']
                if A_ub is not None:
                    for i in range(len(b_ub)):
                        a1, a2 = A_ub[i, 0], A_ub[i, 1]
                        c_color = line_colors[i % len(line_colors)]
                        if abs(a2) > 1e-6:
                            y_line = (b_ub[i] - a1 * x_grid) / a2
                            ax.plot(x_grid, y_line, label=f"Bound {i+1}: {a1:g}x₁+{a2:g}x₂≤{b_ub[i]:g}", linestyle="--", color=c_color, linewidth=1.2)
                        else:
                            ax.axvline(x=b_ub[i]/a1, label=f"Bound {i+1}: x₁≤{b_ub[i]:g}", linestyle="--", color=c_color, linewidth=1.2)

                if A_ge is not None:
                    for i in range(len(b_ge)):
                        a1, a2 = A_ge[i, 0], A_ge[i, 1]
                        if abs(a2) > 1e-6:
                            y_line = (b_ge[i] - a1 * x_grid) / a2
                            ax.plot(x_grid, y_line, label=f"Demand: {a1:g}x₁+{a2:g}x₂≥{b_ge[i]:g}", linestyle="-.", color='#244855', linewidth=1.4)

                steps = [h['x'] for h in res.history]
                xs = [s[0] for s in steps]
                ys = [s[1] for s in steps]

                ax.plot(xs, ys, color='#244855', linestyle='-', marker='o', markersize=5, linewidth=1.8, label="Simplex Trajectory", zorder=8)
                for idx, (px, py) in enumerate(zip(xs, ys)):
                    ax.annotate(f"P{idx}", (px, py), textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold', color='#244855')

                # Optimal Solution Star in Accent Terracotta
                ax.scatter([res.x[0]], [res.x[1]], color='#E64833', s=160, marker='*', zorder=12, label=f"x* ({res.x[0]:.0f}, {res.x[1]:.0f})")

                ax.set_xlim(0, x_max_plot)
                ax.set_ylim(0, y_max_plot)
                ax.set_xlabel(f"{var_names[0]}", fontsize=8.5, fontweight='bold', color='#874F41')
                ax.set_ylabel(f"{var_names[1]}", fontsize=8.5, fontweight='bold', color='#874F41')
                ax.grid(True, linestyle=':', alpha=0.5, color='#E5DFD5')
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
                plt.tight_layout()
                st.pyplot(fig)

        # Engineering Details Expander (Audit, Trace, Presolve, Reference, Roadmap)
        with st.expander("🛠️ **Engineering Details & Mathematical Verification**", expanded=False):
            t_trace, t_audit, t_presolve, t_bench, t_road = st.tabs([
                "Algorithm Trace",
                "Constraint Audit",
                "Presolve Log",
                "External Reference Comparison",
                "Technical Roadmap"
            ])

            with t_trace:
                st.markdown("##### Step-by-Step Simplex Pivot Log")
                if hasattr(res, 'trace_log') and len(res.trace_log) > 0:
                    for step in res.trace_log:
                        st.markdown(f"""
                        <div style="background: #F8FAFC; border-left: 3px solid #2563EB; padding: 6px 10px; margin-bottom: 5px; border-radius: 4px; font-size: 0.85rem;">
                            <b>Pivot {step['iteration']}:</b> {step['explanation']}
                        </div>
                        """, unsafe_allow_html=True)
                    df_trace = pd.DataFrame(res.trace_log).drop(columns=["explanation"])
                    st.dataframe(df_trace, width="stretch", hide_index=True)
                else:
                    st.info("Direct optimal resolution or Branch-and-Bound search tree.")

            with t_audit:
                audit = audit_solution(
                    c=c, A_ub=A_ub, b_ub=b_ub, A_ge=A_ge, b_ge=b_ge, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, integrality=integrality, var_names=var_names,
                    x_sol=res.x, reported_obj=res.fun, maximize=maximize, tol=tolerance
                )
                st.markdown(r"**Objective Recalculation Check ($\mathbf{c}^T \mathbf{x}$):**")
                st.code(f"c @ x = {audit['obj_recalc_display']}", language="python")
                st.markdown("##### Constraint Compliance Table")
                st.dataframe(audit['constraint_df'].drop(columns=["is_passed"], errors="ignore"), width="stretch", hide_index=True)

            with t_presolve:
                st.markdown("##### Presolve Reductions Log")
                if p_res is not None:
                    pr1, pr2, pr3 = st.columns(3)
                    with pr1:
                        st.metric("Original Size", f"{p_res.original_stats['vars']} v × {p_res.original_stats['constraints']} c")
                    with pr2:
                        st.metric("Reduced Size", f"{p_res.reduced_stats['vars']} v × {p_res.reduced_stats['constraints']} c")
                    with pr3:
                        dim_red = (1.0 - (p_res.reduced_stats['vars'] * max(1, p_res.reduced_stats['constraints'])) / max(1, (p_res.original_stats['vars'] * max(1, p_res.original_stats['constraints'])))) * 100.0
                        st.metric("Matrix Reduction", f"{dim_red:.1f}%")

                    st.markdown("**Transformations Executed:**")
                    for entry in p_res.log_entries:
                        st.markdown(f"- `{entry}`")
                else:
                    st.info("No presolve reductions applied for discrete MILP formulations.")

            with t_bench:
                st.markdown("##### External Reference Comparison")
                if verify_info["state"] in ["reference_failed", "reference_unavailable"]:
                    st.warning("⚠️ Reference verification unavailable — solver result is not independently confirmed.")
                elif verify_info["state"] == "mismatch":
                    st.error(f"❌ Verification Discrepancy: {verify_info['message']}")
                else:
                    st.success(f"✅ Verified: {verify_info['message']}")

                comp_data = {
                    "Metric / Property": ["Math Engine Foundation", "Calculated Objective", "Convergence Status", "Algorithm Class"],
                    "IndSolve (Native)": ["First-Principles Linear Programming Core", f"{res.fun:,.4f}" if res.success else res.status, res.status, "Tableau Simplex / B&B"],
                    verify_info["ref_label"]: ["External Reference Solver (Compiled C++)", f"{verify_info['ref_obj']:,.4f}" if verify_info['ref_obj'] is not None else "N/A (Unavailable)", verify_info["ref_status"], "Dual Simplex / Branch-and-Cut"],
                    "Reference Status": ["Native Core", verify_info["verdict_label"], "Identical" if verify_info["state"] == "verified" else "Unconfirmed", "Standard Match"]
                }
                st.table(pd.DataFrame(comp_data))

            with t_road:
                render_engineering_roadmap()
