"""
IndSolve — Scenario Lab View
Decision-first refinery crude procurement and blending simulation.
Includes intentional form submission and honest dual-timing telemetry.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from solver.refinery_simulation import run_what_if_simulation
from solver.audit import audit_solution
from ui.components import (
    render_metric_tile,
    render_decision_box,
    render_model_assumptions_expander,
    format_solver_status,
)


def render_scenario_lab_view(tolerance: float = 1e-7, max_iter: int = 5000) -> None:
    """Renders the interactive Scenario Lab view with intentional form submission and configured solver parameters."""
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <span style="font-size: 1.2rem; font-weight: 700; color: #244855;">Scenario Lab</span><br>
        <span style="font-size: 0.86rem; color: #5C6B73;">
            Optimise an illustrative crude procurement slate under quality, volume, price, and supply constraints.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State (Using distinct keys that do not collide with form widget IDs)
    if "scenario_result" not in st.session_state:
        st.session_state["scenario_result"] = None
        st.session_state["scenario_params"] = None
        st.session_state["scenario_timestamp"] = None
        st.session_state["scenario_e2e_time"] = None
        st.session_state["scenario_first_run_done"] = False

    col_ctrl, col_main = st.columns([1, 1.4], gap="medium")

    with col_ctrl:
        with st.form("scenario_inputs"):
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #244855; border-bottom: 2px solid #E5DFD5; padding-bottom: 4px; margin-bottom: 12px;'>1. Operating Targets</div>", unsafe_allow_html=True)
            
            throughput_input = st.slider(
                "Distillation Throughput Target (bpd):",
                min_value=60000, max_value=120000, value=100000, step=5000,
                help="Total crude distillation capacity required per operating day."
            )
            sulfur_limit_input = st.slider(
                "Blended Feed Sulfur Ceiling (% wt):",
                min_value=0.60, max_value=2.00, value=1.20, step=0.05,
                help="Maximum permissible weighted sulfur content in the blended crude feed for the illustrative refinery unit."
            )

            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #244855; border-bottom: 2px solid #E5DFD5; padding-bottom: 4px; margin-top: 14px; margin-bottom: 12px;'>2. Market & Supply Presets</div>", unsafe_allow_html=True)
            
            price_preset = st.selectbox(
                "Price Shock Scenario:",
                [
                    "Baseline Market ($78 Brent, $72 Arab Light)",
                    "Middle East Escalation (+$15 Arab Light, +$10 Dubai)",
                    "Sweet Crude Premium (+$14 Brent, +$12 Bonny Light)",
                    "Heavy Sour Discount (-$12 Basra, -$8 Dubai)"
                ],
                index=0
            )

            supply_preset = st.selectbox(
                "Supply Disruption Scenario:",
                [
                    "Normal Supplier Availability",
                    "Strait of Hormuz Disruption (-50% Arab Light & Dubai)",
                    "West Africa Logistics Delay (-60% Bonny Light)",
                    "Basra Coker Outage (Basra Heavy Cap = 10,000 bpd)"
                ],
                index=0
            )

            # Parse Preset Values
            preset_price_adjs = {}
            if "Middle East" in price_preset:
                preset_price_adjs = {"Arabian_Light": 15.0, "Dubai_Sour": 10.0}
            elif "Sweet Crude" in price_preset:
                preset_price_adjs = {"Brent": 14.0, "Bonny_Light": 12.0}
            elif "Heavy Sour" in price_preset:
                preset_price_adjs = {"Basra_Heavy": -12.0, "Dubai_Sour": -8.0}

            preset_avail_adjs = {}
            preset_heavy_limit = 25000.0
            if "Hormuz" in supply_preset:
                preset_avail_adjs = {"Arabian_Light": 20000.0, "Dubai_Sour": 22500.0}
            elif "West Africa" in supply_preset:
                preset_avail_adjs = {"Bonny_Light": 12000.0}
            elif "Coker Outage" in supply_preset:
                preset_heavy_limit = 10000.0

            with st.expander("⚙️ Advanced Input Overrides", expanded=False):
                st.markdown("<small style='color: #874F41;'>Fine-tune individual crude spot adjustments, quotas, and carbon policy penalties:</small>", unsafe_allow_html=True)
                
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    brent_adj = st.slider("Brent Spot Shift ($/bbl):", -20.0, 20.0, preset_price_adjs.get("Brent", 0.0), 1.0)
                    arab_adj = st.slider("Arab Light Shift ($/bbl):", -20.0, 20.0, preset_price_adjs.get("Arabian_Light", 0.0), 1.0)
                    basra_adj = st.slider("Basra Heavy Shift ($/bbl):", -20.0, 20.0, preset_price_adjs.get("Basra_Heavy", 0.0), 1.0)
                with c_p2:
                    bonny_adj = st.slider("Bonny Light Shift ($/bbl):", -20.0, 20.0, preset_price_adjs.get("Bonny_Light", 0.0), 1.0)
                    dubai_adj = st.slider("Dubai Sour Shift ($/bbl):", -20.0, 20.0, preset_price_adjs.get("Dubai_Sour", 0.0), 1.0)
                    carbon_penalty = st.slider("Sour Carbon Penalty ($/bbl):", 0.0, 15.0, 0.0, 0.5)

                st.markdown("<hr style='margin: 8px 0; border-color: #E5DFD5;'>", unsafe_allow_html=True)
                c_q1, c_q2 = st.columns(2)
                with c_q1:
                    arab_max = st.slider("Arab Light Max Quota:", 5000, 50000, int(preset_avail_adjs.get("Arabian_Light", 40000.0)), 2500)
                    brent_max = st.slider("Brent Max Quota:", 5000, 45000, int(preset_avail_adjs.get("Brent", 35000.0)), 2500)
                with c_q2:
                    bonny_max = st.slider("Bonny Light Max Quota:", 5000, 40000, int(preset_avail_adjs.get("Bonny_Light", 30000.0)), 2500)
                    dubai_max = st.slider("Dubai Sour Max Quota:", 5000, 50000, int(preset_avail_adjs.get("Dubai_Sour", 45000.0)), 2500)
                
                coker_limit_override = st.slider("Basra Coker Unit Limit (bpd):", 5000, 35000, int(preset_heavy_limit), 2500)

            # Form Submit Action Button (Modern Streamlit width="stretch")
            apply_btn = st.form_submit_button("Apply scenario", type="primary", width="stretch")

    # Execute simulation strictly on Apply button press
    if apply_btn:
        price_adjs = {
            "Brent": brent_adj,
            "Arabian_Light": arab_adj,
            "Basra_Heavy": basra_adj,
            "Bonny_Light": bonny_adj,
            "Dubai_Sour": dubai_adj
        }
        avail_adjs = {
            "Arabian_Light": float(arab_max),
            "Brent": float(brent_max),
            "Bonny_Light": float(bonny_max),
            "Dubai_Sour": float(dubai_max)
        }

        t_start = time.perf_counter()
        sim_res = run_what_if_simulation(
            base_throughput=100000.0,
            base_sulfur=1.20,
            what_if_throughput=float(throughput_input),
            what_if_sulfur=float(sulfur_limit_input),
            price_adjustments=price_adjs,
            avail_adjustments=avail_adjs,
            carbon_penalty=float(carbon_penalty),
            heavy_limit_bpd=float(coker_limit_override),
            tolerance=tolerance,
            max_iter=max_iter,
        )
        t_end = time.perf_counter()
        e2e_time = t_end - t_start

        # Store in session state
        st.session_state["scenario_result"] = sim_res
        st.session_state["scenario_params"] = {
            "throughput": throughput_input,
            "sulfur_limit": sulfur_limit_input,
            "price_preset": price_preset,
            "supply_preset": supply_preset
        }
        st.session_state["scenario_timestamp"] = time.strftime("%H:%M:%S UTC", time.gmtime())
        st.session_state["scenario_e2e_time"] = e2e_time
        st.session_state["scenario_first_run_done"] = True

    # Retrieve current active state
    sim = st.session_state.get("scenario_result")
    e2e_duration = st.session_state.get("scenario_e2e_time", 0.0)

    with col_main:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 800; color: #244855; border-bottom: 2px solid #E5DFD5; padding-bottom: 4px; margin-bottom: 12px;'>Recommended Procurement Slate</div>", unsafe_allow_html=True)

        if sim is None:
            # Clean Empty State before first calculation
            st.markdown("""
            <div style="background: #FAF5ED; border: 1px dashed #90AEAD; border-radius: 6px; padding: 36px 20px; text-align: center; margin: 16px 0;">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">⚙️</div>
                <b style="color: #244855; font-size: 1rem;">No Scenario Evaluated Yet</b>
                <p style="color: #5C6B73; font-size: 0.88rem; max-width: 420px; margin: 6px auto 0 auto;">
                    Set scenario inputs and select <b>Apply scenario</b> to calculate a recommended procurement slate.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Honest Timings & Standardized Feasibility Status
            solve_ms = sim["res_whatif"].solve_time * 1000.0
            e2e_ms = e2e_duration * 1000.0
            ts_str = st.session_state.get("scenario_timestamp", "")
            status_info = format_solver_status(sim["res_whatif"].status, sim["res_whatif"].success)

            if status_info["is_optimal"]:
                st.markdown(f"""
                <div style="background: {status_info['badge_bg']}; border: 1px solid {status_info['badge_border']}; border-radius: 4px; padding: 6px 12px; margin-bottom: 8px; font-size: 0.82rem; color: {status_info['badge_color']}; font-weight: 600; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px;">
                    <span>{status_info['icon']} {status_info['label']}</span>
                    <span>Simplex: <b>{solve_ms:.2f} ms</b> · Pipeline: <b>{e2e_ms:.2f} ms</b> ({ts_str})</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<small style='color: #874F41; display: block; margin-bottom: 10px;'>⏱️ Solver time excludes browser rendering and one-time JIT preparation.</small>", unsafe_allow_html=True)

                if solve_ms > 50.0:
                    st.markdown("<small style='color: #874F41;'>ℹ️ Initial run includes one-time numerical-engine preparation.</small>", unsafe_allow_html=True)

                # Three Concise Decision KPIs Only
                k1, k2, k3 = st.columns(3)
                with k1:
                    delta_sign = '+' if sim['cost_delta'] >= 0 else ''
                    delta_c = '#E64833' if sim['cost_delta'] > 0 else '#244855'
                    st.markdown(f"""
                    <div class="metric-tile metric-tile-primary">
                        <div class="metric-tile-label">Daily Procurement Cost</div>
                        <div class="metric-tile-value">${sim['cost_whatif']:,.0f}</div>
                        <div class="metric-tile-sub" style="color: {delta_c}; font-weight: 700;">
                            {delta_sign}${sim['cost_delta']:,.0f} vs base ({sim['cost_pct']:+.1f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with k2:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Cost per Barrel</div>
                        <div class="metric-tile-value">${sim['cost_per_bbl']:.2f} / bbl</div>
                        <div class="metric-tile-sub">Baseline: ${sim['cost_base_per_bbl']:.2f}/bbl</div>
                    </div>
                    """, unsafe_allow_html=True)

                with k3:
                    hr_sign = '+' if sim['sulfur_headroom_pct_pts'] >= 0 else ''
                    target_s = st.session_state["scenario_params"]["sulfur_limit"]
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Actual Blend Sulfur / Limit</div>
                        <div class="metric-tile-value">{sim['actual_blend_sulfur_pct']:.3f}% <span style="font-size: 0.95rem; color: #874F41;">/ {target_s:.2f}%</span></div>
                        <div class="metric-tile-sub">Headroom: {hr_sign}{sim['sulfur_headroom_pct_pts']:.3f}% pts</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 3-Bullet Decision Reasoning
                render_decision_box("Why This Recommendation Changed (Decision Reasoning)", sim["explanations"])

                # Visual Horizontal Bar Chart (Baseline vs Scenario)
                st.markdown("##### Slate Allocation: Baseline vs Scenario (bpd)")
                fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=130)
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')

                crude_names = [c.replace('_', ' ') for c in sim["df_comp"]["Crude Variety"].values]
                y_pos = np.arange(len(crude_names))
                bar_height = 0.35

                ax.barh(y_pos - bar_height/2, sim["df_comp"]["base_val"].values, bar_height, label='Baseline (100k bpd)', color='#90AEAD', alpha=0.9)
                ax.barh(y_pos + bar_height/2, sim["df_comp"]["whatif_val"].values, bar_height, label='Recommended Scenario Slate', color='#E64833', alpha=0.9)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(crude_names, fontsize=8.5, fontweight='600', color='#244855')
                ax.set_xlabel('Throughput Volume (bpd)', fontsize=8, color='#874F41')
                ax.xaxis.grid(True, linestyle=':', alpha=0.6, color='#E5DFD5')
                ax.set_axisbelow(True)
                ax.legend(loc='lower right', fontsize=7.5, framealpha=0.95)
                for spine in ['top', 'right', 'left']:
                    ax.spines[spine].set_visible(False)
                ax.spines['bottom'].set_color('#90AEAD')
                plt.tight_layout()
                st.pyplot(fig)

                # Allocation Breakdown Table (Modern Streamlit width="stretch")
                st.markdown("##### Crude Allocation Breakdown")
                table_cols = ["Crude Variety", "Crude Quality", "Price ($/bbl)", "Quota (bpd)", "Recommended (bpd)", "Delta (bpd)", "Daily Cost ($)"]
                st.dataframe(sim["df_comp"][table_cols], width="stretch", hide_index=True)

                # Active Bottlenecks Box
                if sim["binding_constraints"]:
                    st.markdown("##### Active Operating Bottlenecks")
                    for bc in sim["binding_constraints"]:
                        st.markdown(f"<small style='color: #244855;'>{bc}</small>", unsafe_allow_html=True)

            else:
                # Honest Non-Optimal / Infeasible Presentation
                st.markdown(f"""
                <div style="background: {status_info['badge_bg']}; border: 1px solid {status_info['badge_border']}; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                    <b style="color: {status_info['badge_color']}; font-size: 1rem;">{status_info['icon']} {status_info['label']}</b>
                    <p style="color: #874F41; font-size: 0.88rem; margin: 6px 0;">
                        The optimization solver halted before an optimal feasible crude blend could be reached.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<small style='color: #874F41; display: block; margin-bottom: 10px;'>⏱️ Solver time excludes browser rendering and one-time JIT preparation.</small>", unsafe_allow_html=True)

                render_decision_box("Diagnostic Cause & Actionable Next Steps", sim["explanations"])

            # Independent Constraint Audit Expander
            if sim["res_whatif"].success:
                res_w = sim["res_whatif"]
                wm = sim["whatif_model"]

                audit_w = audit_solution(
                    c=wm["c"],
                    A_ub=wm["A_ub"],
                    b_ub=wm["b_ub"],
                    A_ge=wm.get("A_ge"),
                    b_ge=wm.get("b_ge"),
                    A_eq=wm["A_eq"],
                    b_eq=wm["b_eq"],
                    bounds=wm["bounds"],
                    integrality=[0] * len(wm["var_names"]),
                    var_names=wm["var_names"],
                    x_sol=res_w.x,
                    reported_obj=res_w.fun,
                    maximize=False,
                    constraint_names_ub=wm.get("constraint_labels_ub"),
                    constraint_names_eq=wm.get("constraint_labels_eq"),
                )

                with st.expander("🔍 **Independent Constraint Audit & Mathematical Proof**", expanded=False):
                    st.markdown(f"**Objective Recalculation Check:** `c @ x = {audit_w['obj_recalc_display']}`")
                    st.dataframe(audit_w['constraint_df'].drop(columns=["is_passed"], errors="ignore"), width="stretch", hide_index=True)

    render_model_assumptions_expander()
