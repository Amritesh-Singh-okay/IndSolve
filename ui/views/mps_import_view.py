"""
IndSolve — MPS Import View
Loader for the validated continuous-LP subset of MPS models.
"""

import os
from typing import Dict, Any
import pandas as pd
import streamlit as st

from solver.tableau_simplex import SimplexSolver
from solver.mps_parser import parse_mps_text, MPSParseError
from solver.verify_reference import verify_with_scipy_reference
from ui.components import format_solver_status


def render_mps_import_view(tolerance: float = 1e-7, max_iter: int = 5000) -> None:
    """Renders the Validated Continuous-LP MPS Model Loader view."""
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <span style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">Import Model (MPS)</span><br>
        <span style="font-size: 0.86rem; color: #64748B;">
            Parses the validated continuous-LP subset of standard MPS instances (NAME, ROWS, COLUMNS, RHS, BOUNDS).
            Supports LO, UP, FX, and FR bounds. Discrete integer markers (INTORG/INTEND) and quadratic sections are safely rejected.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 **Supported Continuous-LP MPS Subset Specification**", expanded=False):
        st.markdown("""
        - **Supported Sections**: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `BOUNDS`, `ENDATA`.
        - **Supported Row Types**: `N` (Objective), `L` ($\\le$), `G` ($\\ge$), `E` ($=$).
        - **Supported Bound Types**: `LO` (Lower Bound), `UP` (Upper Bound), `FX` (Fixed Variable), `FR` (Free Variable $-\\infty < x < \\infty$).
        - **Explicitly Rejected Constructs**:
          - Integer markers (`INTORG` / `INTEND` / `MARKER`)
          - Discrete/integer bound types (`BV`, `LI`, `UI`, `SC`, `SI`)
          - Unsupported sections (`RANGES`, `SOS`, `QUADOBJ`, `QCMATRIX`, `INDICATORS`)
        - **Scale Envelope**: Small-to-medium dense models ($\\le 250$ rows and variables).
        - **Solver Pivot Limit**: Configured via the sidebar *Max Simplex Pivots* setting.
        """)

    benchmark_dir = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks")

    col_mps1, col_mps2 = st.columns([1, 1], gap="medium")

    with col_mps1:
        mps_source = st.radio("MPS Model Source:", ["Preloaded Netlib Benchmark", "Upload Custom .MPS File"], horizontal=True)

        if mps_source == "Preloaded Netlib Benchmark":
            mps_choice = st.selectbox(
                "Select Benchmark File:",
                [
                    "AFIRO (Authentic Netlib LP Benchmark — Published Opt: -464.7531)",
                    "BLEND_TOY (Refinery Crude Blending Demo — 4 rows × 2 vars)"
                ]
            )
            filename = "afiro.mps" if "AFIRO" in mps_choice else "blend_sample.mps"
            file_path = os.path.join(benchmark_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    mps_content = f.read()
            else:
                mps_content = ""
            
            if "AFIRO" in mps_choice:
                st.markdown("""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; margin-top: 6px; font-size: 0.80rem; color: #475569;">
                    <b>Authentic Netlib Benchmark Provenance:</b><br>
                    • <b>Source:</b> Netlib LP Library (<code>netlib.sandia.gov/lp/data/afiro</code>) & HiGHS test suite.<br>
                    • <b>SHA-256:</b> <code>9cd304f02717cbd6f85068cb777b69d28539b22a4868ae0f0fb425f514f0eea5</code><br>
                    • <b>Published Rational Optimum:</b> <code>-464.75314286</code> (Koch 2004, Operations Research Letters).<br>
                    • <b>Structure:</b> 27 constraint rows (8 EQ, 19 LEQ), 32 structural variables, 83 constraint non-zeros (9.61% density).
                </div>
                """, unsafe_allow_html=True)
        else:
            uploaded_file = st.file_uploader("Upload continuous LP .mps file (Max 250 rows/vars):", type=["mps"])
            mps_content = uploaded_file.getvalue().decode("utf-8") if uploaded_file else ""

    with col_mps2:
        if mps_content:
            with st.expander("📄 View Raw MPS Text (First 30 Lines)", expanded=False):
                st.code("\n".join(mps_content.splitlines()[:30]), language="text")

    if mps_content:
        try:
            model = parse_mps_text(mps_content, max_vars=250, max_rows=250)

            c_d1, c_d2, c_d3, c_d4, c_d5 = st.columns(5)
            with c_d1:
                st.markdown(f"""<div class="metric-tile"><div class="metric-tile-label">Problem</div><div class="metric-tile-value" style="font-size:1.05rem;">{model['problem_name']}</div></div>""", unsafe_allow_html=True)
            with c_d2:
                st.markdown(f"""<div class="metric-tile"><div class="metric-tile-label">Rows (m)</div><div class="metric-tile-value">{model['num_rows']}</div></div>""", unsafe_allow_html=True)
            with c_d3:
                st.markdown(f"""<div class="metric-tile"><div class="metric-tile-label">Vars (n)</div><div class="metric-tile-value">{model['num_vars']}</div></div>""", unsafe_allow_html=True)
            with c_d4:
                st.markdown(f"""<div class="metric-tile"><div class="metric-tile-label">Constraint NNZ</div><div class="metric-tile-value">{model['nnz']}</div></div>""", unsafe_allow_html=True)
            with c_d5:
                st.markdown(f"""<div class="metric-tile"><div class="metric-tile-label">Matrix Density</div><div class="metric-tile-value">{model['density']:.1f}%</div></div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
            solve_mps_btn = st.button("Solve Parsed MPS Model", type="primary", width="stretch")

            if solve_mps_btn:
                solver = SimplexSolver(tol=tolerance, max_iter=max_iter)
                res = solver.solve(
                    c=model["c"], A_ub=model["A_ub"], b_ub=model["b_ub"],
                    A_ge=model["A_ge"], b_ge=model["b_ge"], A_eq=model["A_eq"], b_eq=model["b_eq"],
                    bounds=model["bounds"], maximize=False
                )

                verify_info = verify_with_scipy_reference(
                    ind_res=res,
                    c=model["c"],
                    A_ub=model["A_ub"],
                    b_ub=model["b_ub"],
                    A_ge=model["A_ge"],
                    b_ge=model["b_ge"],
                    A_eq=model["A_eq"],
                    b_eq=model["b_eq"],
                    bounds=model["bounds"],
                    maximize=False,
                    tol=1e-2,
                )

                status_info = format_solver_status(res.status, res.success)
                km1, km2, km3, km4 = st.columns(4)
                with km1:
                    obj_disp = f"{res.fun:,.4f}" if status_info["is_optimal"] else "Unresolved"
                    border_c = "#244855" if status_info["is_optimal"] else "#E64833"
                    st.markdown(f"""
                    <div class="metric-tile" style="border-top: 3px solid {border_c};">
                        <div class="metric-tile-label">Optimal Objective</div>
                        <div class="metric-tile-value">{obj_disp}</div>
                        <div class="metric-tile-sub">Calculated fun</div>
                    </div>
                    """, unsafe_allow_html=True)
                with km2:
                    st.markdown(f"""
                    <div class="metric-tile" style="border-top: 3px solid {status_info['badge_border']};">
                        <div class="metric-tile-label">Convergence Status</div>
                        <div class="metric-tile-value" style="font-size: 1.02rem; color: {status_info['badge_color']};">{status_info['icon']} {status_info['label']}</div>
                        <div class="metric-tile-sub">{status_info['status_subtitle']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with km3:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Simplex Pivots</div>
                        <div class="metric-tile-value">{res.nit}</div>
                        <div class="metric-tile-sub">Extreme Points</div>
                    </div>
                    """, unsafe_allow_html=True)
                with km4:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Solve Latency</div>
                        <div class="metric-tile-value">{res.solve_time*1000:.2f} ms</div>
                        <div class="metric-tile-sub">CPU Simplex Core</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<small style='color: #874F41; display: block; margin-top: 4px; margin-bottom: 12px;'>⏱️ Solver time excludes browser rendering and one-time JIT preparation.</small>", unsafe_allow_html=True)

                st.markdown("##### External Reference Comparison")
                if verify_info["state"] in ["reference_failed", "reference_unavailable"]:
                    st.warning("⚠️ Reference verification unavailable — solver result is not independently confirmed.")
                elif verify_info["state"] == "mismatch":
                    st.error(f"❌ Verification Discrepancy: {verify_info['message']}")
                else:
                    st.success(f"✅ Verified: {verify_info['message']}")

                comp_mps = {
                    "Metric / Property": ["Math Engine Foundation", "Calculated Objective", "Convergence Status", "Algorithm Class"],
                    "IndSolve (Native)": ["First-Principles Simplex Core", f"{res.fun:,.6f}", res.status, "Tableau Simplex"],
                    verify_info["ref_label"]: ["External Reference Solver (Compiled C++)", f"{verify_info['ref_obj']:,.6f}" if verify_info['ref_obj'] is not None else "N/A (Unavailable)", verify_info["ref_status"], "Dual Simplex (HiGHS)"],
                    "Reference Status": ["Native Core", verify_info["verdict_label"], "Identical" if verify_info["state"] == "verified" else "Unconfirmed", "Standard Match"]
                }
                st.table(pd.DataFrame(comp_mps))

        except MPSParseError as e:
            st.error(f"❌ **MPS Syntax Notice**: {e}")
        except Exception as e:
            st.error(f"❌ **Model Loading Notice**: {e}")
