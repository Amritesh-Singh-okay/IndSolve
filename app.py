"""
IndSolve — Industrial Optimization Workbench
Linear & Mixed-Integer Programming Decision Support Platform
Problem Statement ID: 26119 (Mangalore Refinery and Petrochemicals Limited)

Main Streamlit Application Entrypoint (Modular Architecture).
"""

import streamlit as st

from ui.theme import apply_theme
from ui.components import render_app_header
from ui.views.scenario_lab import render_scenario_lab_view
from ui.views.model_explorer import render_model_explorer_view
from ui.views.validation_view import render_validation_view
from ui.views.mps_import_view import render_mps_import_view

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IndSolve — Industrial Optimization Workbench",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- APPLY CUSTOM 5-COLOR PALETTE THEME ---
apply_theme()

# --- TOP EXECUTIVE HEADER ---
render_app_header()

# --- TOP LEFT SIDEBAR BRANDING CARD ---
st.sidebar.markdown("""
<div class="sidebar-brand-card">
    <div class="sidebar-brand-title">
        ⚖️ IndSolve
    </div>
    <div class="sidebar-brand-tag">
        Optimization Workbench
    </div>
    <div class="sidebar-brand-sub">
        SIH Problem 26119 · MRPL Refinery
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ENGINE CONTROLS ---
st.sidebar.markdown("### 🎛️ Solver Controls")

with st.sidebar.expander("⚙️ Advanced Solver Settings", expanded=False):
    st.markdown("**Linear Algebra**: Dense Tableau Simplex")
    st.markdown("**Anti-Cycling**: Dantzig with Bland's Tie-Break")
    st.markdown("**Integrality**: DFS Branch-and-Bound")
    st.markdown("**Preprocessing**: 3-Phase Safe Presolve Engine")
    tol_exp = st.slider("Convergence Tolerance (10^x):", min_value=-10, max_value=-4, value=-7, step=1)
    tolerance = 10.0 ** tol_exp
    max_iter_setting = st.number_input("Max Simplex Pivots:", min_value=100, max_value=20000, value=5000, step=500)

if "tolerance" not in locals():
    tolerance = 1e-7

st.sidebar.markdown(
    "<div style='font-size: 0.78rem; color: #90AEAD; margin-top: 15px; border-top: 1px solid #38616F; padding-top: 10px;'>"
    "<b>Scope & Scale:</b> Small-to-medium dense models (< 250 rows/vars). "
    "Designed for MRPL crude evaluation case studies."
    "</div>",
    unsafe_allow_html=True
)

# --- TOP NAVIGATION (SHORT, CLEAN LABELS) ---
app_mode = st.radio(
    "Navigation:",
    [
        "Scenario Lab",
        "Model Explorer",
        "Validation",
        "Import Model"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# --- ROUTE TO ACTIVE VIEW ---
max_pivots = int(max_iter_setting)
if app_mode == "Scenario Lab":
    render_scenario_lab_view(tolerance=tolerance, max_iter=max_pivots)
elif app_mode == "Model Explorer":
    render_model_explorer_view(tolerance=tolerance, max_iter=max_pivots)
elif app_mode == "Validation":
    render_validation_view()
else:
    render_mps_import_view(tolerance=tolerance, max_iter=max_pivots)
