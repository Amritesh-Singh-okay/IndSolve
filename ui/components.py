"""
IndSolve — Shared UI Components
Reusable presentation elements using the customized 5-color palette:
#244855 (Teal), #E64833 (Terracotta), #874F41 (Earth), #90AEAD (Sage), #FBE9D0 (Cream).
"""

from typing import List, Optional, Dict, Any
import streamlit as st


def format_solver_status(status: Optional[str], success: bool) -> Dict[str, Any]:
    """
    Standardizes raw solver status codes into honest UI presentation tokens.
    OPTIMAL -> 'Optimal solution found'
    INFEASIBLE -> 'No feasible solution'
    UNBOUNDED -> 'Objective is unbounded'
    ITERATION_LIMIT -> 'Solve stopped before optimality was proven'
    NODE_LIMIT_* -> 'Node limit reached before optimality was proven'
    """
    raw = (status or "").upper()
    if raw == "OPTIMAL" and success:
        return {
            "label": "Optimal solution found",
            "is_optimal": True,
            "badge_bg": "#F4F8F8",
            "badge_border": "#90AEAD",
            "badge_color": "#244855",
            "icon": "🟢",
            "status_subtitle": "Proven optimal"
        }
    elif raw == "INFEASIBLE":
        return {
            "label": "No feasible solution",
            "is_optimal": False,
            "badge_bg": "#FEF6F4",
            "badge_border": "#E64833",
            "badge_color": "#E64833",
            "icon": "🔴",
            "status_subtitle": "Contradictory bounds"
        }
    elif raw == "UNBOUNDED":
        return {
            "label": "Objective is unbounded",
            "is_optimal": False,
            "badge_bg": "#FEF6F4",
            "badge_border": "#E64833",
            "badge_color": "#E64833",
            "icon": "🔴",
            "status_subtitle": "Infinite descent"
        }
    elif raw in ["ITERATION_LIMIT", "MAX_ITER"]:
        return {
            "label": "Solve stopped before optimality was proven",
            "is_optimal": False,
            "badge_bg": "#FDF7F4",
            "badge_border": "#874F41",
            "badge_color": "#874F41",
            "icon": "⚠️",
            "status_subtitle": "Pivot limit reached"
        }
    elif "NODE_LIMIT" in raw:
        return {
            "label": "Node limit reached before optimality was proven",
            "is_optimal": False,
            "badge_bg": "#FDF7F4",
            "badge_border": "#874F41",
            "badge_color": "#874F41",
            "icon": "⚠️",
            "status_subtitle": "Tree limit reached"
        }
    else:
        return {
            "label": f"Halted ({raw or 'Unknown'})",
            "is_optimal": False,
            "badge_bg": "#FDF7F4",
            "badge_border": "#874F41",
            "badge_color": "#874F41",
            "icon": "⚠️",
            "status_subtitle": "Non-convergent"
        }


def render_app_header() -> None:
    """Renders the top executive header using the custom palette."""
    st.markdown("""
    <div class="app-header">
        <div class="app-title-row">
            <div>
                <span class="app-brand">IndSolve</span>
                <span class="app-tagline">Industrial Optimization Workbench</span>
            </div>
            <div class="app-status">
                Illustrative scenario model · Custom LP engine · External verification available
            </div>
        </div>
        <div class="app-desc">
            Decision-support platform for refinery feedstock procurement, crude slate blending, and operational dispatch planning.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_tile(label: str, value: str, sub: str, sub_color: Optional[str] = None, border_color: Optional[str] = None) -> str:
    """Returns HTML for a metric tile with nowrap protection."""
    style_color = f'style="color: {sub_color};"' if sub_color else ""
    border_style = f'style="border-top: 3px solid {border_color};"' if border_color else ""
    return f"""
    <div class="metric-tile" {border_style}>
        <div class="metric-tile-label">{label}</div>
        <div class="metric-tile-value">{value}</div>
        <div class="metric-tile-sub" {style_color}>{sub}</div>
    </div>
    """


def _format_markdown_bold(text: str) -> str:
    """Converts markdown **bold** pairs into <b>bold</b> for HTML rendering."""
    parts = text.split("**")
    if len(parts) >= 3:
        res = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                res.append(f"<b style='color: #244855; font-weight: 700;'>{part}</b>")
            else:
                res.append(part)
        return "".join(res)
    return text


def render_decision_box(title: str, bullets: List[str]) -> None:
    """Renders a decision explanation box with up to 3 bullets in a single robust container."""
    bullet_items = "".join([
        f"<div style='margin: 4px 0; color: #244855; font-size: 0.85rem; line-height: 1.45;'>• {_format_markdown_bold(b)}</div>"
        for b in bullets
    ])
    st.markdown(f"""
    <div class="decision-box">
        <div class="decision-box-title">{title}</div>
        {bullet_items}
    </div>
    """, unsafe_allow_html=True)


def render_model_assumptions_expander() -> None:
    """Renders the standard model assumptions and operational scope expander."""
    with st.expander("ℹ️ **Model Assumptions, Data Provenance & Operational Scope**", expanded=False):
        st.markdown("""
        - **Illustrative Model Parameters:** Crude prices ($/bbl), sulfur weight percentages, API gravity values, supplier quotas, and coker limits are illustrative parameters intended for decision-support algorithm demonstration.
        - **Refinery Modeling Scope:** This prototype models a static linear crude blending and procurement slate with feed sulfur and coker bounds. It does not model non-linear refinery distillation curves, individual fractionation cuts, secondary hydrotreating unit kinematics, or finished fuel Euro-VI product specifications (e.g., 10 ppm finished diesel).
        - **MRPL / Industry Context:** Mangalore Refinery and Petrochemicals Limited (MRPL) publicly confirms regular utilization of LP models for crude evaluation and procurement planning. This indigenous solver demonstrates the core mathematical programming architecture and can be integrated with verified commercial ERP and laboratory crude assay databases.
        """)


def render_engineering_roadmap() -> None:
    """Renders the 3-phase technical roadmap using the palette."""
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown("""
        <div style="border: 1px solid #90AEAD; border-radius: 6px; padding: 12px; background: #F4F8F8;">
            <b style="color: #244855;">Phase 1: Dense Core</b><br>
            <span style="font-size: 0.72rem; background: #E1ECEB; color: #244855; padding: 2px 6px; border-radius: 4px; font-weight: 600;">CURRENT (v0.2)</span>
            <hr style="margin: 6px 0; border-color: #D6E4E3;">
            • CPU NumPy + Numba JIT<br>
            • Dense Tableau Simplex & B&B<br>
            • Scale: < 250 rows/vars<br>
            • 23/23 Verified Unit Tests
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown("""
        <div style="border: 1px solid #874F41; border-radius: 6px; padding: 12px; background: #FDF7F4;">
            <b style="color: #874F41;">Phase 2: Sparse Engine</b><br>
            <span style="font-size: 0.72rem; background: #F7EAE3; color: #874F41; padding: 2px 6px; border-radius: 4px; font-weight: 600;">ROADMAP (Q4 2026)</span>
            <hr style="margin: 6px 0; border-color: #EBD6CC;">
            • Sparse Revised Simplex (CSC)<br>
            • Sparse LU + Forest-Tomlin<br>
            • Scale: 10,000+ variables<br>
            • Gomory Mixed-Integer Cuts
        </div>
        """, unsafe_allow_html=True)
    with col_r3:
        st.markdown("""
        <div style="border: 1px solid #E64833; border-radius: 6px; padding: 12px; background: #FEF6F4;">
            <b style="color: #E64833;">Phase 3: GPU Acceleration</b><br>
            <span style="font-size: 0.72rem; background: #FCE6E2; color: #E64833; padding: 2px 6px; border-radius: 4px; font-weight: 600;">RESEARCH (2027)</span>
            <hr style="margin: 6px 0; border-color: #F8D0C7;">
            • CUDA cuSPARSE & SYCL kernels<br>
            • Massively parallel SpMV & IPM<br>
            • Multi-GPU Subtree B&B<br>
            • Enterprise Refinery Models
        </div>
        """, unsafe_allow_html=True)
