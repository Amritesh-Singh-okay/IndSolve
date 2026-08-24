"""
IndSolve — Validation Lab View
Automated mathematical test suite and benchmark verification dashboard.
"""

import time
from typing import Dict, Any
import pandas as pd
import streamlit as st

from solver.test_suite import run_full_verification_lab


def render_validation_view() -> None:
    """Renders the Validation Lab view with on-demand execution and failure-first sorting."""
    if "verification_results" not in st.session_state:
        st.session_state["verification_results"] = None
        st.session_state["verification_timestamp"] = None

    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.markdown("""
        <div style="margin-bottom: 5px;">
            <span style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">Validation Lab</span><br>
            <span style="font-size: 0.86rem; color: #64748B;">
                Automated regression test suite validating feasibility, contradiction detection, unboundedness, bound transformations, and exact MILP optimums.
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        run_v_btn = st.button("🔄 Run Verification Suite", type="primary", width="stretch")

    if run_v_btn or st.session_state["verification_results"] is None:
        with st.spinner("Executing Mathematical Unit Test Suite across 23 instances..."):
            st.session_state["verification_results"] = run_full_verification_lab()
            st.session_state["verification_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    test_out = st.session_state["verification_results"]
    ts = st.session_state["verification_timestamp"]
    total_p = test_out["total_passed"]
    total_t = test_out["total_tests"]
    all_p = (total_p == total_t)

    # Obvious Overall Pass/Fail Banner (Prompt 10)
    banner_bg = "#F4F8F8" if all_p else "#FEF6F4"
    banner_border = "#90AEAD" if all_p else "#E64833"
    banner_text = f"✅ ALL {total_t} MATHEMATICAL TESTS PASSING (100%)" if all_p else f"⚠️ {total_t - total_p} TESTS FAILED ({total_p}/{total_t} Passed)"
    banner_color = "#244855" if all_p else "#E64833"

    st.markdown(f"""
    <div style="background: {banner_bg}; border: 1px solid {banner_border}; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; color: {banner_color}; font-size: 0.95rem;">{banner_text}</span>
        <span style="font-size: 0.8rem; color: #874F41;">Engine: <b>IndSolve v0.2</b> | Last Run: {ts}</span>
    </div>
    """, unsafe_allow_html=True)

    # 6-Column Responsive Family Badges
    cols = st.columns(6)
    fam_items = list(test_out["family_stats"].items())
    for idx, (fam_name, s) in enumerate(fam_items):
        p = s["passed"]
        t = s["total"]
        if p == t and t > 0:
            border_c = "#90AEAD"
            bg_c = "#F4F8F8"
            sub_text = f"All Passed ({p}/{t}) ✅"
            sub_c = "#244855"
        elif p > 0:
            border_c = "#874F41"
            bg_c = "#FDF7F4"
            sub_text = f"Incomplete ({p}/{t}) ⚠️"
            sub_c = "#874F41"
        else:
            border_c = "#E64833"
            bg_c = "#FEF6F4"
            sub_text = f"Failed (0/{t}) ❌"
            sub_c = "#E64833"

        with cols[idx]:
            st.markdown(f"""
            <div style="background: {bg_c}; border: 1px solid {border_c}; border-radius: 6px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #874F41; font-weight: 700; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{fam_name}">{fam_name}</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #244855; margin: 3px 0;">{p}/{t}</div>
                <div style="font-size: 0.70rem; font-weight: 600; color: {sub_c};">{sub_text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # Full Test Table with Failure-First Sorting
    df_tests = test_out["results_df"].copy()
    df_tests["Result"] = df_tests["Passed"].apply(lambda p: "✅ PASS" if p else "❌ FAIL")
    df_tests = df_tests.sort_values(by=["Passed", "Family"], ascending=[True, True])

    st.dataframe(
        df_tests[["Family", "Test Name", "IndSolve Status", "Expected Status", "Computed Obj", "Expected Obj", "Latency (ms)", "Result", "Verification Note"]],
        width="stretch",
        hide_index=True
    )
