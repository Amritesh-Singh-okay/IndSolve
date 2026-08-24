"""
IndSolve — UI Theme and Styling System
5-Color Palette:
- Deep Slate Teal : #244855 (Sidebar, App Header, High-Contrast Text, Structure)
- Terracotta Rust : #E64833 (Primary Action Buttons, Highlights, Optimal Markers)
- Earth Umber     : #874F41 (Metric Labels, Secondary Headings, Warnings)
- Sage Slate      : #90AEAD (Baseline Charts, Section Outlines, Inactive Borders)
- Cream Sand      : #FBE9D0 (Canvas Tint, Status Badges, Soft Container Accents)
"""

import streamlit as st


def apply_theme() -> None:
    """Injects responsive, high-contrast CSS ensuring robust typography and zero invisible inputs."""
    st.markdown("""
    <style>
        /* Global Typography & Canvas Reset */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #244855;
            -webkit-font-smoothing: antialiased;
        }

        .stApp {
            background-color: #FAF5ED;
        }

        /* --- DEEP SLATE TEAL SIDEBAR (#244855) --- */
        [data-testid="stSidebar"] {
            background-color: #244855 !important;
            border-right: 1px solid #1A3742;
        }
        [data-testid="stSidebar"] * {
            color: #FBE9D0;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
            font-weight: 700;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background-color: #1D3B47 !important;
            border: 1px solid #38616F !important;
            border-radius: 6px;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            color: #FFFFFF !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] hr {
            border-color: #38616F !important;
        }

        /* Sidebar Brand Emblem */
        .sidebar-brand-card {
            background: #1D3B47;
            border: 1px solid #38616F;
            border-left: 4px solid #E64833;
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 14px;
        }
        .sidebar-brand-title {
            font-size: 1.30rem;
            font-weight: 800;
            color: #FFFFFF !important;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .sidebar-brand-tag {
            font-size: 0.76rem;
            font-weight: 700;
            color: #FBE9D0 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 2px;
        }
        .sidebar-brand-sub {
            font-size: 0.70rem;
            color: #90AEAD !important;
            margin-top: 4px;
        }

        /* --- SIDEBAR INPUTS & STEPPERS (HIGH CONTRAST & TYPED VALUE VISIBILITY) --- */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] div[data-baseweb="input"] {
            background-color: #1D3B47 !important;
            color: #FFFFFF !important;
            border: 1px solid #38616F !important;
            border-radius: 4px !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="input"] input {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
        [data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
        [data-testid="stSidebar"] div[data-baseweb="input"] button {
            background-color: #2A5463 !important;
            color: #FFFFFF !important;
            border: 1px solid #38616F !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="input"] button:hover {
            background-color: #38616F !important;
        }

        /* --- DEEP SLATE TEAL TOP HEADER (#244855) --- */
        .app-header {
            background: #244855;
            border: 1px solid #1A3742;
            border-left: 6px solid #E64833;
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 14px;
            box-shadow: 0 3px 10px rgba(36, 72, 85, 0.10);
        }
        .app-title-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }
        .app-brand {
            font-size: clamp(1.25rem, 2.2vw, 1.55rem);
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
        }
        .app-tagline {
            font-size: clamp(0.82rem, 1.2vw, 0.95rem);
            color: #FBE9D0;
            font-weight: 600;
            margin-left: 6px;
        }
        .app-status {
            font-size: 0.78rem;
            color: #FFFFFF;
            font-weight: 600;
            background: #E64833;
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid #C83B28;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }
        .app-desc {
            font-size: 0.84rem;
            color: #90AEAD;
            margin-top: 4px;
            line-height: 1.35;
        }

        /* --- TOP NAVIGATION BAR STYLING --- */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            background: #FFFFFF;
            border: 1px solid #E5DFD5;
            border-radius: 8px;
            padding: 6px 12px;
            box-shadow: 0 1px 2px rgba(36, 72, 85, 0.04);
            gap: 12px;
            flex-wrap: wrap !important;
        }
        div[data-testid="stRadio"] label {
            font-weight: 700 !important;
            color: #244855 !important;
            font-size: 0.88rem !important;
            white-space: nowrap;
        }

        /* --- MAIN CANVAS FORM INPUTS, SELECTBOXES & DROPDOWNS (HIGH VISIBILITY) --- */
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #244855 !important;
            border: 1px solid #90AEAD !important;
            border-radius: 6px !important;
        }
        div[data-baseweb="select"] * {
            color: #244855 !important;
            font-weight: 600;
        }
        div[data-baseweb="popover"], ul[data-baseweb="menu"] {
            background-color: #FFFFFF !important;
            border: 1px solid #90AEAD !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 12px rgba(36, 72, 85, 0.12) !important;
        }
        li[data-baseweb="menu-item"] {
            color: #244855 !important;
            background-color: #FFFFFF !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
        }
        li[data-baseweb="menu-item"]:hover {
            background-color: #F4F8F8 !important;
            color: #E64833 !important;
            font-weight: 700 !important;
        }
        div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
            background-color: #FFFFFF !important;
            color: #244855 !important;
            border-color: #90AEAD !important;
            font-weight: 600 !important;
        }

        /* --- KPI METRIC TILES (NOWRAP & ADAPTIVE FONT) --- */
        .metric-tile {
            background: #FFFFFF;
            border: 1px solid #E5DFD5;
            border-top: 3px solid #244855;
            border-radius: 6px;
            padding: 10px 12px;
            box-shadow: 0 1px 2px rgba(36, 72, 85, 0.04);
            min-width: 0;
            overflow: hidden;
        }
        .metric-tile-primary {
            border-top: 3px solid #E64833 !important;
        }
        .metric-tile-label {
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #874F41;
            margin-bottom: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-tile-value {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: clamp(1.10rem, 1.6vw, 1.35rem);
            font-weight: 800;
            color: #244855;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-tile-sub {
            font-size: 0.74rem;
            font-weight: 600;
            color: #5C6B73;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* --- DECISION INSIGHT BOX --- */
        .decision-box {
            background: #FDF9F2;
            border: 1px solid #EAE0D3;
            border-left: 4px solid #E64833;
            border-radius: 0 6px 6px 0;
            padding: 10px 14px;
            margin: 10px 0;
        }
        .decision-box-title {
            font-size: 0.80rem;
            font-weight: 700;
            color: #E64833;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 6px;
        }

        /* --- CLEAN TABS STYLING (PREVENT OVERFLOW) --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            border-bottom: 2px solid #E5DFD5;
            padding-bottom: 2px;
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 12px;
            font-weight: 600;
            font-size: 0.82rem;
            border-radius: 4px 4px 0 0;
            color: #5C6B73;
            white-space: nowrap;
        }
        .stTabs [aria-selected="true"] {
            color: #244855 !important;
            font-weight: 700 !important;
            border-bottom-color: #E64833 !important;
        }

        /* --- PRIMARY BUTTONS (TERRACOTTA #E64833) --- */
        div.stButton > button:first-child, div[data-testid="stFormSubmitButton"] > button:first-child {
            background-color: #E64833 !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            box-shadow: 0 2px 4px rgba(230, 72, 51, 0.2) !important;
            transition: background-color 0.15s ease-in-out;
        }
        div.stButton > button:first-child:hover, div[data-testid="stFormSubmitButton"] > button:first-child:hover {
            background-color: #C83B28 !important;
            color: #FFFFFF !important;
        }

        /* --- RESPONSIVE MEDIA QUERIES --- */
        @media (max-width: 1024px) {
            .app-header {
                padding: 10px 14px;
            }
            .app-title-row {
                flex-direction: column;
                align-items: flex-start;
                gap: 4px;
            }
            .metric-tile {
                padding: 8px 10px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 4px 8px;
                font-size: 0.78rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
