import streamlit as st

_CSS = """
<style>
/* ── Metric cards ──────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 20px 24px;
    transition: border-color 0.15s;
}
[data-testid="metric-container"]:hover { border-color: #10b981; }
[data-testid="stMetricLabel"] p {
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    color: #475569 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

/* ── Hide Streamlit chrome ─────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Tabs ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #1e293b;
    gap: 4px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #10b981 !important;
    border-bottom: 2px solid #10b981 !important;
    background: transparent !important;
}

/* ── Primary button ────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #10b981 !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
    border-radius: 8px !important;
    transition: background 0.15s;
}
.stButton > button[kind="primary"]:hover   { background: #059669 !important; }
.stButton > button[kind="primary"]:disabled {
    background: #1e293b !important;
    color: #334155 !important;
}

/* ── Secondary button ──────────────────────────────────────── */
.stButton > button[kind="secondary"] {
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    transition: border-color 0.15s, color 0.15s;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #10b981 !important;
    color: #10b981 !important;
}

/* ── Sidebar ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] [data-testid="metric-container"] {
    padding: 10px 14px;
    border-radius: 8px;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
}

/* ── Divider ───────────────────────────────────────────────── */
hr { border-color: #1e293b !important; margin: 1.25rem 0 !important; }

/* ── Expander ──────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--secondary-background-color) !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    color: #94a3b8 !important;
    border: 1px solid #1e293b !important;
}

/* ── Code blocks ───────────────────────────────────────────── */
.stCodeBlock pre {
    font-size: 0.80rem !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    background: #0a0f1a !important;
}

/* ── Caption / small text ──────────────────────────────────── */
.stCaption { color: #475569 !important; font-size: 0.77rem !important; }

/* ── Status pills ──────────────────────────────────────────── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.pill-green { background:#052e16; color:#10b981; border:1px solid #065f46; }
.pill-red   { background:#2d0404; color:#f87171; border:1px solid #7f1d1d; }
.pill-gray  { background:#0f172a; color:#64748b; border:1px solid #1e293b; }
.pill-blue  { background:#0c1a2e; color:#60a5fa; border:1px solid #1e3a8a; }
.pill-yellow{ background:#1c1400; color:#fbbf24; border:1px solid #78350f; }

/* ── Alert boxes ───────────────────────────────────────────── */
.stAlert { border-radius: 8px !important; }

/* ── Selectbox / multiselect ───────────────────────────────── */
[data-baseweb="select"] { border-radius: 8px !important; }
</style>
"""

def inject():
    st.markdown(_CSS, unsafe_allow_html=True)
