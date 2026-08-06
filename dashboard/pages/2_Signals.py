import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from dashboard.style import inject
from dashboard.components import sidebar, bss_roi_scatter
from dashboard.db import signals_df

st.set_page_config(page_title="Apollo · Signals", page_icon="✅", layout="wide")
inject()
sidebar()

st.title("Signal Registry")

df = signals_df()

# ── Header stats ──────────────────────────────────────────────
if not df.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Validated Signals", len(df[df["status"] == "validated"]))
    avg_bss = df["brier_skill_avg"].mean()
    avg_roi = df["roi_avg"].mean()
    c2.metric("Avg Brier Skill Score", f"{avg_bss:.4f}" if not df["brier_skill_avg"].isna().all() else "—")
    c3.metric("Avg ROI", f"{avg_roi:.3f}" if not df["roi_avg"].isna().all() else "—")
    st.divider()

tab_table, tab_scatter, tab_inspect = st.tabs(["📋  Table", "📊  Quality Scatter", "🔍  Inspect"])

# ── Table tab ─────────────────────────────────────────────────
with tab_table:
    if df.empty:
        st.info("No validated signals yet. Run the discovery pipeline to populate this registry.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        status_opts = df["status"].dropna().unique().tolist()
        status_filter = st.multiselect("Status", status_opts, default=status_opts)
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["brier_skill_avg", "roi_avg", "effect_size_avg", "gate1_p"],
            format_func=lambda x: {
                "brier_skill_avg":  "Brier Skill Score ↓",
                "roi_avg":         "ROI ↓",
                "effect_size_avg": "Effect Size ↓",
                "gate1_p":         "Gate 1 p-value ↑",
            }[x],
        )
    with col3:
        op_opts = df["operator"].dropna().unique().tolist()
        op_filter = st.multiselect("Operator", op_opts, default=op_opts)

    ascending = sort_by == "gate1_p"
    filtered = (
        df[df["status"].isin(status_filter) & df["operator"].isin(op_filter)]
        .sort_values(sort_by, ascending=ascending)
    )

    st.caption(f"{len(filtered)} signal(s)")

    st.dataframe(
        filtered[[
            "signal_id", "description", "operator",
            "brier_skill_avg", "roi_avg", "effect_size_avg",
            "gate1_p", "gate4_p", "status", "validated_at",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "signal_id":       st.column_config.TextColumn("ID", width="small"),
            "description":     st.column_config.TextColumn("Description", width="large"),
            "operator":        st.column_config.TextColumn("Operator", width="small"),
            "brier_skill_avg": st.column_config.NumberColumn("BSS",       format="%.4f"),
            "roi_avg":         st.column_config.NumberColumn("ROI",       format="%.3f"),
            "effect_size_avg": st.column_config.NumberColumn("Effect",    format="%.3f"),
            "gate1_p":         st.column_config.NumberColumn("Gate 1 p",  format="%.2e"),
            "gate4_p":         st.column_config.NumberColumn("Gate 4 p",  format="%.2e"),
            "validated_at":    st.column_config.DatetimeColumn("Validated", format="YYYY-MM-DD"),
        },
    )

# ── Scatter tab ───────────────────────────────────────────────
with tab_scatter:
    if df.empty:
        st.info("No signals to plot.")
    else:
        fig = bss_roi_scatter(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each dot is a validated signal. "
                "Top-right quadrant = best (high BSS, positive ROI). "
                "Reference lines at BSS=0 and ROI=0."
            )
        else:
            st.info("Not enough data to render scatter.")

# ── Inspect tab ───────────────────────────────────────────────
with tab_inspect:
    if df.empty:
        st.info("No signals.")
    else:
        selected_id = st.selectbox("Signal", df["signal_id"].tolist())
        if selected_id:
            row = df[df["signal_id"] == selected_id].iloc[0].to_dict()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BSS",    f"{row.get('brier_skill_avg', 0):.4f}" if row.get("brier_skill_avg") else "—")
            c2.metric("ROI",    f"{row.get('roi_avg', 0):.3f}"         if row.get("roi_avg")         else "—")
            c3.metric("Gate 1 p", f"{row.get('gate1_p', 1):.2e}"       if row.get("gate1_p")         else "—")
            c4.metric("Status", row.get("status", "—"))

            with st.expander("Full record", expanded=True):
                st.json(row)
