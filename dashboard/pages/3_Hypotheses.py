import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from dashboard.style import inject
from dashboard.components import sidebar, pvalue_hist_chart, pass_rate_chart
from dashboard.db import hypotheses_df, hypothesis_runs_df, pvalue_dist_df, pass_rate_by_level_df

st.set_page_config(page_title="Apollo · Hypotheses", page_icon="🧪", layout="wide")
inject()
sidebar()

st.title("Hypothesis Browser")

df = hypotheses_df()

# ── Header KPIs ───────────────────────────────────────────────
if not df.empty:
    total    = len(df)
    tested   = df["highest_gate_tested"].notna().sum()
    gate1_ok = (df["best_gate_passed"] >= 1).sum()
    gate2_ok = (df["best_gate_passed"] >= 2).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Hypotheses", f"{total:,}")
    c2.metric("Tested",           f"{tested:,}")
    c3.metric("Passed Gate 1",    gate1_ok)
    c4.metric("Passed Gate 2+",   gate2_ok)
    st.divider()

tab_browse, tab_analytics, tab_detail = st.tabs(
    ["📋  Browse", "📈  Analytics", "🔍  Hypothesis Detail"]
)

# ── Browse tab ────────────────────────────────────────────────
with tab_browse:
    if df.empty:
        st.info("No hypotheses generated yet. Run a discovery batch first.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        level_filter = st.multiselect("Level", sorted(df["level"].dropna().unique()))
    with col2:
        target_filter = st.multiselect("Target", sorted(df["target"].dropna().unique()))
    with col3:
        gate_filter = st.multiselect("Min gate passed", [0, 1, 2, 3, 4])
    with col4:
        batch_opts = df["generation_batch"].dropna().unique().tolist()
        batch_filter = st.multiselect("Batch", batch_opts)

    search = st.text_input("Search description", placeholder="e.g. form, elo, home advantage")

    filtered = df.copy()
    if level_filter:
        filtered = filtered[filtered["level"].isin(level_filter)]
    if target_filter:
        filtered = filtered[filtered["target"].isin(target_filter)]
    if gate_filter:
        filtered = filtered[filtered["best_gate_passed"] >= min(gate_filter)]
    if batch_filter:
        filtered = filtered[filtered["generation_batch"].isin(batch_filter)]
    if search.strip():
        filtered = filtered[
            filtered["description"].str.contains(search.strip(), case=False, na=False)
        ]

    st.caption(f"{len(filtered):,} / {len(df):,} hypotheses")

    st.dataframe(
        filtered[[
            "hypothesis_id", "description", "level", "target", "operator",
            "best_gate_passed", "best_p_raw", "generation_batch", "created_at",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "hypothesis_id":    st.column_config.TextColumn("ID", width="small"),
            "description":      st.column_config.TextColumn("Description", width="large"),
            "level":            st.column_config.NumberColumn("Lvl", width="small"),
            "best_gate_passed": st.column_config.NumberColumn("Best Gate", width="small"),
            "best_p_raw":       st.column_config.NumberColumn("Best p (raw)", format="%.2e"),
            "created_at":       st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD"),
        },
    )

# ── Analytics tab ─────────────────────────────────────────────
with tab_analytics:
    col_left, col_right = st.columns(2)

    with col_left:
        pv_df = pvalue_dist_df()
        fig = pvalue_hist_chart(pv_df, height=300)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Excess mass near p=0 vs the dashed H₀ uniform baseline indicates "
                "real signal in the feature space."
            )
        else:
            st.info("No Gate 1 p-values to plot.")

    with col_right:
        pr_df = pass_rate_by_level_df()
        fig2 = pass_rate_chart(pr_df, height=300)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Gate 1 raw pass rate (before FDR correction) by hypothesis "
                "level and prediction target."
            )
        else:
            st.info("No Gate 1 results yet.")

# ── Detail tab ────────────────────────────────────────────────
with tab_detail:
    if df.empty:
        st.info("No hypotheses.")
        st.stop()

    selected = st.selectbox(
        "Select hypothesis",
        df["hypothesis_id"].tolist(),
        format_func=lambda x: f"{x}  —  {df.loc[df['hypothesis_id']==x, 'description'].values[0][:80]}",
    )

    if selected:
        hyp = df[df["hypothesis_id"] == selected].iloc[0]

        st.markdown(f"### {hyp['description']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Level",      hyp["level"])
        c2.metric("Target",     hyp["target"])
        c3.metric("Operator",   hyp["operator"])
        c4.metric("Best Gate",  int(hyp["best_gate_passed"] or 0))

        if hyp["best_p_raw"] is not None:
            st.metric("Best raw p-value", f"{hyp['best_p_raw']:.4e}")

        st.divider()
        runs = hypothesis_runs_df(selected)

        if runs.empty:
            st.info("No test runs recorded for this hypothesis.")
        else:
            st.markdown(f"**{len(runs)} test run(s)**")
            st.dataframe(
                runs[[
                    "gate", "league", "passed", "p_value_raw", "p_value_by",
                    "brier_model", "brier_baseline", "brier_skill",
                    "roi_3pct", "n_observations", "rejection_reason", "completed_at",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "passed":      st.column_config.CheckboxColumn("Pass"),
                    "p_value_raw": st.column_config.NumberColumn("p (raw)", format="%.2e"),
                    "p_value_by":  st.column_config.NumberColumn("p (BY)",  format="%.2e"),
                    "brier_model": st.column_config.NumberColumn("Brier",   format="%.4f"),
                    "brier_skill": st.column_config.NumberColumn("BSS",     format="%.4f"),
                    "roi_3pct":    st.column_config.NumberColumn("ROI @3%", format="%.3f"),
                    "completed_at": st.column_config.DatetimeColumn("Completed", format="YYYY-MM-DD HH:mm"),
                },
            )
