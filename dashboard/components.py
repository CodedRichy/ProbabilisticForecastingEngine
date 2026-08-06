"""
Apollo Dashboard — shared UI components.

Every page calls components.sidebar() first, then uses the chart
functions here rather than building Plotly figures inline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.db import overview_stats

# ── Colour palette ────────────────────────────────────────────
GREEN  = "#10b981"
RED    = "#ef4444"
BLUE   = "#3b82f6"
YELLOW = "#f59e0b"
PURPLE = "#8b5cf6"
MUTED  = "#334155"

_COLORS = [GREEN, BLUE, YELLOW, PURPLE, RED]

# ── Shared Plotly layout ──────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#64748b", family="monospace", size=11),
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="#334155",
        font_color="#e2e8f0",
        font_size=12,
    ),
)

def _ax(title: str = "", fmt: str = "", pct: bool = False) -> dict:
    d = dict(
        title=title,
        gridcolor="#1e293b",
        linecolor="#334155",
        tickcolor="#334155",
        title_font=dict(color="#475569", size=11),
        tickfont=dict(color="#64748b"),
        zeroline=False,
    )
    if pct:
        d["tickformat"] = ".0%"
    return d

def _layout(title: str = "", height: int = 300, **kwargs) -> dict:
    d = {
        **_BASE,
        "height": height,
        "margin": dict(l=0, r=0, t=44 if title else 10, b=0),
        "legend": dict(
            orientation="h", y=1.12,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b"),
        ),
    }
    if title:
        d["title"] = dict(
            text=title,
            font=dict(color="#94a3b8", size=13, family="monospace"),
            x=0,
        )
    d.update(kwargs)
    return d


# ── Sidebar ───────────────────────────────────────────────────

def sidebar():
    stats = overview_stats()
    with st.sidebar:
        st.markdown("## 🔭 Apollo")
        st.caption("Quantitative Research Engine")
        st.divider()

        c1, c2 = st.columns(2)
        c1.metric("Hypotheses", f"{stats['n_hypotheses']:,}")
        c2.metric("Signals",    stats['n_signals'])
        c1.metric("Tests",      f"{stats['n_tests']:,}")
        c2.metric("Batches",    stats['n_batches'])

        st.divider()
        if st.button("↺  Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("apollo.db · parquet")


# ── Gate funnel ───────────────────────────────────────────────

def gate_funnel_chart(df: pd.DataFrame, height: int = 280) -> go.Figure | None:
    if df.empty or df["total"].sum() == 0:
        return None

    df = df.copy()
    df["pct"] = (df["passed"] / df["total"].replace(0, 1) * 100).round(1)

    fig = go.Figure()
    fig.add_bar(
        name="Tested",
        x=df["gate"], y=df["total"],
        marker_color=MUTED,
        text=df["total"],
        textposition="outside",
        textfont=dict(color="#64748b", size=10),
    )
    fig.add_bar(
        name="Passed",
        x=df["gate"], y=df["passed"],
        marker_color=GREEN,
        text=[f"{p}%" for p in df["pct"]],
        textposition="outside",
        textfont=dict(color=GREEN, size=10),
    )
    fig.update_layout(
        **_layout("Gate Survival Funnel", height),
        barmode="overlay",
        xaxis=_ax("Gate"),
        yaxis=_ax("Hypotheses"),
    )
    return fig


# ── Tests-over-time line ──────────────────────────────────────

def timeline_chart(df: pd.DataFrame, height: int = 200) -> go.Figure | None:
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_scatter(
        x=df["date"], y=df["cumulative"],
        mode="lines",
        line=dict(color=GREEN, width=2),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.07)",
        hovertemplate="%{x|%b %d, %Y}<br><b>%{y:,}</b> cumulative tests<extra></extra>",
    )
    fig.update_layout(
        **_layout("Cumulative Tests Over Time", height),
        showlegend=False,
        xaxis=_ax(),
        yaxis=_ax("Tests"),
    )
    return fig


# ── P-value histogram ─────────────────────────────────────────

def pvalue_hist_chart(df: pd.DataFrame, height: int = 280) -> go.Figure | None:
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_histogram(
        x=df["p_value_raw"],
        nbinsx=20,
        marker_color=BLUE,
        opacity=0.75,
        histnorm="probability density",
        name="Observed",
        hovertemplate="p ∈ [%{x:.2f}]<br>density %{y:.2f}<extra></extra>",
    )
    fig.add_hline(
        y=1.0,
        line=dict(color=YELLOW, dash="dash", width=1.5),
        annotation_text="H₀ uniform",
        annotation_font=dict(color=YELLOW, size=10),
        annotation_position="bottom right",
    )
    fig.update_layout(
        **_layout("Gate 1 P-value Distribution", height),
        showlegend=False,
        xaxis={**_ax("p-value (raw)"), "range": [0, 1]},
        yaxis=_ax("Density"),
    )
    return fig


# ── Pass rate by level × target ───────────────────────────────

def pass_rate_chart(df: pd.DataFrame, height: int = 260) -> go.Figure | None:
    if df.empty:
        return None

    fig = go.Figure()
    for i, tgt in enumerate(df["target"].unique()):
        sub = df[df["target"] == tgt]
        fig.add_bar(
            x=sub["level"].astype(str),
            y=sub["pass_rate"],
            name=tgt,
            marker_color=_COLORS[i % len(_COLORS)],
            text=[f"{v:.1%}" for v in sub["pass_rate"]],
            textposition="outside",
            textfont=dict(color="#64748b", size=9),
        )
    fig.update_layout(
        **_layout("Pass Rate  ·  Level × Target", height),
        barmode="group",
        xaxis=_ax("Level"),
        yaxis={**_ax("Pass rate"), "tickformat": ".0%", "range": [0, 1.1]},
    )
    return fig


# ── BSS vs ROI scatter ────────────────────────────────────────

def bss_roi_scatter(df: pd.DataFrame, height: int = 340) -> go.Figure | None:
    if df.empty:
        return None

    status_color = df["status"].map({"validated": GREEN}).fillna(RED)

    fig = go.Figure()
    fig.add_scatter(
        x=df["brier_skill_avg"],
        y=df["roi_avg"],
        mode="markers+text",
        marker=dict(
            color=status_color,
            size=11,
            line=dict(color="#0f172a", width=1),
            opacity=0.85,
        ),
        text=df["signal_id"],
        textposition="top right",
        textfont=dict(color="#475569", size=9),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "BSS: %{x:.4f}<br>"
            "ROI: %{y:.3f}<extra></extra>"
        ),
    )
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot", width=1))
    fig.add_vline(x=0, line=dict(color=MUTED, dash="dot", width=1))
    fig.update_layout(
        **_layout("Signal Quality  ·  BSS vs ROI", height),
        showlegend=False,
        xaxis=_ax("Brier Skill Score"),
        yaxis=_ax("ROI"),
    )
    return fig


# ── Probability mini-bar (HTML) ───────────────────────────────

def prob_bar_html(p_home: float, p_draw: float, p_away: float) -> str:
    """Inline stacked bar as an HTML string for use in st.markdown."""
    if any(v is None for v in (p_home, p_draw, p_away)):
        return "—"
    w_h = round(p_home * 100)
    w_d = round(p_draw * 100)
    w_a = 100 - w_h - w_d
    return (
        f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;width:120px">'
        f'<div style="width:{w_h}%;background:#10b981"></div>'
        f'<div style="width:{w_d}%;background:#f59e0b"></div>'
        f'<div style="width:{max(w_a,0)}%;background:#3b82f6"></div>'
        f'</div>'
    )
