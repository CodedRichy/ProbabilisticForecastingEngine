import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from dashboard.style import inject
from dashboard.components import sidebar, prob_bar_html

ROOT       = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / "data" / "models" / "elo_national.json"

st.set_page_config(page_title="Apollo · Predict", page_icon="⚽", layout="wide")
inject()
sidebar()

st.title("Today's Predictions")
st.caption("EloModel probabilities — national team football")

# ── Model guard ───────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error(
        f"No EloModel at `{MODEL_PATH.relative_to(ROOT)}`.\n\n"
        "**Fix:** `python scripts/build_elo.py`"
    )
    st.stop()


@st.cache_resource(show_spinner="Loading EloModel…")
def _load_model():
    from core.elo_model import EloModel
    return EloModel.load(str(MODEL_PATH))


model = _load_model()

# ── Controls ──────────────────────────────────────────────────
col_date, col_btn, _ = st.columns([1, 1, 4])
with col_date:
    target_date = st.date_input("Date", value=date.today())
with col_btn:
    st.write("")  # vertical align
    st.write("")
    fetch = st.button("⚽  Fetch & Predict", type="primary")

if not fetch:
    st.info("Select a date and click **Fetch & Predict** to load fixtures.")
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────
from core.fixtures_fetcher import get_today_fixtures

with st.spinner(f"Fetching fixtures for {target_date}…"):
    try:
        fixtures = get_today_fixtures(date=target_date.isoformat())
    except Exception as e:
        st.error(f"Fixture fetch failed: {e}")
        st.stop()

if not fixtures:
    st.warning(f"No fixtures found for {target_date}.")
    st.stop()

# ── Predict ───────────────────────────────────────────────────
rows, errors = [], []
for f in fixtures:
    try:
        pred = model.predict(f["home"], f["away"], neutral=True)
        ph, pd_, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
        probs = [ph, pd_, pa]
        best_idx = probs.index(max(probs))
        best = [f["home"], "Draw", f["away"]][best_idx]
        confidence = max(probs) - sorted(probs)[-2]  # margin over 2nd choice
        rows.append({
            "home":      f["home"],
            "away":      f["away"],
            "time_utc":  f.get("time_utc", "—"),
            "group":     f.get("group", "—"),
            "p_home":    ph,
            "p_draw":    pd_,
            "p_away":    pa,
            "pick":      best,
            "confidence": confidence,
            "_bar":      prob_bar_html(ph, pd_, pa),
        })
    except Exception as e:
        errors.append({"match": f"{f['home']} vs {f['away']}", "error": str(e)})

# ── Display ───────────────────────────────────────────────────
df = pd.DataFrame(rows)

c1, c2, c3 = st.columns(3)
c1.metric("Fixtures", len(df))
if not df.empty:
    top_pick = df.loc[df["confidence"].idxmax(), "pick"]
    top_conf = df["confidence"].max()
    c2.metric("Highest confidence pick", top_pick)
    c3.metric("Confidence margin",        f"{top_conf:.1%}")

st.divider()

if df.empty:
    st.warning("No predictions generated.")
else:
    # Render table with HTML mini-bars
    def _fmt_p(v):
        if v is None:
            return "—"
        # Green if > 50%, yellow if 35-50%, gray otherwise
        color = "#10b981" if v > 0.50 else "#f59e0b" if v > 0.35 else "#64748b"
        return f'<span style="color:{color};font-weight:700">{v:.1%}</span>'

    html_rows = []
    for _, r in df.iterrows():
        html_rows.append(
            f"<tr>"
            f"<td style='padding:8px 12px'>{r['time_utc']}</td>"
            f"<td style='padding:8px 12px'>{r.get('group','—')}</td>"
            f"<td style='padding:8px 12px;font-weight:600'>{r['home']}</td>"
            f"<td style='padding:8px 12px;color:#475569'>vs</td>"
            f"<td style='padding:8px 12px;font-weight:600'>{r['away']}</td>"
            f"<td style='padding:8px 12px;text-align:center'>{_fmt_p(r['p_home'])}</td>"
            f"<td style='padding:8px 12px;text-align:center'>{_fmt_p(r['p_draw'])}</td>"
            f"<td style='padding:8px 12px;text-align:center'>{_fmt_p(r['p_away'])}</td>"
            f"<td style='padding:8px 12px'>{r['_bar']}</td>"
            f"<td style='padding:8px 12px;color:#10b981;font-weight:700'>{r['pick']}</td>"
            f"</tr>"
        )

    header_style = "padding:8px 12px;color:#475569;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1e293b"
    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;font-family:monospace">
      <thead>
        <tr>
          <th style="{header_style}">Time</th>
          <th style="{header_style}">Group</th>
          <th style="{header_style}" colspan="3">Match</th>
          <th style="{header_style};text-align:center">P(H)</th>
          <th style="{header_style};text-align:center">P(D)</th>
          <th style="{header_style};text-align:center">P(A)</th>
          <th style="{header_style}">Distribution</th>
          <th style="{header_style}">Pick</th>
        </tr>
      </thead>
      <tbody style="color:#e2e8f0">
        {"".join(html_rows)}
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.divider()
    st.caption(
        "🟢 P(Home)  🟡 P(Draw)  🔵 P(Away)  ·  Distribution bar shows relative split.  "
        "Probabilities from EloModel on national team ratings."
    )

# ── Errors ────────────────────────────────────────────────────
if errors:
    with st.expander(f"⚠️  {len(errors)} prediction error(s)"):
        for e in errors:
            st.markdown(f"**{e['match']}** — `{e['error']}`")
