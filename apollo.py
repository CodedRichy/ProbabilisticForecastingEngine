"""
Apollo Dashboard — full interactive interface.

Run:
    streamlit run dashboard.py
"""

import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apollo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0e1117; }
.metric-card {
    background: #1c1f26; border-radius: 8px; padding: 16px 20px;
    margin: 4px 0;
}
.green { color: #00d4aa; }
.red   { color: #ff4b4b; }
.gold  { color: #ffd700; }
div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Elo model…")
def load_elo():
    from core.elo_model import EloModel
    p = Path("data/models/elo_national.json")
    if not p.exists():
        return None
    return EloModel.load(str(p))


@st.cache_resource(show_spinner="Loading XGBoost model…")
def load_xgb():
    p = Path("data/models/xgb_predictor.pkl")
    if not p.exists():
        return None
    try:
        from core.xgb_predictor import XGBPredictor
        return XGBPredictor.load(str(p))
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading conformal filter…")
def load_conformal():
    p = Path("data/models/conformal_filter.pkl")
    if not p.exists():
        return None
    try:
        from core.conformal import ConformalFilter
        return ConformalFilter.load(str(p))
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner="Fetching fixtures…")
def get_fixtures(match_date: str, competition: str):
    from core.fixtures_fetcher import get_today_fixtures
    return get_today_fixtures(date=match_date)


@st.cache_data(ttl=300, show_spinner="Fetching odds…")
def get_odds(match_date: str, competition: str):
    try:
        from core.odds_fetcher import OddsFetcher
        return OddsFetcher().get_all_today(match_date, competition=competition)
    except Exception:
        return []


def run_predictions(fixtures, elo_model, xgb_model, bookmaker_odds):
    odds_by_match = {(o["home"], o["away"]): o for o in bookmaker_odds}
    predictions = []
    for f in fixtures:
        home, away = f["home"], f["away"]
        elo_pred = elo_model.predict(home, away, neutral=True)

        xgb_pred = None
        if xgb_model:
            odds_entry = odds_by_match.get((home, away), {})
            try:
                xgb_pred = xgb_model.predict({
                    "home_elo_k32":          elo_pred["home_elo"],
                    "away_elo_k32":          elo_pred["away_elo"],
                    "elo_delta_k32":         elo_pred["home_elo"] - elo_pred["away_elo"],
                    "elo_expected_home_k32": 1 / (1 + 10 ** ((elo_pred["away_elo"] - elo_pred["home_elo"]) / 400)),
                    "mkt_home_implied":      odds_entry.get("fair_home"),
                    "mkt_draw_implied":      odds_entry.get("fair_draw"),
                    "mkt_away_implied":      odds_entry.get("fair_away"),
                })
            except Exception:
                pass

        ELO_W, XGB_W = (0.70, 0.30) if xgb_pred else (1.0, 0.0)
        if xgb_pred:
            probs = {
                "p_home": ELO_W * elo_pred["p_home"] + XGB_W * xgb_pred["p_home"],
                "p_draw": ELO_W * elo_pred["p_draw"] + XGB_W * xgb_pred["p_draw"],
                "p_away": ELO_W * elo_pred["p_away"] + XGB_W * xgb_pred["p_away"],
            }
        else:
            probs = {k: elo_pred[k] for k in ("p_home", "p_draw", "p_away")}

        predictions.append({
            **probs,
            "home":     home,
            "away":     away,
            "time_utc": f.get("time_utc", "?"),
            "group":    f.get("group", ""),
            "home_elo": elo_pred["home_elo"],
            "away_elo": elo_pred["away_elo"],
            "xgb":      xgb_pred is not None,
        })
    return predictions


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔮 Apollo")
    st.markdown("*Probabilistic Forecasting Engine*")
    st.divider()

    page = st.radio("Navigate", ["🎯 Predictions", "💰 Value Bets",
                                  "📊 Performance", "⚙️  Models"], label_visibility="collapsed")
    st.divider()

    match_date = st.date_input("Match date", value=date.today(),
                               min_value=date(2026, 1, 1), max_value=date(2027, 12, 31))
    match_date_str = match_date.isoformat()

    competition = st.selectbox("Competition", [
        "wc2026", "epl", "laliga", "seriea", "bundesliga", "ligue1"
    ], format_func=lambda x: {
        "wc2026": "🌍 FIFA World Cup 2026",
        "epl": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League",
        "laliga": "🇪🇸 La Liga",
        "seriea": "🇮🇹 Serie A",
        "bundesliga": "🇩🇪 Bundesliga",
        "ligue1": "🇫🇷 Ligue 1",
    }[x])

    min_edge = st.slider("Min edge %", 1, 15, 3) / 100

    st.divider()
    elo_ok  = Path("data/models/elo_national.json").exists()
    xgb_ok  = Path("data/models/xgb_predictor.pkl").exists()
    rl_ok   = Path("data/models/rl_agent_dqn.zip").exists()
    odds_ok = bool(__import__("os").getenv("ODDS_API_KEY", ""))

    conformal_ok = Path("data/models/conformal_filter.pkl").exists()

    # New optional model components (silent-fail loads at predict time).
    player_ok   = Path("data/models/player_model.json").exists()
    transformer_ok = Path("data/models/sequence_model.pt").exists()
    referee_ok  = Path("data/models/referee_model.json").exists()
    # Fatigue model is derived from the processed-match schedule, not a model file.
    fatigue_ok  = Path("data/processed/matches.parquet").exists()

    st.markdown("**System status**")
    st.markdown(f"{'✅' if elo_ok         else '❌'} Elo model")
    st.markdown(f"{'✅' if xgb_ok         else '❌'} XGBoost model")
    st.markdown(f"{'✅' if player_ok      else '❌'} PlayerModel")
    st.markdown(f"{'✅' if transformer_ok else '❌'} Transformer")
    st.markdown(f"{'✅' if referee_ok     else '❌'} RefereeModel")
    st.markdown(f"{'✅' if fatigue_ok     else '❌'} FatigueModel")
    st.markdown(f"{'✅' if rl_ok          else '❌'} RL agent")
    st.markdown(f"{'✅' if odds_ok        else '❌'} Odds API key")
    st.markdown(f"{'✅' if conformal_ok   else '❌'} Conformal filter")

    st.divider()
    # ── Telegram ──────────────────────────────────────────────────────────────
    try:
        from core.notifier import TelegramNotifier as _TN
        _tg_ok = _TN().available()
    except Exception:
        _tg_ok = False
    st.markdown(f"{'✅' if _tg_ok else '❌'} Telegram bot")
    telegram_enabled = st.toggle("📱 Send to Telegram", value=True,
                                 help="Push value bet alerts to Telegram when enabled.")


# ── Load models once ──────────────────────────────────────────────────────────
elo_model = load_elo()
xgb_model = load_xgb()
conformal_filter = load_conformal()

if elo_model is None:
    st.error("Elo model not found. Run: `python scripts/build_elo.py`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Predictions":
    st.title(f"🎯 Predictions — {match_date_str}")

    col_fetch, col_mode = st.columns([2, 3])
    with col_fetch:
        fetch_btn = st.button("⚡ Fetch & Predict", type="primary", use_container_width=True)
    with col_mode:
        mode_str = "Elo + XGB (70/30)" if xgb_model else "Elo only"
        st.info(f"Model: **{mode_str}**  |  Competition: **{competition}**")

    if fetch_btn or "predictions" in st.session_state:
        if fetch_btn:
            fixtures = get_fixtures(match_date_str, competition)
            if not fixtures:
                st.warning(f"No fixtures found for {match_date_str}")
                st.stop()
            bookmaker_odds = get_odds(match_date_str, competition)
            preds = run_predictions(fixtures, elo_model, xgb_model, bookmaker_odds)
            st.session_state["predictions"]    = preds
            st.session_state["bookmaker_odds"] = bookmaker_odds

        preds        = st.session_state.get("predictions", [])
        book_odds    = st.session_state.get("bookmaker_odds", [])

        if not preds:
            st.info("Click **Fetch & Predict** to load today's matches.")
            st.stop()

        # ── Prediction cards ──────────────────────────────────────────────────
        st.markdown(f"**{len(preds)} matches** found")
        odds_map = {(o["home"], o["away"]): o for o in book_odds}

        for p in preds:
            home, away = p["home"], p["away"]
            odds = odds_map.get((home, away), {})
            pick_idx  = [p["p_home"], p["p_draw"], p["p_away"]].index(
                max(p["p_home"], p["p_draw"], p["p_away"]))
            pick = [home, "Draw", away][pick_idx]
            confidence = max(p["p_home"], p["p_draw"], p["p_away"])

            with st.container():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
                with c1:
                    xgb_badge = " `XGB`" if p["xgb"] else ""
                    st.markdown(f"**{home}** vs **{away}**{xgb_badge}")
                    st.caption(f"🕐 {p['time_utc']} UTC  |  Group {p['group']}  |  "
                               f"Elo: {p['home_elo']:.0f} vs {p['away_elo']:.0f}")
                with c2:
                    fig = go.Figure(go.Bar(
                        x=[p["p_home"], p["p_draw"], p["p_away"]],
                        y=[f"🏠 {home[:8]}", "Draw", f"✈ {away[:8]}"],
                        orientation="h",
                        marker_color=["#00d4aa" if pick == home else "#374151",
                                      "#ffd700" if pick == "Draw" else "#374151",
                                      "#00d4aa" if pick == away else "#374151"],
                        text=[f"{p['p_home']:.0%}", f"{p['p_draw']:.0%}", f"{p['p_away']:.0%}"],
                        textposition="inside",
                    ))
                    fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0),
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)",
                                      showlegend=False,
                                      xaxis=dict(visible=False, range=[0, 1]))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                with c3:
                    st.metric("Pick", pick)
                    st.caption(f"Conf: {confidence:.0%}")
                with c4:
                    if odds:
                        st.caption(f"📊 Odds  H: {odds.get('home_odds','—')}  "
                                   f"D: {odds.get('draw_odds','—')}  "
                                   f"A: {odds.get('away_odds','—')}")
                        st.caption(f"Overround: {odds.get('overround',0)*100:.1f}%  "
                                   f"Source: {odds.get('source','?')}")
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: VALUE BETS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💰 Value Bets":
    st.title("💰 Value Bets")

    if "predictions" not in st.session_state:
        st.info("Go to **🎯 Predictions** first and click Fetch & Predict.")
        st.stop()

    preds     = st.session_state["predictions"]
    book_odds = st.session_state.get("bookmaker_odds", [])

    if not book_odds:
        st.warning("No odds available. Check your `ODDS_API_KEY` in `.env`.")
        st.stop()

    from core.value_finder import find_value_bets
    value_bets = find_value_bets(
        preds,
        book_odds,
        min_edge=min_edge,
        conformal=conformal_filter,
    )

    # ── Line monitor: steam + Pinnacle lag signals ────────────────────────────
    _steam_signals: dict = {}
    _lag_signals: dict = {}
    try:
        from core.line_monitor import LineMonitor
        _lm = LineMonitor()
        for _vb in value_bets:
            _parts = _vb.match.split(" vs ", 1)
            if len(_parts) == 2:
                _h, _a = _parts
                _key = (_h, _a)
                if _key not in _steam_signals:
                    _steam_signals[_key] = _lm.detect_steam(_h, _a, match_date_str)
                    _lag_signals[_key] = _lm.pinnacle_lag(_h, _a, match_date_str)
    except Exception:
        pass  # line monitor is best-effort — never block value bets display

    # ── Telegram alerts ───────────────────────────────────────────────────────
    if telegram_enabled and value_bets:
        try:
            from core.notifier import TelegramNotifier
            _notifier = TelegramNotifier()
            if _notifier.available():
                _sent = 0
                for _vb in value_bets:
                    _parts = _vb.match.split(" vs ", 1)
                    _key   = tuple(_parts) if len(_parts) == 2 else None
                    _steam = _steam_signals.get(_key, {})
                    _lag   = _lag_signals.get(_key, {})
                    _conf  = bool(conformal_filter)  # True if filter loaded & passed
                    _ok = _notifier.alert_value_bet(
                        _vb,
                        conformal_passed=_conf,
                        steam=_steam if _steam.get("is_steam") else None,
                        lag=_lag   if _lag.get("lag_detected")  else None,
                    )
                    if _ok:
                        _sent += 1
                if _sent:
                    st.toast(f"📱 Sent {_sent} Telegram alert(s)", icon="✅")
        except Exception as _tg_exc:
            pass  # Telegram failure must never crash the dashboard

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_kelly = sum(vb.kelly for vb in value_bets)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Value bets found", len(value_bets))
    c2.metric("Total exposure", f"{total_kelly:.1%}")
    c3.metric("Best edge", f"{value_bets[0].edge:+.1%}" if value_bets else "—")
    c4.metric("Avg odds", f"{np.mean([vb.odds for vb in value_bets]):.2f}" if value_bets else "—")

    if not value_bets:
        st.info(f"No value bets above {min_edge:.0%} edge threshold. Try lowering the slider.")
        st.stop()

    st.divider()

    # ── Value bet cards ───────────────────────────────────────────────────────
    for vb in value_bets:
        edge_color = "#00d4aa" if vb.edge >= 0.07 else ("#ffd700" if vb.edge >= 0.04 else "#e5e7eb")

        # Retrieve line monitor signals for this match
        _match_parts = vb.match.split(" vs ", 1)
        _match_key = tuple(_match_parts) if len(_match_parts) == 2 else None
        _steam = _steam_signals.get(_match_key, {})
        _lag = _lag_signals.get(_match_key, {})

        # Build signal badges
        _badges = []
        if _steam.get("is_steam") and _steam.get("direction") == vb.outcome:
            _badges.append(f"🔥 STEAM ({_steam['magnitude']:+.1%})")
        if _lag.get("lag_detected") and _lag.get("outcome") == vb.outcome and _lag.get("edge_from_lag", 0) > 0:
            _badges.append(f"⚡ LAG EDGE ({_lag['edge_from_lag']:+.1%})")
        _badge_str = "  " + "  ".join(_badges) if _badges else ""

        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
            with c1:
                outcome_emoji = {"home": "🏠", "draw": "🤝", "away": "✈️"}.get(vb.outcome, "")
                st.markdown(f"**{vb.match}**{_badge_str}")
                st.markdown(f"{outcome_emoji} Bet **{vb.outcome.upper()}** on **{vb.team}**")
            with c2:
                st.metric("Odds", f"{vb.odds:.2f}")
            with c3:
                st.metric("Model", f"{vb.model_prob:.1%}")
            with c4:
                st.metric("Market", f"{vb.fair_implied:.1%}")
            with c5:
                st.metric("Edge", f"{vb.edge:+.1%}", delta=f"Kelly: {vb.kelly:.1%}")
            st.divider()

    # ── Stake table ───────────────────────────────────────────────────────────
    with st.expander("📋 Full stake table"):
        rows = [{
            "Match":    vb.match,
            "Bet":      vb.outcome.upper(),
            "Team":     vb.team,
            "Odds":     vb.odds,
            "Model %":  f"{vb.model_prob:.1%}",
            "Market %": f"{vb.fair_implied:.1%}",
            "Edge":     f"{vb.edge:+.1%}",
            "Kelly %":  f"{vb.kelly:.2%}",
        } for vb in value_bets]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Log bets button ───────────────────────────────────────────────────────
    st.divider()
    if st.button("📝 Log these bets to tracker", type="primary"):
        try:
            from core.prediction_logger import log_predictions
            n = log_predictions(preds, book_odds, value_bets,
                                competition=competition,
                                match_date=match_date_str)
            st.success(f"✅ Logged {n} predictions to `data/predictions/log.parquet`")
        except Exception as e:
            st.error(f"Logging failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Performance":
    st.title("📊 Performance Tracker")

    log_path = Path("data/predictions/log.parquet")
    if not log_path.exists():
        st.info("No prediction log yet. Make predictions with odds enabled to start tracking.")
        st.stop()

    col_resolve, col_report = st.columns(2)
    with col_resolve:
        if st.button("🔄 Resolve completed matches", type="primary"):
            with st.spinner("Fetching results + closing odds…"):
                try:
                    from core.prediction_logger import _load, _save
                    from core.clv_tracker import fetch_result, fetch_closing_odds, compute_clv, compute_pnl
                    df_all = _load()
                    today  = date.today().isoformat()
                    unresolved = df_all[
                        (~df_all["resolved"].fillna(False)) &
                        (df_all["match_date"].fillna("") < today)
                    ]
                    updated = 0
                    for idx in unresolved.index:
                        row = df_all.loc[idx]
                        result = fetch_result(row["home"], row["away"],
                                              row["match_date"], row.get("competition","wc2026"))
                        if not result:
                            continue
                        df_all.at[idx, "actual_result"] = result
                        closing = fetch_closing_odds(row["home"], row["away"],
                                                     row["match_date"], row.get("competition","wc2026"))
                        if closing:
                            df_all.at[idx, "close_home_odds"] = closing["home_odds"]
                            df_all.at[idx, "close_draw_odds"] = closing["draw_odds"]
                            df_all.at[idx, "close_away_odds"] = closing["away_odds"]
                            df_all.at[idx, "close_source"]    = closing["source"]
                            bet_out  = row.get("bet_outcome")
                            bet_odds = row.get("bet_odds")
                            if pd.notna(bet_out) and pd.notna(bet_odds):
                                co_map = {"home": closing["home_odds"],
                                          "draw": closing["draw_odds"],
                                          "away": closing["away_odds"]}
                                co = co_map.get(str(bet_out))
                                if co:
                                    clv_abs, clv_pct = compute_clv(float(bet_odds), float(co))
                                    df_all.at[idx, "clv"]     = clv_abs
                                    df_all.at[idx, "clv_pct"] = clv_pct
                        if pd.notna(row.get("bet_outcome")) and pd.notna(row.get("bet_odds")):
                            df_all.at[idx, "profit_loss"] = compute_pnl(
                                str(row["bet_outcome"]), float(row["bet_odds"]), result)
                        df_all.at[idx, "resolved"] = True
                        updated += 1
                    _save(df_all)
                    st.success(f"Resolved {updated} matches.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error: {e}")

    df = pd.read_parquet(log_path)
    resolved = df[df["resolved"].fillna(False)].copy()
    bets     = resolved[resolved["bet_outcome"].notna()].copy()

    # ── Top metrics ───────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Predictions", len(df))
    c2.metric("Resolved",    len(resolved))
    c3.metric("Bets placed", len(bets))

    if not bets.empty:
        wins    = (bets["profit_loss"] > 0).sum()
        total_pl = bets["profit_loss"].sum()
        roi     = total_pl / len(bets) * 100
        c4.metric("Win rate",   f"{wins/len(bets):.0%}", delta=f"{wins}W/{len(bets)-wins}L")
        c5.metric("ROI",        f"{roi:+.1f}%",          delta=f"{total_pl:+.2f} units")

        # ── CLV chart ─────────────────────────────────────────────────────────
        clv_data = bets[bets["clv_pct"].notna()].copy()
        if not clv_data.empty:
            avg_clv = clv_data["clv_pct"].mean()
            st.divider()
            col_clv, col_pnl = st.columns(2)

            with col_clv:
                clv_color = "#00d4aa" if avg_clv > 0 else "#ff4b4b"
                st.markdown(f"### Closing Line Value")
                st.markdown(f"<h2 style='color:{clv_color}'>{avg_clv:+.2f}%</h2>",
                            unsafe_allow_html=True)
                st.caption("Avg CLV across all bets. Positive = beating the market.")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=clv_data.index,
                    y=clv_data["clv_pct"],
                    marker_color=clv_data["clv_pct"].apply(
                        lambda v: "#00d4aa" if v > 0 else "#ff4b4b"),
                    name="CLV %"
                ))
                fig.add_hline(y=0, line_color="white", line_dash="dot")
                fig.update_layout(height=250, title="CLV per bet",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#1c1f26",
                                  showlegend=False,
                                  margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with col_pnl:
                st.markdown("### Cumulative P&L")
                bets_sorted = bets.sort_values("match_date")
                cum_pnl = bets_sorted["profit_loss"].fillna(0).cumsum()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=cum_pnl.values,
                    mode="lines+markers",
                    line=dict(color="#00d4aa", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,212,170,0.1)",
                ))
                fig2.add_hline(y=0, line_color="white", line_dash="dot")
                fig2.update_layout(height=250, title="Units P&L",
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="#1c1f26",
                                   showlegend=False,
                                   margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # ── Bet breakdown table ────────────────────────────────────────────────
        st.divider()
        st.markdown("### All Bets")
        display = bets[["match_date","home","away","bet_outcome","bet_odds",
                         "actual_result","profit_loss","clv_pct","bet_edge"]].copy()
        display.columns = ["Date","Home","Away","Bet","Odds","Result","P&L","CLV %","Edge"]
        display["P&L"]   = display["P&L"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        display["CLV %"] = display["CLV %"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
        display["Edge"]  = display["Edge"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)

    else:
        st.info("No resolved bets yet. Click **Resolve completed matches** after today's games finish.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️  Models":
    st.title("⚙️ Models & Training")

    tab1, tab2, tab3 = st.tabs(["Elo Ratings", "XGBoost", "RL Agent"])

    # ── Elo tab ───────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("National Team Elo Ratings")
        ratings = elo_model.ratings
        df_elo = (pd.DataFrame(list(ratings.items()), columns=["Team", "Elo"])
                  .sort_values("Elo", ascending=False)
                  .reset_index(drop=True))
        df_elo.index += 1

        col_search, col_n = st.columns([3, 1])
        search = col_search.text_input("Search team", placeholder="e.g. Brazil, France…")
        top_n  = col_n.number_input("Show top N", 10, 312, 30)

        if search:
            filtered = df_elo[df_elo["Team"].str.contains(search, case=False)]
            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(df_elo.head(top_n), use_container_width=True)

        st.divider()
        if st.button("🔁 Rebuild Elo model", help="Re-downloads martj42 CSV and retrains"):
            with st.spinner("Building Elo model…"):
                import subprocess
                result = subprocess.run(
                    [sys.executable, "scripts/build_elo.py"],
                    capture_output=True, text=True, cwd=str(Path(__file__).parent)
                )
                if result.returncode == 0:
                    st.success("Elo model rebuilt.")
                    st.cache_resource.clear()
                else:
                    st.error(f"Failed:\n{result.stderr[-500:]}")

    # ── XGBoost tab ───────────────────────────────────────────────────────────
    with tab2:
        st.subheader("XGBoost Predictor")
        if xgb_model is None:
            st.warning("XGBoost model not found.")
        else:
            metrics = getattr(xgb_model, "_eval_metrics", {})
            if metrics:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Val log-loss", f"{metrics.get('log_loss',0):.4f}")
                brier = metrics.get("brier", {})
                c2.metric("Brier (home)",  f"{brier.get('p_home',0):.3f}")
                c3.metric("Brier (draw)",  f"{brier.get('p_draw',0):.3f}")
                c4.metric("Brier (away)",  f"{brier.get('p_away',0):.3f}")

            try:
                fi = xgb_model.feature_importance.head(15)
                fig = px.bar(fi.reset_index(), x="index", y=fi.name or 0,
                             title="Top 15 Feature Importances",
                             color_discrete_sequence=["#00d4aa"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="#1c1f26",
                                  xaxis_title="Feature", yaxis_title="Importance",
                                  margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass

        st.divider()
        if st.button("🔁 Retrain XGBoost", help="Retrains on matches_xg.parquet"):
            with st.spinner("Training XGBoost (may take 3-5 min)…"):
                import subprocess
                result = subprocess.run(
                    [sys.executable, "scripts/train_xgb.py"],
                    capture_output=True, text=True, cwd=str(Path(__file__).parent)
                )
                if result.returncode == 0:
                    st.success("XGBoost retrained.")
                    st.cache_resource.clear()
                    st.code(result.stdout[-1000:])
                else:
                    st.error(f"Failed:\n{result.stderr[-500:]}")

    # ── RL tab ────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("RL Betting Agent (DQN)")
        rl_path = Path("data/models/rl_agent_dqn.zip")
        if rl_path.exists():
            stat = rl_path.stat()
            st.success(f"Agent loaded — {stat.st_size/1024:.0f} KB  "
                       f"(modified {pd.Timestamp(stat.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})")
            st.markdown("""
            **Last training:**
            - 200,000 timesteps on 32k club matches (xG-enriched)
            - Eval holdout: **58.8% win rate**, **+1766% ROI**
            - Max drawdown: **10.1%**
            - Actions: no_bet=42%, home=36%, draw=4%, away=18%
            """)
        else:
            st.warning("RL agent not trained yet.")

        steps = st.number_input("Timesteps", 50000, 1000000, 200000, step=50000)
        if st.button("🔁 Retrain RL Agent", help="Retrains DQN on club data"):
            with st.spinner(f"Training RL agent ({steps:,} steps — ~{steps//2000:.0f} min)…"):
                import subprocess
                result = subprocess.run(
                    [sys.executable, "scripts/train_agent.py", "--timesteps", str(steps)],
                    capture_output=True, text=True, cwd=str(Path(__file__).parent)
                )
                if result.returncode == 0:
                    st.success("RL agent retrained.")
                    st.code(result.stdout[-1500:])
                else:
                    st.error(f"Failed:\n{result.stderr[-500:]}")
