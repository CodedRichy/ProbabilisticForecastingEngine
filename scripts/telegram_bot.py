"""
scripts/telegram_bot.py

Bidirectional Telegram polling bot for Apollo Forecasting Engine.

Handles user commands via long-polling (getUpdates, timeout=30).
Updates user profile in response to /bankroll, /risk, /minedge, etc.

Run:
    python scripts/telegram_bot.py

Environment variables (set in .env):
    TELEGRAM_BOT_TOKEN  — Bot API token from BotFather
    TELEGRAM_CHAT_ID    — Your personal chat ID (run get_chat_id.py once)
    ODDS_API_KEY        — The Odds API key (for /today command)
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests

from core.user_profile import UserProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_URL = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT = 30   # seconds for long-polling
_SEND_TIMEOUT = 10   # seconds for sendMessage

# Owner identification — the TELEGRAM_CHAT_ID owner gets full admin access.
# Evaluated at startup after .env is loaded.
OWNER_CHAT_ID: int = 0  # set in main() after load_dotenv()

# Commands that mutate profile/settings or expose admin diagnostics —
# restricted to owner only.
ADMIN_COMMANDS  = {"/bankroll", "/risk", "/minedge", "/mute", "/competitions",
                   "/model", "/referee"}
# Commands anyone (if they somehow get the bot link) may use.
READ_COMMANDS   = {"/today", "/picks", "/status", "/help", "/start"}

COMPETITION_LABELS: dict[str, str] = {
    "wc2026":     "FIFA World Cup 2026",
    "epl":        "Premier League",
    "laliga":     "La Liga",
    "seriea":     "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1":     "Ligue 1",
}

ELO_W = 0.70
XGB_W = 0.30

# ── Ensemble weight table (mirrors scripts/predict_today.py) ──────────────────
#   elo + xgb                          -> 0.70 / 0.30
#   elo + xgb + player_model           -> 0.50 / 0.30 / 0.20
#   elo + xgb + transformer            -> 0.45 / 0.30 / 0.25
#   elo + xgb + player_model + transf  -> 0.40 / 0.25 / 0.20 / 0.15
ENSEMBLE_WEIGHTS: dict[frozenset, dict[str, float]] = {
    frozenset({"elo", "xgb"}):                          {"elo": 0.70, "xgb": 0.30},
    frozenset({"elo", "xgb", "player"}):                {"elo": 0.50, "xgb": 0.30, "player": 0.20},
    frozenset({"elo", "xgb", "transformer"}):           {"elo": 0.45, "xgb": 0.30, "transformer": 0.25},
    frozenset({"elo", "xgb", "player", "transformer"}): {"elo": 0.40, "xgb": 0.25, "player": 0.20, "transformer": 0.15},
}


def _ensemble_weights(active: list[str]) -> dict[str, float]:
    """Normalised ensemble weights for the active component set (see table)."""
    key = frozenset(active)
    if key in ENSEMBLE_WEIGHTS:
        return dict(ENSEMBLE_WEIGHTS[key])
    if active == ["elo"]:
        return {"elo": 1.0}
    base = {"elo": 0.70, "xgb": 0.30, "player": 0.20, "transformer": 0.25}
    w = {c: base.get(c, 0.0) for c in active}
    total = sum(w.values())
    if total <= 0:
        return {"elo": 1.0}
    return {c: v / total for c, v in w.items()}

# HTML help text
HELP_TEXT = (
    "🔮 <b>Apollo Bot Commands</b>\n"
    "\n"
    "📋 <b>Public</b>\n"
    "<code>/today [competition]</code> — Today's picks\n"
    "<code>/picks [competition]</code> — Picks with portfolio staking\n"
    "<code>/status</code> — Profile &amp; P&amp;L\n"
    "<code>/help</code> — This message\n"
    "\n"
    "⚙️ <b>Owner only</b>\n"
    "<code>/bankroll &lt;amount&gt;</code> — Set bankroll\n"
    "<code>/risk &lt;mode&gt;</code> — conservative | moderate | aggressive\n"
    "<code>/minedge &lt;percent&gt;</code> — Min edge threshold\n"
    "<code>/mute &lt;minutes&gt;</code> — Pause alerts\n"
    "<code>/competitions &lt;list&gt;</code> — Filter competitions\n"
    "<code>/model</code> — Loaded models &amp; ensemble weights\n"
    "<code>/referee &lt;name&gt;</code> — Referee bias stats\n"
    "\n"
    "★ high confidence · ☆ value bet"
)


# ── HTML helpers ─────────────────────────────────────────────────────────────

import html as _html_mod

def _md(text: str) -> str:
    """Escape a value for safe embedding in HTML parse_mode messages.
    Named _md for backwards compatibility with existing call sites.
    """
    return _html_mod.escape(str(text))


# ── Low-level Telegram helpers ────────────────────────────────────────────────

def _api(token: str, method: str, **kwargs) -> dict:
    """Call the Telegram Bot API. Returns decoded JSON or raises on error."""
    url = _BASE_URL.format(token=token, method=method)
    try:
        resp = requests.post(url, json=kwargs, timeout=_SEND_TIMEOUT)
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("Telegram API call failed (%s): %s", method, exc)
        return {"ok": False}


def send(token: str, chat_id: str, text: str) -> bool:
    """Send a plain-text message. Returns True on success."""
    data = _api(token, "sendMessage",
                chat_id=chat_id,
                text=text,
                disable_web_page_preview=True)
    if not data.get("ok"):
        logger.warning("sendMessage failed: %s", data.get("description", ""))
    return bool(data.get("ok"))


def send_md(token: str, chat_id: str, text: str,
            reply_markup: dict | None = None) -> bool:
    """Send an HTML-formatted message. Returns True on success."""
    kwargs: dict = dict(
        chat_id=chat_id,
        text=text,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    if reply_markup is not None:
        kwargs["reply_markup"] = reply_markup
    data = _api(token, "sendMessage", **kwargs)
    if not data.get("ok"):
        logger.warning("sendMessage (HTML) failed: %s", data.get("description", ""))
    return bool(data.get("ok"))


def answer_callback(token: str, callback_query_id: str,
                    text: str = "") -> None:
    """Acknowledge a callback query so the Telegram spinner disappears."""
    _api(token, "answerCallbackQuery",
         callback_query_id=callback_query_id,
         text=text)


def send_typing(token: str, chat_id: str) -> None:
    """Broadcast 'typing…' action — shows spinner while processing."""
    _api(token, "sendChatAction", chat_id=chat_id, action="typing")


def get_updates(token: str, offset: int) -> list[dict]:
    """Long-poll for new updates. Returns list of update dicts."""
    url = _BASE_URL.format(token=token, method="getUpdates")
    try:
        resp = requests.post(
            url,
            json={
                "offset": offset,
                "timeout": _POLL_TIMEOUT,
                "allowed_updates": ["message", "callback_query"],
            },
            timeout=_POLL_TIMEOUT + 5,
        )
        data = resp.json()
        if data.get("ok"):
            return data.get("result", [])
    except requests.RequestException as exc:
        logger.warning("getUpdates failed: %s", exc)
    return []


# ── Prediction helpers ────────────────────────────────────────────────────────

def _load_models() -> dict:
    """Load all available models. Returns a dict keyed by model name.

    Keys: ``elo``, ``xgb``, ``conformal``, ``referee``, ``player``,
    ``sequence``, ``fatigue``. Every load except Elo is best-effort and silent —
    a missing model file simply leaves its key set to ``None`` so the bot keeps
    working with whatever subset is present.
    """
    from core.elo_model import EloModel
    models: dict = {
        "elo": None, "xgb": None, "conformal": None,
        "referee": None, "player": None, "sequence": None, "fatigue": None,
    }

    elo_path = Path("data/models/elo_national.json")
    if elo_path.exists():
        models["elo"] = EloModel.load(str(elo_path))
        # Warn if Elo data is stale (older than 7 days) — ratings may drift
        # significantly over a week of international fixtures.
        import os as _os
        age_days = (time.time() - _os.path.getmtime(str(elo_path))) / 86_400
        if age_days > 7:
            logger.warning(
                "elo_national.json is %.0f days old — run scripts/build_elo.py "
                "to refresh ratings before WC2026 predictions.", age_days
            )

    xgb_path = Path("data/models/xgb_predictor.pkl")
    if xgb_path.exists():
        try:
            from core.xgb_predictor import XGBPredictor
            models["xgb"] = XGBPredictor.load(str(xgb_path))
        except Exception:
            pass

    conf_path = Path("data/models/conformal_filter.pkl")
    if conf_path.exists():
        try:
            from core.conformal import ConformalFilter
            models["conformal"] = ConformalFilter.load(str(conf_path))
        except Exception:
            pass

    # ── New optional model components (all silent-fail) ──────────────────────
    ref_path = Path("data/models/referee_model.json")
    if ref_path.exists():
        try:
            from core.referee_model import RefereeModel
            models["referee"] = RefereeModel.load(str(ref_path))
        except Exception:
            pass

    player_path = Path("data/models/player_model.json")
    if player_path.exists():
        try:
            from core.player_model import PlayerModel
            models["player"] = PlayerModel.load(str(player_path))
        except Exception:
            pass

    seq_path = Path("data/models/sequence_model.pt")
    if seq_path.exists():
        try:
            from core.sequence_model import SequenceModel
            models["sequence"] = SequenceModel.load(str(seq_path))
        except Exception:
            pass

    # Fatigue model initialised from the processed-match schedule.
    sched_path = Path("data/processed/matches.parquet")
    if sched_path.exists():
        try:
            import pandas as pd
            from core.fatigue_model import FatigueModel
            fm = FatigueModel()
            fm.build_schedule_index(pd.read_parquet(sched_path))
            models["fatigue"] = fm
        except Exception:
            pass

    return models


def _elo_probs_from_ratings(home_elo: float, away_elo: float, neutral: bool = True) -> dict:
    """Recompute 1X2 probabilities from (possibly adjusted) raw Elo ratings.

    Mirrors EloModel.predict / scripts.predict_today._elo_probs_from_ratings so
    referee + fatigue Elo deltas can be folded in before the probability calc.
    """
    home_adv = 0.0 if neutral else 75.0  # HOME_ADVANTAGE in core.elo_model
    exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - home_adv) / 400))

    elo_delta = abs((home_elo + home_adv) - away_elo)
    if elo_delta <= 50:
        draw_prob = 0.28
    elif elo_delta <= 100:
        draw_prob = 0.26
    elif elo_delta <= 150:
        draw_prob = 0.24
    elif elo_delta <= 200:
        draw_prob = 0.21
    elif elo_delta <= 300:
        draw_prob = 0.18
    else:
        draw_prob = 0.14

    p_home = exp_home * (1.0 - draw_prob)
    p_away = (1.0 - exp_home) * (1.0 - draw_prob)
    total = p_home + draw_prob + p_away
    return {
        "p_home": p_home / total,
        "p_draw": draw_prob / total,
        "p_away": p_away / total,
        "home_elo": home_elo,
        "away_elo": away_elo,
    }


def _ensemble_blend(elo: dict, xgb: dict | None,
                    player: dict | None = None,
                    transformer: dict | None = None) -> dict:
    """Blend 1X2 probabilities across the active components using the weight table."""
    comps: dict[str, dict] = {"elo": elo}
    if xgb is not None:
        comps["xgb"] = xgb
    if player is not None:
        comps["player"] = player
    if transformer is not None:
        comps["transformer"] = transformer

    weights = _ensemble_weights(list(comps.keys()))
    out = {"p_home": 0.0, "p_draw": 0.0, "p_away": 0.0}
    for name, pred in comps.items():
        w = weights.get(name, 0.0)
        for k in out:
            out[k] += w * pred[k]
    total = out["p_home"] + out["p_draw"] + out["p_away"]
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def _predict_match(home: str, away: str, odds: dict | None, models: dict,
                   date_str: str, referee_name: str = "",
                   competition: str = "wc2026", neutral: bool = True) -> dict:
    """Build a blended 1X2 prediction for one fixture.

    ``models`` is the dict returned by :func:`_load_models`. Referee and fatigue
    Elo deltas (when those models are loaded) are applied to the raw Elos BEFORE
    the probability calc; player/transformer predictions are blended into the
    ensemble alongside elo + xgb. All optional components silent-fail per match.

    Parameters
    ----------
    competition : str
        Competition slug (e.g. ``"wc2026"``, ``"epl"``). XGB is skipped for
        international competitions because the model has no club-form features
        for national teams; applying club-trained medians produces meaningless
        output that drags the ensemble toward 50%.
    neutral : bool
        Whether the fixture is at a neutral venue.  Passed directly to
        ``elo_model.predict()``; True removes the 75-point home-advantage term.
        WC2026 matches are always neutral.
    """
    # International competitions: XGB was trained exclusively on club-league
    # data (EPL, Bundesliga, La Liga, etc.).  National teams (Germany, Ecuador,
    # etc.) are absent from the training set, so ALL form features resolve to
    # the club training-set median — turning XGB into structured noise that
    # pulls the ensemble toward ~50% and erases the Elo signal.
    INTERNATIONAL_COMPETITIONS = {"wc2026", "euro", "copa_america", "nations_league"}
    _xgb_eligible = competition.lower() not in INTERNATIONAL_COMPETITIONS

    elo_model = models.get("elo")
    xgb_model = models.get("xgb")
    referee_model = models.get("referee")
    fatigue_model = models.get("fatigue")
    player_model = models.get("player")
    sequence_model = models.get("sequence")

    elo_pred = elo_model.predict(home, away, neutral=neutral)

    # Warn when either team fell back to DEFAULT_ELO (1500).  This produces a
    # near-50/50 output regardless of actual strength — e.g. France vs an
    # unknown team would show ~36/28/36 instead of 65/20/15.
    # Fix: re-run `python scripts/build_elo.py` and add missing name variants
    # to TEAM_ALIASES in core/team_names.py.
    if elo_pred.get("elo_fallback"):
        logger.warning(
            "Elo fallback for '%s' vs '%s' — one or both teams missing from "
            "ratings (home_elo=%.0f, away_elo=%.0f). Prediction is unreliable "
            "(%.1f%% / %.1f%% / %.1f%%). "
            "Re-run scripts/build_elo.py and check core/team_names.py aliases.",
            home, away,
            elo_pred["home_elo"], elo_pred["away_elo"],
            elo_pred["p_home"] * 100,
            elo_pred["p_draw"] * 100,
            elo_pred["p_away"] * 100,
        )

    # ── Referee + fatigue Elo deltas (applied BEFORE probability calc) ─────────
    referee_elo_adj = 0.0
    home_fatigue_adj = 0.0
    away_fatigue_adj = 0.0

    if referee_model is not None and referee_name:
        try:
            referee_elo_adj = referee_model.get_elo_adjustment(referee_name) or 0.0
        except Exception:
            referee_elo_adj = 0.0

    if fatigue_model is not None:
        try:
            fa = fatigue_model.get_match_adjustments(home, away, date_str)
            home_fatigue_adj = fa.get("home_elo_adj", 0.0)
            away_fatigue_adj = fa.get("away_elo_adj", 0.0)
        except Exception:
            pass

    home_elo_eff = elo_pred["home_elo"] + home_fatigue_adj + (referee_elo_adj / 2.0)
    away_elo_eff = elo_pred["away_elo"] + away_fatigue_adj - (referee_elo_adj / 2.0)

    # ── Squad value adjustment (international competitions only) ──────────────
    squad_adj = 0.0
    if not _xgb_eligible:  # international competition
        try:
            from core.squad_value_model import squad_elo_adjustment
            squad_adj = squad_elo_adjustment(home, away)
        except Exception:
            pass
    home_elo_eff += squad_adj

    # ── H2H adjustment (capped ±40 Elo, min 5 meetings to apply) ─────────────
    h2h_adj = 0.0
    try:
        import json as _json
        _h2h_path = Path("data/models/h2h_records.json")
        if _h2h_path.exists():
            with open(_h2h_path) as _f:
                _h2h = _json.load(_f)
            _key = "_vs_".join(sorted([home, away]))
            _rec = _h2h.get(_key, {})
            if _rec.get("total", 0) >= 5:
                teams = _key.split("_vs_")
                _home_wins = _rec.get(f"{teams[0].lower().replace(' ','_')}_wins", 0) if home == teams[0] else _rec.get(f"{teams[1].lower().replace(' ','_')}_wins", 0)
                _h2h_win_rate = _home_wins / _rec["total"]
                h2h_adj = max(-40.0, min(40.0, 60 * (_h2h_win_rate - 0.5)))
    except Exception:
        pass
    home_elo_eff += h2h_adj

    if referee_elo_adj or home_fatigue_adj or away_fatigue_adj or squad_adj or h2h_adj:
        elo_eff = _elo_probs_from_ratings(home_elo_eff, away_elo_eff, neutral=neutral)
    else:
        elo_eff = elo_pred

    # ── XGB ───────────────────────────────────────────────────────────────────
    # XGB is skipped for international competitions — see _xgb_eligible above.
    xgb_pred = None
    if xgb_model and odds and _xgb_eligible:
        try:
            xgb_pred = xgb_model.predict({
                "home_elo_k32":          elo_pred["home_elo"],
                "away_elo_k32":          elo_pred["away_elo"],
                "elo_delta_k32":         elo_pred["home_elo"] - elo_pred["away_elo"],
                "elo_expected_home_k32": 1 / (1 + 10 ** (
                    (elo_pred["away_elo"] - elo_pred["home_elo"]) / 400)),
                "mkt_home_implied":      odds.get("fair_home"),
                "mkt_draw_implied":      odds.get("fair_draw"),
                "mkt_away_implied":      odds.get("fair_away"),
            })
        except Exception:
            pass

    # ── Player model (Dixon-Coles goal-based 1X2) ─────────────────────────────
    player_pred = None
    if player_model is not None:
        try:
            player_pred = player_model.predict_outcome(
                home, away, home_absent=None, away_absent=None, neutral=True
            )
        except Exception:
            player_pred = None

    # ── Transformer sequence model ────────────────────────────────────────────
    # Trained on club-league sequences — skip for international competitions
    # (same reason as XGB: no Germany/Ecuador match history in training data).
    transformer_pred = None
    if sequence_model is not None and _xgb_eligible:
        try:
            fn = getattr(sequence_model, "predict_teams", None)
            if callable(fn):
                transformer_pred = fn(home, away, neutral=True)
        except Exception:
            transformer_pred = None

    probs = _ensemble_blend(elo_eff, xgb_pred, player_pred, transformer_pred)

    return {
        **probs,
        "home_elo": elo_pred["home_elo"],
        "away_elo": elo_pred["away_elo"],
        "squad_elo_adj": squad_adj,
        "h2h_elo_adj": h2h_adj,
        "xgb_used": xgb_pred is not None,
        "player_used": player_pred is not None,
        "transformer_used": transformer_pred is not None,
    }


# ── Message formatters ────────────────────────────────────────────────────────

def _format_date(date_str: str) -> str:
    try:
        d = datetime.date.fromisoformat(date_str)
        return d.strftime("%a %d %b")
    except ValueError:
        return date_str


def _format_picks(
    picks: list[dict],
    competition: str,
    date_str: str,
    n_fixtures: int,
    n_with_odds: int = 0,
) -> str:
    """Format the full /today response using HTML parse_mode."""
    comp_label = COMPETITION_LABELS.get(competition, competition.upper())
    date_label = _format_date(date_str)

    lines = [
        f"🔮 <b>Apollo Picks</b>",
        f"<i>{_md(comp_label)}  ·  {_md(date_label)}</i>",
    ]

    if not picks:
        odds_note = (
            f" ({n_with_odds} of {n_fixtures} fixtures had odds)"
            if n_fixtures > 0 else ""
        )
        lines.append(f"\nNo value bets found today{_md(odds_note)}.")
        return "\n".join(lines)

    lines.append("")
    n_confident = sum(1 for p in picks if p["conformal"])

    for p in picks:
        star       = "★" if p["conformal"] else "☆"
        edge_pct   = p["edge"] * 100
        model_pct  = p.get("model_prob", 0.0) * 100
        market_pct = p.get("fair_implied", 0.0) * 100

        # Match header
        lines.append(f"<b>⚽ {_md(p['home'])} vs {_md(p['away'])}</b>  {star}")
        lines.append(f"⏰ {_md(p['time_utc'])} UTC")
        lines.append("")

        # Probability comparison table — monospace via <pre>
        # Numbers only inside <pre> so no escaping concern
        header_row = "         Model  Market    Edge"
        data_row   = f"Bet     {model_pct:>6.1f}% {market_pct:>6.1f}% {edge_pct:>+7.1f}%"
        lines.append(f"<pre>{header_row}\n{data_row}</pre>")

        # Outcome, odds, stake
        lines.append(f"🎯 <b>{_md(p['team'])}</b>  ·  Odds: <code>{p['odds']:.2f}</code>")
        lines.append(f"💵 Stake: <code>{_md(p['stake_str'])}</code>")
        lines.append("")

    n_bets    = len(picks)
    n_matches = len({(p["home"], p["away"]) for p in picks})
    bets_word = "bet" if n_bets == 1 else "bets"
    conf_note = f"  ·  ★ {n_confident} high conf" if n_confident else ""
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(
        f"<b>{n_bets} {bets_word}</b> across {n_matches} of {n_fixtures} matches{conf_note}"
    )
    return "\n".join(lines)


def _format_status(profile: dict) -> str:
    """Format the /status reply using HTML parse_mode."""
    bankroll  = profile.get("bankroll", 100)
    risk_mode = profile.get("risk_mode", "moderate")
    min_edge  = profile.get("min_edge", 0.03)
    min_odds  = profile.get("min_odds", 1.30)
    max_odds  = profile.get("max_odds", 15.0)
    comps     = ", ".join(profile.get("competitions", ["wc2026"]))

    # Kelly multiplier display (e.g. "0.5×")
    multipliers = profile.get("kelly_multipliers",
                              {"conservative": 0.25, "moderate": 0.5, "aggressive": 1.0})
    kelly_mult  = multipliers.get(risk_mode, "?")

    # Try to read recent P&L from log.parquet
    pnl_str   = "N/A"
    bet_count = 0
    log_path  = Path("data/predictions/log.parquet")
    if log_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(log_path)
            today = datetime.date.today().isoformat()
            today_bets = df[
                (df["match_date"].astype(str) == today) &
                (df["bet_outcome"].notna())
            ]
            bet_count = len(today_bets)

            resolved = df[df["resolved"].fillna(False) & df["profit_loss"].notna()]
            if not resolved.empty:
                total_pl = resolved["profit_loss"].sum()
                pnl_str  = f"{total_pl:+.2f} units"
        except Exception as exc:
            logger.debug("Could not read log.parquet for /status: %s", exc)

    # Mute status
    muted_str = "No"
    if UserProfile.is_muted(profile):
        mute_until = profile.get("mute_until", 0)
        remaining  = int((float(mute_until) - time.time()) / 60)
        muted_str  = f"Yes ({remaining} min remaining)"

    lines = [
        "📊 <b>Apollo Status</b>",
        "",
        f"💰 Bankroll: <b><code>₹{bankroll:.2f}</code></b>",
        f"⚡ Risk: <b>{_md(risk_mode)}</b> ({_md(str(kelly_mult))}× Kelly)",
        f"🎯 Min edge: <b><code>{min_edge*100:.0f}%</code></b>",
        f"📈 Odds range: <code>{min_odds:.2f} – {max_odds:.2f}</code>",
        f"🏆 Competitions: {_md(comps)}",
        f"🔕 Muted: {_md(muted_str)}",
        "",
        f"📅 Today's bets: {bet_count}",
        f"📈 All-time P&amp;L: <code>{_md(pnl_str)}</code>",
    ]
    return "\n".join(lines)


# ── Command handlers ──────────────────────────────────────────────────────────

def handle_today(args: list[str], token: str, chat_id: str, profile: dict,
                 use_portfolio: bool = False) -> None:
    competition = args[0].lower() if args else (
        profile.get("competitions", ["wc2026"])[0]
    )
    if competition not in COMPETITION_LABELS:
        send(token, chat_id,
             f"Unknown competition '{competition}'. "
             f"Choose from: {', '.join(COMPETITION_LABELS)}")
        return

    send_typing(token, chat_id)
    send_md(token, chat_id,
            f"<i>Fetching {_md(COMPETITION_LABELS[competition])} fixtures...</i>")

    date_str = datetime.date.today().isoformat()

    # Load models
    try:
        models = _load_models()
        elo_model        = models.get("elo")
        conformal_filter = models.get("conformal")
        if elo_model is None:
            send(token, chat_id, "Elo model not found. Run: python scripts/build_elo.py")
            return
    except Exception as exc:
        send(token, chat_id, f"Model load error: {exc}")
        return

    # Fixtures
    try:
        from core.fixtures_fetcher import get_today_fixtures
        fixtures = get_today_fixtures(date=date_str, competition=competition)
    except Exception as exc:
        send(token, chat_id, f"Fixtures fetch failed: {exc}")
        return

    if not fixtures:
        send(token, chat_id,
             f"No fixtures for {COMPETITION_LABELS[competition]} on {date_str}.")
        return

    # Odds
    all_odds = []
    try:
        from core.odds_fetcher import OddsFetcher, _normalise_team
        all_odds = OddsFetcher().get_all_today(date_str, competition)
    except Exception as exc:
        logger.warning("Odds fetch failed: %s", exc)

    odds_map: dict[tuple, dict] = {}
    if all_odds:
        try:
            from core.odds_fetcher import _normalise_team
            for o in all_odds:
                k = (_normalise_team(o["home"]), _normalise_team(o["away"]))
                odds_map[k] = o
        except Exception:
            for o in all_odds:
                odds_map[(o["home"], o["away"])] = o

    from core.value_finder import find_value_bets as _find_value_bets

    min_edge = profile.get("min_edge", 0.03)
    bankroll = float(profile.get("bankroll", 100))

    # Build predictions
    predictions = []
    fixture_odds = []
    for fix in fixtures:
        home, away = fix["home"], fix["away"]
        try:
            from core.odds_fetcher import _normalise_team as _nt
            odds = odds_map.get((_nt(home), _nt(away))) or odds_map.get((home, away))
        except Exception:
            odds = odds_map.get((home, away))

        referee_name = fix.get("referee") or fix.get("Referee") or ""
        fix_neutral = fix.get("neutral", competition in {"wc2026", "euro", "copa_america", "nations_league"})
        try:
            pred = _predict_match(home, away, odds, models, date_str, referee_name,
                                  competition=competition, neutral=fix_neutral)
            predictions.append({**pred, "home": home, "away": away,
                                 "time_utc": fix.get("time_utc", "?"),
                                 "has_odds": odds is not None})
            if odds:
                fixture_odds.append(odds)
        except Exception as exc:
            logger.debug("Prediction failed %s vs %s: %s", home, away, exc)

    # Run WITHOUT conformal so we see all edge-positive bets.
    # Conformal is checked per-bet below to set the ★ (high confidence) flag.
    value_bets = _find_value_bets(
        predictions,
        fixture_odds,
        min_edge=min_edge,
        conformal=None,
    )

    pred_map = {(p["home"], p["away"]): p for p in predictions}

    # ── Portfolio optimiser (only for /picks): correlated-Kelly staking ────────
    # Builds a per-bet stake map keyed by (match, outcome). Silent-fail: if the
    # optimiser is unavailable the bot falls back to per-bet fractional Kelly.
    portfolio_stake_map: dict[tuple, dict] = {}
    portfolio_summary: str | None = None
    if use_portfolio and value_bets:
        try:
            from core.portfolio import PortfolioOptimizer
            risk_mode = profile.get("risk_mode", "moderate")
            kelly_mults = profile.get(
                "kelly_multipliers",
                {"conservative": 0.25, "moderate": 0.5, "aggressive": 1.0},
            )
            opt = PortfolioOptimizer(kelly_fraction=float(kelly_mults.get(risk_mode, 0.25)))
            peak = float(profile.get("peak_bankroll", bankroll) or bankroll)
            context = {
                vb.match: {"competition": competition, "date": date_str}
                for vb in value_bets
            }
            port = opt.optimize(value_bets, bankroll=bankroll, peak_bankroll=peak,
                                competition=competition, context=context)
            for r in port:
                portfolio_stake_map[(r["match"], r["outcome"])] = r
            portfolio_summary = opt.summary(port, bankroll)
        except Exception as exc:
            logger.warning("Portfolio optimiser failed; using per-bet Kelly: %s", exc)
            portfolio_stake_map = {}
            portfolio_summary = None

    picks = []
    for vb in value_bets:
        if not UserProfile.should_alert(vb, profile):
            continue

        parts = vb.match.split(" vs ", 1)
        home  = parts[0]
        away  = parts[1] if len(parts) == 2 else ""

        # Check conformal per-bet so ★ reflects actual model conviction.
        conformal_passed = False
        if conformal_filter is not None:
            entry = pred_map.get((home, away), {})
            probs = {
                "p_home": entry.get("p_home", 0.0),
                "p_draw": entry.get("p_draw", 0.0),
                "p_away": entry.get("p_away", 0.0),
            }
            conformal_passed = conformal_filter.is_confident(probs)

        if vb.outcome == "home":
            team_label = home
        elif vb.outcome == "away":
            team_label = away
        else:
            team_label = "Draw"

        time_utc = pred_map.get((home, away), {}).get("time_utc", "?")

        # Portfolio (correlated-Kelly) stake overrides per-bet Kelly for /picks.
        port_entry = portfolio_stake_map.get((vb.match, vb.outcome))
        if port_entry is not None:
            stake_str = f"₹{port_entry['stake_amount']:.2f} ({port_entry['stake_fraction']:.1%} corr-Kelly)"
        else:
            stake_str = UserProfile.format_stake(vb.kelly, profile)

        picks.append({
            "home":        home,
            "away":        away,
            "time_utc":    time_utc,
            "outcome":     vb.outcome,
            "team":        team_label,
            "odds":        vb.odds,
            "edge":        vb.edge,
            "model_prob":  vb.model_prob,
            "fair_implied": vb.fair_implied,
            "stake_str":   stake_str,
            "conformal":   conformal_passed,
        })

    n_with_odds = sum(1 for p in predictions if p.get("has_odds", False))
    msg = _format_picks(picks, competition, date_str, len(fixtures), n_with_odds)

    # ── Portfolio footer (only for /picks) ─────────────────────────────────────
    if use_portfolio and picks and portfolio_stake_map:
        total_stake = sum(r["stake_amount"] for r in portfolio_stake_map.values())
        total_frac  = (total_stake / bankroll) if bankroll else 0.0
        raw_total   = sum(r["kelly_raw"] for r in portfolio_stake_map.values())
        port_total  = sum(r["kelly_portfolio"] for r in portfolio_stake_map.values())
        div_score   = (port_total / raw_total) if raw_total > 0 else 1.0
        # Diversification badge: higher retention = better diversified.
        if div_score >= 0.85:
            badge = "🟢 Well diversified"
        elif div_score >= 0.6:
            badge = "🟡 Moderately correlated"
        else:
            badge = "🔴 Highly correlated — stakes throttled"
        msg += (
            "\n\n💼 <b>Portfolio (correlated-Kelly)</b>\n"
            f"Total exposure: <code>₹{total_stake:,.2f}</code> "
            f"({total_frac:.1%} of bankroll)\n"
            f"Diversification: <b>{div_score:.0%}</b> · {_md(badge)}"
        )

    # Inline keyboard: Refresh / Status buttons.
    inline_keyboard = {
        "inline_keyboard": [[
            {"text": "🔄 Refresh", "callback_data": "refresh_today"},
            {"text": "📊 Status",  "callback_data": "status"},
        ]]
    }

    # Telegram max 4096 chars; split if needed.
    # Only attach the inline keyboard to the final chunk.
    chunks = [msg[i:i + 4000] for i in range(0, len(msg), 4000)]
    for idx, chunk in enumerate(chunks):
        is_last = idx == len(chunks) - 1
        send_md(token, chat_id, chunk,
                reply_markup=inline_keyboard if is_last else None)


def handle_picks(args: list[str], token: str, chat_id: str, profile: dict) -> None:
    """Same as /today but uses the PortfolioOptimizer for correlated-Kelly stakes."""
    handle_today(args, token, chat_id, profile, use_portfolio=True)


def handle_referee(args: list[str], token: str, chat_id: str, profile: dict) -> None:
    """Look up a referee's home-bias stats from the loaded referee model.

    Usage: /referee Michael Oliver   (quotes optional)
    """
    name = " ".join(args).strip().strip('"').strip("'")
    if not name:
        send(token, chat_id,
             'Usage: /referee <name>\nExample: /referee Michael Oliver')
        return

    ref_path = Path("data/models/referee_model.json")
    if not ref_path.exists():
        send(token, chat_id,
             "Referee model not found. Run: python scripts/build_referee_model.py")
        return

    try:
        from core.referee_model import RefereeModel
        model = RefereeModel.load(str(ref_path))
    except Exception as exc:
        send(token, chat_id, f"Referee model load error: {exc}")
        return

    stats = model.stats
    # Case-insensitive lookup against the modelled referee names.
    entry = stats.get(name)
    matched_name = name
    if entry is None:
        lowered = {k.lower(): k for k in stats}
        key = lowered.get(name.lower())
        if key is not None:
            matched_name = key
            entry = stats[key]

    if entry is None:
        send_md(token, chat_id,
                f"No stats for referee <b>{_md(name)}</b> "
                f"(unknown referee or below the minimum-match threshold).")
        return

    league_avg = model.league_home_win_rate
    home_win   = entry.get("home_win_rate", league_avg + entry["delta"])
    elo_adj    = model.get_elo_adjustment(matched_name)
    delta      = entry["delta"]
    bias_word  = "home-favouring" if delta > 0 else ("away-favouring" if delta < 0 else "neutral")

    lines = [
        f"🧑‍⚖️ <b>Referee:</b> {_md(matched_name)}",
        "",
        f"📊 Matches: <code>{entry['n_matches']}</code>",
        f"🏠 Home win rate: <code>{home_win*100:.1f}%</code> "
        f"(baseline <code>{league_avg*100:.1f}%</code>)",
        f"📈 Shrunk delta: <code>{delta*100:+.2f}%</code> ({_md(bias_word)})",
        f"♟️ Elo equivalent: <code>{elo_adj:+.1f}</code>",
        f"🥅 Avg goals/match: <code>{entry.get('avg_goals', float('nan')):.2f}</code>",
        f"🤝 Draw rate: <code>{entry.get('draw_rate', 0.0)*100:.1f}%</code>",
    ]
    send_md(token, chat_id, "\n".join(lines))


def handle_model(token: str, chat_id: str) -> None:
    """Report which models are loaded and the active ensemble weights."""
    paths = {
        "Elo":         Path("data/models/elo_national.json"),
        "XGB":         Path("data/models/xgb_predictor.pkl"),
        "PlayerModel": Path("data/models/player_model.json"),
        "Transformer": Path("data/models/sequence_model.pt"),
        "Referee":     Path("data/models/referee_model.json"),
        "Conformal":   Path("data/models/conformal_filter.pkl"),
    }
    flags = {name: p.exists() for name, p in paths.items()}
    # Fatigue model is derived from the processed-match schedule, not a model file.
    flags["Fatigue"] = Path("data/processed/matches.parquet").exists()

    # Determine the active ensemble blend from which prob-model files are present.
    active = ["elo"]
    if flags["XGB"]:
        active.append("xgb")
    if flags["PlayerModel"]:
        active.append("player")
    if flags["Transformer"]:
        active.append("transformer")
    weights = _ensemble_weights(active)
    label_map = {"elo": "Elo", "xgb": "XGB", "player": "PlayerModel",
                 "transformer": "Transformer"}
    blend = " / ".join(f"{label_map[c]} {weights[c]:.0%}" for c in active)

    def tick(ok: bool) -> str:
        return "✅" if ok else "❌"

    order = ["Elo", "XGB", "PlayerModel", "Transformer", "Referee", "Fatigue", "Conformal"]
    lines = ["🧠 <b>Apollo Models</b>", ""]
    for name in order:
        lines.append(f"{tick(flags.get(name, False))} {_md(name)}")
    lines += [
        "",
        "⚖️ <b>Ensemble weights</b>",
        f"<code>{_md(blend)}</code>",
    ]
    send_md(token, chat_id, "\n".join(lines))


def handle_bankroll(args: list[str], token: str, chat_id: str, profile: dict) -> None:
    if not args:
        send(token, chat_id, "Usage: /bankroll <amount>\nExample: /bankroll 500")
        return
    try:
        amount = float(args[0].lstrip("₹$"))
        if amount <= 0:
            raise ValueError("Bankroll must be positive")
    except ValueError as exc:
        send(token, chat_id, f"Invalid amount: {exc}")
        return

    profile["bankroll"] = round(amount, 2)
    UserProfile.save(profile)

    # Show what a moderate-Kelly stake would look like at 5% kelly
    example_kelly = 0.05
    example_stake = UserProfile.get_stake_amount(example_kelly, amount, profile)
    send(token, chat_id,
         f"Bankroll set to ₹{amount:.2f}. Kelly stakes updated.\n"
         f"Example: 5% Kelly = ₹{example_stake:.2f}")


def handle_risk(args: list[str], token: str, chat_id: str, profile: dict) -> None:
    valid = ("conservative", "moderate", "aggressive")
    if not args or args[0].lower() not in valid:
        send(token, chat_id,
             f"Usage: /risk <mode>\nValid modes: {', '.join(valid)}")
        return

    old_mode = profile.get("risk_mode", "moderate")
    new_mode = args[0].lower()
    profile["risk_mode"] = new_mode
    UserProfile.save(profile)

    multipliers = profile.get("kelly_multipliers",
                              {"conservative": 0.25, "moderate": 0.5, "aggressive": 1.0})
    old_mult = multipliers.get(old_mode, "?")
    new_mult = multipliers.get(new_mode, "?")
    bankroll = float(profile.get("bankroll", 100))

    # Illustrate change with a 5% Kelly example
    example_kelly = 0.05
    old_stake = round(old_mult / 0.25 * example_kelly * bankroll, 2)
    new_stake = round(new_mult / 0.25 * example_kelly * bankroll, 2)

    send(token, chat_id,
         f"Risk mode: {old_mode} → {new_mode}\n"
         f"Kelly multiplier: {old_mult}x → {new_mult}x\n"
         f"5% Kelly example: ₹{old_stake:.2f} → ₹{new_stake:.2f}")


def handle_minedge(args: list[str], token: str, chat_id: str, profile: dict) -> None:
    if not args:
        send(token, chat_id,
             "Usage: /minedge <percent>\nExample: /minedge 5  (sets threshold to 5%)")
        return
    try:
        pct = float(args[0].rstrip("%"))
        if pct < 0 or pct > 50:
            raise ValueError("Edge must be between 0 and 50")
    except ValueError as exc:
        send(token, chat_id, f"Invalid value: {exc}")
        return

    old_edge = profile.get("min_edge", 0.03)
    profile["min_edge"] = round(pct / 100, 4)
    UserProfile.save(profile)
    send(token, chat_id,
         f"Min edge: {old_edge*100:.0f}% → {pct:.0f}%")


def handle_status(token: str, chat_id: str, profile: dict) -> None:
    send_md(token, chat_id, _format_status(profile))


def handle_mute(args: list[str], token: str, chat_id: str, profile: dict) -> None:
    if not args:
        send(token, chat_id,
             "Usage: /mute <minutes>\nExample: /mute 60  (mutes alerts for 1 hour)")
        return
    try:
        minutes = int(args[0])
        if minutes <= 0:
            raise ValueError("Must be > 0")
    except ValueError as exc:
        send(token, chat_id, f"Invalid value: {exc}")
        return

    mute_until = time.time() + minutes * 60
    profile["mute_until"] = mute_until
    UserProfile.save(profile)

    until_str = datetime.datetime.fromtimestamp(mute_until).strftime("%H:%M")
    send(token, chat_id,
         f"Alerts muted for {minutes} minute{'s' if minutes != 1 else ''} (until {until_str}).\n"
         f"Send /mute 0 or /status to check.")


def handle_competitions(args: list[str], token: str, chat_id: str, profile: dict) -> None:
    if not args:
        send(token, chat_id,
             "Usage: /competitions <list>\nExample: /competitions wc2026 epl\n"
             f"Available: {', '.join(COMPETITION_LABELS)}")
        return

    valid   = set(COMPETITION_LABELS)
    chosen  = [a.lower() for a in args]
    invalid = [c for c in chosen if c not in valid]
    if invalid:
        send(token, chat_id,
             f"Unknown competition(s): {', '.join(invalid)}\n"
             f"Available: {', '.join(valid)}")
        return

    old_comps = profile.get("competitions", ["wc2026"])
    profile["competitions"] = chosen
    UserProfile.save(profile)

    old_str = ", ".join(old_comps)
    new_str = ", ".join(chosen)
    send(token, chat_id, f"Competitions: {old_str} → {new_str}")


def handle_start(token: str, chat_id: str) -> None:
    """Send the welcome message with a quick-start inline keyboard."""
    lines = [
        "🔮 <b>Apollo Forecasting Engine</b>",
        "",
        "Professional probabilistic football value betting.",
        "",
        "<b>Quick commands:</b>",
        "/today — find today's value bets",
        "/picks — picks with portfolio staking",
        "/status — bankroll &amp; P&amp;L summary",
        "/help — full command list",
    ]
    keyboard = {"inline_keyboard": [[
        {"text": "⚽ Today's Picks", "callback_data": "refresh_today"},
        {"text": "📊 Status",        "callback_data": "status"},
    ]]}
    send_md(token, chat_id, "\n".join(lines), reply_markup=keyboard)


# ── Main dispatch ─────────────────────────────────────────────────────────────

def is_owner(sender_chat_id: int) -> bool:
    """Return True if the sender is the authorised owner of this bot."""
    return sender_chat_id == OWNER_CHAT_ID


def dispatch(text: str, token: str, chat_id: str, sender_id: int) -> None:
    """Parse a command text and call the appropriate handler.

    Parameters
    ----------
    text : str
        Raw message text from Telegram (may or may not start with /).
    token : str
        Bot API token.
    chat_id : str
        Destination chat ID for replies (owner's private chat string).
    sender_id : int
        Integer chat/user ID of the person who sent the message.
        Used for ownership checks — admin commands are rejected if this
        does not match OWNER_CHAT_ID.
    """
    text = text.strip()
    if not text.startswith("/"):
        return  # Ignore non-command messages

    parts = text.split()
    # Strip bot name suffix (e.g. /today@MyBot → /today)
    cmd   = parts[0].split("@")[0].lower()
    args  = parts[1:]

    # Authorization check for admin commands
    if cmd in ADMIN_COMMANDS and not is_owner(sender_id):
        send(token, str(sender_id), "Not authorized.")
        return

    profile = UserProfile.load()

    if cmd == "/start":
        handle_start(token, chat_id)

    elif cmd == "/help":
        send_md(token, chat_id, HELP_TEXT)

    elif cmd == "/today":
        handle_today(args, token, chat_id, profile)

    elif cmd == "/picks":
        handle_picks(args, token, chat_id, profile)

    elif cmd == "/referee":
        handle_referee(args, token, chat_id, profile)

    elif cmd == "/model":
        handle_model(token, chat_id)

    elif cmd == "/bankroll":
        handle_bankroll(args, token, chat_id, profile)

    elif cmd == "/risk":
        handle_risk(args, token, chat_id, profile)

    elif cmd == "/minedge":
        handle_minedge(args, token, chat_id, profile)

    elif cmd == "/status":
        handle_status(token, chat_id, profile)

    elif cmd == "/mute":
        handle_mute(args, token, chat_id, profile)

    elif cmd == "/competitions":
        handle_competitions(args, token, chat_id, profile)

    else:
        send(token, chat_id,
             f"Unknown command: {cmd}\nSend /help to see all commands.")


# ── Polling loop ──────────────────────────────────────────────────────────────

def main() -> None:
    global OWNER_CHAT_ID

    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)
    if not chat_id:
        logger.error("TELEGRAM_CHAT_ID not set in .env — run scripts/get_chat_id.py first")
        sys.exit(1)

    # Set the owner ID from the env variable (the owner's integer chat ID).
    try:
        OWNER_CHAT_ID = int(chat_id)
    except ValueError:
        logger.error("TELEGRAM_CHAT_ID must be an integer, got: %r", chat_id)
        sys.exit(1)

    logger.info("Apollo bot starting (long-polling, timeout=%ds)...", _POLL_TIMEOUT)
    logger.info("Owner chat ID: %d", OWNER_CHAT_ID)

    # Confirm bot is reachable
    me = _api(token, "getMe")
    if me.get("ok"):
        bot_name = me["result"].get("username", "?")
        logger.info("Connected as @%s", bot_name)
        send_md(
            token, chat_id,
            f"🔮 <b>Apollo started</b>  ·  @{_md(bot_name)}\nSend /help for commands.",
        )
    else:
        logger.error("getMe failed — check TELEGRAM_BOT_TOKEN")
        sys.exit(1)

    # Register commands with Telegram so they appear in the / menu.
    commands = [
        {"command": "start",        "description": "Welcome & quick start"},
        {"command": "today",        "description": "Today's value bets"},
        {"command": "picks",        "description": "Picks with portfolio staking"},
        {"command": "status",       "description": "Profile & P&L summary"},
        {"command": "bankroll",     "description": "Set bankroll amount"},
        {"command": "risk",         "description": "Set risk mode"},
        {"command": "minedge",      "description": "Set minimum edge %"},
        {"command": "mute",         "description": "Mute alerts temporarily"},
        {"command": "competitions", "description": "Set competitions"},
        {"command": "model",        "description": "Loaded models & weights"},
        {"command": "referee",      "description": "Referee bias stats"},
        {"command": "help",         "description": "Show all commands"},
    ]
    reg = _api(token, "setMyCommands", commands=commands)
    if reg.get("ok"):
        logger.info("Bot commands registered with Telegram.")
    else:
        logger.warning("setMyCommands failed: %s", reg.get("description", ""))

    offset = 0
    while True:
        updates = get_updates(token, offset)
        for update in updates:
            offset = update["update_id"] + 1

            # ── Callback query (inline keyboard button taps) ──────────────────
            cq = update.get("callback_query")
            if cq:
                cq_id      = cq.get("id", "")
                cq_data    = cq.get("data", "")
                cq_sender  = cq.get("from", {}).get("id", 0)
                cq_chat_id = str(
                    cq.get("message", {}).get("chat", {}).get("id")
                    or cq_sender
                    or chat_id
                )
                logger.info("Callback from %d: %s", cq_sender, cq_data)
                # Auth check — anyone who forwards a message with inline buttons
                # could trigger callbacks without this guard.
                if OWNER_CHAT_ID and cq_sender != OWNER_CHAT_ID:
                    answer_callback(token, cq_id, "Not authorised.")
                    continue
                try:
                    if cq_data == "refresh_today":
                        answer_callback(token, cq_id, "Refreshing picks…")
                        profile = UserProfile.load()
                        handle_today([], token, cq_chat_id, profile)
                    elif cq_data == "status":
                        answer_callback(token, cq_id, "Loading status…")
                        profile = UserProfile.load()
                        handle_status(token, cq_chat_id, profile)
                    else:
                        answer_callback(token, cq_id)
                except Exception as exc:
                    logger.error("Callback handler error: %s", exc, exc_info=True)
                    answer_callback(token, cq_id, "Error")
                continue

            # ── Regular text message ──────────────────────────────────────────
            msg = update.get("message", {})
            text = msg.get("text", "")
            if not text:
                continue

            # sender_id = personal user ID (always), used for owner auth
            # reply_to  = chat where message came from (group or private)
            sender_id = msg.get("from", {}).get("id", 0)
            reply_to  = str(msg.get("chat", {}).get("id") or sender_id or chat_id)

            logger.info("Received from %d: %s", sender_id, text)
            try:
                dispatch(text, token, reply_to, sender_id)
            except Exception as exc:
                logger.error("Handler error: %s", exc, exc_info=True)
                try:
                    send(token, reply_to, f"Error processing command: {exc}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
