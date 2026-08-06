"""
Apollo Telegram Alert Bot
=========================
Fetches today's fixtures, runs predictions through the full Elo+XGB pipeline,
applies the conformal filter, and sends a Telegram message for every match
that has a positive edge vs the bookmaker.

Usage
-----
    # Send today's value bets
    python scripts/telegram_alerts.py

    # Specific date and competition
    python scripts/telegram_alerts.py --date 2026-06-26 --competition epl

    # Lower the minimum edge threshold (default 3%)
    python scripts/telegram_alerts.py --min-edge 2

    # Preview in terminal without sending to Telegram
    python scripts/telegram_alerts.py --dry-run

Environment variables required in .env
---------------------------------------
    TELEGRAM_BOT_TOKEN   — Bot API token from BotFather
    TELEGRAM_CHAT_ID     — Your personal chat ID (run get_chat_id.py once)
    ODDS_API_KEY         — The Odds API key (for bookmaker odds)
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MIN_EDGE = 0.03   # 3% edge over fair bookmaker probability
ELO_W  = 0.70
XGB_W  = 0.30

COMPETITION_LABELS = {
    "wc2026":     "FIFA World Cup 2026",
    "epl":        "Premier League",
    "laliga":     "La Liga",
    "seriea":     "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1":     "Ligue 1",
}

OUTCOME_LABEL = {"home": "Home win", "draw": "Draw", "away": "Away win"}
CONFIDENCE_STARS = {True: "High", False: "Medium"}


# ── Telegram sender ───────────────────────────────────────────────────────────

def send_telegram(text: str, token: str, chat_id: str) -> bool:
    """Send a message via the Telegram Bot API. Returns True on success."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


# ── Prediction pipeline ───────────────────────────────────────────────────────

def load_models():
    """Load Elo, XGB, and conformal filter. Returns (elo, xgb, conformal)."""
    from core.elo_model import EloModel
    from core.xgb_predictor import XGBPredictor
    from core.conformal import ConformalFilter

    elo, xgb, conf = None, None, None

    elo_path = Path("data/models/elo_national.json")
    if elo_path.exists():
        elo = EloModel.load(str(elo_path))
        logger.info("Elo model loaded.")
    else:
        logger.warning("Elo model not found at %s", elo_path)

    xgb_path = Path("data/models/xgb_predictor.pkl")
    if xgb_path.exists():
        xgb = XGBPredictor.load(str(xgb_path))
        logger.info("XGBoost model loaded.")
    else:
        logger.warning("XGBoost model not found at %s", xgb_path)

    conf_path = Path("data/models/conformal_filter.pkl")
    if conf_path.exists():
        conf = ConformalFilter.load(str(conf_path))
        logger.info("Conformal filter loaded.")
    else:
        logger.warning("Conformal filter not found at %s", conf_path)

    return elo, xgb, conf


def predict_match(home: str, away: str, odds: dict, elo_model, xgb_model) -> dict:
    """
    Run Elo + optional XGB prediction for a single match.
    Returns dict with p_home, p_draw, p_away, home_elo, away_elo, xgb_used.
    """
    elo_pred = elo_model.predict(home, away, neutral=True)

    xgb_pred = None
    if xgb_model and odds:
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
        except Exception as exc:
            logger.debug("XGB prediction failed for %s vs %s: %s", home, away, exc)

    w_elo, w_xgb = (ELO_W, XGB_W) if xgb_pred else (1.0, 0.0)
    if xgb_pred:
        probs = {
            "p_home": w_elo * elo_pred["p_home"] + w_xgb * xgb_pred["p_home"],
            "p_draw": w_elo * elo_pred["p_draw"] + w_xgb * xgb_pred["p_draw"],
            "p_away": w_elo * elo_pred["p_away"] + w_xgb * xgb_pred["p_away"],
        }
    else:
        probs = {k: elo_pred[k] for k in ("p_home", "p_draw", "p_away")}

    return {
        **probs,
        "home_elo":  elo_pred["home_elo"],
        "away_elo":  elo_pred["away_elo"],
        "xgb_used":  xgb_pred is not None,
    }


def find_value_bets(pred: dict, odds: dict, min_edge: float) -> list[dict]:
    """
    Compare model probabilities to bookmaker fair implied probabilities.
    Returns a list of value bet dicts (may be empty).
    """
    if not odds:
        return []

    bets = []
    for outcome, model_prob, fair_prob, book_odds in [
        ("home", pred["p_home"], odds.get("fair_home"), odds.get("home_odds")),
        ("draw", pred["p_draw"], odds.get("fair_draw"), odds.get("draw_odds")),
        ("away", pred["p_away"], odds.get("fair_away"), odds.get("away_odds")),
    ]:
        if not fair_prob or not book_odds:
            continue
        edge = model_prob - fair_prob
        if edge >= min_edge:
            bets.append({
                "outcome":    outcome,
                "model_prob": model_prob,
                "fair_prob":  fair_prob,
                "book_odds":  book_odds,
                "edge":       edge,
                "ev":         model_prob * book_odds - 1.0,
            })

    # Sort by edge descending
    bets.sort(key=lambda b: b["edge"], reverse=True)
    return bets


# ── Message formatting ────────────────────────────────────────────────────────

def format_header(date: str, competition: str) -> str:
    comp_label = COMPETITION_LABELS.get(competition, competition.upper())
    try:
        d = datetime.date.fromisoformat(date)
        friendly = d.strftime("%a %d %b")
    except ValueError:
        friendly = date
    return f"Apollo Picks | {comp_label} | {friendly}\n"


def format_bet_card(
    home: str,
    away: str,
    time_utc: str,
    bets: list[dict],
    conformal_passed: bool,
) -> str:
    """One card per match. Only called when there is at least one bet."""
    star = "★" if conformal_passed else "☆"
    lines = [f"{star} {home} vs {away}  {time_utc} UTC"]
    for b in bets:
        if b["outcome"] == "home":
            pick = f"{home} to win"
        elif b["outcome"] == "away":
            pick = f"{away} to win"
        else:
            pick = "Draw"
        lines.append(f"   {pick} @ {b['book_odds']:.2f}  (+{b['edge']*100:.0f}% edge)")
    return "\n".join(lines)


def format_no_fixtures(date: str, competition: str) -> str:
    comp_label = COMPETITION_LABELS.get(competition, competition.upper())
    return f"Apollo | {comp_label}\nNo matches today."


def format_footer(n_bets: int, n_matches_with_bets: int, n_total: int) -> str:
    if n_bets == 0:
        return "Nothing to bet today."
    return (
        f"\n{n_bets} bet{'s' if n_bets > 1 else ''} across "
        f"{n_matches_with_bets} of {n_total} matches\n"
        f"★ = high confidence  ☆ = medium confidence"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Apollo Telegram value bet alerts")
    parser.add_argument("--date", default=None,
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--competition", default="wc2026",
                        choices=list(COMPETITION_LABELS.keys()),
                        help="Competition to analyse (default: wc2026)")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE * 100,
                        help="Minimum edge %% over fair odds to flag a bet (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print messages to terminal instead of sending to Telegram")
    args = parser.parse_args()

    date_str    = args.date or datetime.date.today().isoformat()
    competition = args.competition
    min_edge    = args.min_edge / 100.0

    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not args.dry_run:
        if not token:
            logger.error("TELEGRAM_BOT_TOKEN not set in .env")
            sys.exit(1)
        if not chat_id:
            logger.error("TELEGRAM_CHAT_ID not set in .env — run scripts/get_chat_id.py first")
            sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    elo_model, xgb_model, conformal_filter = load_models()
    if elo_model is None:
        logger.error("Elo model is required. Run: python scripts/build_elo.py")
        sys.exit(1)

    # ── Fetch fixtures ────────────────────────────────────────────────────────
    logger.info("Fetching fixtures for %s [%s]...", date_str, competition)
    from core.fixtures_fetcher import get_today_fixtures
    fixtures = get_today_fixtures(date=date_str, competition=competition)
    logger.info("Found %d fixture(s).", len(fixtures))

    if not fixtures:
        msg = format_no_fixtures(date_str, competition)
        if args.dry_run:
            print(msg)
        else:
            send_telegram(msg, token, chat_id)
        return

    # ── Fetch bookmaker odds ──────────────────────────────────────────────────
    logger.info("Fetching odds...")
    try:
        from core.odds_fetcher import OddsFetcher
        all_odds = OddsFetcher().get_all_today(date_str, competition)
    except Exception as exc:
        logger.warning("Odds fetch failed: %s — continuing without odds", exc)
        all_odds = []

    odds_map = {}
    for o in all_odds:
        from core.odds_fetcher import _normalise_team
        key = (_normalise_team(o["home"]), _normalise_team(o["away"]))
        odds_map[key] = o

    # ── Run pipeline ──────────────────────────────────────────────────────────
    messages = []
    n_value     = 0
    n_confident = 0

    for fix in fixtures:
        home, away = fix["home"], fix["away"]
        time_utc   = fix.get("time_utc", "?")

        # Look up odds
        from core.odds_fetcher import _normalise_team
        odds = odds_map.get((_normalise_team(home), _normalise_team(away)))

        # Predict
        try:
            pred = predict_match(home, away, odds, elo_model, xgb_model)
        except Exception as exc:
            logger.warning("Prediction failed for %s vs %s: %s", home, away, exc)
            continue

        # Conformal filter
        conformal_passed = False
        if conformal_filter:
            try:
                conformal_passed = conformal_filter.is_confident({
                    "p_home": pred["p_home"],
                    "p_draw": pred["p_draw"],
                    "p_away": pred["p_away"],
                })
                if conformal_passed:
                    n_confident += 1
            except Exception as exc:
                logger.debug("Conformal filter error: %s", exc)

        # Value bets
        bets = find_value_bets(pred, odds, min_edge)
        if bets:
            n_value += len(bets)

        # Only include matches that have at least one value bet
        if bets:
            messages.append(
                format_bet_card(home, away, time_utc, bets, conformal_passed)
            )

    # ── Compose and send ──────────────────────────────────────────────────────
    header = format_header(date_str, competition)
    footer = format_footer(n_value, len(messages), len(fixtures))

    if not messages:
        body = "Nothing to bet today."
    else:
        body = "\n\n".join(messages)

    full_message = header + body + footer

    if args.dry_run:
        sys.stdout.buffer.write((full_message + f"\n\n[dry-run] {len(full_message)} chars\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    else:
        logger.info("Sending Telegram alert (%d chars)...", len(full_message))
        # Telegram max message length is 4096 chars — split if needed
        for i in range(0, len(full_message), 4000):
            chunk = full_message[i:i + 4000]
            ok = send_telegram(chunk, token, chat_id)
            if ok:
                logger.info("Message sent (chunk %d).", i // 4000 + 1)
            else:
                logger.error("Failed to send chunk %d.", i // 4000 + 1)


if __name__ == "__main__":
    main()
