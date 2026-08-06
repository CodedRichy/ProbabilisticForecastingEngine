"""
Fetches closing odds + actual results for logged predictions.
Computes Closing Line Value (CLV) — the definitive measure of model edge.

CLV > 0 on a bet means: our opening price was better than the market's
final price (closing). Sustained positive CLV = genuine alpha.

Usage:
    python scripts/track_clv.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
_BASE = "https://api.the-odds-api.com/v4"
_TIMEOUT = 12

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
_COMP_ESPN = {
    "wc2026": "fifa.world",
    "epl":    "eng.1",
    "laliga": "esp.1",
}


# ── Closing odds ─────────────────────────────────────────────────────────────

def fetch_closing_odds(home: str, away: str, match_date: str,
                       competition: str = "wc2026") -> dict | None:
    """
    Fetch closing odds from The Odds API historical endpoint.
    Returns dict with home_odds, draw_odds, away_odds, source or None.
    """
    if not _ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set — cannot fetch closing odds.")
        return None

    sport_key_map = {
        "wc2026": "soccer_fifa_world_cup",
        "epl":    "soccer_epl",
        "laliga": "soccer_spain_la_liga",
        "seriea": "soccer_italy_serie_a",
        "bundesliga": "soccer_germany_bundesliga",
        "ligue1": "soccer_france_ligue_1",
    }
    sport = sport_key_map.get(competition, "soccer_fifa_world_cup")

    # Odds API: historical odds snapshot closest to kickoff (the closing line)
    # Use match_date + 23:59 so we get the final pre-match price
    snapshot_time = f"{match_date}T23:59:00Z"
    url = (f"{_BASE}/sports/{sport}/odds-history/"
           f"?apiKey={_ODDS_API_KEY}&regions=eu&markets=h2h"
           f"&oddsFormat=decimal&date={snapshot_time}")
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.debug("Closing odds API %d: %s", r.status_code, r.text[:100])
            return None
        events = r.json()
        match = _find_event(events, home, away)
        if not match:
            return None
        return _extract_h2h(match, source="odds_api_closing")
    except Exception as e:
        logger.debug("fetch_closing_odds failed: %s", e)
        return None


def _find_event(events: list, home: str, away: str) -> dict | None:
    hn, an = _norm(home), _norm(away)
    for ev in events:
        if isinstance(ev, dict):
            eh = _norm(ev.get("home_team", ""))
            ea = _norm(ev.get("away_team", ""))
            if (hn in eh or eh in hn) and (an in ea or ea in an):
                return ev
    return None


def _extract_h2h(event: dict, source: str) -> dict | None:
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            home_o = outcomes.get(event.get("home_team"))
            away_o = outcomes.get(event.get("away_team"))
            draw_o = outcomes.get("Draw")
            if home_o and draw_o and away_o:
                return {
                    "home_odds": float(home_o),
                    "draw_odds": float(draw_o),
                    "away_odds": float(away_o),
                    "source":    source,
                }
    return None


def _norm(name: str) -> str:
    return name.lower().strip().replace("fc", "").replace("  ", " ").strip()


# ── Actual result ─────────────────────────────────────────────────────────────

def fetch_result(home: str, away: str, match_date: str,
                 competition: str = "wc2026") -> str | None:
    """
    Fetch actual result (H/D/A) from ESPN scoreboard.
    Returns None if match not yet played or not found.
    """
    espn_comp = _COMP_ESPN.get(competition, "fifa.world")
    url = (f"{_ESPN_BASE}/{espn_comp}/scoreboard"
           f"?dates={match_date.replace('-', '')}")
    try:
        r = requests.get(url, timeout=_TIMEOUT,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        events = r.json().get("events", [])
        hn, an = _norm(home), _norm(away)
        for ev in events:
            competitors = ev.get("competitions", [{}])[0].get("competitors", [])
            names = [_norm(c.get("team", {}).get("displayName", "")) for c in competitors]
            if not (any(hn in n or n in hn for n in names) and
                    any(an in n or n in an for n in names)):
                continue
            status = ev.get("competitions", [{}])[0].get("status", {})
            if status.get("type", {}).get("completed") is not True:
                return None  # not finished yet
            scores = {c.get("homeAway"): int(c.get("score", 0)) for c in competitors}
            hg = scores.get("home", 0)
            ag = scores.get("away", 0)
            return "H" if hg > ag else ("D" if hg == ag else "A")
    except Exception as e:
        logger.debug("fetch_result failed: %s", e)
    return None


# ── CLV calculation ───────────────────────────────────────────────────────────

def compute_clv(bet_odds: float, closing_odds: float) -> tuple[float, float]:
    """
    CLV = our implied prob at bet time vs closing implied prob.
    Returns (clv_absolute, clv_pct).
    Positive = we got a better price than closing → real edge.
    """
    our_implied   = 1.0 / bet_odds
    close_implied = 1.0 / closing_odds
    clv_abs = close_implied - our_implied       # positive = we beat closing
    clv_pct = clv_abs / close_implied * 100     # as % of closing prob
    return round(clv_abs, 5), round(clv_pct, 3)


# ── Profit/loss ───────────────────────────────────────────────────────────────

def compute_pnl(bet_outcome: str, bet_odds: float, actual_result: str) -> float:
    """Returns P&L in units (1 unit staked). +N on win, -1 on loss, 0 if no bet."""
    if not bet_outcome or not actual_result:
        return 0.0
    outcome_to_result = {"home": "H", "draw": "D", "away": "A"}
    won = outcome_to_result.get(bet_outcome) == actual_result
    return round((bet_odds - 1.0) if won else -1.0, 4)
