"""
Logs every prediction + bet to data/predictions/log.parquet.
Records opening odds, model probs, bet action at prediction time.
CLV and result are filled in later by clv_tracker.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LOG_PATH = Path(__file__).parent.parent / "data" / "predictions" / "log.parquet"

SCHEMA = {
    "logged_at":          "object",   # ISO timestamp
    "match_date":         "object",   # YYYY-MM-DD
    "home":               "object",
    "away":               "object",
    "competition":        "object",
    # Model probabilities
    "model_p_home":       "float64",
    "model_p_draw":       "float64",
    "model_p_away":       "float64",
    # Opening odds (at prediction time)
    "open_home_odds":     "float64",
    "open_draw_odds":     "float64",
    "open_away_odds":     "float64",
    "open_source":        "object",
    "open_overround":     "float64",
    # Value bet selected (if any)
    "bet_outcome":        "object",   # "home" | "draw" | "away" | None
    "bet_odds":           "float64",
    "bet_edge":           "float64",
    "bet_kelly":          "float64",
    # Closing odds (filled by clv_tracker)
    "close_home_odds":    "float64",
    "close_draw_odds":    "float64",
    "close_away_odds":    "float64",
    "close_source":       "object",
    # Outcome (filled by clv_tracker)
    "actual_result":      "object",   # "H" | "D" | "A"
    "clv":                "float64",  # closing line value (our price - closing price)
    "clv_pct":            "float64",  # CLV as % of closing implied prob
    "profit_loss":        "float64",  # +odds-1 on win, -1 on loss, 0 if no bet
    "resolved":           "bool",
}


def _load() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_parquet(LOG_PATH)
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in SCHEMA.items()})


def _save(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOG_PATH, index=False)


def log_predictions(
    predictions: list[dict],
    bookmaker_odds: list[dict],
    value_bets: list,
    competition: str = "wc2026",
    match_date: str | None = None,
) -> int:
    """
    Log a batch of predictions + bets. One row per value bet; one row per
    no-bet match. Returns number of new rows written.
    """
    odds_by_match = {(o["home"], o["away"]): o for o in bookmaker_odds}
    # Multiple bets can exist per match — group them
    bets_by_match: dict[tuple, list] = {}
    for vb in (value_bets or []):
        key = (vb.match.split(" vs ")[0], vb.match.split(" vs ")[1])
        bets_by_match.setdefault(key, []).append(vb)

    now = datetime.utcnow().isoformat(timespec="seconds")
    today = match_date or now[:10]
    rows = []

    for p in predictions:
        home, away = p["home"], p["away"]
        odds = odds_by_match.get((home, away), {})
        match_bets = bets_by_match.get((home, away), [])

        base = {
            "logged_at":      now,
            "match_date":     today,
            "home":           home,
            "away":           away,
            "competition":    competition,
            "model_p_home":   p.get("p_home"),
            "model_p_draw":   p.get("p_draw"),
            "model_p_away":   p.get("p_away"),
            "open_home_odds": odds.get("home_odds"),
            "open_draw_odds": odds.get("draw_odds"),
            "open_away_odds": odds.get("away_odds"),
            "open_source":    odds.get("source"),
            "open_overround": odds.get("overround"),
            "close_home_odds": None,
            "close_draw_odds": None,
            "close_away_odds": None,
            "close_source":   None,
            "actual_result":  None,
            "clv":            None,
            "clv_pct":        None,
            "profit_loss":    None,
            "resolved":       False,
        }

        if match_bets:
            for vb in match_bets:
                rows.append({**base,
                             "bet_outcome": vb.outcome,
                             "bet_odds":    vb.odds,
                             "bet_edge":    vb.edge,
                             "bet_kelly":   vb.kelly})
        else:
            rows.append({**base,
                         "bet_outcome": None, "bet_odds": None,
                         "bet_edge": None, "bet_kelly": None})

    if not rows:
        return 0

    df = _load()
    existing_keys = set(zip(df["home"], df["away"], df["match_date"].astype(str),
                            df["bet_outcome"].astype(str)))
    new_rows = [r for r in rows
                if (r["home"], r["away"], str(r["match_date"]),
                    str(r["bet_outcome"])) not in existing_keys]

    if not new_rows:
        logger.info("All predictions already logged.")
        return 0

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    _save(df)
    logger.info("Logged %d rows to %s", len(new_rows), LOG_PATH)
    return len(new_rows)


def load_predictions(
    date_from: str | None = None,
    date_to: str | None = None,
    unresolved_only: bool = False,
) -> pd.DataFrame:
    df = _load()
    if df.empty:
        return df
    if date_from:
        df = df[df["match_date"] >= date_from]
    if date_to:
        df = df[df["match_date"] <= date_to]
    if unresolved_only:
        df = df[~df["resolved"].fillna(False)]
    return df.reset_index(drop=True)
