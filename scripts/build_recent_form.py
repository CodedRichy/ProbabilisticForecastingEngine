"""
build_recent_form.py
====================
Compute per-team recent form features from the martj42/international_results
dataset and save to data/models/recent_form.json.

For each team, considers the last 10 matches in the dataset (all competitions).

Output format:
  {
    "Germany": {
      "last_10_results":       ["W","W","D","W","W","L","W","W","W","D"],
      "last_10_win_rate":      0.7,
      "last_10_goals_for":     2.3,
      "last_10_goals_against": 0.9,
      "last_10_xg_proxy":      null,
      "form_points":           7.0,
      "last_match_date":       "2026-06-10"
    },
    ...
  }

form_points = (wins * 3 + draws * 1) / 10  (out of 3.0 max)

Run standalone:
  python scripts/build_recent_form.py
"""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "models" / "recent_form.json"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
LOCAL_CACHE = ROOT / "downloaded_files" / "international_results.csv"

FORM_WINDOW = 10  # number of recent matches to consider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (reuses cache from build_h2h_features if available)
# ---------------------------------------------------------------------------

def _load_results() -> pd.DataFrame:
    if LOCAL_CACHE.exists():
        logger.info("Loading results from local cache: %s", LOCAL_CACHE)
        df = pd.read_csv(LOCAL_CACHE)
    else:
        logger.info("Downloading results from GitHub: %s", RESULTS_URL)
        resp = requests.get(RESULTS_URL, timeout=60)
        resp.raise_for_status()
        LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_CACHE.write_text(resp.text, encoding="utf-8")
        df = pd.read_csv(StringIO(resp.text))

    logger.info("Loaded %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Form computation
# ---------------------------------------------------------------------------

def _result_label(goals_for: float, goals_against: float) -> str:
    if goals_for > goals_against:
        return "W"
    if goals_for == goals_against:
        return "D"
    return "L"


def build_recent_form(df: pd.DataFrame, window: int = FORM_WINDOW) -> dict[str, dict]:
    """
    Build per-team form dict using the last `window` matches.

    The martj42 CSV columns:
      date, home_team, away_team, home_score, away_score, ...
    """
    required = {"home_team", "away_team", "home_score", "away_score", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Expand into one row per team-match
    home_df = df[["date", "home_team", "home_score", "away_score"]].copy()
    home_df.columns = ["date", "team", "goals_for", "goals_against"]

    away_df = df[["date", "away_team", "away_score", "home_score"]].copy()
    away_df.columns = ["date", "team", "goals_for", "goals_against"]

    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df["team"] = long_df["team"].str.strip()
    long_df = long_df.sort_values("date").reset_index(drop=True)

    teams = sorted(long_df["team"].unique())
    logger.info("Computing form for %d teams (window=%d)", len(teams), window)

    form_output: dict[str, dict] = {}

    for team in teams:
        team_df = long_df[long_df["team"] == team].sort_values("date")
        last_n = team_df.tail(window)

        if last_n.empty:
            continue

        results: list[str] = []
        wins = draws = losses = 0
        total_gf = 0.0
        total_ga = 0.0

        for _, row in last_n.iterrows():
            gf = float(row["goals_for"])
            ga = float(row["goals_against"])
            label = _result_label(gf, ga)
            results.append(label)
            total_gf += gf
            total_ga += ga
            if label == "W":
                wins += 1
            elif label == "D":
                draws += 1
            else:
                losses += 1

        n = len(last_n)
        win_rate = round(wins / n, 4)
        avg_gf = round(total_gf / n, 2)
        avg_ga = round(total_ga / n, 2)
        form_pts = round((wins * 3 + draws * 1) / window, 4)
        last_date = str(last_n["date"].iloc[-1].date())

        # Pad results list to length `window` with empty strings if fewer matches exist
        padded = ([""] * (window - n)) + results

        form_output[team] = {
            "last_10_results":       padded,
            "last_10_win_rate":      win_rate,
            "last_10_goals_for":     avg_gf,
            "last_10_goals_against": avg_ga,
            "last_10_xg_proxy":      None,   # martj42 dataset has no xG; placeholder
            "form_points":           form_pts,
            "last_match_date":       last_date,
        }

    return form_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = _load_results()
    form = build_recent_form(df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(form, fh, indent=2, ensure_ascii=False)

    logger.info("Saved recent form for %d teams to %s", len(form), OUTPUT_PATH)

    # Preview key WC2026 nations
    preview_teams = ["Germany", "Spain", "Argentina", "Brazil", "France", "England"]
    for team in preview_teams:
        if team in form:
            rec = form[team]
            logger.info(
                "  %s: form_pts=%.2f  last_10=%s  last_match=%s",
                team,
                rec["form_points"],
                "".join(rec["last_10_results"]),
                rec["last_match_date"],
            )


if __name__ == "__main__":
    main()
