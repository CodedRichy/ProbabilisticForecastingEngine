"""
build_h2h_features.py
=====================
Compute head-to-head (H2H) records for every international team matchup
using the martj42/international_results dataset and save to
data/models/h2h_records.json.

Output format (alphabetically ordered team keys):
  {
    "Argentina_vs_Brazil": {
      "total": 107,
      "argentina_wins": 38,
      "draws": 26,
      "brazil_wins": 43,
      "argentina_win_rate": 0.355,
      "avg_goals_argentina": 1.8,
      "avg_goals_brazil": 2.0,
      "last_5_results": ["W", "L", "D", "W", "L"]
    },
    ...
  }

The key is always "TeamA_vs_TeamB" where TeamA < TeamB (alphabetical).
Results ("W"/"D"/"L") are from TeamA's perspective.

Run standalone:
  python scripts/build_h2h_features.py
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
OUTPUT_PATH = ROOT / "data" / "models" / "h2h_records.json"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
# Local cache to avoid re-downloading when running alongside build_elo.py
LOCAL_CACHE = ROOT / "downloaded_files" / "international_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results() -> pd.DataFrame:
    """Load the martj42 results CSV from local cache or GitHub."""
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
# H2H computation
# ---------------------------------------------------------------------------

def _h2h_key(team_a: str, team_b: str) -> tuple[str, str]:
    """Return (team_a, team_b) alphabetically sorted."""
    if team_a <= team_b:
        return team_a, team_b
    return team_b, team_a


def build_h2h_records(df: pd.DataFrame) -> dict[str, dict]:
    """
    Iterate over all match rows and accumulate H2H statistics.

    The martj42 CSV columns:
      date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
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

    # Accumulator: key → list of (team_a_goals, team_b_goals, date)
    # team_a is alphabetically first
    from collections import defaultdict

    records: dict[str, list[tuple[float, float, str]]] = defaultdict(list)

    for _, row in df.iterrows():
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        home_g = float(row["home_score"])
        away_g = float(row["away_score"])
        date_str = str(row["date"].date())

        first, second = _h2h_key(home, away)
        if first == home:
            records[f"{first}_vs_{second}"].append((home_g, away_g, date_str))
        else:
            # home is actually team_b (alphabetically second)
            records[f"{first}_vs_{second}"].append((away_g, home_g, date_str))

    logger.info("Found %d unique H2H matchups", len(records))

    h2h_output: dict[str, dict] = {}

    for key, matches in records.items():
        total = len(matches)
        team_a_wins = sum(1 for ga, gb, _ in matches if ga > gb)
        draws = sum(1 for ga, gb, _ in matches if ga == gb)
        team_b_wins = total - team_a_wins - draws

        avg_goals_a = round(sum(ga for ga, _, _ in matches) / total, 2)
        avg_goals_b = round(sum(gb for _, gb, _ in matches) / total, 2)

        win_rate_a = round(team_a_wins / total, 4) if total > 0 else 0.0

        # Last 5 results from team_a's perspective
        last_5_raw = matches[-5:]
        last_5_results: list[str] = []
        for ga, gb, _ in last_5_raw:
            if ga > gb:
                last_5_results.append("W")
            elif ga == gb:
                last_5_results.append("D")
            else:
                last_5_results.append("L")
        # Pad with empty strings to always have length 5
        while len(last_5_results) < 5:
            last_5_results.append("")

        parts = key.split("_vs_")
        team_a_name = parts[0]
        team_b_name = parts[1] if len(parts) > 1 else "Unknown"

        # Derive field names from team_a_name (lowercase, replace spaces/special chars)
        a_slug = team_a_name.lower().replace(" ", "_").replace("-", "_")
        b_slug = team_b_name.lower().replace(" ", "_").replace("-", "_")

        h2h_output[key] = {
            "total": total,
            f"{a_slug}_wins": team_a_wins,
            "draws": draws,
            f"{b_slug}_wins": team_b_wins,
            f"{a_slug}_win_rate": win_rate_a,
            f"avg_goals_{a_slug}": avg_goals_a,
            f"avg_goals_{b_slug}": avg_goals_b,
            "last_5_results": last_5_results,
            "last_match_date": matches[-1][2] if matches else None,
        }

    return h2h_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = _load_results()
    h2h = build_h2h_records(df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(h2h, fh, indent=2, ensure_ascii=False)

    logger.info("Saved %d H2H records to %s", len(h2h), OUTPUT_PATH)

    # Preview a well-known matchup
    preview_keys = [k for k in h2h if "Argentina" in k and "Brazil" in k]
    if preview_keys:
        pk = preview_keys[0]
        rec = h2h[pk]
        logger.info("Preview [%s]: total=%d, last_5=%s", pk, rec["total"], rec["last_5_results"])


if __name__ == "__main__":
    main()
