"""
scrape_fifa_rankings.py
=======================
Fetch FIFA Men's World Rankings and save to data/models/fifa_rankings.json.

Strategy (in order):
  1. Try the FIFA API endpoint (JSON).
  2. Try the Kassiesa historical ranking CSV (has FIFA points, updated ~monthly).
  3. Fall back to hardcoded approximate June 2026 values for WC2026 nations.

Output format:
  {
    "Germany": {"rank": 12, "points": 1634.45},
    "Spain":   {"rank": 1,  "points": 1837.0},
    ...
  }

Run standalone:
  python scripts/scrape_fifa_rankings.py
"""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "models" / "fifa_rankings.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback — approximate FIFA ranking points / rank as of June 2026
# (WC2026 participating nations most relevant to Apollo)
# ---------------------------------------------------------------------------
FALLBACK_RANKINGS: dict[str, dict] = {
    "Spain":           {"rank": 1,  "points": 1837.0},
    "Argentina":       {"rank": 2,  "points": 1779.0},
    "France":          {"rank": 3,  "points": 1762.0},
    "England":         {"rank": 4,  "points": 1730.0},
    "Brazil":          {"rank": 5,  "points": 1720.0},
    "Portugal":        {"rank": 6,  "points": 1700.0},
    "Netherlands":     {"rank": 7,  "points": 1680.0},
    "Belgium":         {"rank": 8,  "points": 1672.0},
    "Italy":           {"rank": 9,  "points": 1650.0},
    "Croatia":         {"rank": 10, "points": 1638.0},
    "Switzerland":     {"rank": 11, "points": 1620.0},
    "Germany":         {"rank": 12, "points": 1634.0},
    "Morocco":         {"rank": 13, "points": 1610.0},
    "Colombia":        {"rank": 14, "points": 1611.0},
    "Denmark":         {"rank": 15, "points": 1605.0},
    "United States":   {"rank": 16, "points": 1540.0},
    "Uruguay":         {"rank": 17, "points": 1598.0},
    "Japan":           {"rank": 18, "points": 1555.0},
    "Senegal":         {"rank": 19, "points": 1545.0},
    "Austria":         {"rank": 20, "points": 1542.0},
    "Mexico":          {"rank": 22, "points": 1530.0},
    "Canada":          {"rank": 26, "points": 1494.0},
    "Ecuador":         {"rank": 44, "points": 1421.0},
    "Turkey":          {"rank": 46, "points": 1415.0},
    "Australia":       {"rank": 23, "points": 1510.0},
    "South Korea":     {"rank": 24, "points": 1505.0},
    "Iran":            {"rank": 25, "points": 1500.0},
    "Saudi Arabia":    {"rank": 56, "points": 1385.0},
    "Ghana":           {"rank": 55, "points": 1390.0},
    "Nigeria":         {"rank": 40, "points": 1440.0},
    "Cameroon":        {"rank": 45, "points": 1418.0},
    "Costa Rica":      {"rank": 48, "points": 1410.0},
    "Panama":          {"rank": 60, "points": 1370.0},
    "Honduras":        {"rank": 70, "points": 1340.0},
    "Jamaica":         {"rank": 80, "points": 1310.0},
    "New Zealand":     {"rank": 96, "points": 1260.0},
    "Chile":           {"rank": 31, "points": 1470.0},
    "Peru":            {"rank": 33, "points": 1460.0},
    "Venezuela":       {"rank": 35, "points": 1455.0},
    "Paraguay":        {"rank": 38, "points": 1445.0},
    "Bolivia":         {"rank": 85, "points": 1295.0},
    "Serbia":          {"rank": 27, "points": 1490.0},
    "Czech Republic":  {"rank": 37, "points": 1448.0},
    "Hungary":         {"rank": 28, "points": 1488.0},
    "Scotland":        {"rank": 30, "points": 1475.0},
    "Ukraine":         {"rank": 21, "points": 1535.0},
    "Poland":          {"rank": 29, "points": 1480.0},
    "Romania":         {"rank": 32, "points": 1465.0},
    "Egypt":           {"rank": 34, "points": 1458.0},
    "Algeria":         {"rank": 36, "points": 1452.0},
    "Tunisia":         {"rank": 39, "points": 1442.0},
    "Mali":            {"rank": 41, "points": 1438.0},
    "Ivory Coast":     {"rank": 42, "points": 1432.0},
    "Burkina Faso":    {"rank": 43, "points": 1428.0},
    "Congo DR":        {"rank": 47, "points": 1412.0},
    "South Africa":    {"rank": 53, "points": 1395.0},
    "Tanzania":        {"rank": 120, "points": 1210.0},
    "Uganda":          {"rank": 95, "points": 1265.0},
    "Guatemala":       {"rank": 100, "points": 1250.0},
    "Cuba":            {"rank": 130, "points": 1190.0},
    "Iraq":            {"rank": 58, "points": 1378.0},
    "Qatar":           {"rank": 62, "points": 1365.0},
    "Uzbekistan":      {"rank": 65, "points": 1355.0},
    "Indonesia":       {"rank": 127, "points": 1198.0},
    "China PR":        {"rank": 88, "points": 1282.0},
    "Thailand":        {"rank": 113, "points": 1225.0},
}

# ---------------------------------------------------------------------------
# Strategy 1: FIFA official API
# ---------------------------------------------------------------------------
# FIFA exposes a JSON endpoint with the rankings. The dateId changes each
# publication cycle — we try a range of plausible IDs near mid-2026.
FIFA_API_BASE = "https://www.fifa.com/api/ranking-overview"
FIFA_DATE_IDS = [
    "id13792", "id13800", "id13780", "id13770", "id13760",
    "id13750", "id13740", "id13730", "id13720", "id13710",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.fifa.com/fifa-world-ranking/",
}


def _try_fifa_api() -> dict[str, dict] | None:
    """Attempt to fetch the FIFA ranking JSON for any plausible dateId."""
    session = requests.Session()
    session.headers.update(HEADERS)

    for date_id in FIFA_DATE_IDS:
        url = f"{FIFA_API_BASE}?gender=M&dateId={date_id}"
        try:
            logger.info("Trying FIFA API: %s", url)
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.debug("HTTP %s for %s", resp.status_code, date_id)
                continue

            data = resp.json()
            # The API wraps teams in {"rankings": [...]} or similar.
            rankings_list = (
                data.get("rankings")
                or data.get("items")
                or data.get("data", {}).get("rankings", [])
            )
            if not rankings_list:
                logger.debug("No rankings list found in response for %s", date_id)
                continue

            result: dict[str, dict] = {}
            for entry in rankings_list:
                team = (
                    entry.get("team", {}).get("name")
                    or entry.get("teamName")
                    or entry.get("name")
                )
                rank = (
                    entry.get("rank")
                    or entry.get("ranking")
                    or entry.get("currentRanking")
                )
                points = (
                    entry.get("totalPoints")
                    or entry.get("points")
                    or entry.get("rankingPoints")
                )
                if team and rank is not None:
                    result[team] = {
                        "rank": int(rank),
                        "points": round(float(points), 2) if points is not None else None,
                    }

            if result:
                logger.info("FIFA API returned %d teams (dateId=%s)", len(result), date_id)
                return result

        except Exception as exc:
            logger.debug("FIFA API error for %s: %s", date_id, exc)

    return None


# ---------------------------------------------------------------------------
# Strategy 2: Kassiesa historical CSV
# ---------------------------------------------------------------------------
KASSIESA_URL = "http://kassiesa.net/bert/data/data5/trank2025.csv"


def _try_kassiesa() -> dict[str, dict] | None:
    """Download the Kassiesa FIFA ranking points CSV and parse the latest entry per team."""
    try:
        import pandas as pd

        logger.info("Trying Kassiesa CSV: %s", KASSIESA_URL)
        resp = requests.get(KASSIESA_URL, timeout=30, headers=HEADERS)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text), sep=None, engine="python")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        logger.debug("Kassiesa columns: %s", list(df.columns))

        # Common column names: team / country / nation, rank, points / pts / total_points
        team_col = next(
            (c for c in df.columns if c in {"team", "country", "nation", "name"}), None
        )
        rank_col = next(
            (c for c in df.columns if "rank" in c), None
        )
        pts_col = next(
            (c for c in df.columns if c in {"points", "pts", "total_points", "rating"}), None
        )

        if not team_col:
            logger.warning("Kassiesa: could not identify team column among %s", list(df.columns))
            return None

        # Keep latest row per team (sort by date column if present)
        date_col = next((c for c in df.columns if "date" in c or "year" in c), None)
        if date_col:
            df = df.sort_values(date_col, ascending=False)
        df = df.drop_duplicates(subset=[team_col], keep="first")

        result: dict[str, dict] = {}
        for _, row in df.iterrows():
            team = str(row[team_col]).strip()
            rank = int(row[rank_col]) if rank_col and not pd.isna(row.get(rank_col)) else None
            points = float(row[pts_col]) if pts_col and not pd.isna(row.get(pts_col)) else None
            if team:
                result[team] = {"rank": rank, "points": round(points, 2) if points is not None else None}

        if result:
            logger.info("Kassiesa returned %d teams", len(result))
            return result

    except Exception as exc:
        logger.warning("Kassiesa failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fetch_rankings() -> dict[str, dict]:
    """
    Attempt live fetch; fall through to hardcoded values if all sources fail.
    Returns the rankings dict (team → {rank, points}).
    """
    rankings = _try_fifa_api()
    if rankings:
        return rankings

    rankings = _try_kassiesa()
    if rankings:
        return rankings

    logger.warning(
        "All live sources failed — using hardcoded fallback rankings (%d teams)",
        len(FALLBACK_RANKINGS),
    )
    return FALLBACK_RANKINGS


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rankings = fetch_rankings()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(rankings, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d team rankings to %s", len(rankings), OUTPUT_PATH
    )
    # Quick preview
    top5 = sorted(
        [(t, v) for t, v in rankings.items() if v.get("rank") is not None],
        key=lambda x: x[1]["rank"],
    )[:5]
    for team, val in top5:
        logger.info("  #%d %s — %.2f pts", val["rank"], team, val.get("points") or 0)


if __name__ == "__main__":
    main()
