"""
Download xG data from Understat for top 5 European leagues.

Saves to data/raw/understat_xg.parquet
Merges with data/processed/matches.parquet on (date, home_team, away_team)
Saves enriched dataset to data/processed/matches_xg.parquet

Usage:
    python scripts/fetch_understat.py
    python scripts/fetch_understat.py --seasons 2022 2023 2024   # specific seasons
    python scripts/fetch_understat.py --merge-only               # skip download, just merge
"""

import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Understat league names -> our league names
LEAGUE_MAP = {
    "EPL":        "EPL",
    "La_liga":    "LaLiga",
    "Bundesliga": "Bundesliga",
    "Serie_A":    "SerieA",
    "Ligue_1":    "Ligue1",
}

# Seasons: understat uses start year ("2014" = 2014-15, "2025" = 2025-26)
ALL_SEASONS = [str(y) for y in range(2014, 2026)]

# Team name normalisations (understat -> football-data.co.uk)
TEAM_ALIASES = {
    # EPL
    "Manchester United":       "Man United",
    "Manchester City":         "Man City",
    "Tottenham":               "Tottenham",
    "Tottenham Hotspur":       "Tottenham",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United":         "West Ham",
    "West Bromwich Albion":    "West Brom",
    "Leicester City":          "Leicester",
    "Brighton":                "Brighton",
    "Brighton & Hove Albion":  "Brighton",
    "Nottingham Forest":       "Nott'm Forest",
    "Sheffield United":        "Sheffield United",
    "Luton Town":              "Luton",
    "Queens Park Rangers":     "QPR",
    # Bundesliga
    "Borussia Dortmund":       "Dortmund",
    "RB Leipzig":              "RB Leipzig",
    "RasenBallsport Leipzig":  "RB Leipzig",
    "Bayer Leverkusen":        "Leverkusen",
    "Borussia M.Gladbach":     "M'gladbach",
    "Eintracht Frankfurt":     "Ein Frankfurt",
    "FC Cologne":              "FC Koln",
    "Bayern Munich":           "Bayern Munich",
    "Arminia Bielefeld":       "Bielefeld",
    "FC Heidenheim":           "Heidenheim",
    "Fortuna Duesseldorf":     "Fortuna Dusseldorf",
    "Greuther Fuerth":         "Greuther Furth",
    "Hamburger SV":            "Hamburg",
    "Hannover 96":             "Hannover",
    "Hertha Berlin":           "Hertha",
    "Mainz 05":                "Mainz",
    "Nuernberg":               "Nurnberg",
    "VfB Stuttgart":           "Stuttgart",
    # LaLiga
    "Atletico Madrid":         "Ath Madrid",
    "Athletic Club":           "Ath Bilbao",
    "Real Betis":              "Betis",
    "Celta Vigo":              "Celta",
    "Deportivo Alaves":        "Alaves",
    "Deportivo La Coruna":     "La Coruna",
    "Espanyol":                "Espanol",
    "Espanol":                 "Espanol",
    "Rayo Vallecano":          "Vallecano",
    "Real Valladolid":         "Valladolid",
    "Real Sociedad":           "Sociedad",
    "SD Huesca":               "Huesca",
    "Sporting Gijon":          "Sp Gijon",
    # SerieA
    "AC Milan":                "Milan",
    "AS Roma":                 "Roma",
    "SSC Napoli":              "Napoli",
    "Inter":                   "Inter",
    "Parma Calcio 1913":       "Parma",
    "SPAL 2013":               "Spal",
    # Ligue1
    "Paris Saint Germain":     "Paris SG",
    "Paris Saint-Germain":     "Paris SG",
    "Clermont Foot":           "Clermont",
    "GFC Ajaccio":             "Ajaccio",
    "SC Bastia":               "Bastia",
    "Saint-Etienne":           "St Etienne",
    "Evian Thonon Gaillard":   "Evian TG",
}


def normalise_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


import requests

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
})


def fetch_league_season(league: str, season: str) -> pd.DataFrame:
    # Understat AJAX API (found from league.min.js source)
    url = f"https://understat.com/main/getLeagueData/{league}/{season}"
    SESSION.headers["Referer"] = f"https://understat.com/league/{league}/{season}"
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    matches = data.get("dates", [])

    rows = []
    for m in matches:
        try:
            rows.append({
                "understat_id":  m.get("id"),
                "date":          m.get("datetime", "")[:10],
                "home_team_raw": m.get("h", {}).get("title", ""),
                "away_team_raw": m.get("a", {}).get("title", ""),
                "home_xg":       float(m.get("xG", {}).get("h") or np.nan),
                "away_xg":       float(m.get("xG", {}).get("a") or np.nan),
                "home_goals":    int(m.get("goals", {}).get("h", -1)),
                "away_goals":    int(m.get("goals", {}).get("a", -1)),
                "league":        LEAGUE_MAP[league],
                "season_start":  int(season),
                "is_result":     bool(m.get("isResult", False)),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["is_result"]]  # only completed matches
    df["home_team"] = df["home_team_raw"].apply(normalise_team)
    df["away_team"] = df["away_team_raw"].apply(normalise_team)
    df["date"] = pd.to_datetime(df["date"])
    return df


def download_all(seasons: list) -> pd.DataFrame:
    frames = []
    total = len(LEAGUE_MAP) * len(seasons)
    done = 0
    for league in LEAGUE_MAP:
        for season in seasons:
            done += 1
            try:
                df = fetch_league_season(league, season)
                if not df.empty:
                    frames.append(df)
                    logger.info("[%d/%d] %s %s: %d matches  home_xg=%.2f",
                                done, total, league, season, len(df), df["home_xg"].mean())
                else:
                    logger.warning("[%d/%d] %s %s: no data", done, total, league, season)
                time.sleep(1.2)  # polite
            except Exception as e:
                logger.warning("[%d/%d] %s %s FAILED: %s", done, total, league, season, e)

    if not frames:
        raise RuntimeError("No data downloaded from Understat.")
    return pd.concat(frames, ignore_index=True)


def merge_with_club_data(xg_df: pd.DataFrame, parquet_path: str) -> pd.DataFrame:
    matches = pd.read_parquet(parquet_path)
    matches["date"] = pd.to_datetime(matches["date"])

    xg_slim = xg_df[["date", "home_team", "away_team", "home_xg", "away_xg",
                      "home_team_raw", "away_team_raw"]].copy()

    # Merge on date + normalised team names
    merged = matches.merge(
        xg_slim,
        on=["date", "home_team", "away_team"],
        how="left",
    )

    n_total   = len(merged)
    n_matched = merged["home_xg"].notna().sum()
    logger.info("Merge: %d/%d matches got xG (%.1f%%)", n_matched, n_total, n_matched/n_total*100)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=ALL_SEASONS,
                        help="Start years e.g. 2018 2019 2020")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip download; just merge existing understat parquet")
    parser.add_argument("--out-raw",    default="data/raw/understat_xg.parquet")
    parser.add_argument("--out-merged", default="data/processed/matches_xg.parquet")
    parser.add_argument("--club-data",  default="data/processed/matches.parquet")
    args = parser.parse_args()

    Path(args.out_raw).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_merged).parent.mkdir(parents=True, exist_ok=True)

    raw_path = Path(args.out_raw)

    if not args.merge_only:
        logger.info("Downloading xG from Understat for %d leagues x %d seasons...",
                    len(LEAGUE_MAP), len(args.seasons))
        xg_df = download_all(args.seasons)
        xg_df.to_parquet(raw_path, index=False)
        logger.info("Saved %d xG records -> %s", len(xg_df), raw_path)
    else:
        if not raw_path.exists():
            logger.error("--merge-only but %s not found. Run without --merge-only first.", raw_path)
            sys.exit(1)
        xg_df = pd.read_parquet(raw_path)
        # Re-apply current aliases from raw names so alias updates take effect
        if "home_team_raw" in xg_df.columns:
            xg_df["home_team"] = xg_df["home_team_raw"].apply(normalise_team)
            xg_df["away_team"] = xg_df["away_team_raw"].apply(normalise_team)
        logger.info("Loaded %d existing xG records from %s", len(xg_df), raw_path)

    # Summary stats
    print(f"\nxG data summary:")
    print(f"  Total matches : {len(xg_df)}")
    print(f"  Leagues       : {xg_df['league'].value_counts().to_dict()}")
    print(f"  Seasons       : {xg_df['season_start'].min()} - {xg_df['season_start'].max()}")
    print(f"  Avg home xG   : {xg_df['home_xg'].mean():.3f}")
    print(f"  Avg away xG   : {xg_df['away_xg'].mean():.3f}")

    # Merge with club data
    if Path(args.club_data).exists():
        logger.info("Merging with %s...", args.club_data)
        merged = merge_with_club_data(xg_df, args.club_data)
        merged.to_parquet(args.out_merged, index=False)
        n_xg = merged["home_xg"].notna().sum()
        print(f"\nEnriched dataset:")
        print(f"  Total matches : {len(merged)}")
        print(f"  With xG       : {n_xg} ({n_xg/len(merged)*100:.1f}%)")
        print(f"  Saved         : {args.out_merged}")
    else:
        logger.warning("Club data not found at %s — skipping merge.", args.club_data)


if __name__ == "__main__":
    main()
