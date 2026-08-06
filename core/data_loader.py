"""
Data pipeline for Apollo.

Primary source: Football-Data.co.uk
- Free CSV files with match results and betting odds
- Covers 30+ years for major leagues
- Most reliable free odds data available

Usage:
    python -m core.data_loader download
    python -m core.data_loader process
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

from core.metrics import remove_overround


# ── Source Configuration ──────────────────────────────────────────────

FOOTBALL_DATA_UK = {
    'base_url': 'https://www.football-data.co.uk/mmz4281',
    'leagues': {
        'EPL':        {'code': 'E0', 'country': 'england'},
        'Championship': {'code': 'E1', 'country': 'england'},
        'LaLiga':     {'code': 'SP1', 'country': 'spain'},
        'SerieA':     {'code': 'I1', 'country': 'italy'},
        'Bundesliga': {'code': 'D1', 'country': 'germany'},
        'Ligue1':     {'code': 'F1', 'country': 'france'},
        'Eredivisie': {'code': 'N1', 'country': 'netherlands'},
        'PortugalLiga': {'code': 'P1', 'country': 'portugal'},
    },
    # Season codes: '2324' = 2023-24 season
    'seasons': {
        '2025-26': '2526', '2024-25': '2425', '2023-24': '2324', '2022-23': '2223',
        '2021-22': '2122', '2020-21': '2021', '2019-20': '1920',
        '2018-19': '1819', '2017-18': '1718', '2016-17': '1617',
        '2015-16': '1516', '2014-15': '1415', '2013-14': '1314',
        '2012-13': '1213', '2011-12': '1112', '2010-11': '1011',
    }
}


# ── Column Mapping ────────────────────────────────────────────────────

# Football-Data.co.uk uses inconsistent column names across years
# This maps all variants to our standard schema

COLUMN_MAP = {
    # Date
    'Date': 'date',
    # Teams
    'HomeTeam': 'home_team', 'Home': 'home_team',
    'AwayTeam': 'away_team', 'Away': 'away_team',
    # Goals
    'FTHG': 'home_goals', 'HG': 'home_goals',
    'FTAG': 'away_goals', 'AG': 'away_goals',
    # Result
    'FTR': 'result',
    # Shots (when available)
    'HS': 'home_shots', 'AS': 'away_shots',
    'HST': 'home_shots_target', 'AST': 'away_shots_target',
    # Corners
    'HC': 'home_corners', 'AC': 'away_corners',
    # Cards
    'HY': 'home_yellows', 'AY': 'away_yellows',
    'HR': 'home_reds', 'AR': 'away_reds',
    # Referee
    'Referee': 'referee',
}

# Odds columns — we take the best available closing odds
# Priority: Pinnacle > Bet365 > market average
ODDS_PRIORITY = [
    # Pinnacle closing (most efficient)
    ('PSH', 'PSD', 'PSA', 'pinnacle'),
    # Bet365 closing
    ('B365H', 'B365D', 'B365A', 'bet365'),
    # Market max
    ('BbMxH', 'BbMxD', 'BbMxA', 'market_max'),
    # Market average
    ('BbAvH', 'BbAvD', 'BbAvA', 'market_avg'),
    # Interwetten as fallback
    ('IWH', 'IWD', 'IWA', 'interwetten'),
]


# ── Download ──────────────────────────────────────────────────────────

def build_download_urls(leagues: Optional[List[str]] = None,
                        seasons: Optional[List[str]] = None) -> List[dict]:
    """Build list of URLs to download."""
    urls = []
    
    league_configs = FOOTBALL_DATA_UK['leagues']
    if leagues:
        league_configs = {k: v for k, v in league_configs.items() if k in leagues}
    
    season_configs = FOOTBALL_DATA_UK['seasons']
    if seasons:
        season_configs = {k: v for k, v in season_configs.items() if k in seasons}
    
    for league_name, league_info in league_configs.items():
        for season_name, season_code in season_configs.items():
            url = f"{FOOTBALL_DATA_UK['base_url']}/{season_code}/{league_info['code']}.csv"
            urls.append({
                'url': url,
                'league': league_name,
                'season': season_name,
                'filename': f"{league_name}_{season_name}.csv"
            })
    
    return urls


def download_data(output_dir: str = "data/raw",
                  leagues: Optional[List[str]] = None,
                  seasons: Optional[List[str]] = None,
                  force_seasons: Optional[List[str]] = None):
    """
    Download CSVs from Football-Data.co.uk.

    force_seasons: list of season names (e.g. ['2024-25', '2025-26']) that
                   will be re-downloaded even when a local file already exists.

    NOTE: Requires network access. Run once, then work offline.
    """
    import requests

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    urls = build_download_urls(leagues, seasons)

    for item in urls:
        filepath = output_path / item['filename']

        if filepath.exists():
            if force_seasons and item['season'] in force_seasons:
                print(f"  Force re-download: {item['filename']}")
                filepath.unlink()
            else:
                print(f"  Skip (exists): {item['filename']}")
                continue

        print(f"  Downloading: {item['url']}")
        try:
            resp = requests.get(item['url'], timeout=30)
            if resp.status_code == 200 and len(resp.content) > 100:
                filepath.write_bytes(resp.content)
                print(f"    Saved: {item['filename']}")
            else:
                print(f"    FAILED: status={resp.status_code}")
        except Exception as e:
            print(f"    ERROR: {e}")


# ── Processing ────────────────────────────────────────────────────────

def process_single_csv(filepath: str, league: str, season: str) -> Optional[pd.DataFrame]:
    """
    Process a single raw CSV into standardized format.
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
    except Exception as e:
        print(f"  Failed to read {filepath}: {e}")
        return None
    
    if len(df) < 10:
        return None
    
    # Rename columns
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    
    # Require minimum columns
    required = ['home_team', 'away_team', 'home_goals', 'away_goals']
    if not all(col in df.columns for col in required):
        print(f"  Skipping {filepath}: missing required columns")
        return None
    
    # Add metadata
    df['league'] = league
    df['season'] = season
    df['match_id'] = df.apply(
        lambda r: f"{league}_{season}_{r['home_team']}_{r['away_team']}_{r.get('date', '')}",
        axis=1
    )
    
    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    # Compute result if missing
    if 'result' not in df.columns:
        df['result'] = np.where(
            df['home_goals'] > df['away_goals'], 'H',
            np.where(df['home_goals'] < df['away_goals'], 'A', 'D')
        )
    
    # Binary outcome columns
    df['home_win'] = (df['result'] == 'H').astype(int)
    df['draw'] = (df['result'] == 'D').astype(int)
    df['away_win'] = (df['result'] == 'A').astype(int)
    
    # Find best available odds
    df = _extract_best_odds(df)
    
    # Select and order final columns
    final_cols = [
        'match_id', 'date', 'season', 'league',
        'home_team', 'away_team',
        'home_goals', 'away_goals', 'result',
        'home_win', 'draw', 'away_win',
        'home_odds', 'draw_odds', 'away_odds',
        'home_implied', 'draw_implied', 'away_implied',
        'odds_source',
    ]
    
    # Add optional columns if present
    optional = ['referee', 'home_shots', 'away_shots', 'home_shots_target', 'away_shots_target',
                'home_corners', 'away_corners', 'home_yellows', 'away_yellows',
                'home_reds', 'away_reds']
    for col in optional:
        if col in df.columns:
            final_cols.append(col)
    
    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]
    
    # Drop rows with missing essential data
    df = df.dropna(subset=['home_goals', 'away_goals'])
    
    return df


def _extract_best_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract best available closing odds from the raw data.
    Uses priority list: Pinnacle > Bet365 > Market max > Market avg.
    """
    original_cols = df.columns.tolist()
    
    for home_col, draw_col, away_col, source_name in ODDS_PRIORITY:
        if all(col in original_cols for col in [home_col, draw_col, away_col]):
            df['home_odds'] = pd.to_numeric(df[home_col], errors='coerce')
            df['draw_odds'] = pd.to_numeric(df[draw_col], errors='coerce')
            df['away_odds'] = pd.to_numeric(df[away_col], errors='coerce')
            df['odds_source'] = source_name
            
            # Remove overround to get fair implied probabilities
            valid_odds = (df['home_odds'] > 1) & (df['draw_odds'] > 1) & (df['away_odds'] > 1)
            
            implied = df.loc[valid_odds].apply(
                lambda r: remove_overround(r['home_odds'], r['draw_odds'], r['away_odds']),
                axis=1, result_type='expand'
            )
            implied.columns = ['home_implied', 'draw_implied', 'away_implied']
            
            df.loc[valid_odds, 'home_implied'] = implied['home_implied']
            df.loc[valid_odds, 'draw_implied'] = implied['draw_implied']
            df.loc[valid_odds, 'away_implied'] = implied['away_implied']
            
            print(f"    Odds source: {source_name}")
            return df
    
    # No odds found
    df['home_odds'] = np.nan
    df['draw_odds'] = np.nan
    df['away_odds'] = np.nan
    df['home_implied'] = np.nan
    df['draw_implied'] = np.nan
    df['away_implied'] = np.nan
    df['odds_source'] = 'none'
    
    return df


def process_all(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    """
    Process all raw CSVs into standardized parquet files.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_frames = []
    
    for csv_file in sorted(raw_path.glob("*.csv")):
        # Parse league and season from filename: EPL_2023-24.csv
        parts = csv_file.stem.split('_')
        if len(parts) >= 2:
            league = parts[0]
            season = '_'.join(parts[1:])
        else:
            continue
        
        print(f"Processing: {csv_file.name}")
        df = process_single_csv(str(csv_file), league, season)
        
        if df is not None and len(df) > 0:
            all_frames.append(df)
            print(f"  -> {len(df)} matches")
    
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)
        
        # Save as single parquet
        out_file = output_path / "matches.parquet"
        combined.to_parquet(out_file, index=False)
        
        print(f"\nTotal: {len(combined)} matches")
        print(f"Leagues: {combined['league'].nunique()}")
        print(f"Seasons: {combined['season'].nunique()}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Saved to: {out_file}")
        
        # Summary stats
        print(f"\nOdds coverage:")
        print(combined['odds_source'].value_counts().to_string())
        
        return combined
    else:
        print("No data processed.")
        return None


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m core.data_loader download")
        print("  python -m core.data_loader process")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "download":
        print("Downloading data from Football-Data.co.uk...")
        download_data(force_seasons=['2024-25', '2025-26'])
    elif command == "process":
        print("Processing raw data...")
        process_all()
    else:
        print(f"Unknown command: {command}")
