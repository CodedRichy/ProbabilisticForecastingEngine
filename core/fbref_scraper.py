"""
FBref scraper for Apollo.

Scrapes season schedule pages to get xG data per match.
Saves to data/raw/fbref/{league}_{season}.parquet

FBref is protected by Cloudflare. To authenticate, grab the cf_clearance
cookie from your browser after visiting any FBref page once:
  1. Open https://fbref.com in Chrome/Firefox
  2. Open DevTools > Application > Cookies > fbref.com
  3. Copy the value of 'cf_clearance'
  4. Set it as an environment variable: FBREF_CF_CLEARANCE=<value>
     OR pass it directly: --cookie <value>

Usage:
    python -m core.fbref_scraper download
    python -m core.fbref_scraper download --league EPL
    python -m core.fbref_scraper download --league EPL --season 2023-24
    python -m core.fbref_scraper download --league EPL --season 2023-24 --cookie <cf_clearance>
    python -m core.fbref_scraper process
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


# ── League / Season Config ────────────────────────────────────────────

FBREF_LEAGUES = {
    'EPL':        {'id': 9,  'slug': 'Premier-League'},
    'LaLiga':     {'id': 12, 'slug': 'La-Liga'},
    'SerieA':     {'id': 11, 'slug': 'Serie-A'},
    'Bundesliga': {'id': 20, 'slug': 'Bundesliga'},
    'Ligue1':     {'id': 13, 'slug': 'Ligue-1'},
}

FBREF_SEASONS = [
    '2017-18', '2018-19', '2019-20', '2020-21',
    '2021-22', '2022-23', '2023-24', '2024-25',
]

# Seconds between requests — FBref asks scrapers to be polite
RATE_LIMIT_SLEEP = 4

BASE_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;'
        'q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
}


# ── Team Name Normalization ───────────────────────────────────────────

NAME_MAP = {
    'Manchester Utd':  'Manchester United',
    'Newcastle Utd':   'Newcastle United',
    'Wolves':          'Wolverhampton',
    'Sheffield Utd':   'Sheffield United',
    "Nott'ham Forest": 'Nottingham Forest',
    'West Brom':       'West Bromwich Albion',
    'Brighton':        'Brighton',
    'Luton Town':      'Luton',
    'Ipswich Town':    'Ipswich',
}


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    return NAME_MAP.get(name.strip(), name.strip())


# ── URL Builder ───────────────────────────────────────────────────────

def build_url(league: str, season: str) -> str:
    cfg  = FBREF_LEAGUES[league]
    lid  = cfg['id']
    slug = cfg['slug']
    return (
        f"https://fbref.com/en/comps/{lid}/{season}/schedule"
        f"/{season}-{slug}-Scores-and-Fixtures"
    )


# ── Session Factory ───────────────────────────────────────────────────

def make_session(cf_clearance: Optional[str] = None) -> requests.Session:
    """
    Build a requests.Session.  If cf_clearance is provided the Cloudflare
    challenge cookie is injected so FBref returns 200 instead of 403.
    """
    s = requests.Session()
    s.headers.update(BASE_HEADERS)

    if cf_clearance:
        s.cookies.set('cf_clearance', cf_clearance, domain='fbref.com')
    else:
        # Try environment variable
        env_cookie = os.environ.get('FBREF_CF_CLEARANCE', '').strip()
        if env_cookie:
            s.cookies.set('cf_clearance', env_cookie, domain='fbref.com')

    return s


# ── Core Fetch + Parse ────────────────────────────────────────────────

def fetch_schedule(url: str, session: requests.Session) -> Optional[pd.DataFrame]:
    """
    Fetch the FBref schedule page and return the raw parsed table.
    Returns None on failure.
    """
    try:
        resp = session.get(url, timeout=30)
    except Exception as e:
        print(f"    Request error: {e}")
        return None

    if resp.status_code == 429:
        print("    Rate limited (429) — sleeping 60s then retrying...")
        time.sleep(60)
        try:
            resp = session.get(url, timeout=30)
        except Exception as e:
            print(f"    Retry failed: {e}")
            return None

    if resp.status_code == 403:
        if 'cf_clearance' not in session.cookies:
            print(
                "    HTTP 403 — Cloudflare challenge blocked the request.\n"
                "    To fix: get the cf_clearance cookie from your browser after\n"
                "    visiting https://fbref.com, then pass it via:\n"
                "      --cookie <value>   OR   env var FBREF_CF_CLEARANCE=<value>"
            )
        else:
            print("    HTTP 403 — cf_clearance cookie may have expired. Refresh it from your browser.")
        return None

    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code} for {url}")
        return None

    # Prefer the table with id="sched_all"; fall back to largest table
    try:
        tables = pd.read_html(resp.text, attrs={'id': 'sched_all'})
    except Exception:
        try:
            tables = pd.read_html(resp.text)
        except Exception as e:
            print(f"    read_html failed: {e}")
            return None

    if not tables:
        print("    No tables found on page.")
        return None

    df = max(tables, key=len)
    return df


def parse_schedule(df: pd.DataFrame, league: str, season: str) -> Optional[pd.DataFrame]:
    """
    Parse the raw FBref schedule table into the normalized xG schema.

    FBref column layout (approx):
      Wk | Day | Date | Time | Home | xG | Score | xG | Away | Attendance | Venue | Referee | ...
    The two xG columns share the name "xG"; pandas disambiguates them as
    "xG" and "xG.1" when the table has a flat header, or as multi-index
    tuples when the header has two levels.
    """
    # Flatten any MultiIndex headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(str(c) for c in col).strip() for col in df.columns]

    col_lower = {c: c.lower() for c in df.columns}

    def find_col(candidates):
        for cand in candidates:
            for orig, low in col_lower.items():
                if cand == low:
                    return orig
        return None

    date_col  = find_col(['date'])
    home_col  = find_col(['home'])
    away_col  = find_col(['away'])
    score_col = find_col(['score'])

    if not all([date_col, home_col, away_col, score_col]):
        print(f"    Missing essential columns. Found: {list(df.columns)}")
        return None

    # xG columns — both named "xg" / "xg.1" in flat headers
    xg_cols = [c for c in df.columns if c.lower() in ('xg', 'xg.1')]
    home_xg_col = xg_cols[0] if len(xg_cols) >= 1 else None
    away_xg_col = xg_cols[1] if len(xg_cols) >= 2 else None

    df = df.copy()

    # Drop repeated header rows that FBref embeds every ~10 rows
    df = df[df[date_col].astype(str) != 'Date']
    df = df[df[date_col].astype(str) != 'nan']

    # Keep only rows with a completed score (contains a digit)
    df = df[df[score_col].notna()]
    df = df[df[score_col].astype(str).str.contains(r'\d', na=False)]

    if len(df) == 0:
        print("    No completed matches after filtering.")
        return None

    out = pd.DataFrame()
    out['date']      = pd.to_datetime(df[date_col], errors='coerce')
    out['season']    = season
    out['league']    = league
    out['home_team'] = df[home_col].apply(normalize_team)
    out['away_team'] = df[away_col].apply(normalize_team)

    out['home_xg'] = (
        pd.to_numeric(df[home_xg_col], errors='coerce')
        if home_xg_col else float('nan')
    )
    out['away_xg'] = (
        pd.to_numeric(df[away_xg_col], errors='coerce')
        if away_xg_col else float('nan')
    )

    out['source'] = 'fbref'

    out['fbref_match_id'] = out.apply(
        lambda r: f"{league}_{season}_{r['home_team']}_{r['away_team']}",
        axis=1
    )

    out = out[['fbref_match_id', 'date', 'season', 'league',
               'home_team', 'away_team', 'home_xg', 'away_xg', 'source']]

    out = out.dropna(subset=['date'])
    return out.reset_index(drop=True)


# ── Download ──────────────────────────────────────────────────────────

def download(
    output_dir: str = "data/raw/fbref",
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    cf_clearance: Optional[str] = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_leagues = leagues or list(FBREF_LEAGUES.keys())
    target_seasons = seasons or FBREF_SEASONS

    unknown = [l for l in target_leagues if l not in FBREF_LEAGUES]
    if unknown:
        print(f"Unknown league(s): {unknown}. Valid: {list(FBREF_LEAGUES.keys())}")
        sys.exit(1)

    session = make_session(cf_clearance)

    has_cookie = 'cf_clearance' in session.cookies
    if not has_cookie:
        print(
            "WARNING: No cf_clearance cookie found.\n"
            "FBref is behind Cloudflare and will return 403 without it.\n"
            "To fix:\n"
            "  1. Visit https://fbref.com in your browser\n"
            "  2. Copy the cf_clearance cookie value from DevTools > Application > Cookies\n"
            "  3. Re-run with: --cookie <value>\n"
            "     OR set environment variable: FBREF_CF_CLEARANCE=<value>\n"
        )

    tasks = [(lg, s) for lg in target_leagues for s in target_seasons]

    for league, season in tqdm(tasks, desc="FBref", unit="page"):
        parquet_file = out_path / f"{league}_{season}.parquet"

        if parquet_file.exists():
            print(f"  Skip (exists): {parquet_file.name}")
            continue

        url = build_url(league, season)
        print(f"  Fetching {league} {season}: {url}")

        raw = fetch_schedule(url, session)
        if raw is None:
            print(f"    WARNING: failed to fetch {league} {season}, skipping.")
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        print(f"    Raw columns ({len(raw.columns)}): {list(raw.columns)}")

        parsed = parse_schedule(raw, league, season)
        if parsed is None or len(parsed) == 0:
            print(f"    WARNING: no data parsed for {league} {season}, skipping.")
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        xg_present = parsed['home_xg'].notna().any()
        print(f"    Parsed {len(parsed)} matches | xG present: {xg_present}")

        parsed.to_parquet(parquet_file, index=False)
        print(f"    Saved: {parquet_file}")

        time.sleep(RATE_LIMIT_SLEEP)


# ── Process ───────────────────────────────────────────────────────────

def process(
    raw_dir: str = "data/raw/fbref",
    output_dir: str = "data/processed",
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquets = sorted(raw_path.glob("*.parquet"))
    if not parquets:
        print(f"No parquet files found in {raw_path}")
        return None

    frames = []
    for pq in parquets:
        try:
            df = pd.read_parquet(pq)
            frames.append(df)
            print(f"  Loaded: {pq.name} ({len(df)} rows)")
        except Exception as e:
            print(f"  Failed to read {pq.name}: {e}")

    if not frames:
        print("No data loaded.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=['fbref_match_id'])
    after = len(combined)
    if before != after:
        print(f"  Dropped {before - after} duplicate rows.")

    out_file = out_path / "fbref_xg.parquet"
    combined.to_parquet(out_file, index=False)

    print(f"\nTotal: {len(combined)} matches")
    print(f"Leagues: {combined['league'].nunique()} — {sorted(combined['league'].unique())}")
    print(f"Seasons: {combined['season'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    xg_pct = combined['home_xg'].notna().mean() * 100
    print(f"xG coverage: {xg_pct:.1f}%")
    print(f"Saved to: {out_file}")

    return combined


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FBref xG scraper for Apollo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "FBref requires a cf_clearance cookie (Cloudflare).\n"
            "Get it from your browser after visiting https://fbref.com once,\n"
            "then pass via --cookie or set env var FBREF_CF_CLEARANCE."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Scrape FBref schedule pages")
    dl.add_argument("--league",  help="League key, e.g. EPL, LaLiga, SerieA, Bundesliga, Ligue1")
    dl.add_argument("--season",  help="Season, e.g. 2023-24")
    dl.add_argument("--cookie",  help="cf_clearance cookie value from browser")
    dl.add_argument("--out-dir", default="data/raw/fbref", help="Output directory for parquets")

    proc = sub.add_parser("process", help="Merge parquets into fbref_xg.parquet")
    proc.add_argument("--raw-dir",    default="data/raw/fbref",   help="Input parquet directory")
    proc.add_argument("--output-dir", default="data/processed",   help="Output directory")

    args = parser.parse_args()

    if args.command == "download":
        leagues = [args.league] if args.league else None
        seasons = [args.season] if args.season else None
        print("Downloading xG data from FBref...")
        download(
            output_dir=args.out_dir,
            leagues=leagues,
            seasons=seasons,
            cf_clearance=args.cookie,
        )

    elif args.command == "process":
        print("Processing FBref parquets...")
        process(raw_dir=args.raw_dir, output_dir=args.output_dir)

    else:
        parser.print_help()
        sys.exit(1)
