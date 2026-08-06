"""
activate_fbref_xg.py
--------------------
Standalone script to check, fetch, and activate xG data in the main
matches dataset (data/processed/matches_xg.parquet).

Workflow:
  1. Load data/processed/matches_xg.parquet and inspect xG coverage.
  2. If real xG columns already exist with reasonable coverage, report and exit.
  3. If real xG is missing, attempt to scrape FBref for the last 2 seasons
     (2023-24, 2024-25) across EPL, Bundesliga, LaLiga, SerieA, then merge
     the FBref data into matches_xg.parquet.
  4. As a fallback proxy, fill any remaining null home_xg / away_xg values
     using shots-on-target (xg = shots_on_target × 0.33) where available.
  5. Print a coverage summary: "X matches with xG out of Y total".

Run from project root:
    python scripts/activate_fbref_xg.py
"""

import os
import sys
import time
from pathlib import Path

# Allow imports from the project root (e.g. core.fbref_scraper)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).parent.parent
MATCHES_XG_PATH = PROJECT_ROOT / "data" / "processed" / "matches_xg.parquet"
FBREF_XG_PATH   = PROJECT_ROOT / "data" / "processed" / "fbref_xg.parquet"
FBREF_RAW_DIR   = PROJECT_ROOT / "data" / "raw" / "fbref"

# Seasons and leagues to scrape when real xG is absent
TARGET_SEASONS = ["2023-24", "2024-25"]
TARGET_LEAGUES = ["EPL", "Bundesliga", "LaLiga", "SerieA"]

# xG proxy coefficient: xG ≈ shots_on_target × 0.33
XG_PROXY_COEFF = 0.33


# ── Helper: detect xG column names ───────────────────────────────────────────

def find_xg_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Return (home_xg_col, away_xg_col) for whichever naming convention is present.
    Checks 'home_xg' / 'away_xg' first, then 'xg_home' / 'xg_away'.
    Returns (None, None) if neither pair is found.
    """
    cols = set(df.columns)
    if "home_xg" in cols and "away_xg" in cols:
        return "home_xg", "away_xg"
    if "xg_home" in cols and "xg_away" in cols:
        return "xg_home", "xg_away"
    return None, None


# ── Helper: coverage report ───────────────────────────────────────────────────

def print_coverage(df: pd.DataFrame, home_xg_col: str) -> None:
    """Print xG coverage statistics."""
    total   = len(df)
    with_xg = int(df[home_xg_col].notna().sum())
    pct     = with_xg / total * 100 if total > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  xG Coverage Report")
    print(f"{'='*55}")
    print(f"  {with_xg} matches with xG out of {total} total ({pct:.1f}%)")

    if with_xg > 0:
        xg_rows = df[df[home_xg_col].notna()]
        date_col = next(
            (c for c in ("date", "match_date", "Date") if c in df.columns), None
        )
        if date_col:
            try:
                dates = pd.to_datetime(xg_rows[date_col], errors="coerce").dropna()
                if not dates.empty:
                    print(f"  Date range : {dates.min().date()} → {dates.max().date()}")
            except Exception:
                pass

        if "league" in df.columns:
            league_counts = xg_rows["league"].value_counts()
            print(f"  By league  :")
            for lg, cnt in league_counts.items():
                print(f"    {lg:<15} {cnt} matches")

    print(f"{'='*55}\n")


# ── Step 1: load main dataset ─────────────────────────────────────────────────

def load_matches() -> pd.DataFrame | None:
    """Load data/processed/matches_xg.parquet. Return None if it doesn't exist."""
    if not MATCHES_XG_PATH.exists():
        print(f"[ERROR] Main dataset not found: {MATCHES_XG_PATH}")
        return None
    df = pd.read_parquet(MATCHES_XG_PATH)
    print(f"[INFO] Loaded matches_xg.parquet — {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── Step 2: check existing xG coverage ───────────────────────────────────────

def check_existing_xg(df: pd.DataFrame) -> tuple[bool, str | None, str | None]:
    """
    Inspect the dataframe for xG columns.

    Returns:
        (has_xg_columns, home_xg_col, away_xg_col)
        has_xg_columns is True even if the columns exist but are all-null.
    """
    home_col, away_col = find_xg_columns(df)
    has_columns = home_col is not None
    return has_columns, home_col, away_col


# ── Step 3: scrape FBref and merge ───────────────────────────────────────────

def scrape_and_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scrape FBref for TARGET_LEAGUES × TARGET_SEASONS, build fbref_xg.parquet,
    then left-join home_xg / away_xg into df on (date, home_team, away_team).

    Returns the updated dataframe (xg columns added / filled).
    """
    from core.fbref_scraper import (
        make_session,
        build_url,
        fetch_schedule,
        parse_schedule,
        RATE_LIMIT_SLEEP,
    )

    print("\n[SCRAPE] Starting FBref xG download...")
    print(f"         Leagues : {TARGET_LEAGUES}")
    print(f"         Seasons : {TARGET_SEASONS}")

    # Check for Cloudflare cookie
    cf_clearance = os.environ.get("FBREF_CF_CLEARANCE", "").strip() or None
    if not cf_clearance:
        print(
            "[WARNING] FBREF_CF_CLEARANCE env var not set.\n"
            "          FBref is protected by Cloudflare and may return 403.\n"
            "          To fix: export FBREF_CF_CLEARANCE=<cookie from browser>\n"
        )

    FBREF_RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session(cf_clearance)

    frames = []

    for league in TARGET_LEAGUES:
        for season in TARGET_SEASONS:
            parquet_path = FBREF_RAW_DIR / f"{league}_{season}.parquet"

            # Use cached file if it already exists
            if parquet_path.exists():
                try:
                    cached = pd.read_parquet(parquet_path)
                    print(f"[CACHE ] {league} {season} — loaded {len(cached)} rows from cache")
                    frames.append(cached)
                    continue
                except Exception as e:
                    print(f"[WARN  ] Cache read failed for {parquet_path.name}: {e}")

            # Fetch fresh
            url = build_url(league, season)
            print(f"[FETCH ] {league} {season}: {url}")

            raw = fetch_schedule(url, session)
            if raw is None:
                print(f"[SKIP  ] Could not fetch {league} {season}")
                time.sleep(RATE_LIMIT_SLEEP)
                continue

            parsed = parse_schedule(raw, league, season)
            if parsed is None or len(parsed) == 0:
                print(f"[SKIP  ] No parseable data for {league} {season}")
                time.sleep(RATE_LIMIT_SLEEP)
                continue

            xg_ok = parsed["home_xg"].notna().sum()
            print(f"[OK    ] {league} {season} — {len(parsed)} matches, {xg_ok} with xG")

            parsed.to_parquet(parquet_path, index=False)
            frames.append(parsed)

            time.sleep(RATE_LIMIT_SLEEP)

    if not frames:
        print("[SCRAPE] No FBref data collected. Skipping merge.")
        return df

    fbref_df = pd.concat(frames, ignore_index=True)
    fbref_df = fbref_df.drop_duplicates(subset=["fbref_match_id"])

    # Save combined fbref_xg.parquet
    fbref_df.to_parquet(FBREF_XG_PATH, index=False)
    print(f"\n[SAVED ] fbref_xg.parquet — {len(fbref_df):,} rows → {FBREF_XG_PATH}")

    # --- Merge into main dataset ---
    # Normalise dates for joining
    date_col = next(
        (c for c in ("date", "match_date", "Date") if c in df.columns), None
    )
    if date_col is None:
        print("[WARN  ] No date column found in matches_xg — cannot merge by date.")
        return df

    df = df.copy()
    df["_date_norm"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    fbref_df["_date_norm"] = pd.to_datetime(
        fbref_df["date"], errors="coerce"
    ).dt.normalize()

    # FBref uses 'home_team' / 'away_team' — align with main dataset column names
    home_team_col = next(
        (c for c in ("home_team", "home", "HomeTeam") if c in df.columns), None
    )
    away_team_col = next(
        (c for c in ("away_team", "away", "AwayTeam") if c in df.columns), None
    )

    if home_team_col is None or away_team_col is None:
        print("[WARN  ] Cannot identify team columns in matches_xg — skipping merge.")
        return df

    # Build a lookup: (date, home_team, away_team) → (home_xg, away_xg)
    lookup = fbref_df[
        ["_date_norm", "home_team", "away_team", "home_xg", "away_xg"]
    ].rename(
        columns={
            "home_team": "_fbref_home",
            "away_team": "_fbref_away",
        }
    ).drop_duplicates(subset=["_date_norm", "_fbref_home", "_fbref_away"])

    # Merge on date + teams (case-insensitive strip for safety)
    df["_home_key"] = df[home_team_col].astype(str).str.strip()
    df["_away_key"] = df[away_team_col].astype(str).str.strip()
    lookup["_home_key"] = lookup["_fbref_home"].astype(str).str.strip()
    lookup["_away_key"] = lookup["_fbref_away"].astype(str).str.strip()

    merged = df.merge(
        lookup[["_date_norm", "_home_key", "_away_key", "home_xg", "away_xg"]],
        on=["_date_norm", "_home_key", "_away_key"],
        how="left",
    )

    # If home_xg column already existed, fill nulls; otherwise use new column
    if "home_xg" in df.columns:
        merged["home_xg"] = merged["home_xg_x"].fillna(merged["home_xg_y"])
        merged["away_xg"] = merged["away_xg_x"].fillna(merged["away_xg_y"])
        merged = merged.drop(
            columns=[c for c in merged.columns if c.endswith(("_x", "_y"))
                     and c not in ("home_xg", "away_xg")],
            errors="ignore",
        )
    # else: merged already has home_xg / away_xg from the right side

    # Drop temp columns
    merged = merged.drop(
        columns=["_date_norm", "_home_key", "_away_key", "_fbref_home", "_fbref_away"],
        errors="ignore",
    )

    matched = int(merged["home_xg"].notna().sum())
    print(f"[MERGE ] FBref xG matched {matched:,} rows in matches_xg")

    return merged


# ── Step 4: xG proxy from shots on target ────────────────────────────────────

def apply_xg_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill null home_xg / away_xg using shots-on-target proxy if available.
    Formula: xg = shots_on_target × XG_PROXY_COEFF (0.33)
    """
    df = df.copy()

    # Detect shots-on-target column names
    home_sot = next(
        (c for c in ("home_shots_target", "HS", "HST", "home_shots_on_target")
         if c in df.columns), None
    )
    away_sot = next(
        (c for c in ("away_shots_target", "AS", "AST", "away_shots_on_target")
         if c in df.columns), None
    )

    proxy_applied = 0

    if home_sot and "home_xg" in df.columns:
        null_mask = df["home_xg"].isna() & df[home_sot].notna()
        df.loc[null_mask, "home_xg"] = (
            pd.to_numeric(df.loc[null_mask, home_sot], errors="coerce") * XG_PROXY_COEFF
        )
        proxy_applied += int(null_mask.sum())
        if null_mask.any():
            print(
                f"[PROXY ] home_xg filled via {home_sot} × {XG_PROXY_COEFF} "
                f"for {null_mask.sum():,} rows"
            )

    if away_sot and "away_xg" in df.columns:
        null_mask = df["away_xg"].isna() & df[away_sot].notna()
        df.loc[null_mask, "away_xg"] = (
            pd.to_numeric(df.loc[null_mask, away_sot], errors="coerce") * XG_PROXY_COEFF
        )
        proxy_applied += int(null_mask.sum())
        if null_mask.any():
            print(
                f"[PROXY ] away_xg filled via {away_sot} × {XG_PROXY_COEFF} "
                f"for {null_mask.sum():,} rows"
            )

    if proxy_applied == 0:
        print("[PROXY ] No shots-on-target columns found or no nulls to fill.")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\nactivate_fbref_xg.py — xG activation script")
    print("=" * 55)

    # 1. Load the main dataset
    df = load_matches()
    if df is None:
        return 1

    # 2. Check existing xG coverage
    has_xg_cols, home_xg_col, away_xg_col = check_existing_xg(df)

    if has_xg_cols:
        existing_coverage = df[home_xg_col].notna().sum()
        total = len(df)
        print(
            f"[INFO] xG columns found: '{home_xg_col}' / '{away_xg_col}' "
            f"({existing_coverage:,} / {total:,} non-null)"
        )

        if existing_coverage > 0:
            # Already have real xG data — just report and exit
            print("[INFO] xG data already present. Reporting coverage.")
            print_coverage(df, home_xg_col)
            return 0

        # Columns exist but are all-null — fall through to scrape
        print("[INFO] xG columns are all-null. Attempting FBref scrape...")
    else:
        print("[INFO] No xG columns found. Attempting FBref scrape...")

    # 3. Scrape FBref and merge
    df = scrape_and_merge(df)

    # Refresh column detection after merge
    home_xg_col, away_xg_col = find_xg_columns(df)

    # 4. Apply xG proxy for any remaining nulls
    if home_xg_col:
        df = apply_xg_proxy(df)
        home_xg_col, _ = find_xg_columns(df)  # re-detect (shouldn't change)

    # If we still have no xG columns after everything, add empty ones so the
    # file schema is consistent for downstream consumers
    if home_xg_col is None:
        print("[WARN  ] No xG data could be obtained. Adding null home_xg/away_xg columns.")
        df["home_xg"] = float("nan")
        df["away_xg"] = float("nan")
        home_xg_col = "home_xg"

    # 5. Save updated dataset
    df.to_parquet(MATCHES_XG_PATH, index=False)
    print(f"\n[SAVED ] matches_xg.parquet updated → {MATCHES_XG_PATH}")

    # 6. Final coverage report
    print_coverage(df, home_xg_col)

    return 0


if __name__ == "__main__":
    sys.exit(main())
