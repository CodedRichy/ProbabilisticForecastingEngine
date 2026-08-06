"""
build_player_model.py - Train and save the player-parametric Dixon-Coles model.

Loads historical match xG (or goals as fallback), fits PlayerModel team
attack/defense parameters, saves to data/models/player_model.json, and prints
the top attack and best defense rankings.

Usage:
    python scripts/build_player_model.py
    python scripts/build_player_model.py --seasons 5
    python scripts/build_player_model.py --data data/processed/matches_xg.parquet \
        --out data/models/player_model.json --seasons 6
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.player_model import PlayerModel

logger = logging.getLogger(__name__)

_DEFAULT_DATA = "data/processed/matches_xg.parquet"
_FALLBACK_DATA = "data/processed/matches.parquet"
_DEFAULT_OUT = "data/models/player_model.json"


def _load_data(data_path: str) -> pd.DataFrame:
    """Load the requested parquet, falling back to matches.parquet if missing."""
    p = pathlib.Path(data_path)
    if not p.exists():
        fb = pathlib.Path(_FALLBACK_DATA)
        if data_path == _DEFAULT_DATA and fb.exists():
            logger.warning("%s not found; falling back to %s", data_path, fb)
            p = fb
        else:
            raise FileNotFoundError(f"Match data not found: {data_path}")
    logger.info("Loading match data from %s", p)
    return pd.read_parquet(p)


def _select_seasons(df: pd.DataFrame, n_seasons: int) -> pd.DataFrame:
    """Keep only the last `n_seasons` seasons if a season column exists."""
    if n_seasons <= 0 or "season" not in df.columns:
        return df
    seasons = sorted(s for s in df["season"].dropna().unique())
    keep = set(seasons[-n_seasons:])
    out = df[df["season"].isin(keep)].copy()
    logger.info("Filtered to last %d seasons: %s (%d matches)",
                n_seasons, sorted(keep), len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the player-parametric Dixon-Coles model."
    )
    parser.add_argument("--data", default=_DEFAULT_DATA,
                        help=f"Input parquet (default: {_DEFAULT_DATA})")
    parser.add_argument("--out", default=_DEFAULT_OUT,
                        help=f"Output JSON path (default: {_DEFAULT_OUT})")
    parser.add_argument("--seasons", type=int, default=5,
                        help="Number of most-recent seasons to train on (default: 5)")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="Max MLE iterations (default: 20)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = _load_data(args.data)

    # Require valid goals on every training row (xG used preferentially inside fit).
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    if df.empty:
        raise SystemExit("No matches with non-null goals to train on.")

    df = _select_seasons(df, args.seasons)
    if df.empty:
        raise SystemExit("No matches remain after season filtering.")

    print(f"Fitting PlayerModel on {len(df)} matches...")
    model = PlayerModel().fit(df, max_iter=args.max_iter)

    out_path = pathlib.Path(args.out)
    model.save(str(out_path))
    print(f"\nModel saved -> {out_path}")
    print(f"  league_avg_goals = {model._league_avg_goals:.3f}")
    print(f"  home_advantage   = {model._home_advantage:.3f}")
    print(f"  teams fitted     = {len(model._team_attack)}")

    print("\n" + model.ranking(top_n=20))


if __name__ == "__main__":
    main()
