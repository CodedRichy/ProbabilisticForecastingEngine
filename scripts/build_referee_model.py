"""
build_referee_model.py - Fit and save the Apollo referee bias model.

Loads historical matches, fits per-referee Bayesian-shrunk home-bias deltas,
writes the model to JSON, and prints a human-readable summary.

Usage:
    python scripts/build_referee_model.py
    python scripts/build_referee_model.py --data data/processed/matches.parquet \
        --out data/models/referee_model.json --min-matches 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from core.referee_model import RefereeModel

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "data" / "processed" / "matches.parquet"
DEFAULT_OUT = ROOT / "data" / "models" / "referee_model.json"


def _resolve(path_str: str, default: Path) -> Path:
    """Resolve a user-supplied path; relative paths anchor at project root."""
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Apollo referee bias model.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to the processed matches parquet file.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output path for the serialized referee model JSON.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=10,
        help="Minimum matches before a referee is modelled.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    data_path = _resolve(args.data, DEFAULT_DATA)
    out_path = _resolve(args.out, DEFAULT_OUT)

    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 1

    print(f"Loading match data from {data_path} ...")
    df = pd.read_parquet(data_path)
    total_matches = len(df)

    # Normalise column name — data_loader stores it as 'referee' (lowercase)
    if "referee" in df.columns and "Referee" not in df.columns:
        df = df.rename(columns={"referee": "Referee"})

    if "Referee" not in df.columns:
        print(
            "WARNING: 'Referee' column missing from the dataset. "
            "The model will be empty.",
            file=sys.stderr,
        )
        df["Referee"] = pd.NA
        ref_matches = 0
    else:
        ref_mask = df["Referee"].notna() & (
            df["Referee"].astype(str).str.strip() != ""
        )
        ref_matches = int(ref_mask.sum())
        if ref_matches == 0:
            print(
                "WARNING: 'Referee' column is present but entirely null/blank. "
                "The model will be empty.",
                file=sys.stderr,
            )

    print("Fitting referee model ...")
    model = RefereeModel().fit(df, min_matches=args.min_matches)

    model.save(str(out_path))
    print(f"\nModel saved -> {out_path}")

    print()
    print(model.summary())

    pct = (ref_matches / total_matches * 100) if total_matches else 0.0
    print()
    print(
        f"Matches with referee data: {ref_matches:,} / {total_matches:,} "
        f"({pct:.1f}%)"
    )
    print(f"Referees modelled (>= {args.min_matches} matches): {len(model.stats)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
