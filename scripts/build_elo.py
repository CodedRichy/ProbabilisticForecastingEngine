"""
build_elo.py - Train and save the Elo model to disk.

Run once to train, then use the saved model for predictions.
Usage: python scripts/build_elo.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.elo_model import EloModel


def main():
    print("Downloading match data and fitting Elo model...")
    model = EloModel.from_csv(
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    )

    ratings = model.ratings
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 30 Teams by Elo Rating:")
    print(f"{'Rank':<6} {'Team':<30} {'Elo':>8}")
    print("-" * 46)
    for rank, (team, elo) in enumerate(sorted_teams[:30], start=1):
        print(f"{rank:<6} {team:<30} {elo:>8.1f}")

    output_path = pathlib.Path("data/models/elo_national.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    print(f"\nModel saved. {len(ratings)} teams rated -> {output_path}")


if __name__ == "__main__":
    main()
