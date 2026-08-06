"""
update_elo_from_eloratings.py

Replaces the team Elo ratings in elo_national.json with live data from
eloratings.net — the gold-standard cross-continental calibrated Elo source.

Why: martj42-trained Elo suffers from continental isolation bias. Ecuador at
1895 (built from CONMEBOL-only record) is not comparable to Germany at 1993
(built from UEFA record). eloratings.net cross-calibrates all continents via
inter-continental tournament results going back to 1950.

Run:
    python scripts/update_elo_from_eloratings.py

Leaves the K-factor / margin-of-victory logic in elo_model.py untouched.
Only replaces the ratings dict.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from core.team_names import normalize_team

ROOT      = Path(__file__).parent.parent
ELO_PATH  = ROOT / "data" / "models" / "elo_national.json"
ELONET_URL = "https://www.eloratings.net/World.tsv"

# Name overrides: eloratings.net name -> canonical martj42/Elo name
# (so both sources share the same canonical form in our system)
_ELONET_OVERRIDES = {
    "United States":    "United States",
    "Korea Republic":   "South Korea",
    "Korea DPR":        "Korea DPR",
    "IR Iran":          "IR Iran",
    "China PR":         "China",
    "Côte d'Ivoire":    "Ivory Coast",
    "Cote d'Ivoire":    "Ivory Coast",
    "Czech Republic":   "Czech Republic",
    "Türkiye":          "Turkey",
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
    "Trinidad & Tobago":  "Trinidad and Tobago",
    "Cape Verde Islands": "Cape Verde",
}


def fetch_ratings() -> dict[str, float]:
    """Download eloratings.net TSV and parse into {team: elo} dict."""
    resp = requests.get(ELONET_URL, timeout=15)
    resp.raise_for_status()

    ratings: dict[str, float] = {}
    for line in resp.text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            # Format: Rank  Team  Elo  [+/-  Matches  ...]
            rank_str, team_raw, elo_str = parts[0], parts[1], parts[2]
            int(rank_str)           # skip header / blank rows
            elo = float(elo_str)
            team = _ELONET_OVERRIDES.get(team_raw, team_raw)
            team = normalize_team(team)
            ratings[team] = elo
        except (ValueError, IndexError):
            continue

    return ratings


def main() -> None:
    print(f"Fetching ratings from {ELONET_URL} ...")
    new_ratings = fetch_ratings()
    print(f"  {len(new_ratings)} teams downloaded.")

    if not ELO_PATH.exists():
        print(f"ERROR: {ELO_PATH} not found. Run scripts/build_elo.py first.")
        sys.exit(1)

    with open(ELO_PATH, encoding="utf-8") as f:
        old_data = json.load(f)

    old_count = len(old_data)

    # Merge: eloratings.net wins for any team present in both.
    # Teams only in martj42 data (tiny nations) are kept.
    merged = dict(old_data)
    merged.update(new_ratings)

    with open(ELO_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"  Updated {len(new_ratings)} ratings from eloratings.net.")
    print(f"  Total teams in model: {old_count} -> {len(merged)}")

    # Sanity check the key WC2026 match
    print()
    print("Sanity check — key WC2026 ratings:")
    for team in ["Germany", "Ecuador", "France", "England", "Brazil",
                 "Argentina", "Spain", "United States", "Turkey", "Ivory Coast"]:
        r = merged.get(team, merged.get(normalize_team(team)))
        print(f"  {team:<25} {r:.1f}" if r else f"  {team:<25} NOT FOUND")

    # Show Germany vs Ecuador gap
    ger = merged.get("Germany", 0)
    ecu = merged.get("Ecuador", 0)
    if ger and ecu:
        diff = ger - ecu
        import math
        p_ger = 1 / (1 + 10 ** (-diff / 400))
        p_draw = 0.22 if diff > 150 else (0.24 if diff > 100 else 0.26)
        p_ger_win = p_ger * (1 - p_draw)
        p_ecu_win = (1 - p_ger) * (1 - p_draw)
        print()
        print(f"Ecuador vs Germany (neutral, eloratings.net):")
        print(f"  Elo gap: {diff:+.0f} pts")
        print(f"  Germany win: {p_ger_win*100:.1f}%  Draw: {p_draw*100:.0f}%  Ecuador win: {p_ecu_win*100:.1f}%")

    print(f"\nSaved -> {ELO_PATH}")


if __name__ == "__main__":
    main()
